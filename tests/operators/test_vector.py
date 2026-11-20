from tests.operators.inspection import inspect_result
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from src.operators import TensorSpec, Vector, simulate_vector
from src.hardware.vector import VectorExecution, load_vector_execution
from src.hardware.ascend import AscendSpec
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.hardware.tsp_cim import TSPCIMSpec
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.operator_workload import parse_vector_workload
from tests.operators.test_gemm import specs


ROOT = Path(__file__).resolve().parents[2]


def task(operation="fp32_add", n=17):
    arity = 1 if operation in {"fp32_exp", "fp32_rsqrt"} else 2
    dtype = "INT32" if operation == "int32_add" else "FP32"
    return Vector(operation, tuple(TensorSpec(f"x{i}", (n,), dtype) for i in range(arity)))


def raw(architecture):
    path = {"ascend": "ascend910_reference", "tpu": "tpu_v4_reference", "tsp": "tsp"}[architecture]
    return json.loads((ROOT / f"configs/hardware/{path}.json").read_text(encoding="utf-8"))


def synthetic(architecture):
    cfg = raw(architecture)
    if architecture == "ascend":
        cfg["clock_freq_ghz"]["value"] = 1
        cfg["chip"]["hbm_bandwidth_bytes_per_second"]["value"] = 16e9 * cfg["chip"]["architectural_cores"]["value"]
        cfg["memory"]["UB"]["usable_bytes"]["value"] = 96
        cfg["vector"]["base_ops_per_cycle"]["value"] = 4
        return AscendSpec(cfg)
    values = cfg["parameters"]
    if architecture == "tpu":
        for key, value in (("clock_hz", 10**9), ("hbm_bandwidth_bytes_per_second", 32e9),
                           ("vmem_bandwidth_bytes_per_second", 32e9), ("vmem_usable_bytes", 96),
                           ("vector_base_ops_per_cycle", 4)):
            values[key]["value"] = value
        return TPUSpec(cfg)
    for key, value in (("clock_ghz", 1), ("mem_read_bpc", 16), ("mem_write_bpc", 16), ("vxm_base_ops_per_cycle", 4)):
        values[key]["value"] = value
    return TSPSpec(cfg)


def execution(architecture):
    config = copy.deepcopy(load_vector_execution(architecture).config)
    for p in config["parameters"].values():
        p["value"] = 32
    return VectorExecution(config, architecture)


class VectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def test_all_five_operations_run_without_matrix_kernels(self):
        with patch("src.core.ascend.simulator.AscendSimulator.simulate", side_effect=AssertionError("GEMM called")), \
             patch("src.core.tpu.simulator.TPUSimulator.simulate", side_effect=AssertionError("GEMM called")), \
             patch("src.core.tsp.simulator.TSPSimulator.simulate", side_effect=AssertionError("GEMM called")), \
             patch("src.core.tsp_cim.simulator.TSPCIMSimulator.simulate", side_effect=AssertionError("GEMM called")), \
             patch("src.core.cim.macro_model.CIMMacroModel.compute", side_effect=AssertionError("Macro called")):
            for arch, spec in self.hardware.items():
                for operation in ("int32_add", "fp32_add", "fp32_mul", "fp32_exp", "fp32_rsqrt"):
                    with self.subTest(arch=arch, operation=operation):
                        op = task(operation, 1025)
                        counted = inspect_result(simulate_vector(op, arch, spec, include_details=True))
                        timed = inspect_result(simulate_vector(op, arch, spec, mode="timing", include_details=True))
                        self.assertEqual(counted["action_ledger"], timed["action_ledger"])
                        self.assertEqual(counted["native_report"]["memory"], timed["native_report"]["memory"])
                        self.assertIsNone(counted["latency"]["cycle_count"])
                        self.assertGreater(timed["latency"]["cycle_count"], 0)
                        self.assertEqual(timed["operator"]["outputs"][0]["logical_bytes"], 4100)
                        self.assertEqual(timed["work_counts"]["logical"]["operation_counts"], {operation: 1025})
                        self.assertIsNone(timed["work_counts"]["issued"]["operations"])
                        timing = timed["native_report"]["timing"]
                        self.assertEqual(sum(timing["elapsed_cycles"]["by_stage"].values()), timing["total_cycles"])
                        self.assertEqual(timing["service_cycles"]["by_stage"], timing["elapsed_cycles"]["by_stage"])
                        entries = timed["action_ledger"]["entries"]
                        self.assertFalse(any(a["category"] in {"matrix", "cim_internal"} for a in entries))
                        self.assertEqual(next(a["count"] for a in entries if a["id"] == operation), 1025)
                        json.dumps(timed, allow_nan=False)

    def test_hand_schedule_includes_both_dma_endpoints_and_tail(self):
        for arch in ("ascend", "tpu", "tsp"):
            spec = synthetic(arch)
            policy = dict(tile_elements=8, execution=execution(arch))
            if arch == "tsp":
                policy.update(mem_slices=[0], streams=1)
            for operation, expected in (("fp32_add", 20 if arch == "tsp" else 28),
                                        ("fp32_exp", 74 if arch == "tsp" else 80)):
                with self.subTest(arch=arch, operation=operation):
                    report = inspect_result(simulate_vector(task(operation), arch, spec, mode="timing", **policy, include_details=True))
                    self.assertEqual(report["latency"]["cycle_count"], expected)
                    self.assertEqual(report["native_report"]["schedule"]["tile_groups"],
                                     [{"elements": 8, "repetitions": 2}, {"elements": 1, "repetitions": 1}])
                    self.assertEqual(report["execution_contract"]["requested_tile_elements"], 8)
                    if arch == "tsp" and operation == "fp32_add":
                        counts = report["native_report"]["counts"]
                        self.assertEqual((counts["mem_read_bytes"], counts["mem_write_bytes"]), (136, 68))
                        self.assertEqual((counts["input_stream_bytes"], counts["output_stream_bytes"]), (1920, 960))

    def test_cim_and_native_share_one_vxm_even_across_hemispheres(self):
        c = TSPCIMConfigLoader().load_hardware_config()
        c["_native_reference_config"] = synthetic("tsp").config
        cim = TSPCIMSpec(c)
        op = task()
        kwargs = dict(mode="timing", tile_elements=8, mem_slices=[0], streams=1)
        a = inspect_result(simulate_vector(op, "tsp", cim.native_reference_spec, **kwargs, include_details=True))
        for hem in (0, 1):
            b = inspect_result(simulate_vector(op, "tsp_cim", cim, hemisphere=hem, **kwargs, include_details=True))
            self.assertEqual(a["latency"], b["latency"])
            self.assertEqual(a["native_report"]["counts"], b["native_report"]["counts"])
            self.assertFalse(b["boundary"]["macro_execution_included"])
            self.assertEqual(b["boundary"]["selected_hemisphere"], hem)
            self.assertEqual(b["coverage"]["offchip_memory"], "outside_native_boundary")

    def test_local_fallback_and_full_tensor_capacity(self):
        for arch in ("ascend", "tpu"):
            result = inspect_result(simulate_vector(task(), arch, synthetic(arch), tile_elements=999, include_details=True))
            self.assertEqual(result["execution_contract"]["selected_tile_elements"], 8)
            cfg = raw(arch)
            if arch == "ascend":
                cfg["memory"]["UB"]["usable_bytes"]["value"] = 11
                spec = AscendSpec(cfg)
            else:
                cfg["parameters"]["vmem_usable_bytes"]["value"] = 11
                spec = TPUSpec(cfg)
            with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
                simulate_vector(task(), arch, spec, include_details=True)
            cfg = raw(arch)
            (cfg["chip"] if arch == "ascend" else cfg["parameters"])["hbm_usable_bytes"]["value"] = 203
            spec = AscendSpec(cfg) if arch == "ascend" else TPUSpec(cfg)
            with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
                simulate_vector(task(), arch, spec, tile_elements=1, include_details=True)
        s = self.hardware["tsp"]
        n = s.mem_slice_capacity_bytes // 12
        simulate_vector(task(n=n), "tsp", s, mem_slices=[0], include_details=True)
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_vector(task(n=n+1), "tsp", s, mem_slices=[0], tile_elements=1, include_details=True)
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_vector(task(n=n), "tsp", s, mem_slices=[0], reserved_bytes=12, include_details=True)

    def test_only_used_services_are_required_and_count_needs_no_rates(self):
        for arch in ("ascend", "tpu", "tsp"):
            cfg = raw(arch)
            if arch == "ascend":
                params = cfg["vector"]
                exp, basic = "fp32_exp_cost", "fp32_basic_op_cost"
                cfg["cube"]["macs_per_cycle"].update(value=None, status="unknown")
                cls = AscendSpec
            else:
                params = cfg["parameters"]
                prefix = "vector" if arch == "tpu" else "vxm"
                exp, basic = f"{prefix}_fp32_exp_cost", f"{prefix}_fp32_basic_op_cost"
                key = "int8_macs_per_cell_cycle" if arch == "tpu" else "mxm_first_cycles"
                params[key].update(value=None, status="unknown")
                cls = TPUSpec if arch == "tpu" else TSPSpec
            params[exp].update(value=None, status="unknown")
            spec = cls(cfg)
            simulate_vector(task(), arch, spec, mode="timing", include_details=True)
            simulate_vector(task("fp32_exp"), arch, spec, mode="count", include_details=True)
            with self.assertRaisesRegex(ValueError, "missing"):
                simulate_vector(task("fp32_exp"), arch, spec, mode="timing", include_details=True)
            params[exp].update(value=15, status="assumed")
            params[basic].update(value=None, status="unknown")
            simulate_vector(task("fp32_exp"), arch, cls(cfg), mode="timing", include_details=True)

    def test_missing_io_and_clock_are_explicit(self):
        for arch in ("ascend", "tpu", "tsp"):
            cfg = raw(arch)
            if arch == "ascend":
                cfg["chip"]["hbm_bandwidth_bytes_per_second"].update(value=None, status="unknown")
                spec = AscendSpec(cfg)
            else:
                name = "vmem_bandwidth_bytes_per_second" if arch == "tpu" else "mem_read_bpc"
                cfg["parameters"][name].update(value=None, status="unknown")
                spec = TPUSpec(cfg) if arch == "tpu" else TSPSpec(cfg)
            simulate_vector(task(), arch, spec, include_details=True)
            with self.assertRaisesRegex(ValueError, "Missing"):
                simulate_vector(task(), arch, spec, mode="timing", include_details=True)
        cfg = raw("tsp")
        cfg["parameters"]["clock_ghz"].update(value=None, status="unknown")
        result = inspect_result(simulate_vector(task(), "tsp", TSPSpec(cfg), mode="timing", include_details=True))
        self.assertEqual(result["latency"]["status"], "cycles_only")
        self.assertIsNone(result["latency"]["latency_s"])

    def test_shared_budget_changes_only_compute_and_exp_cost_is_work_not_exp_count(self):
        op = task("fp32_exp", 80)
        cfg = raw("tsp")
        a = inspect_result(simulate_vector(op, "tsp", TSPSpec(cfg), mode="timing", include_details=True))
        cfg["parameters"]["vxm_base_ops_per_cycle"]["value"] = 160
        b = inspect_result(simulate_vector(op, "tsp", TSPSpec(cfg), mode="timing", include_details=True))
        self.assertEqual(a["action_ledger"], b["action_ledger"])
        self.assertEqual(a["native_report"]["counts"]["fp32_exp"], 80)
        self.assertEqual(a["native_report"]["timing"]["service_cycles"]["by_stage"]["vector_compute"], 4)
        self.assertEqual(b["native_report"]["timing"]["service_cycles"]["by_stage"]["vector_compute"], 8)

    def test_invalid_ops_shapes_aliases_and_allocations_fail(self):
        for op, tensors in (("unknown", (TensorSpec("a", (1,), "FP32"),)),
                            ("fp32_add", (TensorSpec("a", (1,), "FP32"),)),
                            ("fp32_exp", (TensorSpec("a", (1,), "INT8"),)),
                            ("fp32_add", (TensorSpec("a", (1,), "FP32"), TensorSpec("b", (2,), "FP32"))),
                            ("fp32_add", (TensorSpec("a", (1,), "FP32"), TensorSpec("a", (1,), "FP32")))):
            with self.assertRaises(ValueError):
                Vector(op, tensors)
        for options in ({"mem_slices": []}, {"mem_slices": [0, 0]}, {"mem_slices": [True]},
                        {"streams": 0}, {"reserved_bytes": -1}, {"hemisphere": 2},
                        {"tile_elements": True}, {"tile_elements": 0}):
            with self.assertRaises(ValueError):
                simulate_vector(task(), "tsp", self.hardware["tsp"], **options, include_details=True)
        valid = {"schema_version": 1, "operation": "vector", "op": "fp32_exp",
                 "inputs": [{"name": "x", "shape": [2], "dtype": "FP32"}]}
        self.assertEqual(parse_vector_workload(valid).numel, 2)
        for changes in ({"schema_version": True}, {"unknown": 1}, {"inputs": []}, {"op": "softmax"}):
            with self.assertRaises(ValueError):
                parse_vector_workload({**valid, **changes})

    def test_startup_policy_and_large_tile_count_are_aggregated(self):
        profile = execution("tsp").config
        profile["policies"]["tsp"]["transfer_startup_cycles"] = 2
        profile["policies"]["tsp"]["compute_startup_cycles"] = 3
        report = inspect_result(simulate_vector(task(), "tsp", synthetic("tsp"), mode="timing", tile_elements=8,
                                 mem_slices=[0], streams=1, execution=VectorExecution(profile, "tsp"), include_details=True))
        self.assertEqual(report["latency"]["cycle_count"], 20 + 3 * (3*2 + 3))
        report = inspect_result(simulate_vector(task(n=1000000), "tpu", self.hardware["tpu"], tile_elements=1, include_details=True))
        self.assertEqual(report["native_report"]["schedule"]["tile_groups"], [{"elements": 1, "repetitions": 1000000}])
