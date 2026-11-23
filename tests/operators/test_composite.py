from tests.operators.inspection import inspect_result
import json
import unittest
from unittest.mock import patch

from src.operators import RMSNorm, Softmax, TensorSpec, simulate_composite
from src.hardware.vector import VectorExecution, load_vector_execution
from src.hardware.ascend import AscendSpec
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.hardware.tsp_cim import TSPCIMSpec
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.operator_workload import parse_composite_workload
from tests.operators.test_gemm import specs
from tests.operators.test_vector import synthetic, execution, raw


def task(kind, width=17, rows=1):
    x = TensorSpec("x", (rows, width), "FP32")
    return RMSNorm(x, TensorSpec("weight", (width,), "FP32")) if kind == "rmsnorm" else Softmax(x)


class CompositeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def test_four_paths_do_not_call_gemm_or_standalone_vector(self):
        with patch("src.operators.vector.simulate_vector", side_effect=AssertionError("standalone Vector")), \
             patch("src.core.ascend.simulator.AscendSimulator.simulate", side_effect=AssertionError("GEMM")), \
             patch("src.core.tpu.simulator.TPUSimulator.simulate", side_effect=AssertionError("GEMM")), \
             patch("src.core.tsp.simulator.TSPSimulator.simulate", side_effect=AssertionError("GEMM")), \
             patch("src.core.tsp_cim.simulator.TSPCIMSimulator.simulate", side_effect=AssertionError("GEMM")), \
             patch("src.core.cim.macro_model.CIMMacroModel.compute", side_effect=AssertionError("Macro")):
            for arch, spec in self.hardware.items():
                for kind in ("rmsnorm", "softmax"):
                    with self.subTest(arch=arch, kind=kind):
                        op = task(kind, 1025, 2)
                        a = inspect_result(simulate_composite(op, arch, spec, include_details=True))
                        b = inspect_result(simulate_composite(op, arch, spec, mode="timing", include_details=True))
                        self.assertEqual(a["action_ledger"], b["action_ledger"])
                        self.assertEqual(a["resource_requirements"], b["resource_requirements"])
                        self.assertIsNone(a["latency"]["cycle_count"])
                        self.assertGreater(b["latency"]["cycle_count"], 0)
                        self.assertEqual(b["operator"]["outputs"][0]["logical_bytes"], 8200)
                        self.assertFalse(any(e["category"] in {"matrix", "cim_internal"} for e in b["action_ledger"]["entries"]))
                        t = b["native_report"]["timing"]
                        self.assertEqual(sum(t["elapsed_cycles"]["by_stage"].values()), t["total_cycles"])
                        self.assertEqual(t["service_cycles"]["by_stage"], t["elapsed_cycles"]["by_stage"])
                        json.dumps(b, allow_nan=False)

    def test_arithmetic_counts_are_width_minus_one_reductions_for_every_tiling(self):
        for kind in ("rmsnorm", "softmax"):
            for width in (1, 2, 3, 17, 1025):
                rows = 3
                for tile in (1, 2, 8, 1024):
                    with self.subTest(kind=kind, width=width, tile=tile):
                        r = inspect_result(simulate_composite(task(kind, width, rows), "tsp", self.hardware["tsp"], tile_elements=tile, include_details=True))
                        actual = r["work_counts"]["logical"]["operation_counts"]
                        expected = ({"fp32_mul": rows * (3 * width + 1), "fp32_add": rows * width, "fp32_rsqrt": rows}
                                    if kind == "rmsnorm" else
                                    {"fp32_max": rows * (width - 1), "fp32_sub": rows * width, "fp32_exp": rows * width,
                                     "fp32_add": rows * (width - 1), "fp32_rcp": rows, "fp32_mul": rows * width})
                        self.assertEqual(actual, {k: v for k, v in expected.items() if v})

    def test_hand_single_element_timing_and_accesses(self):
        for arch in ("ascend", "tpu", "tsp"):
            for kind in ("rmsnorm", "softmax"):
                with self.subTest(arch=arch, kind=kind):
                    kwargs = {"mem_slices": [0], "streams": 1} if arch == "tsp" else {}
                    r = inspect_result(simulate_composite(task(kind, 1), arch, synthetic(arch), mode="timing",
                                           execution=execution(arch), **kwargs, include_details=True))
                    expected = {"rmsnorm": 24, "softmax": 22} if arch == "tsp" else {"rmsnorm": 26, "softmax": 25}
                    self.assertEqual(r["latency"]["cycle_count"], expected[kind])
                    c = r["native_report"]["counts"]
                    if arch == "tsp":
                        reads, writes = (36, 28) if kind == "rmsnorm" else (32, 24)
                        self.assertEqual((c["mem_read_bytes"], c["mem_write_bytes"]), (reads, writes))
                        self.assertEqual((c["input_stream_bytes"], c["output_stream_bytes"]), (reads // 4 * 320, writes // 4 * 320))
                    else:
                        self.assertEqual(c["hbm_read_bytes"], 12)
                        self.assertEqual(c["hbm_write_bytes"], 4 if kind == "rmsnorm" else 8)

    def test_external_traffic_and_capacity_do_not_materialize_all_intermediates(self):
        for arch in ("ascend", "tpu"):
            for kind in ("rmsnorm", "softmax"):
                op = task(kind, 17, 3)
                r = inspect_result(simulate_composite(op, arch, self.hardware[arch], tile_elements=8, include_details=True))
                c, m = r["native_report"]["counts"], r["resource_requirements"]["memory"]
                self.assertEqual(c["hbm_read_bytes"], 12 * 51)
                self.assertEqual(c["hbm_write_bytes"], (4 if kind == "rmsnorm" else 8) * 51)
                self.assertEqual(m["hbm"]["logical_footprint_bytes"], 8 * 51 + (4 * 17 if kind == "rmsnorm" else 0))
                local = m["ub" if arch == "ascend" else "vmem"]
                self.assertEqual(local["allocated_bytes"], 68 if kind == "rmsnorm" else 40)

    def test_reduction_dependency_is_not_flat_throughput(self):
        for arch in ("ascend", "tpu", "tsp"):
            for kind, rounds in (("rmsnorm", 8), ("softmax", 16)):
                cfg = execution(arch).config
                kwargs = dict(mode="timing", tile_elements=8)
                a = inspect_result(simulate_composite(task(kind), arch, synthetic(arch), execution=VectorExecution(cfg, arch), **kwargs, include_details=True))
                cfg["composite"]["parameters"]["reduction_round_cycles"]["value"] = 5
                b = inspect_result(simulate_composite(task(kind), arch, synthetic(arch), execution=VectorExecution(cfg, arch), **kwargs, include_details=True))
                self.assertEqual(b["latency"]["cycle_count"] - a["latency"]["cycle_count"], rounds * 4)
                self.assertEqual(a["action_ledger"], b["action_ledger"])
                self.assertEqual(b["native_report"]["timing"]["elapsed_cycles"]["by_stage"]["reduction_dependency"], rounds * 4)

    def test_batch_and_head_axes_scale_rows_without_expanding_schedule(self):
        for kind in ("rmsnorm", "softmax"):
            a = inspect_result(simulate_composite(task(kind), "tpu", self.hardware["tpu"], mode="timing", tile_elements=8, include_details=True))
            op = task(kind, rows=6)
            if kind == "softmax":
                op = Softmax(TensorSpec("x", (2, 3, 17), "FP32"))
            b = inspect_result(simulate_composite(op, "tpu", self.hardware["tpu"], mode="timing", tile_elements=8, include_details=True))
            self.assertEqual(b["latency"]["cycle_count"], 6 * a["latency"]["cycle_count"])
            self.assertEqual(b["native_report"]["schedule"]["groups_per_row"], a["native_report"]["schedule"]["groups_per_row"])
            for key, value in a["native_report"]["counts"].items():
                self.assertEqual(b["native_report"]["counts"][key], 6 * value)
        r = inspect_result(simulate_composite(task("softmax", 1, 1000000), "tpu", self.hardware["tpu"], include_details=True))
        self.assertLess(len(r["native_report"]["schedule"]["groups_per_row"]), 10)

    def test_capacity_fallback_accounts_row_state_and_complete_tensors(self):
        for arch in ("ascend", "tpu"):
            for kind, tile in (("rmsnorm", 11), ("softmax", 22)):
                r = inspect_result(simulate_composite(task(kind, 33), arch, synthetic(arch), include_details=True))
                self.assertEqual(r["execution_contract"]["selected_tile_elements"], tile)
            cfg = raw(arch)
            local = cfg["memory"]["UB"] if arch == "ascend" else cfg["parameters"]
            local["usable_bytes" if arch == "ascend" else "vmem_usable_bytes"]["value"] = 11
            cls = AscendSpec if arch == "ascend" else TPUSpec
            for kind in ("rmsnorm", "softmax"):
                with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
                    simulate_composite(task(kind), arch, cls(cfg), include_details=True)
            cfg = raw(arch)
            (cfg["chip"] if arch == "ascend" else cfg["parameters"])["hbm_usable_bytes"]["value"] = 135
            with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
                simulate_composite(task("softmax"), arch, cls(cfg), tile_elements=1, include_details=True)
        spec = self.hardware["tsp"]
        # 136 bytes X/Y + 8 row state + exactly 4 bytes scratch available.
        reserve = spec.mem_slice_capacity_bytes - 148
        r = inspect_result(simulate_composite(task("softmax"), "tsp", spec, mem_slices=[0], reserved_bytes=reserve, include_details=True))
        self.assertEqual(r["execution_contract"]["selected_tile_elements"], 1)
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_composite(task("softmax"), "tsp", spec, mem_slices=[0], reserved_bytes=reserve + 1, include_details=True)

    def test_only_used_costs_are_required_and_count_runs_with_unknowns(self):
        for arch, spec in self.hardware.items():
            cfg = load_vector_execution(arch).config
            for value in cfg["composite"]["parameters"].values():
                value.update(value=None, status="unknown")
            profile = VectorExecution(cfg, arch)
            for kind in ("rmsnorm", "softmax"):
                simulate_composite(task(kind), arch, spec, execution=profile, include_details=True)
                with self.assertRaisesRegex(ValueError, "Missing"):
                    simulate_composite(task(kind), arch, spec, execution=profile, mode="timing", include_details=True)
            # Width-one RMSNorm has no reduction and does not use SUB/MAX/RCP.
            simulate_composite(task("rmsnorm", 1), arch, spec, execution=profile, mode="timing", include_details=True)
            cfg = load_vector_execution(arch).config
            cfg["composite"]["parameters"]["fp32_max_cost"].update(value=None, status="unknown")
            simulate_composite(task("softmax", 1), arch, spec, execution=VectorExecution(cfg, arch), mode="timing", include_details=True)
            with self.assertRaisesRegex(ValueError, "Missing"):
                simulate_composite(task("softmax", 2), arch, spec, execution=VectorExecution(cfg, arch), mode="timing", include_details=True)

    def test_unknown_io_clock_and_unrelated_matrix_services(self):
        for arch in ("ascend", "tpu", "tsp"):
            cfg = raw(arch)
            cls = {"ascend": AscendSpec, "tpu": TPUSpec, "tsp": TSPSpec}[arch]
            if arch == "ascend":
                cfg["cube"]["macs_per_cycle"].update(value=None, status="unknown")
                io = cfg["chip"]["hbm_bandwidth_bytes_per_second"]
            else:
                cfg["parameters"]["int8_macs_per_cell_cycle" if arch == "tpu" else "mxm_first_cycles"].update(value=None, status="unknown")
                io = cfg["parameters"]["vmem_bandwidth_bytes_per_second" if arch == "tpu" else "mem_read_bpc"]
            simulate_composite(task("softmax"), arch, cls(cfg), mode="timing", include_details=True)
            io.update(value=None, status="unknown")
            simulate_composite(task("softmax"), arch, cls(cfg), include_details=True)
            with self.assertRaisesRegex(ValueError, "Missing"):
                simulate_composite(task("softmax"), arch, cls(cfg), mode="timing", include_details=True)
        cfg = raw("tsp")
        cfg["parameters"]["clock_ghz"].update(value=None, status="unknown")
        r = inspect_result(simulate_composite(task("softmax"), "tsp", TSPSpec(cfg), mode="timing", include_details=True))
        self.assertEqual(r["latency"]["status"], "cycles_only")

    def test_cim_shares_vxm_and_cost_budget(self):
        cfg = TSPCIMConfigLoader().load_hardware_config()
        cfg["_native_reference_config"] = synthetic("tsp").config
        spec = TSPCIMSpec(cfg)
        for kind in ("rmsnorm", "softmax"):
            a = inspect_result(simulate_composite(task(kind), "tsp", spec.native_reference_spec, mode="timing", tile_elements=8, include_details=True))
            b = inspect_result(simulate_composite(task(kind), "tsp_cim", spec, mode="timing", tile_elements=8, hemisphere=1, include_details=True))
            self.assertEqual(a["latency"], b["latency"])
            self.assertEqual(a["native_report"]["counts"], b["native_report"]["counts"])
            self.assertFalse(b["boundary"]["macro_execution_included"])
        profile = execution("tsp").config
        a = inspect_result(simulate_composite(task("softmax", 1), "tsp", synthetic("tsp"), mode="timing", execution=VectorExecution(profile, "tsp"), include_details=True))
        profile["composite"]["parameters"]["fp32_rcp_cost"]["value"] = 15
        b = inspect_result(simulate_composite(task("softmax", 1), "tsp", synthetic("tsp"), mode="timing", execution=VectorExecution(profile, "tsp"), include_details=True))
        self.assertEqual(b["latency"]["cycle_count"] - a["latency"]["cycle_count"], 2)
        self.assertEqual(a["action_ledger"], b["action_ledger"])

    def test_strict_shapes_axes_and_workloads(self):
        for x in (TensorSpec("x", (), "FP32"), TensorSpec("x", (2,), "INT32")):
            with self.assertRaises(ValueError):
                Softmax(x)
        x, w = TensorSpec("x", (2, 17), "FP32"), TensorSpec("w", (17,), "FP32")
        for axis in (0, True, -2):
            with self.assertRaises(ValueError):
                Softmax(x, axis)
        for weight in (TensorSpec("x", (17,), "FP32"), TensorSpec("w", (1, 17), "FP32"), TensorSpec("w", (16,), "FP32")):
            with self.assertRaises(ValueError):
                RMSNorm(x, weight)
        for eps in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RMSNorm(x, w, eps)
        valid = dict(schema_version=1, operation="softmax", inputs=[dict(name="x", shape=[2, 17], dtype="FP32")])
        self.assertEqual(parse_composite_workload(valid).rows, 2)
        for changes in (dict(axis=0), dict(epsilon=1e-5), dict(mask=True), dict(schema_version=True), dict(inputs=[])):
            with self.assertRaises(ValueError):
                parse_composite_workload({**valid, **changes})
        for options in (dict(tile_elements=True), dict(tile_elements=0), dict(mem_slices=[0, 0]), dict(streams=0), dict(hemisphere=2)):
            with self.assertRaises(ValueError):
                simulate_composite(task("softmax"), "tsp", self.hardware["tsp"], **options, include_details=True)
