"""Independent work, memory, view and boundary checks for Decode projections."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.tpu import TPUSpec
from src.operators import (TensorSpec, RMSNormINT8, QKVProjection, AttentionOutputProjection,
                           simulate_block, simulate_gemm, simulate_composite, plan_connection)
from src.operators.blocks import block_steps
from src.operators.primitives import Gemm
from src.operators.projections import _StorageBoundary
from src.core.block_elementwise import make_path, simulate
from tests.operators.test_gemm import specs
from tests.operators.test_vector import raw, synthetic, execution
from src.utils.operator_workload import parse_block_workload

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("rmsnorm_int8", "qkv_projection", "attention_output_projection")


def config(kind):
    return json.loads((ROOT / f"configs/workload/operators/{kind}.json").read_text(encoding="utf-8"))


class ProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_block(self, kind, arch="tpu", **kw):
        return simulate_block(parse_block_workload(config(kind)), arch, self.hardware[arch], include_details=True, **kw)

    def test_all_architectures_count_timing_and_serial_aggregation(self):
        for arch in self.hardware:
            for kind in KINDS:
                with self.subTest(arch=arch, kind=kind):
                    count = self.run_block(kind, arch)
                    timed = self.run_block(kind, arch, mode="timing")
                    self.assertEqual(count.action_ledger, timed.action_ledger)
                    self.assertEqual(count.native_report["memory"], timed.native_report["memory"])
                    self.assertIsNone(count.cycle_count)
                    self.assertGreater(timed.cycle_count, 0)
                    self.assertFalse(timed.boundary["chip_execution_included"])
                    self.assertEqual(timed.cycle_count, sum(n["cycle_count"] for n in timed.native_report["schedule"]["nodes"]))
                    json.dumps(timed.to_detailed_dict(), allow_nan=False)

    def test_qkv_hand_counts_and_gqa_output_shapes(self):
        for arch in self.hardware:
            r = self.run_block("qkv_projection", arch)
            self.assertEqual([t.shape for t in r.operator.outputs], [(2, 4, 1, 7), (2, 2, 1, 7), (2, 2, 1, 7)])
            self.assertEqual(r.work_counts["logical"]["useful_macs"], 2 * 17 * (4 + 2 * 2) * 7)
            self.assertEqual(r.work_counts["logical"]["operations"], 2 * 2 * 17 * 56)
            self.assertEqual(r.work_counts["logical"]["output_bytes"], 112)
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 112, "boundary_to_int8": 112})
            self.assertFalse(dict(r.operator.attributes)["positional_encoding_included"])

    def test_qkv_persistent_outputs_and_exact_global_peak(self):
        r = self.run_block("qkv_projection").native_report
        life = r["memory"]["lifetime_validation"]
        self.assertEqual([s["storage_bytes"] for s in life["steps"]], [1210, 1266, 1154, 1182, 1182, 1210])
        self.assertEqual(life["peak_bytes"], 1266)
        self.assertTrue({"query", "key", "value"} <= set(life["steps"][-1]["live_tensors"]))
        self.assertEqual(len(life["lifetimes"]), 10)  # 4 inputs, 3 accumulators, 3 INT8 outputs.
        actions = r["action_counts"]
        self.assertEqual(sum(a["count"] for k, a in actions.items() if k.endswith("hbm_read_bytes")), 1502)
        self.assertEqual(sum(a["count"] for k, a in actions.items() if k.endswith("hbm_write_bytes")), 560)

    def test_output_projection_concat_is_one_backing(self):
        r = self.run_block("attention_output_projection")
        self.assertEqual(r.work_counts["logical"]["useful_macs"], 2 * 28 * 17)
        self.assertEqual(r.operator.outputs[0].shape, (2, 17))
        self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 34, "boundary_to_int8": 34})
        life = r.native_report["memory"]["lifetime_validation"]
        self.assertEqual([s["storage_bytes"] for s in life["steps"]], [668, 702])
        self.assertEqual(len(life["lifetimes"]), 4)
        self.assertEqual(r.native_report["schedule"]["storage_views"],
                         [{"tensor": "context", "matrix_shape": [2, 28], "head_shape": [2, 4, 1, 7], "binding": "same_input_backing"}])
        self.assertFalse(any("reshape" in a.id or "transpose" in a.id for a in r.action_ledger.entries))

    def test_native_gemm_actions_roles_and_counts_preserved(self):
        for kind in KINDS[1:]:
            op = parse_block_workload(config(kind))
            for arch, spec in self.hardware.items():
                result = self.run_block(kind, arch)
                actions = {a.id: a for a in result.action_ledger.entries}
                for node, child in block_steps(op):
                    if not isinstance(child, Gemm):
                        continue
                    native = simulate_gemm(child, arch, spec)
                    for a in native.action_ledger.entries:
                        b = actions[node + "." + a.id]
                        self.assertEqual((a.count, a.unit, a.dtype, a.role), (b.count, b.unit, b.dtype, b.role))
                        self.assertEqual(b.summarizes, tuple(node + "." + v for v in a.summarizes))

    def test_rmsnorm_fp32_arithmetic_and_boundary_counts(self):
        for arch in self.hardware:
            result = self.run_block("rmsnorm_int8", arch)
            self.assertEqual(result.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 34, "boundary_to_int8": 34})
            self.assertEqual(result.work_counts["logical"]["operation_counts"], {"fp32_mul": 104, "fp32_add": 34, "fp32_rsqrt": 2})
            self.assertTrue(all(a.dtype == "FP32" for a in result.action_ledger.entries if a.id.startswith("rmsnorm.") and a.category in {"vector", "nonlinear"}))
            child = block_steps(parse_block_workload(config("rmsnorm_int8")))[1][1]
            native = simulate_composite(child, arch, self.hardware[arch])
            actions = {a.id: a for a in result.action_ledger.entries}
            for a in native.action_ledger.entries:
                self.assertEqual(actions["rmsnorm." + a.id].count, a.count)

    def test_direct_fp32_conversion_uses_destination_without_extra_scratch(self):
        op = _StorageBoundary(TensorSpec("x", (7,), "INT8"), TensorSpec("y", (7,), "FP32"))
        spec, profile = synthetic("ascend"), execution("ascend")
        path = make_path(op, "ascend", spec, profile, "timing", 35, 1000, None, None)
        result = simulate(op, path, profile, load_block_execution(), "timing").report
        self.assertEqual(result["memory"]["ub"]["allocated_bytes"], 35)
        self.assertEqual(result["memory"]["ub"]["scratch_tile_bytes"], 0)
        # Local read/write: 1+1; compute: 2; 16 B/cycle HBM DMA: 1+2 = 7.
        self.assertEqual(result["timing"]["total_cycles"], 7)
        self.assertEqual(result["counts"], {"hbm_read_bytes": 7, "ub_dma_write_bytes": 7, "ub_read_bytes": 7,
                                          "ub_write_bytes": 28, "boundary_to_fp32": 7, "hbm_write_bytes": 28, "ub_dma_read_bytes": 28})

    def test_resident_norm_accounts_for_striped_scalars_and_scratch(self):
        op = RMSNormINT8(TensorSpec("x", (1, 7), "INT8"), TensorSpec("w", (7,), "FP32"))
        for arch in ("tsp", "tsp_cim"):
            result = simulate_block(op, arch, self.hardware[arch], mem_slices=[0, 1, 2], streams=2,
                                    reserved_bytes=17, tile_elements=3, include_details=True)
            life = result.native_report["memory"]["lifetime_validation"]
            norm = life["steps"][1]
            self.assertEqual((norm["storage_bytes"], norm["scratch_bytes"], norm["peak_bytes"]), (117, 18, 135))
            self.assertEqual(life["reserved_bytes"], 18)
            self.assertEqual(life["peak_bytes"], 135)

    def test_global_hbm_capacity_rejects_qkv_even_when_native_nodes_fit(self):
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 1265
        spec = TPUSpec(cfg)
        op = parse_block_workload(config("qkv_projection"))
        for _, child in block_steps(op):
            if isinstance(child, Gemm):
                simulate_gemm(child, "tpu", spec)
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tpu", spec)
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 1266
        simulate_block(op, "tpu", TPUSpec(cfg))

    def test_resident_norm_minimum_temporary_capacity(self):
        op = RMSNormINT8(TensorSpec("x", (1, 7), "INT8"), TensorSpec("w", (7,), "FP32"))
        spec = copy.deepcopy(self.hardware["tsp"])
        spec.mem_total_bytes = spec.mem_n_slices * 36  # 108 B on three slices; live=99, minimum scratch/scalar=12.
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tsp", spec, mem_slices=[0, 1, 2])
        spec.mem_total_bytes = spec.mem_n_slices * 37
        r = simulate_block(op, "tsp", spec, mem_slices=[0, 1, 2], include_details=True)
        self.assertEqual(r.resource_requirements["mem"]["required_bytes"], 111)

    def test_weight_dtype_axis_epsilon_and_projection_shape_rejections(self):
        norm = parse_block_workload(config("rmsnorm_int8"))
        qkv = parse_block_workload(config("qkv_projection"))
        out = parse_block_workload(config("attention_output_projection"))
        cases = [(norm, {"x": TensorSpec("x", (2, 17), "FP32")}), (norm, {"weight": TensorSpec("w", (17,), "INT8")}),
                 (norm, {"epsilon": float("nan")}), (norm, {"axis": 0}), (norm, {"output_name": "x"}),
                 (qkv, {"query_heads": 3}), (qkv, {"kv_heads": True}), (qkv, {"head_dim": 0}),
                 (qkv, {"key_weight": TensorSpec("wk", (17, 28), "INT8")}),
                 (qkv, {"value_weight": qkv.key_weight}), (qkv, {"output_names": ["q", "q", "v"]}),
                 (qkv, {"output_names": ["x", "k", "v"]}), (qkv, {"output_names": ["q", "k"]}),
                 (out, {"context": TensorSpec("context", (2, 4, 2, 7), "INT8")}),
                 (out, {"weight": TensorSpec("w", (27, 17), "INT8")})]
        for op, changes in cases:
            with self.subTest(kind=op.operation, fields=list(changes)), self.assertRaises(ValueError):
                replace(op, **changes)

    def test_head_counts_mha_mqa_and_internal_name_collisions(self):
        for q, kv in ((1, 1), (4, 4), (4, 1)):
            op = QKVProjection(TensorSpec("qkv_projection.query_accumulator", (1, 5), "INT8"),
                               TensorSpec("wq", (5, q * 3), "INT8"), TensorSpec("wk", (5, kv * 3), "INT8"),
                               TensorSpec("wv", (5, kv * 3), "INT8"), q, kv, 3, ["query", "key", "value"])
            r = simulate_block(op, "tpu", self.hardware["tpu"])
            self.assertEqual(r.work_counts["logical"]["useful_macs"], 5 * (q + 2 * kv) * 3)

    def test_profile_sensitivity_and_unknown_boundary_cost(self):
        op = parse_block_workload(config("rmsnorm_int8"))
        profile = copy.deepcopy(load_block_execution().config)
        before = simulate_block(op, "tpu", self.hardware["tpu"], mode="timing")
        profile["parameters"]["to_int8_cost"]["value"] = 3000
        after = simulate_block(op, "tpu", self.hardware["tpu"], mode="timing", block_execution=BlockExecution(profile))
        self.assertEqual(before.action_ledger, after.action_ledger)
        self.assertGreater(after.cycle_count, before.cycle_count)
        profile["parameters"]["to_fp32_cost"].update(value=None, status="unknown")
        profile = BlockExecution(profile)
        simulate_block(op, "tpu", self.hardware["tpu"], block_execution=profile)
        with self.assertRaises(ValueError):
            simulate_block(op, "tpu", self.hardware["tpu"], mode="timing", block_execution=profile)

    def test_connections_require_physical_layout_validation(self):
        qkv = self.run_block("qkv_projection")
        out = self.run_block("attention_output_projection")
        connection = plan_connection(qkv, out)
        self.assertIn("layout_validation", connection["requirements"])
        self.assertFalse(connection["residency"]["reuse_assumed"])
        self.assertIsNone(connection["latency_s"])

    def test_strict_workload_fields(self):
        for kind in KINDS:
            cfg = config(kind)
            cfg["bias"] = True
            with self.assertRaises(ValueError):
                parse_block_workload(cfg)
        cfg = config("qkv_projection")
        del cfg["query_heads"]
        with self.assertRaises(ValueError):
            parse_block_workload(cfg)


if __name__ == "__main__":
    unittest.main()
