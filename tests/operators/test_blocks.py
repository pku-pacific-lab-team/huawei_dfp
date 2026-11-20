"""Independent arithmetic, traffic, dependency, storage and service checks."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from src.core.block_schedule import storage_schedule
from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.operators import TensorSpec, Residual, SwiGLU, SwiGLUFFN, simulate_block, simulate_gemm
from src.operators.blocks import block_steps
from src.utils.operator_workload import parse_block_workload
from tests.operators.test_gemm import specs
from tests.operators.test_vector import synthetic, execution, raw

ROOT = Path(__file__).resolve().parents[2]


def ffn(m=2, h=17, i=7):
    return SwiGLUFFN(TensorSpec("x", (m, h), "INT8"), TensorSpec("g", (h, i), "INT8"),
                     TensorSpec("u", (h, i), "INT8"), TensorSpec("d", (i, h), "INT8"))


def residual(n=7, dtype="INT8"):
    return Residual(TensorSpec("x", (n,), dtype), TensorSpec("r", (n,), dtype))


def swiglu(n=7, dtype="INT32", output="INT8"):
    return SwiGLU(TensorSpec("g", (n,), dtype), TensorSpec("u", (n,), dtype), output)


class BlockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_block(self, op, arch="tpu", **kw):
        return simulate_block(op, arch, self.hardware[arch], include_details=True, **kw)

    def test_count_timing_consistency_all_blocks_and_architectures(self):
        for arch in self.hardware:
            for op in (residual(), residual(dtype="FP32"), swiglu(), swiglu(dtype="FP32", output="FP32"), ffn()):
                with self.subTest(arch=arch, operation=op.operation, dtype=op.inputs[0].dtype):
                    count, timed = (self.run_block(op, arch, mode=m) for m in ("count", "timing"))
                    self.assertEqual(count.action_ledger, timed.action_ledger)
                    self.assertEqual(count.native_report["memory"], timed.native_report["memory"])
                    self.assertIsNone(count.cycle_count)
                    self.assertGreater(timed.cycle_count, 0)
                    self.assertFalse(timed.boundary["chip_execution_included"])
                    self.assertEqual(timed.cycle_count, sum(n["cycle_count"] for n in timed.native_report["schedule"]["nodes"]))
                    json.dumps(timed.to_detailed_dict(), allow_nan=False)

    def test_residual_hand_counts_and_tiled_cycles(self):
        r = simulate_block(residual(), "ascend", synthetic("ascend"), execution=execution("ascend"),
                           mode="timing", include_details=True)
        # 96-byte UB / (2 input + 1 output + 12 scratch B/element) => tiles 6+1.
        # Tile 6: DMA 3 + reads 5 + writes 4 + compute (2+2+2+5)=11 =>23.
        # Tile 1: DMA 3 + reads 4 + writes 4 + compute 4 =>15.
        self.assertEqual(r.cycle_count, 38)
        actions = r.native_report["action_counts"]
        expected = {"hbm_read_bytes": 14, "hbm_write_bytes": 7, "ub_read_bytes": 98,
                    "ub_write_bytes": 91, "ub_dma_write_bytes": 14, "ub_dma_read_bytes": 7,
                    "fp32_add": 7, "boundary_to_fp32": 14, "boundary_to_int8": 7}
        self.assertEqual({k.removeprefix("residual."): v["count"] for k, v in actions.items()}, expected)
        self.assertEqual(r.resource_requirements["ub"]["required_bytes"], 90)
        self.assertEqual(r.resource_requirements["hbm"]["required_bytes"], 21)

    def test_fp32_residual_has_no_storage_boundary_work(self):
        r = simulate_block(residual(dtype="FP32"), "ascend", synthetic("ascend"), execution=execution("ascend"), mode="timing")
        self.assertEqual(r.cycle_count, 11)
        self.assertEqual(r.work_counts["internal"]["boundary_elements"], {})
        self.assertTrue(all(a.dtype == "FP32" for a in r.action_ledger.entries if a.category == "vector"))

    def test_swiglu_formula_and_storage_boundaries(self):
        for arch in self.hardware:
            r = self.run_block(swiglu(7), arch)
            self.assertEqual(r.work_counts["logical"]["operation_counts"], {"fp32_mul": 21, "fp32_add": 7, "fp32_exp": 7, "fp32_rcp": 7})
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 14, "boundary_to_int8": 7})
            # Arithmetic always FP32 despite INT8 final storage.
            self.assertTrue(all(a.dtype == "FP32" for a in r.action_ledger.entries if a.id.split(".")[-1].startswith("fp32_")))

    def test_ffn_reuses_native_gemm_work_and_keeps_roles(self):
        for arch, spec in self.hardware.items():
            r = self.run_block(ffn(), arch, mode="timing")
            self.assertEqual(r.work_counts["logical"]["useful_macs"], 3 * 2 * 17 * 7)
            self.assertEqual(r.work_counts["logical"]["operations"], 6 * 2 * 17 * 7 + 6 * 2 * 7)
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 2 * 2 * 7 + 2 * 17, "boundary_to_int8": 2 * 7 + 2 * 17})
            indexed = {a.id: a for a in r.action_ledger.entries}
            for node_id, op in block_steps(ffn()):
                if node_id not in {"gate", "up", "down"}:
                    continue
                standalone = simulate_gemm(op, arch, spec, mode="timing")
                for action in standalone.action_ledger.entries:
                    composed = indexed[node_id + "." + action.id]
                    self.assertEqual((composed.count, composed.unit, composed.dtype, composed.role),
                                     (action.count, action.unit, action.dtype, action.role))
                    self.assertEqual(composed.summarizes, tuple(node_id + "." + k for k in action.summarizes))

    def test_ffn_hbm_materialization_counted_exactly_once_per_access(self):
        # TPU GEMMs use logical HBM traffic. X is read by gate and up separately.
        r = self.run_block(ffn()).native_report
        actions = r["action_counts"]
        read = sum(a["count"] for k, a in actions.items() if k.endswith("hbm_read_bytes"))
        write = sum(a["count"] for k, a in actions.items() if k.endswith("hbm_write_bytes"))
        self.assertEqual(read, 2 * 34 + 3 * 119 + 2 * 56 + 14 + 136)
        self.assertEqual(write, 2 * 56 + 14 + 136 + 34)
        self.assertFalse(any("edge" in k for k in actions))

    def test_ffn_branch_lifetimes_and_peak_not_sum_of_nodes(self):
        r = self.run_block(ffn()).native_report
        schedule = r["memory"]["lifetime_validation"]
        live = {s["node"]: s for s in schedule["steps"]}
        self.assertIn("ffn.gate", live["up"]["live_tensors"])
        self.assertIn("ffn.gate", live["activation"]["live_tensors"])
        self.assertNotIn("ffn.gate", live["down"]["live_tensors"])
        self.assertNotIn("ffn.up", live["down"]["live_tensors"])
        self.assertNotIn("ffn.activation", live["output"]["live_tensors"])
        # Persistent X+three weights=391; final INT32 down 136 overlaps INT8 output34.
        self.assertEqual([s["storage_bytes"] for s in schedule["steps"]], [447, 503, 517, 541, 561])
        self.assertEqual(schedule["peak_bytes"], 561)
        self.assertEqual(r["memory"]["hbm"]["peak_bytes"], 561)
        for name in ("vmem", "cmem"):
            required = [n["native_report"]["memory"].get(name, {}).get("allocated_bytes", 0) for n in r["schedule"]["nodes"]]
            self.assertEqual(r["memory"][name]["peak_bytes"], max(required))

    def test_global_capacity_rejects_graph_whose_children_fit(self):
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 550
        spec = TPUSpec(cfg)
        for _, op in block_steps(ffn()):
            if hasattr(op, "useful_macs"):
                simulate_gemm(op, "tpu", spec)
        with self.assertRaisesRegex(ValueError, "live block tensors"):
            simulate_block(ffn(), "tpu", spec)

    def test_resident_padding_reserved_and_scratch(self):
        for arch in ("tsp", "tsp_cim"):
            r = self.run_block(ffn(), arch, mem_slices=[0, 1, 2], reserved_bytes=1).native_report
            schedule = r["memory"]["lifetime_validation"]
            self.assertEqual(schedule["reserved_bytes"], 3)
            for life in schedule["lifetimes"]:
                self.assertEqual(life["bytes"] % 3, 0)
            for step, node in zip(schedule["steps"], r["schedule"]["nodes"]):
                if node["id"] in {"gate", "up", "down"}:
                    self.assertEqual(node["native_report"]["memory"]["resident_bytes"], step["storage_bytes"])
                else:
                    self.assertEqual(step["scratch_bytes"] % 3, 0)
                    self.assertEqual(node["native_report"]["memory"]["resident_bytes"], step["peak_bytes"])

    def test_resident_capacity_reduces_tile_and_rejects_one_element_short(self):
        spec = copy.deepcopy(self.hardware["tsp"])
        # n=7: inputs/output 21 B plus12 B scratch/element; one selected slice.
        spec.mem_total_bytes = 45 * spec.mem_n_slices
        r = simulate_block(residual(), "tsp", spec, mem_slices=[0], include_details=True)
        node = r.native_report["schedule"]["nodes"][0]["native_report"]
        self.assertEqual(node["execution_contract"]["selected_tile_elements"], 2)
        self.assertEqual(r.resource_requirements["mem"]["required_bytes"], 45)
        spec.mem_total_bytes = 32 * spec.mem_n_slices
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(residual(), "tsp", spec, mem_slices=[0])

    def test_tsp_cim_share_one_vxm_not_macro_count(self):
        for op in (residual(65), swiglu(65)):
            native = self.run_block(op, "tsp", mode="timing", tile_elements=16)
            cim = self.run_block(op, "tsp_cim", mode="timing", tile_elements=16)
            self.assertEqual(native.cycle_count, cim.cycle_count)
            self.assertEqual(native.to_dict()["action_counts"], cim.to_dict()["action_counts"])

    def test_tile_partition_preserves_logical_work_and_hbm_payload(self):
        a = self.run_block(swiglu(65), "ascend", tile_elements=7)
        b = self.run_block(swiglu(65), "ascend", tile_elements=65)
        self.assertEqual(a.to_dict()["action_counts"], b.to_dict()["action_counts"])
        self.assertEqual(a.work_counts, b.work_counts)

    def test_unknown_proxy_fails_only_when_timing_uses_it(self):
        cfg = copy.deepcopy(load_block_execution().config)
        cfg["parameters"]["to_int8_cost"].update(value=None, status="unknown")
        profile = BlockExecution(cfg)
        self.run_block(residual(), block_execution=profile)
        self.run_block(residual(dtype="FP32"), block_execution=profile, mode="timing")
        with self.assertRaisesRegex(ValueError, "to_int8_cost"):
            self.run_block(residual(), block_execution=profile, mode="timing")

    def test_proxy_sensitivity_and_native_arithmetic_independent(self):
        cfg = copy.deepcopy(load_block_execution().config)
        cfg["parameters"]["to_int8_cost"]["value"] = 300
        base = self.run_block(residual(1024), mode="timing")
        more = self.run_block(residual(1024), mode="timing", block_execution=BlockExecution(cfg))
        self.assertEqual(base.to_dict()["action_counts"], more.to_dict()["action_counts"])
        self.assertGreater(more.cycle_count, base.cycle_count)

    def test_batch_scaling_and_cross_k_native_allocation(self):
        for arch in self.hardware:
            # Cross the native K tile and CIM domain boundary, with both K/N tails.
            a = self.run_block(ffn(1, 321, 33), arch)
            b = self.run_block(ffn(3, 321, 33), arch)
            self.assertEqual(b.work_counts["logical"]["operations"], 3 * a.work_counts["logical"]["operations"])
            self.assertEqual(b.work_counts["internal"]["boundary_elements"],
                             {k: 3 * v for k, v in a.work_counts["internal"]["boundary_elements"].items()})
            for result in (a, b):
                for req in result.resource_requirements.values():
                    if req["capacity_bytes"] is not None:
                        self.assertLessEqual(req["required_bytes"], req["capacity_bytes"])

    def test_missing_native_budget_and_clock_semantics(self):
        cfg = raw("tsp")
        cfg["parameters"]["clock_ghz"].update(value=None, status="unknown")
        result = simulate_block(residual(), "tsp", TSPSpec(cfg), mode="timing")
        self.assertGreater(result.cycle_count, 0)
        self.assertIsNone(result.latency_s)
        self.assertEqual(result.timing_status, "cycles_only")
        cfg["parameters"]["vxm_base_ops_per_cycle"].update(value=None, status="unknown")
        simulate_block(residual(), "tsp", TSPSpec(cfg))
        with self.assertRaisesRegex(ValueError, "budget|vxm_base_ops"):
            simulate_block(residual(), "tsp", TSPSpec(cfg), mode="timing")

    def test_shape_dtype_and_alias_rejection(self):
        cases = [lambda: Residual(TensorSpec("x", (2,), "INT8"), TensorSpec("y", (3,), "INT8")),
                 lambda: residual(dtype="INT32"), lambda: swiglu(output="FP16"),
                 lambda: Residual(TensorSpec("x", (2,), "INT8"), TensorSpec("x", (2,), "INT8")),
                 lambda: SwiGLUFFN(TensorSpec("x", (2, 3), "INT8"), TensorSpec("g", (4, 5), "INT8"),
                                    TensorSpec("u", (4, 5), "INT8"), TensorSpec("d", (5, 3), "INT8"))]
        for case in cases:
            with self.assertRaises(ValueError):
                case()

    def test_internal_names_cannot_alias_external_tensors(self):
        op = ffn()
        op = SwiGLUFFN(TensorSpec("ffn.gate", op.x.shape, "INT8"), op.gate_weight, op.up_weight,
                       op.down_weight, output_name="ffn.up")
        r = self.run_block(op)
        lives = r.native_report["memory"]["lifetime_validation"]["lifetimes"]
        self.assertEqual(len(lives), 9)

    def test_schedule_rejects_cycles_alias_and_orphan(self):
        valid = [{"id": "a", "inputs": ["x"], "output": "a"}, {"id": "b", "inputs": ["a"], "output": "b"}]
        for steps in ([], list(reversed(valid)), [{"id": "a", "inputs": ["x"], "output": "x"}],
                      [valid[0], {"id": "b", "inputs": ["x"], "output": "b"}]):
            with self.assertRaises(ValueError):
                storage_schedule(["x"], ["b"], steps, {"x": 1, "a": 1, "b": 1})

    def test_workload_and_profile_strictness_and_provenance(self):
        for kind in ("residual", "swiglu", "swiglu_ffn"):
            cfg = json.loads((ROOT / f"configs/workload/operators/{kind}.json").read_text())
            self.assertEqual(parse_block_workload(cfg).operation, kind)
            for key, value in (("junk", 1), ("schema_version", True), ("inputs", [])):
                invalid = copy.deepcopy(cfg)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    parse_block_workload(invalid)
        profile = load_block_execution()
        audit = json.loads((ROOT / "calibration/operator_execution_audit.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(profile.config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(audit["block_execution_profiles"]["configs/execution/blocks.json"]["canonical_json_sha256"], digest)
        for key, value in (("schedule", "parallel"), ("schema_version", True), ("status", "measured")):
            invalid = copy.deepcopy(profile.config)
            invalid[key] = value
            with self.assertRaises(ValueError):
                BlockExecution(invalid)


if __name__ == "__main__":
    unittest.main()
