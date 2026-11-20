"""Independent Decode RoPE, embedding and FP32 vocabulary-output accounting."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.tpu import TPUSpec
from src.hardware.vector import VectorExecution, load_vector_execution
from src.operators import (TensorSpec, RoPE, Embedding, VocabularyProjection, QKVProjection, KVCacheSpec,
                           DecodeAttention, simulate_block, simulate_gemm, simulate_attention, plan_connection)
from src.operators.blocks import block_steps
from src.utils.operator_workload import parse_block_workload
from tests.operators.test_gemm import specs
from tests.operators.test_vector import raw, synthetic, execution

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("rope", "embedding", "vocabulary_projection")


def config(kind):
    return json.loads((ROOT / f"configs/workload/operators/{kind}.json").read_text(encoding="utf-8"))


def rope(batch=2, qheads=4, kheads=2, dim=10, positions=9, position=5):
    return RoPE(TensorSpec("q", (batch, qheads, 1, dim), "INT8"), TensorSpec("k", (batch, kheads, 1, dim), "INT8"),
                TensorSpec("cos", (positions, dim // 2), "FP32"), TensorSpec("sin", (positions, dim // 2), "FP32"), position)


def sum_actions(result, suffix):
    return sum(a.count for a in result.action_ledger.entries if a.id.endswith(suffix) and a.role == "action")


class PositionVocabularyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_block(self, op, arch="tpu", **kw):
        return simulate_block(op, arch, self.hardware[arch], include_details=True, **kw)

    def test_four_architectures_count_timing_and_summary(self):
        for arch in self.hardware:
            for kind in KINDS:
                with self.subTest(arch=arch, kind=kind):
                    op = parse_block_workload(config(kind))
                    c, t = (self.run_block(op, arch, mode=mode, tile_elements=4) for mode in ("count", "timing"))
                    self.assertEqual(c.action_ledger, t.action_ledger)
                    self.assertEqual(c.native_report["memory"], t.native_report["memory"])
                    self.assertIsNone(c.cycle_count)
                    self.assertGreater(t.cycle_count, 0)
                    self.assertEqual(t.cycle_count, sum(n["cycle_count"] for n in t.native_report["schedule"]["nodes"]))
                    self.assertFalse(t.boundary["chip_execution_included"])
                    self.assertEqual(t.operator.outputs[0].dtype, "FP32" if kind == "vocabulary_projection" else "INT8")
                    json.dumps(t.to_detailed_dict(), allow_nan=False)

    def test_rope_hand_arithmetic_bytes_and_lifetimes(self):
        for arch in ("ascend", "tpu"):
            r = self.run_block(rope(), arch, tile_elements=4)
            self.assertEqual(r.work_counts["logical"]["operation_counts"], {"fp32_mul": 240, "fp32_sub": 60, "fp32_add": 60})
            self.assertEqual(r.work_counts["logical"]["operations"], 360)
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 120, "boundary_to_int8": 120})
            self.assertEqual(sum_actions(r, "hbm_read_bytes"), 600)
            self.assertEqual(sum_actions(r, "hbm_write_bytes"), 120)
            local = "ub" if arch == "ascend" else "vmem"
            self.assertEqual(sum_actions(r, local + "_read_bytes"), 3480)
            self.assertEqual(sum_actions(r, local + "_write_bytes"), 2040)
            self.assertEqual(r.resource_requirements[local]["required_bytes"], 72)
            life = r.native_report["memory"]["lifetime_validation"]
            self.assertEqual([s["storage_bytes"] for s in life["steps"]], [560, 600])
            self.assertIn("rotated_query", life["steps"][1]["live_tensors"])
            self.assertEqual(len(life["lifetimes"]), 6)  # Four inputs and two outputs; tables are not copied per head.

    def test_rope_single_pair_hand_cycles(self):
        op = rope(1, 1, 1, 2, 1, 0)
        for arch in ("ascend", "tpu"):
            r = simulate_block(op, arch, synthetic(arch), execution=execution(arch), mode="timing", include_details=True)
            # Each head: loads=3; cast=3; four MULs=12; SUB+ADD=6; INT8 cast=4; export=1.
            self.assertEqual(r.cycle_count, 2 * 29)
            self.assertEqual(r.work_counts["logical"]["operations"], 12)
            self.assertEqual(r.resource_requirements["hbm"]["required_bytes"], 16)
            self.assertEqual(r.resource_requirements["ub" if arch == "ascend" else "vmem"]["required_bytes"], 36)

    def test_rope_complete_pairs_tail_and_scratch_reuse(self):
        r = self.run_block(rope(), tile_elements=5)
        node = r.native_report["schedule"]["nodes"][0]["native_report"]
        self.assertEqual(node["execution_contract"]["requested_tile_elements"], 5)
        self.assertEqual(node["execution_contract"]["selected_tile_elements"], 4)
        groups = node["schedule"]["tile_groups"]
        self.assertEqual([(g["pairs"], g["repetitions"]) for g in groups], [(2, 16), (1, 8)])
        labels = [s["stage"] for s in groups[0]["stages"]]
        self.assertLess(labels.index("multiply_3"), labels.index("rotate_even"))
        self.assertLess(labels.index("rotate_odd"), labels.index("output_to_int8"))
        self.assertEqual(node["schedule"]["scratch_bytes_per_pair"], 24)
        for tile in (0, 1, True):
            with self.assertRaises(ValueError):
                self.run_block(rope(), tile_elements=tile)

    def test_rope_table_capacity_independent_of_position_and_traffic(self):
        a, b, c = (self.run_block(op) for op in (rope(), rope(position=8), rope(positions=18)))
        self.assertEqual(a.action_ledger, b.action_ledger)
        self.assertEqual(a.action_ledger, c.action_ledger)
        self.assertEqual(c.resource_requirements["hbm"]["required_bytes"] - a.resource_requirements["hbm"]["required_bytes"], 360)
        self.assertFalse(dict(a.operator.attributes)["table_generation_included"])

    def test_rope_head_batch_scaling_without_template_expansion(self):
        small = self.run_block(rope(batch=1, qheads=4, kheads=1))
        large = self.run_block(rope(batch=1000, qheads=4, kheads=1))
        self.assertEqual(large.work_counts["logical"]["operations"], 1000 * small.work_counts["logical"]["operations"])
        self.assertEqual(len(large.native_report["schedule"]["nodes"]), 2)
        for node in large.native_report["schedule"]["nodes"]:
            self.assertEqual(len(node["native_report"]["schedule"]["tile_groups"]), 1)

    def test_rope_striped_global_and_temporary_peak(self):
        for arch in ("tsp", "tsp_cim"):
            r = self.run_block(rope(), arch, mem_slices=[0, 1, 2], streams=2, reserved_bytes=17, tile_elements=4)
            life = r.native_report["memory"]["lifetime_validation"]
            self.assertEqual([s["storage_bytes"] for s in life["steps"]], [582, 624])
            self.assertEqual([s["scratch_bytes"] for s in life["steps"]], [48, 48])
            self.assertEqual(life["peak_bytes"], 672)
            self.assertFalse(any(a.category in {"matrix", "cim_internal"} for a in r.action_ledger.entries))

    def test_rope_minimum_resident_and_local_workspace_capacity(self):
        op = rope(1, 1, 1, 2, 1, 0)
        spec = copy.deepcopy(self.hardware["tsp"])
        spec.mem_total_bytes = spec.mem_n_slices * 15
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tsp", spec, mem_slices=[0, 1, 2])
        spec.mem_total_bytes = spec.mem_n_slices * 16
        r = simulate_block(op, "tsp", spec, mem_slices=[0, 1, 2])
        self.assertEqual(r.resource_requirements["mem"]["required_bytes"], 48)
        cfg = raw("tpu")
        cfg["parameters"]["vmem_usable_bytes"]["value"] = 35
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tpu", TPUSpec(cfg))
        cfg["parameters"]["vmem_usable_bytes"]["value"] = 36
        simulate_block(op, "tpu", TPUSpec(cfg))

    def test_rope_shape_position_table_and_alias_rejections(self):
        op = rope()
        changes = [{"query": TensorSpec("q", (2, 4, 1, 9), "INT8"), "key": TensorSpec("k", (2, 2, 1, 9), "INT8")},
                   {"query": TensorSpec("q", (2, 4, 2, 10), "INT8")}, {"key": TensorSpec("k", (1, 2, 1, 10), "INT8")},
                   {"key": TensorSpec("k", (2, 3, 1, 10), "INT8")}, {"cos": TensorSpec("cos", (9, 10), "FP32")},
                   {"sin": TensorSpec("sin", (9, 5), "INT8")}, {"sin": op.cos}, {"position": -1}, {"position": 9},
                   {"position": True}, {"position": 1.0}, {"output_names": ["q", "out"]}, {"output_names": ["x", "x"]}]
        for change in changes:
            with self.subTest(fields=list(change)), self.assertRaises(ValueError):
                replace(op, **change)

    def test_embedding_hand_counts_capacity_and_cycles(self):
        op = parse_block_workload(config("embedding"))
        for arch in ("ascend", "tpu"):
            r = simulate_block(op, arch, synthetic(arch), execution=execution(arch), tile_elements=4,
                               mode="timing", include_details=True)
            self.assertEqual(r.cycle_count, 12)  # Per token: index DMA+read=2, two row DMA read/write pairs=4.
            self.assertEqual(r.work_counts["logical"]["operations"], 0)
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {})
            self.assertEqual(sum_actions(r, "hbm_read_bytes"), 22)
            self.assertEqual(sum_actions(r, "hbm_write_bytes"), 14)
            self.assertEqual(r.resource_requirements["hbm"]["required_bytes"], 155)
            self.assertEqual(r.resource_requirements["ub" if arch == "ascend" else "vmem"]["required_bytes"], 8)
            self.assertTrue(all(a.dtype == ("INT32" if ".index." in a.id else "INT8") for a in r.action_ledger.entries))

    def test_embedding_table_size_and_batch_scale_separately(self):
        op = parse_block_workload(config("embedding"))
        a = self.run_block(op)
        b = self.run_block(replace(op, weight=TensorSpec("weight", (38, 7), "INT8")))
        c = self.run_block(replace(op, indices=TensorSpec("indices", (4,), "INT32")))
        self.assertEqual(a.action_ledger, b.action_ledger)
        self.assertEqual(b.resource_requirements["hbm"]["required_bytes"] - a.resource_requirements["hbm"]["required_bytes"], 133)
        self.assertEqual(sum_actions(c, "hbm_read_bytes"), 2 * sum_actions(a, "hbm_read_bytes"))
        self.assertFalse(dict(c.operator.attributes)["duplicate_row_reuse"])

    def test_embedding_full_table_capacity_is_checked_on_all_architectures(self):
        op = parse_block_workload(config("embedding"))
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 154
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tpu", TPUSpec(cfg))
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 155
        simulate_block(op, "tpu", TPUSpec(cfg))
        huge = Embedding(TensorSpec("ids", (1,), "INT32"), TensorSpec("table", (100000000, 1024), "INT8"))
        for arch in self.hardware:
            with self.subTest(arch=arch), self.assertRaisesRegex(ValueError, "unsupported_capacity"):
                self.run_block(huge, arch)

    def test_embedding_striping_and_minimum_local_index_buffer(self):
        op = parse_block_workload(config("embedding"))
        r = self.run_block(op, "tsp", mem_slices=[0, 1, 2], reserved_bytes=17)
        self.assertEqual(r.resource_requirements["mem"]["required_bytes"], 177)
        self.assertEqual(sum_actions(r, "mem_read_bytes"), 22)
        self.assertEqual(sum_actions(r, "mem_write_bytes"), 14)
        cfg = raw("tpu")
        cfg["parameters"]["vmem_usable_bytes"]["value"] = 4
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tpu", TPUSpec(cfg))
        cfg["parameters"]["vmem_usable_bytes"]["value"] = 5
        r = simulate_block(op, "tpu", TPUSpec(cfg))
        self.assertEqual(r.resource_requirements["vmem"]["required_bytes"], 5)

    def test_embedding_shape_alias_and_index_range_contract(self):
        op = parse_block_workload(config("embedding"))
        for change in ({"indices": TensorSpec("i", (2,), "FP32")}, {"indices": TensorSpec("i", (2, 1), "INT32")},
                       {"weight": TensorSpec("weight", (19, 7), "FP32")}, {"output_name": "weight"},
                       {"weight": TensorSpec("weight", (2**31+1, 7), "INT8")}):
            with self.assertRaises(ValueError):
                replace(op, **change)
        self.assertEqual(dict(op.spec.attributes)["index_values"], "symbolic_assumed_in_range")

    def test_vocabulary_hand_work_and_fp32_boundary(self):
        op = parse_block_workload(config("vocabulary_projection"))
        for arch in self.hardware:
            r = self.run_block(op, arch)
            self.assertEqual(r.work_counts["logical"]["useful_macs"], 266)
            self.assertEqual(r.work_counts["logical"]["operations"], 532)
            self.assertEqual(r.work_counts["logical"]["output_bytes"], 152)
            self.assertEqual(r.work_counts["internal"]["boundary_elements"], {"boundary_to_fp32": 38})
            self.assertEqual(r.operator.outputs[0].shape, (2, 19))
            self.assertFalse(dict(op.spec.attributes)["softmax_or_sampling_included"])
        r = self.run_block(op)
        self.assertEqual(sum_actions(r, "hbm_read_bytes"), 299)
        self.assertEqual(sum_actions(r, "hbm_write_bytes"), 304)
        self.assertEqual([s["storage_bytes"] for s in r.native_report["memory"]["lifetime_validation"]["steps"]], [299, 451])

    def test_vocabulary_native_matrix_actions_preserved(self):
        op = parse_block_workload(config("vocabulary_projection"))
        gemm = block_steps(op)[0][1]
        for arch, spec in self.hardware.items():
            result = self.run_block(op, arch)
            indexed = {a.id: a for a in result.action_ledger.entries}
            for a in simulate_gemm(gemm, arch, spec).action_ledger.entries:
                b = indexed["projection." + a.id]
                self.assertEqual((a.count, a.unit, a.dtype, a.role), (b.count, b.unit, b.dtype, b.role))
                self.assertEqual(b.summarizes, tuple("projection." + name for name in a.summarizes))

    def test_vocabulary_output_overlap_capacity_and_invalid_inputs(self):
        op = parse_block_workload(config("vocabulary_projection"))
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 450
        simulate_gemm(block_steps(op)[0][1], "tpu", TPUSpec(cfg))
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tpu", TPUSpec(cfg))
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 451
        simulate_block(op, "tpu", TPUSpec(cfg))
        for change in ({"weight": TensorSpec("w", (8, 19), "INT8")}, {"x": TensorSpec("x", (2, 7), "FP32")}, {"output_name": "x"}):
            with self.assertRaises(ValueError):
                replace(op, **change)

    def test_used_costs_sensitive_unknown_costs_only_reject_consumers(self):
        cfg = copy.deepcopy(load_block_execution().config)
        cfg["parameters"]["to_fp32_cost"].update(value=None, status="unknown")
        block = BlockExecution(cfg)
        self.run_block(parse_block_workload(config("embedding")), mode="timing", block_execution=block)
        for op in (rope(), parse_block_workload(config("vocabulary_projection"))):
            self.run_block(op, block_execution=block)
            with self.assertRaises(ValueError):
                self.run_block(op, mode="timing", block_execution=block)
        v = copy.deepcopy(load_vector_execution("tpu").config)
        v["composite"]["parameters"]["fp32_sub_cost"]["value"] = 3000
        a = self.run_block(rope(), mode="timing")
        b = self.run_block(rope(), mode="timing", execution=VectorExecution(v, "tpu"))
        self.assertEqual(a.action_ledger, b.action_ledger)
        self.assertGreater(b.cycle_count, a.cycle_count)

    def test_qkv_rope_and_cache_shape_interfaces_without_graph_execution(self):
        qkv = QKVProjection(TensorSpec("x", (2, 7), "INT8"), TensorSpec("wq", (7, 40), "INT8"),
                            TensorSpec("wk", (7, 20), "INT8"), TensorSpec("wv", (7, 20), "INT8"), 4, 2, 10)
        rotation = RoPE(*qkv.spec.outputs[:2], TensorSpec("cos", (9, 5), "FP32"), TensorSpec("sin", (9, 5), "FP32"), 5)
        a, b = self.run_block(qkv), self.run_block(rotation)
        for i in (0, 1):
            edge = plan_connection(a, b, output_index=i, input_index=i)
            self.assertIn("layout_validation", edge["requirements"])
            self.assertIsNone(edge["latency_s"])
        attention = DecodeAttention(rotation.spec.outputs[0], KVCacheSpec(2, 2, 10, 5, 9))
        c = simulate_attention(attention, "tpu", self.hardware["tpu"])
        self.assertEqual(plan_connection(b, c)["conversion"]["status"], "not_required")
        self.assertEqual(rotation.spec.outputs[1].shape, attention.cache_access.inputs[2].shape)

    def test_strict_workload_schema_and_required_position(self):
        for kind in KINDS:
            cfg = config(kind)
            cfg["unknown"] = 1
            with self.assertRaises(ValueError):
                parse_block_workload(cfg)
        cfg = config("rope")
        del cfg["position"]
        with self.assertRaisesRegex(ValueError, "position"):
            parse_block_workload(cfg)


if __name__ == "__main__":
    unittest.main()
