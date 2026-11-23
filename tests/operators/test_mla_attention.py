"""Absorbed MLA: matrix/FP32 work, state binding and full-block memory accounting."""
import copy
from dataclasses import replace
import json
import math
import unittest

from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.operators import (TensorSpec, MLASpec, MLACacheSpec, MLACacheAccess, MLADecodeAttention,
    AttentionOutputProjection, simulate_mla_attention, simulate_mla_cache, simulate_block, simulate_gemm, plan_connection)
from src.operators.mla import mla_attention_steps
from src.operators.blocks import block_steps
from src.operators.primitives import Gemm
from src.utils.operator_workload import parse_mla_attention_workload
from tests.operators.test_gemm import specs
from tests.operators.test_mla import config, preprocess, actions
from tests.operators.test_vector import raw


def task(batch=2, heads=3, length=2, capacity=9, append=1, projected=True, dn=3, dr=4, rk=7, dv=6, hidden=17):
    d = MLASpec(hidden, heads, 5, rk, dn, dr, dv)
    return MLADecodeAttention(TensorSpec("q", (batch, heads, 1, dn), "INT8"),
        TensorSpec("qp", (batch, heads, 1, dr), "INT8"), TensorSpec("wkvb", (heads, dn + dv, rk), "INT8"),
        d, MLACacheSpec(batch, rk, dr, length, capacity), append,
        TensorSpec("wo", (heads * dv, hidden), "INT8") if projected else None)


class MLAAttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_attention(self, op=None, arch="tpu", **kw):
        return simulate_mla_attention(op or task(), arch, self.hardware[arch], include_details=True, **kw)

    def test_four_architectures_count_timing_context_and_projection(self):
        for arch in self.hardware:
            for projected in (False, True):
                with self.subTest(arch=arch, projected=projected):
                    a, b = (self.run_attention(task(projected=projected), arch, mode=m, tile_elements=2) for m in ("count", "timing"))
                    self.assertEqual(a.action_ledger, b.action_ledger)
                    self.assertEqual(a.native_report["memory"], b.native_report["memory"])
                    self.assertIsNone(a.cycle_count)
                    self.assertGreater(b.cycle_count, 0)
                    self.assertGreater(b.latency_s, 0)
                    nodes = b.native_report["schedule"]["nodes"]
                    self.assertEqual(b.cycle_count, sum(n["repetitions"] * n["cycle_count_per_repetition"] for n in nodes))
                    self.assertEqual(b.operator.outputs[0].shape, (2, 17) if projected else (2, 3, 1, 6))
                    self.assertFalse(b.boundary["chip_execution_included"])
                    json.dumps(b.to_detailed_dict(), allow_nan=False)

    def test_hand_macs_fp32_arithmetic_and_conversion_elements(self):
        for projected in (False, True):
            for arch in self.hardware:
                r = self.run_attention(task(projected=projected), arch)
                logical, internal = r.work_counts["logical"], r.work_counts["internal"]
                # Five GEMMs/head: 3*7 + 7*3 + 4*3 + 3*7 + 7*6 = 117 MAC.
                macs = 6 * 117 + (612 if projected else 0)
                self.assertEqual(logical["useful_macs"], macs)
                self.assertEqual(logical["operations"], 2 * macs + 120)
                self.assertEqual(logical["operation_counts"], {"fp32_add": 30, "fp32_mul": 36,
                    "fp32_max": 12, "fp32_sub": 18, "fp32_exp": 18, "fp32_rcp": 6})
                self.assertEqual(internal["boundary_elements"], {"boundary_to_fp32": 156 + (34 if projected else 0),
                                                                "boundary_to_int8": 138 + (34 if projected else 0)})
                self.assertEqual(logical["output_bytes"], 34 if projected else 36)

    def test_hand_tpu_reads_writes_and_global_peak(self):
        r = self.run_attention()
        # Per head: 329 B read, 199 B write. Append 22 B each direction.
        # Output GEMM reads 36+306 B, writes136 B; conversion reads136 B, writes34 B.
        self.assertEqual(actions(r, "hbm_read_bytes"), 6 * 329 + 22 + 342 + 136)
        self.assertEqual(actions(r, "hbm_write_bytes"), 6 * 199 + 22 + 136 + 34)
        life = r.native_report["memory"]["lifetime_validation"]
        self.assertEqual(life["global_allocations"], {"query_nope": 18, "query_rope": 24, "kv_up_weight": 189,
            "latent_cache": 126, "rope_cache": 72, "context": 36, "reserved": 0, "output_weight": 306})
        self.assertEqual(life["cache_append_peak_bytes"], 793)
        self.assertEqual(r.resource_requirements["hbm"]["required_bytes"], 941)
        self.assertEqual([s["storage_bytes"] for s in life["output_projection"]["steps"]], [907, 941])

    def test_cache_append_once_and_two_latent_reads_per_head(self):
        for arch in self.hardware:
            r = self.run_attention(arch=arch)
            schedule = r.native_report["schedule"]
            nodes = {n["id"]: n for n in schedule["nodes"]}
            self.assertEqual(nodes["cache_append"]["repetitions"], 1)
            self.assertTrue(all(n["repetitions"] == 6 for name, n in nodes.items() if name not in {"cache_append", "output.projection", "output.output_storage"}))
            self.assertEqual(nodes["cache_append"]["native_report"]["work_counts"]["logical"]["cache_read_bytes"], 0)
            self.assertFalse(any("read" in g["phase"] for g in nodes["cache_append"]["native_report"]["schedule"]["tile_groups"]))
            self.assertEqual(r.work_counts["logical"]["cache_read_payload_bytes"], 6 * 3 * (7 + 4 + 7))
            self.assertEqual(r.work_counts["logical"]["cache_appended_bytes"], 22)
            self.assertEqual(r.to_detailed_dict()["io_contract"]["state_aliases"], r.native_report["cache_state"]["output_aliases"])

    def test_cache_capacity_shared_across_heads_and_weights_shared_across_batch(self):
        one = self.run_attention(task(batch=1, heads=1, projected=False))
        heads = self.run_attention(task(batch=1, heads=3, projected=False))
        batches = self.run_attention(task(batch=2, heads=3, projected=False))
        allocations = [r.native_report["memory"]["lifetime_validation"]["global_allocations"] for r in (one, heads, batches)]
        for key in ("latent_cache", "rope_cache"):
            self.assertEqual(allocations[0][key], allocations[1][key])
            self.assertEqual(2 * allocations[1][key], allocations[2][key])
        self.assertEqual(allocations[0]["kv_up_weight"] * 3, allocations[1]["kv_up_weight"])
        self.assertEqual(allocations[1]["kv_up_weight"], allocations[2]["kv_up_weight"])
        self.assertEqual(heads.work_counts["logical"]["cache_read_payload_bytes"], 3 * one.work_counts["logical"]["cache_read_payload_bytes"])

    def test_capacity_does_not_change_work_but_filled_length_does(self):
        a = self.run_attention(task(capacity=9))
        b = self.run_attention(task(capacity=29))
        self.assertEqual(a.action_ledger, b.action_ledger)
        self.assertEqual(b.resource_requirements["hbm"]["required_bytes"] - a.resource_requirements["hbm"]["required_bytes"], 2 * 20 * 11)
        longer = self.run_attention(task(length=5))
        self.assertEqual(longer.work_counts["logical"]["useful_macs"] - a.work_counts["logical"]["useful_macs"], 6 * 3 * 18)
        self.assertEqual(longer.work_counts["logical"]["cache_appended_bytes"], 22)

    def test_split_scores_convert_before_fp32_add_and_scale(self):
        _, steps = mla_attention_steps(task())
        stages = dict(steps)
        add = stages["score_add"]
        self.assertEqual([t.dtype for t in add.inputs], ["FP32", "FP32"])
        self.assertEqual([t.name for t in add.inputs], [stages["content_to_fp32"].output.name, stages["position_to_fp32"].output.name])
        self.assertEqual(stages["scale"].scale, 1 / math.sqrt(7))
        life = self.run_attention().native_report["memory"]["lifetime_validation"]
        by_name = {v["tensor"]: v for v in life["lifetimes"]}
        self.assertEqual(by_name[stages["content_score"].output_name]["last_step"], 4)
        self.assertEqual(by_name[stages["position_score"].output_name]["last_step"], 5)
        self.assertEqual(by_name[add.output_name]["last_step"], 7)
        self.assertIn(stages["content_to_fp32"].output.name, life["steps"][5]["live_tensors"])

    def test_explicit_scale_multiplier_is_not_latent_rank_or_an_extra_multiply(self):
        base = task()
        extension = replace(base, scale_multiplier=2.25, scale_source="test explicit squared YaRN mscale")
        self.assertAlmostEqual(extension.scale, 2.25 / math.sqrt(7))
        a, b = (self.run_attention(o, mode="timing") for o in (base, extension))
        self.assertEqual(a.action_ledger, b.action_ledger)
        self.assertEqual(a.cycle_count, b.cycle_count)
        self.assertEqual(b.native_report["execution_contract"]["scale_multiplier"], 2.25)
        cfg = config("mla_decode_attention")
        cfg["dimensions"]["kv_lora_rank"] = 11
        cfg["cache"]["kv_lora_rank"] = 11
        cfg["kv_up_weight"]["shape"] = [3, 9, 11]
        cfg["new_latent"]["shape"] = [2, 1, 1, 11]
        changed = parse_mla_attention_workload(cfg)
        self.assertEqual(changed.scale, base.scale)

    def test_native_gemm_and_existing_output_projection_actions_reused(self):
        op = task()
        _, steps = mla_attention_steps(op)
        output_steps = block_steps(AttentionOutputProjection(op.context, op.output_weight, op.output_name))
        for arch, hardware in self.hardware.items():
            r = self.run_attention(op, arch)
            nodes = {n["id"]: n["native_report"] for n in r.native_report["schedule"]["nodes"]}
            for name, stage in steps + [("output." + n, s) for n, s in output_steps]:
                if isinstance(stage, Gemm):
                    native = simulate_gemm(stage, arch, hardware, include_details=True).native_report
                    self.assertEqual(nodes[name]["action_counts"], native["action_counts"])
                    self.assertEqual(nodes[name]["work_counts"], native["work_counts"])
            standalone = simulate_block(AttentionOutputProjection(op.context, op.output_weight, op.output_name), arch, hardware, include_details=True)
            output_nodes = standalone.native_report["schedule"]["nodes"]
            for node in output_nodes:
                self.assertEqual(nodes["output." + node["id"]]["action_counts"], node["native_report"]["action_counts"])

    def test_context_views_reserved_once_for_native_output_projection(self):
        op = task()
        for arch in self.hardware:
            r = self.run_attention(op, arch)
            life = r.native_report["memory"]["lifetime_validation"]
            output_life = life["output_projection"]
            context_bytes = next(v["bytes"] for v in output_life["lifetimes"] if v["tensor"] == op.context.name)
            self.assertGreaterEqual(life["global_allocations"]["context"], context_bytes)
            self.assertEqual(output_life["view_bindings"][op.context.name], "context")
            self.assertEqual(life["view_bindings"]["head.latent_score_view"], life["view_bindings"]["head.latent_value_view"])
            partitions = life["kv_weight_partitions"]
            self.assertEqual(life["global_allocations"]["kv_up_weight"],
                3 * (partitions["key_bytes_per_head"] + partitions["value_bytes_per_head"]))

    def test_global_capacity_checked_through_final_projection(self):
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 940
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_mla_attention(task(), "tpu", TPUSpec(cfg))
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 941
        simulate_mla_attention(task(), "tpu", TPUSpec(cfg))
        # The native output GEMM alone fits; the complete live block is what fails.
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 500
        simulate_gemm(Gemm(2, 18, 17), "tpu", TPUSpec(cfg))
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_mla_attention(task(), "tpu", TPUSpec(cfg))

    def test_resident_striping_scratch_and_minimum_global_capacity(self):
        op = task(length=33, capacity=35, rk=17, dv=9)
        for arch in ("tsp", "tsp_cim"):
            r = self.run_attention(op, arch, mem_slices=[0, 1, 2], streams=2, reserved_bytes=17, tile_elements=4)
            life = r.native_report["memory"]["lifetime_validation"]
            self.assertEqual(life["global_allocations"]["reserved"], 18)
            rows = life["steps"] + life["output_projection"]["steps"]
            self.assertTrue(all(row["storage_bytes"] % 3 == 0 for row in rows))
            self.assertTrue(any(row["scratch_bytes"] > 0 for row in rows))
            self.assertEqual(life["peak_bytes"], max([life["cache_append_peak_bytes"]] + [r["peak_bytes"] for r in rows]))
        hw = copy.deepcopy(self.hardware["tsp"])
        hw.mem_total_bytes = hw.mem_n_slices * 64
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_mla_attention(task(), "tsp", hw, mem_slices=[0])

    def test_empty_cache_append_and_read_only_full_cache(self):
        first = self.run_attention(task(batch=1, heads=1, length=0, capacity=1))
        self.assertEqual(first.native_report["cache_state"]["write_token_interval"], [0, 1])
        read = self.run_attention(task(batch=1, heads=1, length=1, capacity=1, append=0))
        self.assertEqual(read.native_report["cache_state"]["write_token_interval"], [1, 1])
        self.assertFalse(any(n["id"] == "cache_append" for n in read.native_report["schedule"]["nodes"]))
        self.assertEqual(read.work_counts["logical"]["cache_appended_bytes"], 0)
        self.assertEqual(first.work_counts["logical"]["useful_macs"], read.work_counts["logical"]["useful_macs"])

    def test_preprocess_binding_position_new_tokens_and_output_connections(self):
        p, shape = preprocess(), task()
        op = MLADecodeAttention.from_preprocess(p, shape.kv_up_weight, shape.cache, output_weight=shape.output_weight)
        self.assertEqual(op.new_latent, p.spec.outputs[2])
        self.assertEqual(op.new_rope, p.spec.outputs[3])
        before = simulate_block(p, "tpu", self.hardware["tpu"])
        after = self.run_attention(op)
        for i, name in enumerate(t.name for t in p.spec.outputs):
            index = next(j for j, t in enumerate(op.inputs) if t.name == name)
            edge = plan_connection(before, after, output_index=i, input_index=index)
            self.assertEqual(edge["conversion"]["status"], "not_required")
            self.assertIn("layout_validation", edge["requirements"])
        with self.assertRaisesRegex(ValueError, "position"):
            MLADecodeAttention.from_preprocess(replace(p, position=3), shape.kv_up_weight, shape.cache)
        readonly = MLADecodeAttention.from_preprocess(p, shape.kv_up_weight, replace(shape.cache, length=3), append_tokens=0)
        self.assertIsNone(readonly.new_latent)
        context = self.run_attention(task(projected=False))
        projection = simulate_block(AttentionOutputProjection(context.operator.outputs[0], shape.output_weight), "tpu", self.hardware["tpu"])
        self.assertEqual(plan_connection(context, projection)["conversion"]["status"], "not_required")

    def test_custom_cache_tokens_are_validated_and_aliases_rejected(self):
        c = task().cache
        kv = TensorSpec("custom_kv", (2, 1, 1, 7), "INT8")
        pe = TensorSpec("custom_pe", (2, 1, 1, 4), "INT8")
        op = MLACacheAccess(c, 1, False, kv, pe)
        r = simulate_mla_cache(op, "tpu", self.hardware["tpu"])
        self.assertEqual(r.operator.inputs[-2:], (kv, pe))
        for args in ((c, 1, False, kv, None), (c, 0, True, kv, pe),
                     (c, 1, False, TensorSpec("x", (2, 1, 1, 8), "INT8"), pe),
                     (c, 1, False, TensorSpec(c.buffers[0].name + ".after", kv.shape, "INT8"), pe)):
            with self.assertRaises(ValueError):
                MLACacheAccess(*args)

    def test_invalid_shapes_state_names_scale_and_position_rejected(self):
        op = task()
        for changes in ({"query_nope": TensorSpec("q", (2, 3, 2, 3), "INT8")},
            {"query_rope": TensorSpec("qp", (2, 3, 1, 4), "FP32")},
            {"kv_up_weight": TensorSpec("wkvb", (3, 9, 7), "FP32")},
            {"output_weight": TensorSpec("wo", (18, 18), "INT8")},
            {"dimensions": replace(op.dimensions, v_head_dim=8)}, {"position": 3}, {"position": True},
            {"scale_multiplier": -1}, {"scale_multiplier": True}, {"scale_multiplier": float("nan")},
            {"scale_multiplier": 2, "scale_source": None}, {"scale_source": ""},
            {"output_name": op.cache.buffers[0].name + ".after"}, {"output_name": "wkvb"},
            {"query_nope": TensorSpec(op.cache.buffers[0].name + ".after", op.query_nope.shape, "INT8")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(op, **changes)
        for opts in ({"length": 9}, {"length": 0, "append": 0}, {"append": True}):
            with self.assertRaises(ValueError):
                task(**opts)

    def test_missing_conversion_cost_and_unknown_clock_propagate(self):
        cfg = copy.deepcopy(load_block_execution().config)
        cfg["parameters"]["to_int8_cost"].update(value=None, status="unknown")
        block = BlockExecution(cfg)
        self.run_attention(block_execution=block)
        with self.assertRaisesRegex(ValueError, "to_int8_cost"):
            self.run_attention(block_execution=block, mode="timing")
        cfg = raw("tsp")
        cfg["parameters"]["clock_ghz"].update(value=None, status="unknown")
        r = simulate_mla_attention(task(), "tsp", TSPSpec(cfg), mode="timing")
        self.assertEqual(r.timing_status, "cycles_only")
        self.assertIsNone(r.latency_s)

    def test_strict_workload_fields(self):
        for key, value in (("unknown", 1), ("schema_version", True), ("output_weight", None),
                           ("dimensions", {}), ("position", -1), ("new_rope", {}), ("description", "")):
            cfg = config("mla_decode_attention")
            cfg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_mla_attention_workload(cfg)
        cfg = config("mla_decode_attention")
        del cfg["output_weight"]
        self.assertIsNone(parse_mla_attention_workload(cfg).output_weight)
        cfg["cache"]["query_heads"] = 3
        with self.assertRaises(ValueError):
            parse_mla_attention_workload(cfg)


if __name__ == "__main__":
    unittest.main()
