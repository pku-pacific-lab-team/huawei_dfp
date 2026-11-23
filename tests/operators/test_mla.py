"""MLA branch work, shared compressed state and native-service accounting."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.tpu import TPUSpec
from src.operators import (MLASpec, MLAPreprocess, MLACacheSpec, MLACacheAccess, TensorSpec,
                           simulate_block, simulate_mla_cache, simulate_gemm, plan_connection)
from src.operators.blocks import block_steps
from src.operators.mla import _ChannelBoundary
from src.operators.primitives import Gemm
from src.runtime.blocks import gemm_storage
from src.utils.operator_workload import parse_block_workload, parse_attention_workload
from tests.operators.test_gemm import specs
from tests.operators.test_vector import raw, synthetic, execution

ROOT = Path(__file__).resolve().parents[2]
KINDS = ("mla_preprocess", "mla_cache")


def config(kind):
    return json.loads((ROOT / f"configs/workload/operators/{kind}.json").read_text(encoding="utf-8"))


def preprocess():
    return parse_block_workload(config("mla_preprocess"))


def cache():
    return parse_attention_workload(config("mla_cache"))


def actions(result, suffix):
    return sum(a.count for a in result.action_ledger.entries if a.role == "action" and a.id.endswith(suffix))


class MLATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_preprocess(self, op=None, arch="tpu", **kw):
        return simulate_block(op or preprocess(), arch, self.hardware[arch], include_details=True, **kw)

    def run_cache(self, op=None, arch="tpu", **kw):
        return simulate_mla_cache(op or cache(), arch, self.hardware[arch], include_details=True, **kw)

    def test_four_architectures_count_timing_and_serial_work(self):
        for arch in self.hardware:
            for run in (self.run_preprocess, self.run_cache):
                with self.subTest(arch=arch, run=run.__name__):
                    a, b = (run(arch=arch, mode=m, tile_elements=4) for m in ("count", "timing"))
                    self.assertEqual(a.action_ledger, b.action_ledger)
                    self.assertEqual(a.native_report["memory"], b.native_report["memory"])
                    self.assertIsNone(a.cycle_count)
                    self.assertGreater(b.cycle_count, 0)
                    self.assertGreater(b.latency_s, 0)
                    self.assertFalse(b.boundary["chip_execution_included"])
                    self.assertTrue(all(t.dtype == "INT8" for t in b.operator.outputs))
                    json.dumps(b.to_detailed_dict(), allow_nan=False)
                    if run.__name__ == "run_preprocess":
                        nodes = b.native_report["schedule"]["nodes"]
                        self.assertEqual(len(nodes), 14)
                        self.assertEqual(b.cycle_count, sum(n["cycle_count"] for n in nodes))

    def test_independent_dimensions_and_precise_shapes(self):
        op = preprocess()
        self.assertEqual([t.shape for t in op.spec.outputs], [(2, 3, 1, 3), (2, 3, 1, 4), (2, 1, 1, 7), (2, 1, 1, 4)])
        self.assertEqual(op.epsilon, 1e-6)
        # Value dimension belongs to the future attention core, not these three projections.
        changed = replace(op, dimensions=replace(op.dimensions, v_head_dim=19))
        a, b = self.run_preprocess(op), self.run_preprocess(changed)
        self.assertEqual(a.action_ledger, b.action_ledger)
        self.assertEqual(a.resource_requirements, b.resource_requirements)

    def test_hand_calculated_macs_normalization_rotation_and_boundaries(self):
        for arch in self.hardware:
            r = self.run_preprocess(arch=arch)
            work = r.work_counts
            self.assertEqual(work["logical"]["useful_macs"], 2 * (17 * 5 + 5 * 21 + 17 * 11))
            self.assertEqual(work["logical"]["operation_counts"],
                             {"fp32_mul": 140, "fp32_add": 40, "fp32_rsqrt": 4, "fp32_sub": 16})
            self.assertEqual(work["logical"]["operations"], 2 * 754 + 200)
            self.assertEqual(work["logical"]["output_bytes"], 64)
            self.assertEqual(work["internal"]["boundary_elements"],
                             {"boundary_to_fp32": 106, "boundary_to_int8": 106})

    def test_selected_rows_account_payload_without_duplicate_full_reads(self):
        r = self.run_preprocess(tile_elements=2)
        nodes = {n["id"]: n["native_report"] for n in r.native_report["schedule"]["nodes"]}
        for name, elements, dtype in (("q_nope_split", 18, "INT8"), ("q_rope_split", 24, "INT8"),
                                      ("kv_split", 14, "FP32"), ("k_rope_split", 8, "INT8")):
            report = nodes[name]
            self.assertEqual(report["action_counts"]["hbm_read_bytes"]["count"], 4 * elements)
            self.assertEqual(report["action_counts"]["hbm_write_bytes"]["count"], elements * (4 if dtype == "FP32" else 1))
        groups = nodes["q_nope_split"]["schedule"]["tile_groups"]
        self.assertEqual([(g["elements"], g["repetitions"]) for g in groups], [(2, 6), (1, 6)])
        self.assertEqual(nodes["q_rope"]["work_counts"]["logical"]["operations"], 72)
        self.assertEqual(nodes["k_rope"]["work_counts"]["logical"]["operations"], 24)

    def test_accumulators_live_through_last_branch_and_outputs_persist(self):
        op = preprocess()
        r = self.run_preprocess(op)
        steps = dict(block_steps(op))
        life = r.native_report["memory"]["lifetime_validation"]
        lifetimes = {v["tensor"]: v for v in life["lifetimes"]}
        self.assertEqual(lifetimes[steps["q_up"].output_name]["last_step"], 6)
        self.assertEqual(lifetimes[steps["kv_down"].output_name]["last_step"], 12)
        self.assertEqual(lifetimes[op.spec.outputs[0].name]["last_step"], 13)
        self.assertEqual(len(lifetimes), 8 + 14)
        for row in life["steps"]:
            self.assertEqual(row["storage_bytes"], sum(lifetimes[n]["bytes"] for n in row["live_tensors"]))
        self.assertEqual(life["peak_bytes"], 845)

    def test_matrix_work_owned_by_native_kernels(self):
        op = preprocess()
        for arch, hardware in self.hardware.items():
            report = self.run_preprocess(op, arch).native_report
            nodes = {n["id"]: n["native_report"] for n in report["schedule"]["nodes"]}
            for name, stage in block_steps(op):
                if isinstance(stage, Gemm):
                    native = simulate_gemm(stage, arch, hardware, include_details=True).native_report
                    self.assertEqual(nodes[name]["action_counts"], native["action_counts"])
                    self.assertEqual(nodes[name]["work_counts"], native["work_counts"])

    def test_hbm_peak_and_full_table_capacity_rejected(self):
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 844
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(preprocess(), "tpu", TPUSpec(cfg))
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 845
        simulate_block(preprocess(), "tpu", TPUSpec(cfg))
        op = preprocess()
        op = replace(op, cos=TensorSpec("cos", (1000, 2), "FP32"), sin=TensorSpec("sin", (1000, 2), "FP32"))
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(op, "tpu", TPUSpec(cfg))

    def test_resident_storage_includes_disjoint_scratch_and_striping(self):
        r = self.run_preprocess(arch="tsp", mem_slices=[0, 1, 2], streams=2, reserved_bytes=17, tile_elements=4)
        life = r.native_report["memory"]["lifetime_validation"]
        self.assertEqual(life["reserved_bytes"], 18)
        self.assertTrue(all(v["bytes"] % 3 == 0 for v in life["lifetimes"]))
        self.assertTrue(any(v["scratch_bytes"] > 0 for v in life["steps"]))
        self.assertEqual(life["peak_bytes"], max(v["storage_bytes"] + v["scratch_bytes"] for v in life["steps"]))
        hardware = copy.deepcopy(self.hardware["tsp"])
        hardware.mem_total_bytes = hardware.mem_n_slices * (life["peak_bytes"] - 3) // 3
        smaller = simulate_block(preprocess(), "tsp", hardware, mem_slices=[0, 1, 2], streams=2,
                                 reserved_bytes=17, tile_elements=4, include_details=True)
        self.assertLess(smaller.resource_requirements["mem"]["required_bytes"], life["peak_bytes"])
        hardware.mem_total_bytes = hardware.mem_n_slices * (max(s["storage_bytes"] for s in life["steps"]) - 3) // 3
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_block(preprocess(), "tsp", hardware, mem_slices=[0, 1, 2], streams=2, reserved_bytes=17, tile_elements=4)

    def test_used_conversion_profile_and_rope_tile_failures(self):
        cfg = copy.deepcopy(load_block_execution().config)
        cfg["parameters"]["to_int8_cost"].update(value=None, status="unknown")
        profile = BlockExecution(cfg)
        self.run_preprocess(block_execution=profile, mode="count")
        with self.assertRaises(ValueError):
            self.run_preprocess(block_execution=profile, mode="timing")
        with self.assertRaises(ValueError):
            self.run_preprocess(tile_elements=1)

    def test_dimensions_and_preprocess_contract_rejections(self):
        op = preprocess()
        for field in op.dimensions.to_dict():
            for value in (0, -1, True, 2.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(op.dimensions, **{field: value})
        with self.assertRaises(ValueError):
            replace(op.dimensions, qk_rope_head_dim=3)
        for changes in ({"position": 9}, {"position": True}, {"epsilon": 0},
                        {"q_norm_weight": TensorSpec("q_norm", (5,), "INT8")},
                        {"kv_down_weight": TensorSpec("wkva", (17, 7), "INT8")},
                        {"q_up_weight": TensorSpec("wqb", (5, 21), "FP32")},
                        {"output_names": ("a", "a", "b", "c")}, {"output_names": ("x", "a", "b", "c")},
                        {"sin": TensorSpec("sin", (8, 2), "FP32")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(op, **changes)

    def test_channel_selection_rejects_invalid_geometry(self):
        s = dict(block_steps(preprocess()))["q_nope_split"]
        for change in ({"offset": 6}, {"rows": 5}, {"row_stride": 8}, {"row_elements": 4}, {"offset": True}):
            with self.assertRaises(ValueError):
                replace(s, **change)

    def test_cache_hand_bytes_and_state_aliases(self):
        for arch in self.hardware:
            r = self.run_cache(arch=arch, tile_elements=4)
            key = "hbm" if arch in {"ascend", "tpu"} else "mem"
            self.assertEqual(actions(r, key + "_read_bytes"), 22 + 66)
            self.assertEqual(actions(r, key + "_write_bytes"), 22)
            self.assertEqual(r.work_counts["logical"]["operations"], 0)
            state = r.native_report["cache_state"]
            self.assertEqual(state["filled_bytes_before"], 44)
            self.assertEqual(state["filled_bytes_after"], 66)
            self.assertEqual(state["write_token_interval"], [2, 3])
            self.assertEqual(state["output_aliases"], r.to_detailed_dict()["io_contract"]["state_aliases"])
            self.assertEqual(sum(v["logical_capacity_bytes"] for v in state["allocations"].values()), 198)
        self.assertEqual(self.run_cache().resource_requirements["hbm"]["required_bytes"], 220)

    def test_cache_reads_capacity_and_batch_scale_independently(self):
        op = cache()
        base = self.run_cache(op)
        more_capacity = self.run_cache(replace(op, cache=replace(op.cache, capacity=19)))
        self.assertEqual(base.action_ledger, more_capacity.action_ledger)
        self.assertEqual(more_capacity.resource_requirements["hbm"]["required_bytes"], 2 * 19 * 11 + 22)
        more_length = self.run_cache(replace(op, cache=replace(op.cache, length=5)))
        self.assertEqual(base.resource_requirements["hbm"], more_length.resource_requirements["hbm"])
        self.assertEqual(actions(more_length, "hbm_read_bytes"), 22 + 132)
        one_batch = self.run_cache(replace(op, cache=replace(op.cache, batch=1)))
        self.assertEqual(actions(base, "hbm_read_bytes"), 2 * actions(one_batch, "hbm_read_bytes"))
        self.assertEqual(base.resource_requirements["hbm"]["required_bytes"], 2 * one_batch.resource_requirements["hbm"]["required_bytes"])

    def test_cache_append_only_read_only_and_empty_to_full(self):
        initial = MLACacheAccess(MLACacheSpec(1, 3, 2, 0, 1))
        first = self.run_cache(initial)
        self.assertEqual(first.work_counts["logical"]["cache_read_bytes"], 5)
        read = self.run_cache(MLACacheAccess(initial.cache_after, 0, True))
        self.assertEqual(actions(read, "hbm_write_bytes"), 0)
        self.assertEqual(read.native_report["cache_state"]["write_token_interval"], [1, 1])
        append = self.run_cache(replace(initial, read_cache=False))
        self.assertEqual(actions(append, "hbm_read_bytes"), 5)
        self.assertEqual(actions(append, "hbm_write_bytes"), 5)
        self.assertEqual(append.work_counts["logical"]["cache_read_bytes"], 0)
        self.assertEqual(append.native_report["cache_state"]["filled_bytes_before"], 0)

    def test_cache_native_padding_covers_both_latent_roles_once(self):
        op = MLACacheAccess(MLACacheSpec(2, 17, 6, 1, 35))
        for arch, hardware in self.hardware.items():
            r = self.run_cache(op, arch)
            alloc = r.native_report["cache_state"]["allocations"]
            roles = [gemm_storage(Gemm(1, 17, 35), arch, hardware)["B"],
                     gemm_storage(Gemm(1, 35, 17), arch, hardware)["B"]]
            self.assertEqual(list(alloc["latent"]["native_view_bytes"].values()), roles)
            self.assertGreaterEqual(alloc["latent"]["bytes_per_batch"], max(roles))
            self.assertEqual(alloc["latent"]["allocated_bytes"], 2 * alloc["latent"]["bytes_per_batch"])
            self.assertEqual(set(alloc), {"latent", "rope"})

    def test_cache_hand_cycles_reuses_memory_services(self):
        op = MLACacheAccess(MLACacheSpec(1, 3, 2, 0, 1))
        for arch in ("ascend", "tpu", "tsp"):
            r = simulate_mla_cache(op, arch, synthetic(arch), execution=execution(arch), mode="timing", include_details=True)
            # Two components, each: append load + store + one filled read; all fit one service cycle.
            self.assertEqual(r.cycle_count, 6)
            self.assertEqual(r.work_counts["logical"]["appended_bytes"], 5)

    def test_cache_exact_hbm_capacity_and_resident_reservation(self):
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 219
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_mla_cache(cache(), "tpu", TPUSpec(cfg))
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 220
        simulate_mla_cache(cache(), "tpu", TPUSpec(cfg))
        r = self.run_cache(arch="tsp_cim", mem_slices=[0, 1, 2], streams=2, reserved_bytes=17, hemisphere=1)
        life = r.native_report["memory"]["lifetime_validation"]
        self.assertEqual(life["reserved_bytes"], 18)
        self.assertEqual(life["new_token_allocated_bytes"], 30)
        self.assertEqual(r.boundary["selected_hemisphere"], 1)

    def test_cache_rejects_invalid_states_and_noop(self):
        op = cache()
        for change in ({"length": -1}, {"length": 10}, {"capacity": 0}, {"batch": True},
                       {"kv_lora_rank": 0}, {"qk_rope_head_dim": 3}, {"dtype": "FP32"}):
            with self.assertRaises(ValueError):
                replace(op.cache, **change)
        for change in ({"append_tokens": True}, {"append_tokens": 2}, {"read_cache": 1},
                       {"append_tokens": 0, "read_cache": False}, {"cache": replace(op.cache, length=9)}):
            with self.assertRaises(ValueError):
                replace(op, **change)
        with self.assertRaisesRegex(ValueError, "empty"):
            MLACacheAccess(replace(op.cache, length=0), 0)

    def test_preprocess_cache_connection_preserves_shared_shapes(self):
        p, c = self.run_preprocess(), self.run_cache()
        for out, inp in ((2, 2), (3, 3)):
            edge = plan_connection(p, c, output_index=out, input_index=inp)
            self.assertEqual(edge["conversion"]["status"], "not_required")
            self.assertIn("layout_validation", edge["requirements"])
            self.assertIsNone(edge["additional_action_counts"])
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            plan_connection(p, c, output_index=1, input_index=3)

    def test_strict_workload_parsing(self):
        for kind, parser in (("mla_preprocess", parse_block_workload), ("mla_cache", parse_attention_workload)):
            for key, value in (("unknown", 1), ("schema_version", True)):
                cfg = config(kind)
                cfg[key] = value
                with self.assertRaises(ValueError):
                    parser(cfg)
        for field in MLASpec.__dataclass_fields__:
            cfg = config("mla_preprocess")
            del cfg["dimensions"][field]
            with self.assertRaises(ValueError):
                parse_block_workload(cfg)
        cfg = config("mla_cache")
        cfg["cache"]["query_heads"] = 128
        with self.assertRaises(ValueError):
            parse_attention_workload(cfg)


if __name__ == "__main__":
    unittest.main()
