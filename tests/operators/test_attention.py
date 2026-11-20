"""Hand-derived Decode/GQA work, cache traffic, live storage and strict boundaries."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from src.hardware.attention import AttentionExecution, load_attention_execution
from src.hardware.blocks import BlockExecution, load_block_execution
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.operators import (TensorSpec, KVCacheSpec, KVCacheAccess, DecodeAttention,
                           simulate_attention, simulate_kv_cache, simulate_gemm, simulate_composite, plan_connection)
from src.operators.attention import attention_steps
from src.utils.operator_workload import parse_attention_workload
from tests.operators.test_gemm import specs
from tests.operators.test_vector import raw, synthetic, execution

ROOT = Path(__file__).resolve().parents[2]


def task(batch=2, heads=4, kv_heads=2, d=7, length=5, capacity=9, append=1):
    c = KVCacheSpec(batch, kv_heads, d, length, capacity)
    return DecodeAttention(TensorSpec("q", (batch, heads, 1, d), "INT8"), c, append)


class AttentionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def run_attention(self, op=None, arch="tpu", **kw):
        return simulate_attention(task() if op is None else op, arch, self.hardware[arch], include_details=True, **kw)

    def test_four_architecture_count_timing_and_serial_template(self):
        for arch in self.hardware:
            for op in (task(), task(batch=1, heads=1, kv_heads=1, length=0, capacity=1)):
                with self.subTest(arch=arch, heads=op.query_heads):
                    count, timed = (self.run_attention(op, arch, mode=m) for m in ("count", "timing"))
                    self.assertEqual(count.action_ledger, timed.action_ledger)
                    self.assertEqual(count.native_report["memory"], timed.native_report["memory"])
                    self.assertIsNone(count.cycle_count)
                    self.assertGreater(timed.cycle_count, 0)
                    report = timed.native_report
                    self.assertEqual(timed.cycle_count, sum(n["cycle_count_per_repetition"] * n["repetitions"] for n in report["schedule"]["nodes"]))
                    self.assertFalse(timed.boundary["chip_execution_included"])
                    self.assertNotIn("lifetime_validation", timed.resource_requirements)
                    self.assertEqual(report["memory"]["lifetime_validation"]["status"], "validated_symbolic_schedule")
                    json.dumps(timed.to_detailed_dict(), allow_nan=False)

    def test_hand_attention_arithmetic_and_dtype_boundaries(self):
        for arch in self.hardware:
            result = self.run_attention(arch=arch)
            logical = result.work_counts["logical"]
            # B=2,Hq=4,L=6,D=7; QK/PV each B*Hq*L*D MAC.
            self.assertEqual(logical["useful_macs"], 672)
            self.assertEqual(logical["operations"], 8 * (4 * 6 * 7 + 6 * 6 - 1))
            self.assertEqual(logical["operation_counts"], {"fp32_mul": 96, "fp32_max": 40, "fp32_sub": 48,
                                                         "fp32_exp": 48, "fp32_add": 40, "fp32_rcp": 8})
            self.assertEqual(result.work_counts["internal"]["boundary_elements"],
                             {"boundary_to_fp32": 104, "boundary_to_int8": 104})
            self.assertEqual(result.operator.outputs[0].dtype, "INT8")
            for action in result.action_ledger.entries:
                if action.name.startswith("fp32_"):
                    self.assertEqual(action.dtype, "FP32")

    def test_hand_attention_hbm_traffic_no_extra_cache_read(self):
        result = self.run_attention()
        actions = result.native_report["action_counts"]
        reads = sum(a["count"] for k, a in actions.items() if k.endswith("hbm_read_bytes"))
        writes = sum(a["count"] for k, a in actions.items() if k.endswith("hbm_write_bytes"))
        # Per head: reads =5D+2LD+21L, writes=5D+17L. Append adds56 B each.
        self.assertEqual(reads, 2016)
        self.assertEqual(writes, 1152)
        self.assertEqual(actions["cache_append.append.hbm_write_bytes"]["count"], 56)
        self.assertFalse(any(k.startswith("cache_append.read.") for k in actions))
        self.assertEqual(result.work_counts["logical"]["cache_read_payload_bytes"], 672)

    def test_gqa_shares_storage_but_not_native_reads(self):
        a, b = self.run_attention(task(heads=2)), self.run_attention(task(heads=4))
        self.assertEqual(a.native_report["cache_state"]["allocations"], b.native_report["cache_state"]["allocations"])
        self.assertEqual(a.native_report["cache_state"]["appended_bytes"], b.native_report["cache_state"]["appended_bytes"])
        self.assertEqual(b.work_counts["logical"]["useful_macs"], 2 * a.work_counts["logical"]["useful_macs"])
        self.assertEqual(b.work_counts["logical"]["cache_read_payload_bytes"], 2 * a.work_counts["logical"]["cache_read_payload_bytes"])
        self.assertEqual(b.native_report["schedule"]["group_size"], 2)

    def test_batch_sequence_and_capacity_have_distinct_scaling(self):
        a = self.run_attention(task(batch=1, append=0, length=3))
        batch = self.run_attention(task(batch=2, append=0, length=3))
        seq = self.run_attention(task(batch=1, append=0, length=6))
        cap = self.run_attention(task(batch=1, append=0, length=3, capacity=18))
        self.assertEqual(batch.work_counts["logical"]["operations"], 2 * a.work_counts["logical"]["operations"])
        self.assertEqual(seq.work_counts["logical"]["useful_macs"], 2 * a.work_counts["logical"]["useful_macs"])
        self.assertEqual(a.to_dict()["action_counts"], cap.to_dict()["action_counts"])
        self.assertEqual(a.work_counts, cap.work_counts)
        self.assertGreater(cap.resource_requirements["hbm"]["required_bytes"], a.resource_requirements["hbm"]["required_bytes"])

    def test_live_peak_one_workspace_and_state_aliases(self):
        r = self.run_attention().native_report
        life = r["memory"]["lifetime_validation"]
        # Q56+K252+V252+output56=616 persistent; append inputs56 =>672 peak.
        self.assertEqual(life["global_allocations"], {"query": 56, "key_cache": 252, "value_cache": 252, "output": 56, "reserved": 0})
        self.assertEqual([s["storage_bytes"] for s in life["steps"]], [640, 664, 664, 646, 650, 644])
        self.assertEqual(life["peak_bytes"], 672)
        self.assertEqual(len(life["view_bindings"]), 4)
        self.assertNotIn("head.scores", life["steps"][2]["live_tensors"])
        self.assertNotIn("head.probabilities_fp32", life["steps"][4]["live_tensors"])
        io = self.run_attention().to_detailed_dict()["io_contract"]
        self.assertEqual(io["state_aliases"], {"kv_cache.key.after": "kv_cache.key", "kv_cache.value.after": "kv_cache.value"})

    def test_all_heads_share_one_workspace_without_report_expansion(self):
        a = self.run_attention(task(batch=1, heads=4, kv_heads=1, append=0))
        b = self.run_attention(task(batch=2, heads=256, kv_heads=1, append=0))
        self.assertEqual(len(a.native_report["schedule"]["nodes"]), len(b.native_report["schedule"]["nodes"]))
        self.assertEqual(a.resource_requirements["vmem"]["required_bytes"], b.resource_requirements["vmem"]["required_bytes"])
        self.assertEqual(b.native_report["schedule"]["head_repetitions"], 512)

    def test_native_gemm_and_softmax_counts_reused_with_repetitions(self):
        op = task(length=33, capacity=35, d=17, append=0)
        for arch, spec in self.hardware.items():
            report = self.run_attention(op, arch).native_report
            nodes = {n["id"]: n for n in report["schedule"]["nodes"]}
            _, steps = attention_steps(op)
            for node_id, child in steps:
                if node_id in ("qk", "pv"):
                    result = simulate_gemm(child, arch, spec)
                elif node_id == "softmax":
                    result = simulate_composite(child, arch, spec)
                else:
                    continue
                for action in result.action_ledger.entries:
                    self.assertEqual(report["action_counts"][node_id + "." + action.id]["count"], 8 * action.count)
                self.assertEqual(nodes[node_id]["repetitions"], 8)

    def test_native_cim_summary_roles_survive_repetition(self):
        result = self.run_attention(arch="tsp_cim")
        index = {a.id: a for a in result.action_ledger.entries}
        refs = [a for a in index.values() if a.role == "reference"]
        summaries = [a for a in index.values() if a.role == "summary"]
        self.assertTrue(refs and summaries)
        for action in summaries:
            self.assertEqual(action.count, sum(index[k].count for k in action.summarizes))

    def test_cache_capacity_counts_unused_slots_but_reads_only_filled_range(self):
        cache = KVCacheSpec(2, 2, 7, 5, 9)
        for arch, spec in self.hardware.items():
            a = simulate_kv_cache(KVCacheAccess(cache), arch, spec, include_details=True)
            b = simulate_kv_cache(KVCacheAccess(KVCacheSpec(2, 2, 7, 5, 100)), arch, spec, include_details=True)
            self.assertEqual(a.to_dict()["action_counts"], b.to_dict()["action_counts"])
            self.assertEqual(a.work_counts["logical"]["appended_bytes"], 56)
            self.assertEqual(a.work_counts["logical"]["cache_read_bytes"], 336)
            self.assertEqual(a.native_report["cache_state"]["write_token_interval"], [5, 6])
            memory = "hbm" if arch in {"ascend", "tpu"} else "mem"
            self.assertGreater(b.resource_requirements[memory]["required_bytes"], a.resource_requirements[memory]["required_bytes"])
            self.assertNotIn("lifetime_validation", a.resource_requirements)

    def test_cache_copy_and_read_hand_cycles_all_memory_paths(self):
        access = KVCacheAccess(KVCacheSpec(2, 2, 7, 5, 9))
        for arch in ("ascend", "tpu", "tsp"):
            opts = {"mem_slices": [0], "streams": 1} if arch == "tsp" else {}
            r = simulate_kv_cache(access, arch, synthetic(arch), execution=execution(arch), mode="timing", include_details=True, **opts)
            # Eight K/V heads: append read/write 1+1; filled42-byte read ceil(42/16)=3.
            self.assertEqual(r.cycle_count, 40)
            counts = r.native_report["action_counts"]
            memory = "mem" if arch == "tsp" else "hbm"
            self.assertEqual(counts[f"append.{memory}_read_bytes"]["count"], 56)
            self.assertEqual(counts[f"append.{memory}_write_bytes"]["count"], 56)
            self.assertEqual(counts[f"read.{memory}_read_bytes"]["count"], 336)
            self.assertFalse(any("mac" in k for k in counts))

    def test_cache_states_are_immutable_and_append_bytes_constant(self):
        cache = KVCacheSpec(1, 1, 7, 0, 3)
        for index in range(3):
            access = KVCacheAccess(cache)
            r = simulate_kv_cache(access, "tpu", self.hardware["tpu"])
            self.assertEqual(cache.length, index)
            self.assertEqual(r.work_counts["logical"]["appended_bytes"], 14)
            self.assertEqual(r.work_counts["logical"]["cache_read_bytes"], 14 * (index + 1))
            cache = access.cache_after
        with self.assertRaisesRegex(ValueError, "exceeds capacity"):
            KVCacheAccess(cache)
        read_only = simulate_kv_cache(KVCacheAccess(cache, 0), "tpu", self.hardware["tpu"])
        self.assertTrue(all(a.name == "read" or a.name == "write" for a in read_only.action_ledger.entries))
        self.assertFalse(any(a.id.startswith("append.") for a in read_only.action_ledger.entries))

    def test_read_only_mha_gqa_mqa_and_one_token_softmax(self):
        for heads, kv in ((4, 4), (4, 2), (4, 1)):
            result = self.run_attention(task(batch=1, heads=heads, kv_heads=kv, length=1, capacity=1, append=0))
            self.assertEqual(result.work_counts["logical"]["useful_macs"], 56)
            self.assertEqual(result.native_report["cache_state"]["appended_bytes"], 0)
            self.assertEqual(len(result.native_report["schedule"]["nodes"]), 6)

    def test_capacity_failure_even_when_individual_kernels_fit(self):
        cfg = raw("tpu")
        cfg["parameters"]["hbm_usable_bytes"]["value"] = 670
        spec = TPUSpec(cfg)
        for node_id, op in attention_steps(task())[1]:
            if node_id in ("qk", "pv"):
                simulate_gemm(op, "tpu", spec)
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_attention(task(), "tpu", spec)
        # With append disabled, the 664-byte graph fits this capacity.
        simulate_attention(task(length=6, append=0), "tpu", spec)

    def test_resident_striping_reserved_and_scratch(self):
        for arch in ("tsp", "tsp_cim"):
            r = self.run_attention(arch=arch, mem_slices=[0, 1, 2], streams=2, reserved_bytes=1, hemisphere=1).native_report
            life = r["memory"]["lifetime_validation"]
            self.assertEqual(life["global_allocations"]["reserved"], 3)
            for item in r["cache_state"]["allocations"].values():
                self.assertEqual(item["bytes_per_head"] % 3, 0)
            nodes = {n["id"]: n["native_report"] for n in r["schedule"]["nodes"]}
            for step in life["steps"]:
                self.assertEqual(nodes[step["node"]]["memory"]["resident_bytes"], step["peak_bytes"])
                self.assertEqual(step["peak_bytes"] % 3, 0)
            self.assertGreater(life["steps"][2]["scratch_bytes"], 0)

    def test_resident_capacity_includes_softmax_scratch(self):
        op = task(batch=1, heads=1, kv_heads=1, append=0)
        spec = copy.deepcopy(self.hardware["tsp"])
        r = simulate_attention(op, "tsp", spec, mem_slices=[0], tile_elements=1, include_details=True)
        steps = r.native_report["memory"]["lifetime_validation"]["steps"]
        maximum_storage = max(s["storage_bytes"] for s in steps)
        spec.mem_total_bytes = maximum_storage * spec.mem_n_slices
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            simulate_attention(op, "tsp", spec, mem_slices=[0], tile_elements=1)

    def test_tile_policy_changes_rounding_not_logical_work(self):
        a, b = (self.run_attention(mode="timing", tile_elements=n) for n in (1, 32))
        # Reduction tile boundaries change local row-state accesses, not HBM payload.
        select = lambda r: {k: v for k, v in r.to_dict()["action_counts"].items() if "hbm_" in k or k.endswith("matrix_mac")}
        self.assertEqual(select(a), select(b))
        self.assertEqual(a.work_counts, b.work_counts)
        self.assertGreater(a.cycle_count, b.cycle_count)

    def test_missing_conversion_and_clock_values(self):
        cfg = copy.deepcopy(load_block_execution().config)
        cfg["parameters"]["to_int8_cost"].update(value=None, status="unknown")
        profile = BlockExecution(cfg)
        self.run_attention(block_execution=profile)
        with self.assertRaisesRegex(ValueError, "to_int8_cost"):
            self.run_attention(block_execution=profile, mode="timing")
        cfg = raw("tsp")
        cfg["parameters"]["clock_ghz"].update(value=None, status="unknown")
        r = simulate_attention(task(), "tsp", TSPSpec(cfg), mode="timing")
        self.assertEqual(r.timing_status, "cycles_only")

    def test_descriptor_rejects_prefill_invalid_grouping_and_alias(self):
        cases = [lambda: task(heads=3), lambda: task(length=9), lambda: task(append=2), lambda: task(append=True),
                 lambda: task(length=0, append=0), lambda: KVCacheSpec(1, 1, 8, -1, 9),
                 lambda: KVCacheSpec(1, 1, 8, 1, 9, dtype="FP32"),
                 lambda: DecodeAttention(TensorSpec("q", (2, 4, 2, 7), "INT8"), task().cache),
                 lambda: DecodeAttention(TensorSpec("kv_cache.key", (2, 4, 1, 7), "INT8"), task().cache),
                 lambda: DecodeAttention(TensorSpec("kv_cache.key.after", (2, 4, 1, 7), "INT8"), task().cache),
                 lambda: DecodeAttention(task().query, task().cache, output_name="kv_cache.key.after")]
        for case in cases:
            with self.assertRaises(ValueError):
                case()

    def test_connection_requires_head_layout_validation(self):
        result = self.run_attention(task(append=0))
        edge = plan_connection(result, result)
        self.assertIn("layout_validation", edge["requirements"])
        self.assertIsNone(edge["latency_s"])

    def test_strict_workload_profile_and_source_audit(self):
        for name in ("kv_cache", "decode_attention"):
            cfg = json.loads((ROOT / f"configs/workload/operators/{name}.json").read_text())
            self.assertEqual(parse_attention_workload(cfg).operation, name)
            for key, value in (("schema_version", True), ("mask", "causal"), ("cache", None), ("append_tokens", -1)):
                bad = copy.deepcopy(cfg)
                bad[key] = value
                with self.assertRaises(ValueError):
                    parse_attention_workload(bad)
        profile = load_attention_execution()
        for key, value in (("schema_version", True), ("reuse", "shared_free"), ("layout", "real_transpose")):
            bad = copy.deepcopy(profile.config)
            bad[key] = value
            with self.assertRaises(ValueError):
                AttentionExecution(bad)
        audit = json.loads((ROOT / "calibration/operator_execution_audit.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(profile.config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(audit["attention_execution_profiles"]["configs/execution/attention.json"]["canonical_json_sha256"], digest)


if __name__ == "__main__":
    unittest.main()
