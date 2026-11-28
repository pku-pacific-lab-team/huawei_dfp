import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.models.chip import load_hardware
from src.models.block_graph import BlockCapacityError
from src.models.config import DecodeRequest, load_model
from src.models.llama_block import _run_block, run_llama_block


ROOT = Path(__file__).resolve().parents[2]
TINY = {**load_model("llama3_70b"), "hidden_size": 32, "num_attention_heads": 4,
        "num_key_value_heads": 2, "intermediate_size": 48}


def nodes(step):
    return {node["id"]: node for node in step["nodes"]}


class LlamaBlockTest(unittest.TestCase):
    def test_tiny_two_steps_shapes_dependencies_and_reductions(self):
        request = DecodeRequest(2, 2, 2)
        report = _run_block(TINY, request, "tpu", load_hardware("tpu"), 2, 0, "timing")
        for length, step in zip((3, 4), next(iter(report["profiles"].values()))):
            n = nodes(step)
            t = step["tensors"]
            self.assertEqual(t["output"]["shape"], [2, 32])
            self.assertEqual(n["attention.residual"]["inputs"], ["input", "attention.reduce.to_int8"])
            self.assertEqual(n["output"]["inputs"], ["attention.residual", "ffn.reduce.to_int8"])
            for prefix, projection in (("attention", "attention.output.partial"), ("ffn", "ffn.down.partial")):
                reduce = prefix+".reduce"
                self.assertEqual(t[projection]["dtype"], "INT32")
                self.assertEqual(n[reduce]["depends_on"], [projection])
                self.assertEqual(n[reduce+".to_int8"]["depends_on"], [reduce])
                self.assertEqual(n[reduce]["payload_bytes_per_rank"], 2*32*4)
                self.assertEqual(n[reduce]["group_int32_adds"], 2*32)
                self.assertIsNone(n[reduce]["communication_cycles"])
            self.assertEqual(n["ffn.swiglu"]["depends_on"], ["ffn.gate", "ffn.up"])
            self.assertEqual(t["ffn.up.weight"]["shape"], [32,24])
            self.assertEqual(t["ffn.down.partial.weight"]["shape"], [24,32])
            self.assertEqual(n["kv.append"]["token_interval"], [length-1, length])
            self.assertEqual(n["kv.append"]["updates"], ["key.cache","value.cache"])
            self.assertEqual(n["kv.append"]["appended_bytes"], 32)
            for key in ("pv","qk.partial"):
                self.assertIn("kv.append", n[key]["depends_on"])
                self.assertEqual(n[key]["cache_read_payload_bytes"], 2*2*length*8)
            self.assertEqual(n["qk.sum"]["action_counts"]["int32_add"]["count"], 2*2*length)
            for name in ("attention_norm","ffn_norm"):
                a = n[name]["action_counts"]
                self.assertEqual(a["fp32_add"]["count"], 2*32)
                self.assertEqual(a["fp32_mul"]["count"], 2*(3*32+1))
                self.assertEqual(a["fp32_rsqrt"]["count"], 2)
                self.assertEqual(n[name]["normalizer_multiplier"], 1/32)
            soft = n["softmax"]["action_counts"]
            for op in ("fp32_add","fp32_max"):
                self.assertEqual(soft[op]["count"], 4*(length-1))
            self.assertEqual(soft["fp32_exp"]["count"], 4*length)
            self.assertEqual(soft["fp32_rcp"]["count"], 4)
            mac = 2*(32*(16+8+8) + 16*32 + 3*32*24 + 2*16*length)
            self.assertEqual(step["useful_macs"], mac)
            self.assertEqual(step["local_compute_cycles"], sum(row["local_compute_cycles"] for row in n.values()))

    def test_kv_replication_and_uneven_head_groups(self):
        for tp in (1, 3, 4):
            with self.subTest(tp=tp):
                r = _run_block(TINY, DecodeRequest(1, 6, 1), "tpu", load_hardware("tpu"), tp, 0, "count")
                pairs = [(q, kv) for c in r["chips"] for q,kv in zip(range(*c["query_heads"]), c["query_to_kv"])]
                self.assertEqual(pairs, [(0,0),(1,0),(2,1),(3,1)])
                heads = sum(c["kv_heads"][1]-c["kv_heads"][0] for c in r["chips"])
                self.assertEqual(heads, {1:2, 3:3, 4:4}[tp])
                total = r["step_totals"][0]
                self.assertEqual(total["physical_kv_append_bytes"], heads*8*2)
                self.assertEqual(total["replica_projection_macs"], 2*32*8*(heads-2))
                filled = 0
                for chip in r["chips"]:
                    step = r["profiles"][chip["profile"]][0]
                    filled += sum(step["tensors"][t]["logical_bytes"] for t in ("key.cache","value.cache"))
                    self.assertIsNone(step["local_compute_cycles"])
                self.assertEqual(filled, heads*2*8*7)

    def test_exact_and_one_byte_short_chip_capacity(self):
        spec = load_hardware("tpu")
        request = DecodeRequest(1, 3, 1)
        def run():
            return next(iter(_run_block(TINY, request, "tpu", spec, 2, 0, "count")["profiles"].values()))[0]
        step = run()
        peak = step["capacity"]["peak"]["required_bytes"]
        spec.values["hbm_usable_bytes"] = peak
        self.assertEqual(run()["capacity"]["peak"]["required_bytes"], peak)
        spec.values["hbm_usable_bytes"] -= 1
        with self.assertRaises(BlockCapacityError) as raised:
            run()
        diagnostic = raised.exception.to_dict()
        self.assertEqual((diagnostic["required_bytes"], diagnostic["capacity_bytes"]), (peak, peak-1))
        self.assertEqual(sum(diagnostic["tensors"].values()), peak)
        self.assertEqual((diagnostic["level"], diagnostic["chip"]), ("HBM", 0))
        self.assertIn("key.cache", diagnostic["tensors"])
        self.assertIn("ffn.down.partial.weight", diagnostic["tensors"])
        self.assertNotIn("input", nodes(step)["ffn.swiglu"]["live_tensors"])

    def test_real_half_capacity_fails_before_simulation(self):
        with patch("src.models.decoder_kernels.simulate_gemm", side_effect=AssertionError("ran GEMM")):
            with self.assertRaises(BlockCapacityError) as raised:
                run_llama_block(DecodeRequest(1,127,2), "tsp", load_hardware("tsp"), tp=2)
        self.assertEqual(raised.exception.level, "MEM")
        self.assertIn(raised.exception.unit, (0,1))

    def test_private_workspaces_and_resident_tile_fallback(self):
        for arch in ("ascend", "tpu"):
            spec = load_hardware(arch)
            if arch == "ascend":
                spec.capacities["L0A"] = 511
                expected = ("L0A", 512, 511)
            else:
                spec.values["vmem_usable_bytes"] = 35
                expected = ("vmem", 36, 35)  # 12 stored and 24 FP32 scratch bytes per RoPE pair.
            with self.subTest(arch=arch), self.assertRaises(BlockCapacityError) as raised:
                _run_block(TINY, DecodeRequest(1,3,1), arch, spec, 2, 0, "count")
            error = raised.exception
            self.assertEqual((error.level, error.required_bytes, error.capacity_bytes), expected)
        request = DecodeRequest(1,127,2)
        spec = load_hardware("tsp")
        def ffn_stage():
            result = run_llama_block(request, "tsp", spec)
            return nodes(next(iter(result["profiles"].values()))[0])["ffn.swiglu"]
        original = ffn_stage()
        spec.mem_total_bytes = original["required_bytes"] - spec.mem_n_slices
        fitted = ffn_stage()
        self.assertLessEqual(fitted["required_bytes"], spec.mem_total_bytes)
        self.assertLess(sum(fitted["scratch_per_unit"]), sum(original["scratch_per_unit"]))

    def test_four_real_rank_shards_count_and_timing(self):
        request = DecodeRequest(1,127,2)
        for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
            with self.subTest(arch=arch):
                spec = load_hardware(arch)
                previous = run_llama_block(request, arch, spec, mode="count")
                timed = run_llama_block(request, arch, spec, mode="timing")
                self.assertEqual(timed["chips"][1]["query_heads"], [16,32])
                self.assertEqual(timed["chips"][1]["kv_heads"], [2,4])
                for a,b in zip(next(iter(previous["profiles"].values())), next(iter(timed["profiles"].values()))):
                    self.assertEqual(a["useful_macs"], b["useful_macs"])
                    self.assertGreater(b["local_compute_cycles"], 0)
                    self.assertEqual([n["action_counts"] for n in a["nodes"]], [n["action_counts"] for n in b["nodes"]])
                    self.assertEqual(a["capacity"], b["capacity"])
                    self.assertEqual(nodes(b)["kv.append"]["appended_bytes"], 512)
                    self.assertEqual(b["tensors"]["key.cache"]["shape"], [1,2,129,128])
                self.assertEqual([t["logical_useful_macs"] for t in timed["step_totals"]], [857735168,857751552])

    def test_cli_diagnostic_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "block.json"
            cmd = [sys.executable, str(ROOT/"scripts/run_llama_block.py"), "--architecture", "tsp",
                   "--mode", "count", "--output", str(target)]
            subprocess.run(cmd, cwd=tmp, capture_output=True, check=True)
            content = target.read_bytes()
            self.assertEqual(json.loads(content)["kind"], "llama_decoder_block")
            failed = subprocess.run([*cmd, "--tp", "2"], cwd=tmp, capture_output=True)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(json.loads(failed.stderr)["status"], "insufficient_capacity")
            self.assertEqual(target.read_bytes(), content)
