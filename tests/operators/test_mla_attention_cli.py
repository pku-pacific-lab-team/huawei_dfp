"""MLA Attention entry points, profiles, state errors and output protection."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.blocks import load_block_execution
from src.hardware.attention import load_attention_execution
from tests.operators.test_mla import ROOT, config


class MLAAttentionCLITest(unittest.TestCase):
    def invoke(self, arch, folder, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args], cwd=folder,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_four_entries_with_profiles_reservation_and_scaled_attention(self):
        with tempfile.TemporaryDirectory() as folder:
            out, details, workload, block, attention = (Path(folder) / n for n in ("r.json", "d.json", "w.json", "b.json", "a.json"))
            cfg = config("mla_decode_attention")
            cfg.update(scale_multiplier=2.25, scale_source="explicit squared mscale test")
            workload.write_text(json.dumps(cfg), encoding="utf-8")
            profile = copy.deepcopy(load_block_execution().config)
            profile["parameters"]["to_fp32_cost"]["value"] = 30
            block.write_text(json.dumps(profile), encoding="utf-8")
            attention.write_text(json.dumps(load_attention_execution().config), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                opts = ["--mem-slices", "0", "1", "2", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    opts += ["--hemisphere", "1"]
                p = self.invoke(arch, folder, "--workload", str(workload), "--mode", "timing", "--block-config", str(block),
                    "--attention-config", str(attention), "--vector-tile-elements", "2", "--output", str(out), "--details-output", str(details), *opts)
                self.assertEqual(p.returncode, 0, p.stderr)
                result = json.loads(out.read_text(encoding="utf-8"))
                native = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                self.assertEqual(result["operator"]["outputs"][0]["shape"], [2, 17])
                self.assertEqual(native["execution_contract"]["scale_multiplier"], 2.25)
                self.assertEqual(native["block_execution_profile"], profile)
                self.assertGreater(result["cycle_count"], 0)
                if opts:
                    self.assertEqual(native["memory"]["lifetime_validation"]["global_allocations"]["reserved"], 18)
                if arch == "tsp_cim":
                    self.assertEqual(result["boundary"]["selected_hemisphere"], 1)

    def test_context_only_read_only_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            workload, out = (Path(folder) / n for n in ("w.json", "r.json"))
            cfg = config("mla_decode_attention")
            for name in ("output_weight", "new_latent", "new_rope"):
                cfg.pop(name)
            cfg.update(append_tokens=0, position=1)
            workload.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                p = self.invoke(arch, folder, "--workload", str(workload), "--mode", "count", "--output", str(out))
                self.assertEqual(p.returncode, 0, p.stderr)
                result = json.loads(out.read_text(encoding="utf-8"))
                self.assertIsNone(result["cycle_count"])
                self.assertEqual(result["operator"]["outputs"][0]["shape"], [2, 3, 1, 6])
                self.assertFalse(any(k.startswith("cache_append.") for k in result["action_counts"]))

    def test_rejected_position_state_scale_and_options_preserve_files(self):
        with tempfile.TemporaryDirectory() as folder:
            out, details, bad = (Path(folder) / n for n in ("r.json", "d.json", "bad.json"))
            out.write_text("summary", encoding="utf-8")
            details.write_text("details", encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for change in ({"position": 3}, {"scale_multiplier": 2, "scale_source": None},
                               {"append_tokens": 0}, {"kv_up_weight": {"name": "w", "shape": [3, 7, 9], "dtype": "INT8"}}):
                    cfg = config("mla_decode_attention")
                    cfg.update(change)
                    bad.write_text(json.dumps(cfg), encoding="utf-8")
                    p = self.invoke(arch, folder, "--workload", str(bad), "--output", str(out), "--details-output", str(details))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(out.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")
                for opts in (["--vector-tile-elements", "0"], ["--report-format", "legacy"], ["--block-config", str(bad)]):
                    p = self.invoke(arch, folder, "--workload", "operators/mla_decode_attention.json", *opts,
                                    "--output", str(out), "--details-output", str(details))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(out.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")
            p = self.invoke("tsp", folder, "--workload", "operators/mla_decode_attention.json", "--scope", "chip", "--output", "")
            self.assertEqual(p.returncode, 2)
            self.assertIn("chip execution is not modeled", p.stderr)


if __name__ == "__main__":
    unittest.main()
