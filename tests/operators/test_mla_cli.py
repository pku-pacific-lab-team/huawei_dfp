"""MLA CLI dispatch, explicit profiles and rejected-output preservation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.blocks import load_block_execution
from src.hardware.attention import load_attention_execution
from tests.operators.test_mla import ROOT, KINDS, config


class MLACLITest(unittest.TestCase):
    def invoke(self, arch, directory, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args], cwd=directory,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_profiles_other_directory_and_allocation_on_four_entries(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, block, attention = (Path(folder) / n for n in ("r.json", "d.json", "b.json", "a.json"))
            cfg = copy.deepcopy(load_block_execution().config)
            cfg["parameters"]["to_fp32_cost"]["value"] = 30
            block.write_text(json.dumps(cfg), encoding="utf-8")
            attention.write_text(json.dumps(load_attention_execution().config), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                options = ["--mem-slices", "0", "1", "2", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    options += ["--hemisphere", "1"]
                for kind in KINDS:
                    profile = ["--block-config", str(block)] if kind == "mla_preprocess" else ["--attention-config", str(attention)]
                    p = self.invoke(arch, folder, "--workload", f"operators/{kind}.json", "--mode", "timing", *profile,
                                    "--vector-tile-elements", "4", "--output", str(output), "--details-output", str(details), *options)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    result = json.loads(output.read_text(encoding="utf-8"))
                    report = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                    self.assertEqual(result["operator"]["kind"], kind)
                    self.assertGreater(result["cycle_count"], 0)
                    if kind == "mla_preprocess":
                        self.assertEqual(report["block_execution_profile"], cfg)
                    else:
                        self.assertTrue(report["execution_contract"]["shared_across_query_heads"])
                    if options:
                        self.assertEqual(report["memory"]["lifetime_validation"]["reserved_bytes"], 18)
                    if arch == "tsp_cim":
                        self.assertEqual(result["boundary"]["selected_hemisphere"], 1)

    def test_invalid_contracts_and_options_preserve_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, bad = (Path(folder) / n for n in ("r.json", "d.json", "bad.json"))
            cfg = config("mla_preprocess")
            cfg["dimensions"]["kv_lora_rank"] = 5
            bad.write_text(json.dumps(cfg), encoding="utf-8")
            output.write_text("summary", encoding="utf-8")
            details.write_text("details", encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for args in (["--workload", str(bad)],
                             ["--workload", "operators/mla_preprocess.json", "--attention-config", str(bad)],
                             ["--workload", "operators/mla_preprocess.json", "--vector-tile-elements", "1"],
                             ["--workload", "operators/mla_cache.json", "--block-config", str(bad)],
                             ["--workload", "operators/mla_cache.json", "--vector-tile-elements", "0"],
                             ["--workload", "operators/mla_cache.json", "--report-format", "legacy"]):
                    p = self.invoke(arch, folder, *args, "--output", str(output), "--details-output", str(details))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(output.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")

    def test_full_cache_overflow_and_chip_execution_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder) / "full.json"
            cfg = config("mla_cache")
            cfg["cache"]["length"] = cfg["cache"]["capacity"]
            bad.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                p = self.invoke(arch, folder, "--workload", str(bad), "--output", "")
                self.assertEqual(p.returncode, 2)
                self.assertIn("exceeds capacity", p.stderr)
            for kind in KINDS:
                p = self.invoke("tsp", folder, "--workload", f"operators/{kind}.json", "--scope", "chip", "--output", "")
                self.assertEqual(p.returncode, 2)
                self.assertIn("chip execution is not modeled", p.stderr)


if __name__ == "__main__":
    unittest.main()
