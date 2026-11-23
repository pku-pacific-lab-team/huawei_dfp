"""Four-entry Decode/cache configuration and failure-path integration."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.attention import load_attention_execution
from src.hardware.blocks import load_block_execution

ROOT = Path(__file__).resolve().parents[2]


class AttentionCLITest(unittest.TestCase):
    def invoke(self, arch, directory, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args], cwd=directory,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_profiles_and_allocations_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, profile, block = (Path(folder) / f for f in ("r.json", "d.json", "a.json", "b.json"))
            cfg = copy.deepcopy(load_attention_execution().config)
            cfg["profile_id"] = "test_decode_profile"
            profile.write_text(json.dumps(cfg), encoding="utf-8")
            b = copy.deepcopy(load_block_execution().config)
            b["parameters"]["to_int8_cost"]["value"] = 30
            block.write_text(json.dumps(b), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                options = ["--mem-slices", "0", "1", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    options += ["--hemisphere", "1"]
                proc = self.invoke(arch, folder, "--workload", "operators/decode_attention.json", "--attention-config", str(profile),
                                   "--block-config", str(block), "--vector-tile-elements", "3", "--output", str(output),
                                   "--details-output", str(details), *options)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                r = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                self.assertEqual(r["attention_execution_profile"], cfg)
                self.assertEqual(r["block_execution_profile"], b)
                self.assertEqual(r["cache_state"]["after"]["length"], 6)
                if options:
                    self.assertEqual(r["memory"]["lifetime_validation"]["global_allocations"]["reserved"], 18)
                if arch == "tsp_cim":
                    self.assertEqual(r["boundary"]["selected_hemisphere"], 1)

    def test_invalid_cache_query_and_profiles_preserve_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, bad = (Path(folder) / f for f in ("r.json", "d.json", "bad.json"))
            output.write_text("summary", encoding="utf-8")
            details.write_text("details", encoding="utf-8")
            cfg = json.loads((ROOT / "configs/workload/operators/decode_attention.json").read_text())
            cfg["query"]["shape"][1] = 3
            bad.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for args in (["--workload", str(bad)],
                             ["--workload", "operators/kv_cache.json", "--block-config", str(bad)],
                             ["--workload", "operators/decode_attention.json", "--attention-config", str(bad)],
                             ["--workload", "operators/kv_cache.json", "--vector-tile-elements", "0"],
                             ["--workload", "operators/decode_attention.json", "--report-format", "legacy"],
                             ["--workload", "operators/residual.json", "--attention-config", str(bad)]):
                    proc = self.invoke(arch, folder, *args, "--output", str(output), "--details-output", str(details))
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertEqual(output.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")

    def test_empty_cache_first_token_and_chip_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = json.loads((ROOT / "configs/workload/operators/decode_attention.json").read_text())
            cfg["cache"].update(length=0, capacity=1)
            work, output = Path(folder) / "first.json", Path(folder) / "r.json"
            work.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                p = self.invoke(arch, folder, "--workload", str(work), "--mode", "count", "--output", str(output))
                self.assertEqual(p.returncode, 0, p.stderr)
                r = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(r["operator"]["attributes"]["length_after"], 1)
            p = self.invoke("tsp", folder, "--workload", str(work), "--scope", "chip", "--output", "")
            self.assertEqual(p.returncode, 2)
            self.assertIn("chip execution is not modeled", p.stderr)


if __name__ == "__main__":
    unittest.main()
