"""Projection workload dispatch and profile/error handling through all four CLIs."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.blocks import load_block_execution
from tests.operators.test_projections import ROOT, KINDS, config


class ProjectionCLITest(unittest.TestCase):
    def invoke(self, arch, directory, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args], cwd=directory,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_four_entries_from_other_directory_with_custom_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, profile = (Path(folder) / f for f in ("r.json", "d.json", "b.json"))
            cfg = copy.deepcopy(load_block_execution().config)
            cfg["parameters"]["to_int8_cost"]["value"] = 30
            profile.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                options = ["--mem-slices", "0", "1", "2", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    options += ["--hemisphere", "1"]
                for kind in KINDS:
                    proc = self.invoke(arch, folder, "--workload", f"operators/{kind}.json", "--block-config", str(profile),
                                       "--vector-tile-elements", "3", "--output", str(output), "--details-output", str(details), *options)
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    summary = json.loads(output.read_text(encoding="utf-8"))
                    r = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                    self.assertEqual(r["block_execution_profile"], cfg)
                    self.assertEqual(summary["operator"]["kind"], kind)
                    self.assertGreater(summary["cycle_count"], 0)
                    self.assertEqual(len(summary["operator"]["outputs"]), 3 if kind == "qkv_projection" else 1)
                    if options:
                        self.assertEqual(r["memory"]["lifetime_validation"]["reserved_bytes"], 18)
                    if arch == "tsp_cim":
                        self.assertEqual(r["boundary"]["selected_hemisphere"], 1)

    def test_invalid_shapes_options_and_profiles_preserve_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, bad = (Path(folder) / f for f in ("r.json", "d.json", "bad.json"))
            output.write_text("summary", encoding="utf-8")
            details.write_text("details", encoding="utf-8")
            cfg = config("qkv_projection")
            cfg["query_heads"] = 3
            bad.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for args in (["--workload", str(bad)],
                             ["--workload", "operators/rmsnorm_int8.json", "--block-config", str(bad)],
                             ["--workload", "operators/qkv_projection.json", "--attention-config", str(bad)],
                             ["--workload", "operators/attention_output_projection.json", "--vector-tile-elements", "0"],
                             ["--workload", "operators/qkv_projection.json", "--report-format", "legacy"],
                             ["--workload", "operators/rmsnorm_int8.json", "--tile", "16", "16", "16"]):
                    proc = self.invoke(arch, folder, *args, "--output", str(output), "--details-output", str(details))
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertEqual(output.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")


if __name__ == "__main__":
    unittest.main()
