"""All four CLI entry points, explicit profiles and rejected-output preservation."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.blocks import load_block_execution
from src.hardware.vector import load_vector_execution
from tests.operators.test_position_vocabulary import ROOT, KINDS, config


class PositionVocabularyCLITest(unittest.TestCase):
    def invoke(self, arch, directory, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args], cwd=directory,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_profiles_other_directory_and_allocation_on_four_entries(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, profile = (Path(folder) / f for f in ("r.json", "d.json", "b.json"))
            cfg = copy.deepcopy(load_block_execution().config)
            cfg["parameters"]["to_fp32_cost"]["value"] = 30
            profile.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                options = ["--mem-slices", "0", "1", "2", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    options += ["--hemisphere", "1"]
                for kind in KINDS:
                    p = self.invoke(arch, folder, "--workload", f"operators/{kind}.json", "--block-config", str(profile),
                                    "--vector-tile-elements", "5", "--output", str(output), "--details-output", str(details), *options)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    summary = json.loads(output.read_text(encoding="utf-8"))
                    report = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                    self.assertEqual(summary["operator"]["kind"], kind)
                    self.assertEqual(report["block_execution_profile"], cfg)
                    if kind == "rope":
                        self.assertEqual(summary["operator"]["attributes"]["position"], 5)
                        child = report["schedule"]["nodes"][0]["native_report"]
                        self.assertEqual(child["execution_contract"]["selected_tile_elements"], 4)
                    if options:
                        self.assertEqual(report["memory"]["lifetime_validation"]["reserved_bytes"], 18)
                    if arch == "tsp_cim":
                        self.assertEqual(summary["boundary"]["selected_hemisphere"], 1)

    def test_invalid_position_dtype_options_and_profiles_preserve_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, bad_rope, bad_embedding = (Path(folder) / f for f in ("r.json", "d.json", "rope.json", "embedding.json"))
            output.write_text("summary", encoding="utf-8")
            details.write_text("details", encoding="utf-8")
            cfg = config("rope")
            cfg["position"] = 9
            bad_rope.write_text(json.dumps(cfg), encoding="utf-8")
            cfg = config("embedding")
            cfg["inputs"][0]["dtype"] = "FP32"
            bad_embedding.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for args in (["--workload", str(bad_rope)], ["--workload", str(bad_embedding)],
                             ["--workload", "operators/rope.json", "--vector-tile-elements", "1"],
                             ["--workload", "operators/rope.json", "--attention-config", str(bad_rope)],
                             ["--workload", "operators/embedding.json", "--block-config", str(bad_rope)],
                             ["--workload", "operators/vocabulary_projection.json", "--report-format", "legacy"],
                             ["--workload", "operators/vocabulary_projection.json", "--vector-tile-elements", "0"]):
                    p = self.invoke(arch, folder, *args, "--output", str(output), "--details-output", str(details))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(output.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")

    def test_rope_requires_composite_profile_and_chip_mode_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder) / "v1.json"
            cfg = copy.deepcopy(load_vector_execution("tpu").config)
            cfg["schema_version"] = 1
            cfg.pop("composite")
            profile.write_text(json.dumps(cfg), encoding="utf-8")
            p = self.invoke("tpu", folder, "--workload", "operators/rope.json", "--vector-config", str(profile), "--output", "")
            self.assertEqual(p.returncode, 2)
            self.assertIn("schema 2", p.stderr)
            for kind in KINDS:
                p = self.invoke("tsp", folder, "--workload", f"operators/{kind}.json", "--scope", "chip", "--output", "")
                self.assertEqual(p.returncode, 2)
                self.assertIn("chip execution is not modeled", p.stderr)


if __name__ == "__main__":
    unittest.main()
