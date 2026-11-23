"""Block CLI allocation/profile propagation and rejected-input output safety."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.blocks import load_block_execution

ROOT = Path(__file__).resolve().parents[2]


class BlockCLITest(unittest.TestCase):
    def invoke(self, arch, directory, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args],
                              cwd=directory, capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_allocations_profile_and_details_from_external_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, profile = (Path(folder) / name for name in ("r.json", "d.json", "b.json"))
            cfg = copy.deepcopy(load_block_execution().config)
            cfg["parameters"]["to_int8_cost"]["value"] = 30
            profile.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                options = ["--mem-slices", "0", "1", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    options += ["--hemisphere", "1"]
                proc = self.invoke(arch, folder, "--workload", "operators/swiglu_ffn.json", "--block-config", str(profile),
                                   "--vector-tile-elements", "3", "--output", str(output), "--details-output", str(details), *options)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                r = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                self.assertEqual(r["block_execution_profile"], cfg)
                self.assertEqual(r["schedule"]["nodes"][2]["native_report"]["execution_contract"]["selected_tile_elements"], 3)
                if options:
                    self.assertEqual(r["boundary"]["mem_slices"], [0, 1])
                    self.assertEqual(r["memory"]["lifetime_validation"]["reserved_bytes"], 18)
                if arch == "tsp_cim":
                    self.assertEqual(r["boundary"]["selected_hemisphere"], 1)

    def test_invalid_options_workload_and_profile_preserve_existing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output, details, bad = (Path(folder) / name for name in ("r.json", "d.json", "bad.json"))
            output.write_text("summary sentinel", encoding="utf-8")
            details.write_text("details sentinel", encoding="utf-8")
            bad.write_text('{"schema_version":1,"operation":"swiglu_ffn","inputs":[]}', encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for args in (["--workload", str(bad)],
                             ["--workload", "operators/residual.json", "--block-config", str(bad)],
                             ["--workload", "operators/swiglu_ffn.json", "--vector-tile-elements", "0"],
                             ["--workload", "operators/swiglu_ffn.json", "--report-format", "legacy"],
                             ["--workload", "operators/fp32_add.json", "--block-config", str(bad)]):
                    proc = self.invoke(arch, folder, *args, "--output", str(output), "--details-output", str(details))
                    self.assertEqual(proc.returncode, 2, proc.stderr)
                    self.assertTrue(proc.stderr)
                    self.assertEqual(output.read_text(), "summary sentinel")
                    self.assertEqual(details.read_text(), "details sentinel")

    def test_chip_execution_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = self.invoke("tsp", folder, "--workload", "operators/swiglu_ffn.json", "--scope", "chip", "--output", "")
            self.assertEqual(proc.returncode, 2)
            self.assertIn("chip execution is not modeled", proc.stderr)


if __name__ == "__main__":
    unittest.main()
