from tests.operators.inspection import inspect_cli_report
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class VectorCLITest(unittest.TestCase):
    def run_entry(self, arch, folder, *options):
        return subprocess.run([sys.executable, str(ROOT / f"{arch}.py"), *options, "--details-output", str(Path(folder) / "diagnostics.json")], cwd=folder,
                              text=True, encoding="utf-8", capture_output=True, timeout=30)

    def test_all_entries_run_vector_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "nested/result.json"
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for op in ("fp32_exp", "int32_add"):
                    for mode in ("count", "timing"):
                        with self.subTest(arch=arch, op=op, mode=mode):
                            result = self.run_entry(arch, folder, "--workload", f"operators/{op}.json", "--mode", mode,
                                                    "--vector-tile-elements", "80", "--output", str(output))
                            self.assertEqual(result.returncode, 0, result.stderr)
                            report = inspect_cli_report(output, Path(folder) / "diagnostics.json")
                            self.assertEqual(report["kind"], "operator_result")
                            self.assertEqual(report["operator"]["kind"], op)
                            self.assertEqual(report["execution_contract"]["selected_tile_elements"], 80)
                            self.assertEqual(report["mode"], mode)
                            self.assertIn("[Report]", result.stdout)

    def test_gemm_flags_rejected_before_output_is_touched(self):
        options = {"ascend": (("--tile", "16", "32", "16"), ("--trace",), ("--buf", "1")),
                   "tpu": (("--cmem-mode", "bypass"), ("--schedule", "serial")),
                   "tsp": (("--loop-order", "KNM"), ("--scope", "chip")),
                   "tsp_cim": (("--compare-mxm",), ("--load-overlap", "serialized"), ("--tile-m", "8"))}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "keep.json"
            output.write_text("keep", encoding="utf-8")
            for arch, cases in options.items():
                for case in (*cases, ("--report-format", "legacy")):
                    result = self.run_entry(arch, folder, "--workload", "operators/fp32_exp.json", "--output", str(output), *case)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(output.read_text(encoding="utf-8"), "keep")
                result = self.run_entry(arch, folder, "--vector-tile-elements", "1", "--output", str(output))
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(output.read_text(encoding="utf-8"), "keep")

    def test_explicit_profile_failure_and_tsp_resource_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            result = self.run_entry("tsp_cim", folder, "--workload", "operators/fp32_rsqrt.json", "--hemisphere", "1",
                                    "--mem-slices", "0", "1", "--streams", "2", "--reserved-bytes", "17",
                                    "--report-format", "operator", "--output", str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = inspect_cli_report(output, Path(folder) / "diagnostics.json")
            self.assertEqual(report["boundary"]["selected_hemisphere"], 1)
            self.assertEqual(report["native_report"]["memory"]["mem_slices"], [0, 1])
            self.assertEqual(report["native_report"]["memory"]["reserved_per_slice_bytes"], 9)
            saved = output.read_bytes()
            result = self.run_entry("tsp_cim", folder, "--workload", "operators/fp32_rsqrt.json",
                                    "--vector-config", str(Path(folder) / "missing.json"), "--output", str(output))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_bytes(), saved)
