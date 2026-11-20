import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class OperatorReportCLITest(unittest.TestCase):
    def test_opt_in_report_preserves_default_report_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            workload = folder / "workload.json"
            workload.write_text(json.dumps({"gemm": {"M": 1, "K": 17, "N": 33}, "data_format": "INT8"}), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for mode in ("count", "timing"):
                    with self.subTest(arch=arch, mode=mode):
                        reports = []
                        for options in ([], ["--report-format", "operator", "--details-output", str(folder / "diagnostics.json")]):
                            target = folder / "result.json"
                            process = subprocess.run([sys.executable, str(ROOT / f"{arch}.py"),
                                "--workload", str(workload), "--mode", mode, "--output", str(target), *options],
                                cwd=folder, text=True, encoding="utf-8", capture_output=True, timeout=30)
                            self.assertEqual(process.returncode, 0, process.stderr)
                            reports.append(json.loads(target.read_text(encoding="utf-8")))
                        old, new = reports
                        self.assertNotIn("kind", old)
                        self.assertEqual(new["kind"], "operator_result")
                        self.assertNotIn("native_report", new)
                        self.assertEqual(json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))["native_report"], old)

    def test_invalid_workload_does_not_replace_operator_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            workload, target = folder / "bad.json", folder / "existing.json"
            workload.write_text(json.dumps({"operation": "softmax", "data_format": "FP32"}), encoding="utf-8")
            target.write_text("preserve", encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                process = subprocess.run([sys.executable, str(ROOT / f"{arch}.py"), "--workload", str(workload),
                    "--report-format", "operator", "--output", str(target)], cwd=folder,
                    capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertEqual(target.read_text(encoding="utf-8"), "preserve")
