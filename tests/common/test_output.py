"""CLI output paths are checked in an isolated project, never in user results."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENTRIES = ("ascend", "tpu", "tsp", "tsp_cim")


class OutputPathTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project"
        self.cwd = Path(self.temp.name) / "working-directory"
        self.project.mkdir()
        self.cwd.mkdir()
        # 复制最小运行工程，既验证真实 CLI，也避免测试覆盖用户的 results。
        for folder in ("src", "configs"):
            shutil.copytree(ROOT / folder, self.project / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for entry in ENTRIES:
            shutil.copy2(ROOT / f"{entry}.py", self.project / f"{entry}.py")
        self.workload = self.cwd / "workload.json"
        self.task = {"operation": "gemm", "gemm": {"M": 1, "K": 17, "N": 33}, "data_format": "INT8"}
        self.workload.write_text(json.dumps(self.task), encoding="utf-8")

    def run_entry(self, entry, *options):
        return subprocess.run(
            [sys.executable, str(self.project / f"{entry}.py"), "--workload", str(self.workload), *options],
            cwd=self.cwd, capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_defaults_create_project_results_from_another_directory(self):
        for entry in ENTRIES:
            timed = None
            for mode, options in (("timing", []), ("count", ["--mode", "count"])):
                with self.subTest(entry=entry, mode=mode):
                    result = self.run_entry(entry, *options)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    target = self.project / "results" / f"{entry}.json"
                    report = json.loads(target.read_text(encoding="utf-8"))
                    self.assertEqual(report["counts"]["useful_ops"], 2 * 17 * 33)
                    self.assertEqual(report["mode"], mode)
                    if mode == "timing":
                        timed = report
                        timing = report["timing"]
                        self.assertGreater(timing["total_cycles"], 0)
                        self.assertGreater(timing["time_ns"], 0)
                        self.assertEqual(timing["total_cycles"], sum(timing["stages_cycles"].values()))
                        self.assertTrue(timing["parameters"])
                        self.assertTrue(timing["assumptions"])
                    else:
                        self.assertIsNone(report["timing"]["total_cycles"])
                        self.assertEqual(report["counts"], timed["counts"])
                        self.assertEqual(report["action_counts"], timed["action_counts"])
                    self.assertIn(f"[Report] {target}", result.stdout)
        self.assertFalse((self.cwd / "results").exists())

    def test_explicit_relative_and_absolute_paths_override_defaults(self):
        for entry in ENTRIES:
            for relative in (True, False):
                with self.subTest(entry=entry, relative=relative):
                    target = self.cwd / entry / str(relative) / "nested" / "custom.json"
                    option = target.relative_to(self.cwd) if relative else target
                    result = self.run_entry(entry, "--output", str(option))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    report = json.loads(target.read_text(encoding="utf-8"))
                    self.assertEqual(report["counts"]["useful_ops"], 2 * 17 * 33)
        self.assertFalse((self.project / "results").exists())

    def test_failed_runs_do_not_create_or_replace_default_reports(self):
        for entry in ENTRIES:
            with self.subTest(entry=entry):
                target = self.project / "results" / f"{entry}.json"
                invalid = dict(self.task, operation="transformer")
                self.workload.write_text(json.dumps(invalid), encoding="utf-8")
                result = self.run_entry(entry)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(target.exists())
                self.workload.write_text(json.dumps(self.task), encoding="utf-8")
                self.assertEqual(self.run_entry(entry).returncode, 0)
                previous = target.read_bytes()
                self.workload.write_text(json.dumps(invalid), encoding="utf-8")
                self.assertEqual(self.run_entry(entry).returncode, 2)
                self.assertEqual(target.read_bytes(), previous)
