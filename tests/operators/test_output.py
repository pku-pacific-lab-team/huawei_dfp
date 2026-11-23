"""Real operator CLI output routing in an isolated project."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURES = ("ascend", "tpu", "tsp", "tsp_cim")


class OperatorOutputTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.project = Path(directory.name) / "project"
        self.cwd = Path(directory.name) / "working"
        self.project.mkdir()
        self.cwd.mkdir()
        for folder in ("src", "configs"):
            shutil.copytree(ROOT / folder, self.project / folder,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for arch in ARCHITECTURES:
            shutil.copy2(ROOT / f"{arch}.py", self.project / f"{arch}.py")

    def run_entry(self, arch, kind, *options):
        return subprocess.run([sys.executable, str(self.project / f"{arch}.py"),
                               "--workload", f"operators/{kind}.json", "--mode", "count", *options],
                              cwd=self.cwd, text=True, encoding="utf-8", capture_output=True, timeout=30)

    def test_defaults_separate_architectures_operations_and_preserve_failed_outputs(self):
        results = self.project / "results"
        results.mkdir()
        for arch in ARCHITECTURES:
            legacy = results / f"{arch}.json"
            legacy.write_text("existing GEMM report", encoding="utf-8")
            for kind in ("fp32_add", "rmsnorm", "softmax"):
                with self.subTest(architecture=arch, operation=kind):
                    process = self.run_entry(arch, kind)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    target = results / f"{arch}_{kind}.json"
                    report = json.loads(target.read_text(encoding="utf-8"))
                    self.assertEqual(report["operator"]["kind"], kind)
                    self.assertEqual(report["architecture"], arch)
                    self.assertIn(f"[Report] {target}", process.stdout)
                    before = target.read_bytes()
                    failed = self.run_entry(arch, kind, "--vector-config", str(self.cwd / "missing.json"))
                    self.assertEqual(failed.returncode, 2)
                    self.assertEqual(target.read_bytes(), before)
            self.assertEqual(legacy.read_text(encoding="utf-8"), "existing GEMM report")
        self.assertEqual(len(list(results.glob("*.json"))), 16)
        self.assertFalse((self.cwd / "results").exists())

    def test_explicit_relative_absolute_and_empty_outputs_override_defaults(self):
        for arch in ARCHITECTURES:
            for absolute in (False, True):
                with self.subTest(architecture=arch, absolute=absolute):
                    target = self.cwd / arch / str(absolute) / "custom.json"
                    path = target if absolute else target.relative_to(self.cwd)
                    # Preserve argparse's long-option abbreviation and = syntax.
                    option = f"--outp={path}" if absolute else f"--output={path}"
                    process = self.run_entry(arch, "fp32_exp", option)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["operator"]["kind"], "fp32_exp")
            process = self.run_entry(arch, "fp32_exp", "--output", "")
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertNotIn("[Report]", process.stdout)
        self.assertFalse((self.project / "results").exists())
