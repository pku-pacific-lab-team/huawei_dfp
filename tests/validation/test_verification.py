import copy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import verification as common
from scripts import verify_issue2 as issue2
from scripts import verify_issue3 as issue3


def test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from test_ids(test)
        else:
            yield test.id()


class VerificationTest(unittest.TestCase):
    def test_issue_suites_are_disjoint_and_scope_covers_operator_modules(self):
        native = set(test_ids(common.discover(common.ISSUE2_TEST_DIRS)))
        operators = set(test_ids(common.discover(common.ISSUE3_TEST_DIRS)))
        self.assertTrue(native and operators)
        self.assertFalse(native & operators)
        self.assertFalse(any(t.startswith(("tests.operators.", "tests.validation.")) for t in native))
        manifest = json.loads(issue3.MANIFEST.read_text(encoding="utf-8"))
        declared = {m for r in manifest["requirements"].values() for m in r["test_modules"]}
        self.assertEqual(declared, {t.rsplit(".", 2)[0] for t in operators})
        commands = [n for r in manifest["requirements"].values() for n in issue3.command_names(r["cli_operations"])]
        self.assertEqual(len(commands), 192)
        self.assertEqual(len(commands), len(set(commands)))

    def test_owned_changes_and_shared_dependencies_have_distinct_snapshots(self):
        native, operators = issue2.snapshot(), issue3.snapshot()
        self.assertNotIn("configs/execution/vector.json", native)
        self.assertNotIn("src/operators/types.py", native)
        self.assertIn("src/operators/types.py", issue2.dependency_snapshot())
        for path in ("src/runtime/simulation.py", "src/runtime/backends.py", "src/core/operator_result.py",
                     "src/core/vector_resources.py", "src/core/moe.py"):
            self.assertNotIn(path, native)
            self.assertIn(path, issue2.dependency_snapshot())
            self.assertIn(path, operators["issue3"])
        self.assertIn("validation/issue3/baselines/layering.json", operators["issue3"])
        self.assertIn("configs/execution/vector.json", operators["issue3"])
        self.assertIn("src/core/tpu/simulator.py", operators["shared_dependencies"])
        self.assertFalse(any(k.startswith("validation/issue2/") for group in operators.values() for k in group))
        read = Path.read_text
        changed = common.ROOT / "configs/execution/vector.json"
        def altered(path, *args, **kwargs):
            return read(path, *args, **kwargs) + ("\n" if path == changed else "")
        with mock.patch.object(Path, "read_text", altered):
            self.assertEqual(issue2.snapshot(), native)
            self.assertNotEqual(issue3.snapshot()["issue3"], operators["issue3"])
        changed = common.ROOT / "src/core/tpu/simulator.py"
        with mock.patch.object(Path, "read_text", altered):
            self.assertNotEqual(issue2.snapshot(), native)
            self.assertNotEqual(issue3.snapshot()["shared_dependencies"], operators["shared_dependencies"])

    def test_cross_issue_output_is_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for current, other in (("issue2", "issue3"), ("issue3", "issue2")):
                for path in (root, root / "validation", root / "validation" / other,
                             root / "validation" / other / "run", root / "validation" / current / ".." / other / "run"):
                    with self.assertRaises(ValueError):
                        common.output_directory(path, current, root)
                allowed = root / "validation" / current / "run"
                self.assertEqual(common.output_directory(allowed, current, root), allowed.resolve())
            self.assertEqual(list(root.iterdir()), [])
        for runner, other in ((issue2, "issue3"), (issue3, "issue2")):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                runner.main(["--output-dir", str(common.ROOT / "validation" / other / "run")])
            self.assertEqual(error.exception.code, 2)

    def test_missing_failed_and_unmapped_evidence_cannot_pass(self):
        manifest = {"requirements": {"one": {"title": "example", "test_modules": ["tests.example"], "cli_operations": ["gemm"]}}}
        records = {"tests.example.Case.test_ok": {"status": "passed"}}
        commands = {name: {"passed": True} for name in issue3.command_names(["gemm"])}
        self.assertEqual(issue3.evaluate_conditions(records, commands, manifest)["one"]["status"], "passed")
        self.assertEqual(issue3.evaluate_conditions({}, commands, manifest)["one"]["status"], "failed")
        for status in ("failed", "error", "skipped", "expected_failure"):
            bad = {"tests.example.Case.test_ok": {"status": status}}
            self.assertEqual(issue3.evaluate_conditions(bad, commands, manifest)["one"]["status"], "failed")
        missing = copy.deepcopy(commands)
        missing.pop(next(iter(missing)))
        self.assertEqual(issue3.evaluate_conditions(records, missing, manifest)["one"]["status"], "failed")
        extra = {**records, "tests.unknown.Case.test_new": {"status": "passed"}}
        self.assertEqual(issue3.evaluate_conditions(extra, commands, manifest)["coverage_mapping"]["status"], "failed")

    def test_subtest_failure_and_skips_do_not_become_success(self):
        class Broken(unittest.TestCase):
            def runTest(self):
                with self.subTest(case="bad"):
                    self.fail("expected verifier test")
                with self.subTest(case="good"):
                    self.assertTrue(True)
        class Skipped(unittest.TestCase):
            def runTest(self):
                self.skipTest("expected verifier test")
        with tempfile.TemporaryDirectory() as directory:
            _, counts, passed = common.run_tests(unittest.TestSuite((Broken(), Skipped())), Path(directory))
            self.assertFalse(passed)
            self.assertEqual(counts, {"failed": 1, "skipped": 1})

    def test_cli_timeout_and_stale_reports_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "tpu_gemm_count.json").write_text('{"stale":true}', encoding="utf-8")
            with mock.patch.object(issue3, "ARCHITECTURES", ("tpu",)), mock.patch.object(issue3, "OPERATIONS", ("gemm",)), \
                 mock.patch.object(issue3.subprocess, "run", side_effect=subprocess.TimeoutExpired("python", 60)):
                commands = issue3.run_commands(out)
            self.assertEqual(len(commands), 4)
            self.assertTrue(all(not c["passed"] for c in commands.values()))
            self.assertTrue(all("report" not in c for c in commands.values()))

    def test_execution_audit_is_separate_from_hardware_audit(self):
        hardware = json.loads((common.ROOT / "calibration/parameter_source_audit.json").read_text(encoding="utf-8"))
        operators = json.loads((common.ROOT / "calibration/operator_execution_audit.json").read_text(encoding="utf-8"))
        self.assertNotIn("vector_execution_profiles", hardware)
        self.assertIn("vector_execution_profiles", operators)
        self.assertFalse(set(operators["sources"]) & set(hardware["sources"]))
        self.assertTrue(operators["issue"].endswith("/issues/3"))
