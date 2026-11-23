import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
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
    def test_main_regression_and_delivery_modes_return_distinct_exit_codes(self):
        manifest = json.loads(issue3.MANIFEST.read_text(encoding="utf-8"))
        records = {n: {"status": "passed"} for n in test_ids(common.discover(common.ISSUE3_TEST_DIRS))}
        conditions = {k: {"status": "passed"} for k in manifest["requirements"]}
        tests = SimpleNamespace(records=records, testsRun=len(records))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            source = out / "scope.json"
            for complete, strict in ((False, False), (False, True), (True, True)):
                with self.subTest(complete=complete, strict=strict):
                    source.write_text(json.dumps({**manifest, "pending_work": [] if complete else ["probe pending block"]}), encoding="utf-8")
                    argv = ["--output-dir", str(out / "run")]
                    if strict:
                        argv.append("--require-complete")
                    stream = io.StringIO()
                    with mock.patch.object(issue3, "MANIFEST", source), \
                         mock.patch.object(issue3, "snapshot", return_value={}), \
                         mock.patch.object(issue3, "run_tests", return_value=(tests, {"passed": len(records)}, True)), \
                         mock.patch.object(issue3, "run_commands", return_value={}), \
                         mock.patch.object(issue3, "evaluate_conditions", return_value=conditions), redirect_stdout(stream):
                        exit_code = issue3.main(argv)
                    report = json.loads((out / "run/acceptance.json").read_text(encoding="utf-8"))
                    self.assertEqual(exit_code, int(strict and not complete))
                    self.assertEqual(report["exit_code"], exit_code)
                    self.assertTrue(report["implemented_scope_passed"])
                    self.assertEqual(report["full_issue_complete"], complete)
                    self.assertEqual(report["acceptance_mode"], "delivery" if strict else "regression")
                    if not complete:
                        self.assertIn("probe pending block", stream.getvalue())
                        self.assertTrue(report["delivery_failures"])

    def test_delivery_rejects_missing_replaced_duplicate_and_invalid_block_ids(self):
        manifest = json.loads(issue3.MANIFEST.read_text(encoding="utf-8"))
        coverage = json.loads((common.ROOT / "validation/issue3/model_coverage.json").read_text(encoding="utf-8"))
        records = {n: {"status": "passed"} for n in test_ids(common.discover(common.ISSUE3_TEST_DIRS))}
        conditions = {k: {"status": "passed"} for k in manifest["requirements"]}
        for model_index in (0, 1):
            for case in ("missing", "replacement", "extra", "duplicate", "invalid", "missing_id"):
                with self.subTest(model=model_index, case=case):
                    bad = copy.deepcopy(coverage)
                    model = bad["models"][model_index]
                    block = next(b for b in model["blocks"] if b["id"] == "embedding")
                    if case == "missing":
                        model["blocks"].remove(block)
                    elif case == "replacement":
                        block["id"] = "unexpected_replacement"
                    elif case == "duplicate":
                        model["blocks"].append(copy.deepcopy(block))
                    elif case == "extra":
                        model["blocks"].append({**block, "id": "unexpected_extra"})
                    elif case == "invalid":
                        block["id"] = ["embedding"]
                    else:
                        del block["id"]
                    audit = issue3.evaluate_delivery(manifest, bad, conditions, records)
                    self.assertEqual(audit["status"], "failed")
                    detail = audit["inventory"]["models"][model["name"]]
                    if case == "duplicate":
                        self.assertEqual(detail["duplicate_ids"], ["embedding"])
                    elif case != "extra":
                        self.assertEqual(detail["missing_ids"], ["embedding"])
                    if case == "replacement":
                        self.assertEqual(detail["unexpected_ids"], ["unexpected_replacement"])
                    if case == "extra":
                        self.assertEqual(detail["missing_ids"], [])
                        self.assertEqual(detail["unexpected_ids"], ["unexpected_extra"])
                    self.assertTrue(audit["failures"])

    def test_inventory_rejects_missing_duplicate_and_unexpected_models(self):
        coverage = json.loads((common.ROOT / "validation/issue3/model_coverage.json").read_text(encoding="utf-8"))
        for case in ("missing", "duplicate", "replacement", "invalid"):
            with self.subTest(case=case):
                bad = copy.deepcopy(coverage)
                name = bad["models"][0]["name"]
                if case == "missing":
                    bad["models"].pop(0)
                elif case == "duplicate":
                    bad["models"].append(copy.deepcopy(bad["models"][0]))
                elif case == "replacement":
                    bad["models"][0]["name"] = "unexpected_model"
                else:
                    bad["models"][0]["name"] = []
                audit = common.issue3_inventory(bad)
                self.assertEqual(audit["status"], "failed")
                self.assertIn(name, audit["duplicate_models"] if case == "duplicate" else audit["missing_models"])
                if case == "replacement":
                    self.assertEqual(audit["unexpected_models"], ["unexpected_model"])

    def test_delivery_verdict_requires_live_evidence_for_every_model_block(self):
        manifest = json.loads(issue3.MANIFEST.read_text(encoding="utf-8"))
        coverage = json.loads((common.ROOT / "validation/issue3/model_coverage.json").read_text(encoding="utf-8"))
        records = {n: {"status": "passed"} for n in test_ids(common.discover(common.ISSUE3_TEST_DIRS))}
        conditions = {k: {"status": "passed"} for k in manifest["requirements"]}
        conditions["coverage_mapping"] = {"status": "passed"}
        audit = issue3.evaluate_delivery(manifest, coverage, conditions, records)
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(len(audit["model_blocks"]), 23)
        self.assertFalse(audit["github_state_changed"])
        self.assertEqual(issue3.evaluate_delivery(manifest, coverage, conditions, {})["status"], "failed")
        for field, value in (("status", "pending"), ("owner", "unknown"), ("implementation", ["missing.py"]), ("tests", [])):
            bad = copy.deepcopy(coverage)
            bad["models"][0]["blocks"][0][field] = value
            self.assertEqual(issue3.evaluate_delivery(manifest, bad, conditions, records)["status"], "failed")

    def test_delivery_verdict_rejects_incomplete_acceptance_or_pending_scope(self):
        manifest = json.loads(issue3.MANIFEST.read_text(encoding="utf-8"))
        coverage = json.loads((common.ROOT / "validation/issue3/model_coverage.json").read_text(encoding="utf-8"))
        records = {n: {"status": "passed"} for n in test_ids(common.discover(common.ISSUE3_TEST_DIRS))}
        conditions = {k: {"status": "passed"} for k in manifest["requirements"]}
        for change in ({"pending_work": ["MoE incomplete"]}, {"acceptance_mapping": {}},
                       {"acceptance_mapping": {"independent_vector": ["missing"]}}):
            self.assertEqual(issue3.evaluate_delivery({**manifest, **change}, coverage, conditions, records)["status"], "failed")
        conditions["I3_vector"] = {"status": "failed"}
        self.assertEqual(issue3.evaluate_delivery(manifest, coverage, conditions, records)["status"], "failed")

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
        self.assertEqual(len(commands), 216)
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
