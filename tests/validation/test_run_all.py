"""Exercise the real batch entry using fast, isolated simulator/verifier fixtures."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import run_all


ROOT = Path(__file__).resolve().parents[2]
STUB = '''import argparse, hashlib, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
for key in ('output', 'output-dir', 'workload', 'mode', 'report-format', 'details-output'):
    p.add_argument('--' + key)
a = p.parse_args()
name = Path(__file__).stem
mode = os.environ.get('RUN_ALL_STUB_MODE', '')
if name.startswith('verify_'):
    issue = name.removeprefix('verify_')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pending = mode == 'review' and issue == 'issue2'
    conditions = {k: {'status': 'needs_review' if pending and k in ('A1','A5') else 'passed'}
                  for k in (('A1','A2','A3','A4','A5') if issue == 'issue2' else ('operators',))}
    test = 'tests.test_fixture.' + ('Issue2Case' if issue == 'issue2' else 'Issue3Case') + '.test_ok'
    (out/'tests.json').write_text(json.dumps({'tests': {test: {'status': 'passed'}}}), encoding='utf-8')
    (out/'commands.json').write_text(json.dumps({'smoke': {'passed': True}}), encoding='utf-8')
    report = {'issue': 'https://github.com/pku-pacific-lab-team/huawei_dfp/issues/' + issue[-1],
              'snapshot_unchanged_during_run': True, 'conditions': conditions,
              'unit_tests': {'passed': True, 'total': 1, 'status_counts': {'passed': 1}},
              'local_acceptance_passed': not pending, 'implemented_scope_passed': True,
              'full_issue_complete': False, 'pending_work': ['remaining operators']}
    if pending:
        report['review_context'] = {'github_review_status': 'reviewed_and_merged',
                                   'summary': 'PR #7 reviewed and merged; local evidence is separate.'}
    (out/'acceptance.json').write_text(json.dumps(report), encoding='utf-8')
    sys.exit(1 if pending else 0)
operation = Path(a.workload).stem
operation = 'gemm' if operation == 'tiny_gemm' else operation
if name == 'ascend' and operation == 'gemm' and a.report_format is None and mode in ('failure','stale'):
    sys.exit(2 if mode == 'failure' else 0)
native = {'mode': 'timing', 'timing': {'total_cycles': 1}}
report = ({'schema_version': 2, 'kind': 'operator_result', 'status': 'success', 'architecture': name,
           'operator': {'kind': operation, 'inputs': []}, 'mode': 'timing', 'cycle_count': 1, 'latency_s': 1e-9,
           'boundary': {'scope': 'test'}, 'resource_requirements': {}, 'action_counts': {}} if a.report_format == 'operator' else native)
target = Path(a.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report), encoding='utf-8')
if a.details_output:
    detail = Path(a.details_output)
    detail.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(',',':')).encode('utf-8')).hexdigest()
    detail.write_text(json.dumps({'result_sha256': digest, 'native_report': native}), encoding='utf-8')
'''


class RunAllTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "project with spaces"
        self.cwd = Path(directory.name) / "working"
        self.root.mkdir()
        self.cwd.mkdir()
        scripts = self.root / "scripts"
        scripts.mkdir()
        for name in ("run_all.ps1", "run_all.py", "verification.py"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        for arch in run_all.ARCHITECTURES:
            (self.root / f"{arch}.py").write_text(STUB, encoding="utf-8")
        for issue in ("issue2", "issue3"):
            (scripts / f"verify_{issue}.py").write_text(STUB, encoding="utf-8")
        tests = self.root / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("", encoding="utf-8")
        (tests / "test_fixture.py").write_text(
            "import unittest\nclass Issue2Case(unittest.TestCase):\n def test_ok(self): pass\n"
            "class Issue3Case(unittest.TestCase):\n def test_ok(self): pass\n", encoding="utf-8")

    def run_batch(self, scope=None, mode="", powershell=False, details=False):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if powershell and shell:
            command = [shell, "-NoProfile", "-File", str(self.root / "scripts/run_all.ps1")]
            if scope is not None:
                command += ["-Scope", scope]
        else:
            command = [sys.executable, str(self.root / "scripts/run_all.py")]
            if scope is not None:
                command += ["--scope", scope]
        if details:
            command += ["-Details" if powershell and shell else "--details"]
        env = dict(os.environ, RUN_ALL_STUB_MODE=mode)
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        process = subprocess.run(command, cwd=self.cwd, env=env, capture_output=True,
                                 text=True, encoding="utf-8", timeout=90)
        report = self.root / "results" / f"run_all_{scope or 'all'}.json"
        self.assertTrue(report.exists(), process.stderr + process.stdout)
        return process, json.loads(report.read_text(encoding="utf-8"))

    def test_scope_matrix_and_default_powershell_from_another_directory(self):
        previous_native = None
        for scope, simulations, issues in (("issue2", 4, {"issue2"}), ("issue3", 64, {"issue3"}),
                                           (None, 68, {"issue2", "issue3"})):
            with self.subTest(scope=scope):
                process, report = self.run_batch(scope, powershell=True)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                self.assertEqual(report["simulation_count"], simulations)
                self.assertEqual(set(report["verification"]), issues)
                self.assertEqual(len(report["steps"]), simulations + len(issues))
                self.assertTrue(all(s["status"] == "passed" for s in report["steps"]))
                self.assertTrue(report["snapshot_unchanged_during_run"])
                if scope == "issue2":
                    previous_native = run_all.file_stamp(self.root / "results/ascend.json")
                    self.assertFalse((self.root / "validation/issue3").exists())
                elif scope == "issue3":
                    self.assertEqual(run_all.file_stamp(self.root / "results/ascend.json"), previous_native)
                    self.assertTrue((self.root / "results/ascend_gemm_operator.json").exists())
                else:
                    self.assertEqual(report["combined_test_coverage"]["status"], "passed")
                    self.assertEqual(report["combined_test_coverage"]["recorded_tests"], 2)
        self.assertFalse((self.cwd / "results").exists())

    def test_optional_details_and_operator_table(self):
        process, report = self.run_batch("issue3", powershell=True, details=True)
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        steps = [s for s in report["steps"] if s["kind"] == "simulation"]
        self.assertTrue(all(s["fresh_details"] for s in steps))
        with Path(report["operator_table"]).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 64)
        self.assertEqual({r["status"] for r in rows}, {"passed"})
        self.assertEqual({r["cycle_count"] for r in rows}, {"1"})
        self.assertEqual({r["latency_s"] for r in rows}, {"1e-09"})

    def test_pending_review_still_finishes_issue3_and_returns_distinct_exit(self):
        process, report = self.run_batch("all", "review")
        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(report["verification"]["issue2"]["status"], "needs_review")
        self.assertEqual(report["verification"]["issue3"]["status"], "passed")
        self.assertEqual(report["combined_test_coverage"]["status"], "passed")
        self.assertEqual(len(report["steps"]), 70)
        self.assertEqual(report["verification"]["issue2"]["review_context"]["github_review_status"],
                         "reviewed_and_merged")
        self.assertIn("PR #7 reviewed and merged", process.stdout)
        self.assertNotIn("review is still pending", process.stdout)

    def test_failed_and_stale_simulations_do_not_stop_remaining_steps(self):
        process, report = self.run_batch("all", "failure")
        self.assertEqual(process.returncode, 1)
        self.assertEqual(report["steps"][0]["status"], "failed")
        self.assertEqual(report["steps"][-1]["status"], "passed")
        self.assertEqual(len(report["steps"]), 70)
        old = self.root / "results/ascend.json"
        old.write_text('{"mode":"timing","timing":{"total_cycles":1}}', encoding="utf-8")
        before = run_all.file_stamp(old)
        process, report = self.run_batch("issue2", "stale")
        self.assertEqual(process.returncode, 1)
        self.assertFalse(report["steps"][0]["fresh_report"])
        self.assertEqual(run_all.file_stamp(old), before)

    def test_timeout_and_uncovered_or_duplicate_tests_cannot_pass(self):
        step = run_all.build_plan("issue2", self.root)[0]
        with mock.patch.object(run_all.subprocess, "run", side_effect=subprocess.TimeoutExpired("python", 1)):
            record = run_all.run_step(step, self.cwd, self.root)
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["fresh_report"])
        (self.cwd / "tests.json").write_text(json.dumps({"tests": {"unittest.case.FunctionTestCase (<lambda>)": {"status": "passed"}}}), encoding="utf-8")
        records = [{"kind": "verification", "fresh_report": True, "report": str(self.cwd / "acceptance.json")}] * 2
        suite = unittest.TestSuite((unittest.FunctionTestCase(lambda: None),))
        with mock.patch.object(run_all, "discover", return_value=suite):
            coverage = run_all.combined_coverage(records, self.root)
        self.assertEqual(coverage["status"], "failed")
        self.assertTrue(coverage["duplicate_tests"])
        suite = unittest.TestSuite((unittest.FunctionTestCase(lambda: None),))
        (self.cwd / "tests.json").write_text('{"tests":{}}', encoding="utf-8")
        with mock.patch.object(run_all, "discover", return_value=suite):
            self.assertTrue(run_all.combined_coverage(records, self.root)["uncovered_tests"])
