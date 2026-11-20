"""Verify implemented Issue #3 operator scope independently of Issue #2 review."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verification import (ISSUE3_TEST_DIRS, discover, fingerprints, issue3_owned,
                                  output_directory, run_tests, save)


ARCHITECTURES = ("ascend", "tpu", "tsp", "tsp_cim")
VECTOR_OPS = ("int32_add", "fp32_add", "fp32_mul", "fp32_exp", "fp32_rsqrt")
OPERATIONS = ("gemm", *VECTOR_OPS, "rmsnorm", "softmax", "residual", "swiglu", "swiglu_ffn", "kv_cache", "decode_attention")
NEGATIVE_CASES = ("reject_null_profile", "reject_unsupported")
MANIFEST = ROOT / "validation/issue3/scope.json"


def snapshot():
    paths = set()
    for folder, pattern in (("src", "*.py"), ("tests/operators", "*.py"),
                            ("tests/validation", "*.py"), ("configs", "*.json"),
                            ("validation/issue3/baselines", "*.json")):
        paths.update((ROOT / folder).rglob(pattern))
    paths.update(ROOT / name for name in ("scripts/verify_issue3.py", "scripts/verification.py", "scripts/verify_issue2.py",
                 "scripts/run_all.py", "scripts/run_all.ps1",
                 "ascend.py", "tpu.py", "tsp.py", "tsp_cim.py", "requirements.txt", "tests/__init__.py",
                 "README.md", "ARCHITECTURE.md", "calibration/parameter_source_audit.json",
                 "calibration/operator_execution_audit.json", "validation/issue3/scope.json", "validation/issue3/model_coverage.json",
                 "validation/issue3/README.md", "validation/report_output_review.md"))
    owned = {p for p in paths if issue3_owned(p.relative_to(ROOT).as_posix())}
    return {"issue3": fingerprints(owned), "shared_dependencies": fingerprints(paths - owned)}


def command_names(kinds):
    return [f"{arch}_{kind}" + ("" if kind in NEGATIVE_CASES else f"_{mode}")
            for arch in ARCHITECTURES for kind in kinds
            for mode in ((None,) if kind in NEGATIVE_CASES else ("count", "timing"))]


def evaluate_conditions(records, commands, manifest):
    conditions, assigned_tests, assigned_commands = {}, set(), set()
    for key, requirement in manifest["requirements"].items():
        modules = requirement["test_modules"]
        selected = {name: result["status"] for name, result in records.items()
                    if name.rsplit(".", 2)[0] in modules}
        missing_modules = [m for m in modules if not any(n.startswith(m + ".") for n in selected)]
        expected = command_names(requirement["cli_operations"])
        cli = {name: commands.get(name, {}).get("passed", False) for name in expected}
        passed = bool(selected) and not missing_modules and all(s == "passed" for s in selected.values()) and all(cli.values())
        conditions[key] = {"title": requirement["title"], "status": "passed" if passed else "failed",
                           "tests": selected, "missing_test_modules": missing_modules, "commands": cli}
        assigned_tests.update(selected)
        assigned_commands.update(expected)
    unmapped_tests = sorted(set(records) - assigned_tests)
    unmapped_commands = sorted(set(commands) - assigned_commands)
    conditions["coverage_mapping"] = {"status": "failed" if unmapped_tests or unmapped_commands else "passed",
                                       "unmapped_tests": unmapped_tests, "unmapped_commands": unmapped_commands}
    return conditions


def report_checks(report, arch, kind, mode, details):
    scope = {"ascend": "single_core", "tpu": "single_tensorcore", "tsp": "single_hem", "tsp_cim": "single_hem"}[arch]
    op, native = report["operator"], details["native_report"]
    digest = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    checks = {"operator_schema": report["schema_version"] == 2 and report["kind"] == "operator_result",
              "architecture": report["architecture"] == arch, "operation": op["kind"] == kind,
              "mode": report["mode"] == mode, "scope": report["boundary"]["scope"] == scope,
              "actions_present": bool(report["action_counts"]),
              "compact_result": not {"native_report", "action_ledger", "io_contract", "execution_contract"} & report.keys(),
              "details_binding": details["kind"] == "operator_details" and details["result_sha256"] == digest,
              "io_contract": details["io_contract"]["inputs"][0]["dtype"] == op["inputs"][0]["dtype"],
              "time_status": report["timing_status"] == ("not_evaluated" if mode == "count" else "analytical_estimate"),
              "no_claimed_energy": report["energy"]["total_pj"] is None}
    cycles = report["cycle_count"]
    checks["time_value"] = (cycles is None and report["latency_s"] is None) if mode == "count" else (
        cycles > 0 and report["latency_s"] > 0)
    checks["native_cycles"] = cycles == native["timing"]["total_cycles"]
    checks["native_actions"] = set(report["action_counts"]) == set(native["action_counts"]) and all(
        action["count"] == native["action_counts"][key]["count"] and
        action["unit"] == native["action_counts"][key]["unit"] for key, action in report["action_counts"].items())
    dtype = "INT8" if kind in ("residual", "swiglu", "swiglu_ffn", "kv_cache", "decode_attention") else "INT32" if kind in ("gemm", "int32_add") else "FP32"
    checks["stored_output_dtype"] = op["outputs"][0]["dtype"] == dtype
    if kind in ("kv_cache", "decode_attention"):
        checks["lifetime_validated"] = native["memory"]["lifetime_validation"]["status"] == "validated_symbolic_schedule"
        checks["single_resource"] = report["boundary"]["chip_execution_included"] is False
        state = native["cache_state"]
        checks["append_semantics"] = state["after"]["length"] == state["before"]["length"] + 1
        checks["state_aliases"] = details["io_contract"]["state_aliases"] == state["output_aliases"]
    if kind in ("residual", "swiglu", "swiglu_ffn"):
        checks["lifetime_validated"] = native["memory"]["lifetime_validation"]["status"] == "validated_symbolic_schedule"
        checks["single_resource"] = report["boundary"]["chip_execution_included"] is False
        checks["block_plan"] = len(native["schedule"]["steps"]) == (5 if kind == "swiglu_ffn" else 1)
    if kind in ("rmsnorm", "softmax"):
        checks["axis_semantics"] = op["attributes"]["axis"] == len(op["inputs"][0]["shape"]) - 1
        checks["lifetime_validated"] = native["memory"]["lifetime_validation"]["status"] == "validated_symbolic_schedule"
        if kind == "rmsnorm":
            checks["epsilon_semantics"] = op["attributes"]["epsilon"] == 1e-5
    return checks


def run_commands(out):
    commands = {}
    fixture = out / "fixtures"
    save(fixture / "gemm.json", {"operation": "gemm", "gemm": {"M": 2, "K": 17, "N": 7}, "data_format": "INT8"})
    profile = json.loads((ROOT / "configs/execution/vector.json").read_text(encoding="utf-8"))
    profile["composite"] = None
    save(fixture / "null_profile.json", profile)
    save(fixture / "unsupported.json", {"schema_version": 1, "operation": "attention"})

    def invoke(name, arch, options, kind=None, mode=None, expected=0):
        with tempfile.TemporaryDirectory(prefix="issue3-") as directory:
            target = Path(directory) / "report.json"
            sentinel = "Existing report must survive rejected input.\n"
            if expected:
                target.write_text(sentinel, encoding="utf-8")
            arguments = [sys.executable, str(ROOT / f"{arch}.py"), *options, "--output", str(target)]
            detail_target = Path(directory) / "details.json"
            if not expected:
                arguments += ["--details-output", str(detail_target)]
            record = {"arguments": arguments, "expected_exit_code": expected}
            checks, stdout, stderr = {}, "", ""
            try:
                proc = subprocess.run(arguments, cwd=directory, capture_output=True, text=True, encoding="utf-8", timeout=60)
                stdout, stderr = proc.stdout, proc.stderr
                record["exit_code"] = proc.returncode
                checks["exit_code"] = proc.returncode == expected
                if expected:
                    checks["explicit_error"] = bool(stderr.strip())
                    checks["existing_output_preserved"] = target.exists() and target.read_text(encoding="utf-8") == sentinel
                else:
                    checks["report_created"] = target.exists()
                    if target.exists():
                        report = json.loads(target.read_text(encoding="utf-8"))
                        save(out / f"{name}.json", report)
                        record["report"] = f"{name}.json"
                        details = json.loads(detail_target.read_text(encoding="utf-8"))
                        save(out / "details" / f"{name}.json", details)
                        record["details_report"] = f"details/{name}.json"
                        checks.update(report_checks(report, arch, kind, mode, details))
            except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
                checks["execution_and_report_valid"] = False
                record["error"] = str(exc)
            (out / f"{name}.stdout.txt").write_text(stdout, encoding="utf-8")
            (out / f"{name}.stderr.txt").write_text(stderr, encoding="utf-8")
            commands[name] = {**record, "checks": checks, "passed": bool(checks) and all(checks.values()),
                              "stdout": f"{name}.stdout.txt", "stderr": f"{name}.stderr.txt"}

    for arch in ARCHITECTURES:
        for kind in OPERATIONS:
            workload = fixture / "gemm.json" if kind == "gemm" else ROOT / f"configs/workload/operators/{kind}.json"
            for mode in ("count", "timing"):
                invoke(f"{arch}_{kind}_{mode}", arch, ["--workload", str(workload), "--mode", mode,
                       "--report-format", "operator"], kind, mode)
        invoke(f"{arch}_reject_null_profile", arch, ["--workload", "operators/rmsnorm.json",
               "--vector-config", str(fixture / "null_profile.json")], expected=2)
        invoke(f"{arch}_reject_unsupported", arch, ["--workload", str(fixture / "unsupported.json")], expected=2)
    save(out / "commands.json", commands)
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation/issue3/run")
    args = parser.parse_args(argv)
    try:
        out = output_directory(args.output_dir, "issue3")
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONUTF8"] = "1"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before = snapshot()
    tests, counts, tests_passed = run_tests(discover(ISSUE3_TEST_DIRS), out)
    commands = run_commands(out)
    conditions = evaluate_conditions(tests.records, commands, manifest)
    unchanged = snapshot() == before
    passed = tests_passed and unchanged and all(c["status"] == "passed" for c in conditions.values())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    summary = {"schema_version": 1, "issue": manifest["issue"], "scope": manifest["scope"],
               "reference_commit": manifest["reference_commit"], "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "repository_head": git.stdout.strip(), "snapshot": before,
               "snapshot_unchanged_during_run": unchanged, "test_directories": list(ISSUE3_TEST_DIRS),
               "unit_tests": {"total": tests.testsRun, "status_counts": counts, "passed": tests_passed,
                              "details": "tests.json", "log": "unittest.txt"},
               "cli_checks": {"total": len(commands), "passed": sum(c["passed"] for c in commands.values()), "details": "commands.json"},
               "conditions": conditions, "implemented_scope_passed": passed,
               "full_issue_complete": False, "pending_work": manifest["pending_work"],
               "limits": ["Exit 0 applies only to the declared implemented scope, not completion of the whole issue.",
                          "No Issue #2 review or acceptance result is imported, renewed, or overwritten.",
                          "Shared native dependencies are fingerprinted separately; analytical tests do not calibrate hardware."]}
    save(out / "acceptance.json", summary)
    print(f"Issue #3 implemented scope: {'passed' if passed else 'failed'}; full issue remains open")
    print(f"Unit tests: {counts}; CLI checks: {summary['cli_checks']['passed']}/{len(commands)}; unchanged snapshot: {unchanged}")
    for key, result in conditions.items():
        print(f"{key}: {result['status']}")
    print(f"Evidence: {out / 'acceptance.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
