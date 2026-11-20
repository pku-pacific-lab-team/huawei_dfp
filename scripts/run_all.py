"""Run selected simulations and verification; retain each issue's own verdict."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verification import discover, fingerprints, save


ARCHITECTURES = ("ascend", "tpu", "tsp", "tsp_cim")
OPERATIONS = ("int32_add", "fp32_add", "fp32_mul", "fp32_exp", "fp32_rsqrt", "rmsnorm", "softmax",
              "residual", "swiglu", "swiglu_ffn", "kv_cache", "decode_attention",
              "rmsnorm_int8", "qkv_projection", "attention_output_projection", "rope", "embedding", "vocabulary_projection",
              "mla_preprocess", "mla_cache")


def build_plan(scope, root=ROOT, *, details=False):
    if scope not in ("issue2", "issue3", "all"):
        raise ValueError("Unknown run scope")
    steps = []
    for issue in (("issue2", "issue3") if scope == "all" else (scope,)):
        for arch in ARCHITECTURES:
            for operation in (("gemm",) if issue == "issue2" else ("gemm", *OPERATIONS)):
                operator_gemm = issue == "issue3" and operation == "gemm"
                filename = f"{arch}.json" if issue == "issue2" else (
                    f"{arch}_gemm_operator.json" if operator_gemm else f"{arch}_{operation}.json")
                report = root / "results" / filename
                workload = "tiny_gemm.json" if operation == "gemm" else f"operators/{operation}.json"
                options = ["--workload", workload, "--mode", "timing", "--output", str(report)]
                if issue == "issue3":
                    options += ["--report-format", "operator"]
                detail_report = root / "results" / "details" / filename if issue == "issue3" and details else None
                if detail_report:
                    options += ["--details-output", str(detail_report)]
                steps.append({"name": f"{issue}_{arch}_{operation}", "issue": issue, "kind": "simulation",
                              "architecture": arch, "operation": operation, "report": report,
                              "arguments": [sys.executable, "-B", "-X", "utf8", str(root / f"{arch}.py"), *options]})
                if detail_report:
                    steps[-1]["details_report"] = detail_report
        evidence = root / "validation" / issue / "run"
        steps.append({"name": f"verify_{issue}", "issue": issue, "kind": "verification",
                      "report": evidence / "acceptance.json",
                      "arguments": [sys.executable, "-B", "-X", "utf8", str(root / f"scripts/verify_{issue}.py"),
                                    "--output-dir", str(evidence)]})
    return steps


def source_snapshot(root=ROOT):
    paths = set()
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("scripts", "*.py"),
                            ("scripts", "*.ps1"), ("configs", "*.json"), ("calibration", "*.json"),
                            ("validation/issue3/baselines", "*.json")):
        paths.update((root / folder).rglob(pattern))
    paths.update(p for p in (root / name for name in (
        "ascend.py", "tpu.py", "tsp.py", "tsp_cim.py", "README.md", "ARCHITECTURE.md", "requirements.txt",
        "validation/README.md", "validation/report_output_review.md", "validation/issue2/review.json", "validation/issue3/README.md",
        "validation/issue3/scope.json", "validation/issue3/model_coverage.json", "validation/issue2/review_context.json")) if p.exists())
    return fingerprints(paths, root)


def file_stamp(path):
    if not path.exists():
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def verification_verdict(step, report, exit_code):
    expected_issue = "https://github.com/pku-pacific-lab-team/huawei_dfp/issues/" + step["issue"][-1]
    tests = report["unit_tests"]
    commands = json.loads((step["report"].parent / "commands.json").read_text(encoding="utf-8"))
    conditions = {key: value["status"] for key, value in report["conditions"].items()}
    automatic = (report["issue"] == expected_issue and report["snapshot_unchanged_during_run"] is True
                 and tests["passed"] is True and tests["total"] > 0
                 and tests["status_counts"] == {"passed": tests["total"]}
                 and bool(commands) and all(c["passed"] is True for c in commands.values()) and bool(conditions))
    details = {"unit_tests": tests["total"], "cli_checks": len(commands), "conditions": conditions,
               "automatic_checks_passed": automatic and all(s in ("passed", "passed_with_recorded_review", "needs_review")
                                                              for s in conditions.values())}
    if step["issue"] == "issue3":
        details.update(full_issue_complete=report["full_issue_complete"], pending_work=report["pending_work"])
        passed = (automatic and exit_code == 0 and report["implemented_scope_passed"] is True
                  and all(s == "passed" for s in conditions.values()))
        return ("passed" if passed else "failed"), details
    details["review_context"] = report.get("review_context")
    passed = automatic and exit_code == 0 and report["local_acceptance_passed"] is True and all(
        s in ("passed", "passed_with_recorded_review") for s in conditions.values())
    pending = {key for key, status in conditions.items() if status == "needs_review"}
    needs_review = (automatic and exit_code == 1 and report["local_acceptance_passed"] is False
                    and bool(pending) and pending <= {"A1", "A5"}
                    and all(s in ("passed", "passed_with_recorded_review", "needs_review") for s in conditions.values()))
    return ("passed" if passed else "needs_review" if needs_review else "failed"), details


def run_step(step, log_dir, root=ROOT):
    record = {k: str(v) if isinstance(v, Path) else v for k, v in step.items()}
    record.update(status="failed", fresh_report=False)
    stdout, stderr = "", ""
    started = time.monotonic()
    try:
        before = file_stamp(step["report"])
        detail_before = file_stamp(step["details_report"]) if "details_report" in step else None
        proc = subprocess.run(step["arguments"], cwd=root, capture_output=True, text=True,
                              encoding="utf-8", timeout=1800 if step["kind"] == "verification" else 300)
        stdout, stderr = proc.stdout, proc.stderr
        record["exit_code"] = proc.returncode
        after = file_stamp(step["report"])
        record["fresh_report"] = after is not None and after != before
        if not record["fresh_report"]:
            record["error"] = "No fresh report; an earlier result cannot satisfy this run."
        else:
            report = json.loads(step["report"].read_text(encoding="utf-8"))
            if step["kind"] == "verification":
                record["status"], record["verification"] = verification_verdict(step, report, proc.returncode)
            else:
                operator_view = step["issue"] == "issue3"
                cycles = report["cycle_count"] if operator_view else report["timing"]["total_cycles"]
                valid = proc.returncode == 0 and report["mode"] == "timing" and cycles > 0
                if operator_view:
                    valid = valid and report["schema_version"] == 2 and report["kind"] == "operator_result" and report["status"] == "success" and (
                        report["architecture"] == step["architecture"] and report["operator"]["kind"] == step["operation"])
                    valid = valid and (isinstance(report["operator"]["inputs"], list)
                                       and isinstance(report["boundary"]["scope"], str)
                                       and report["latency_s"] > 0
                                       and isinstance(report["resource_requirements"], dict)
                                       and isinstance(report["action_counts"], dict)
                                       and all({"count", "unit", "role"} <= action.keys()
                                               for action in report["action_counts"].values()))
                if "details_report" in step:
                    detail_after = file_stamp(step["details_report"])
                    record["fresh_details"] = detail_after is not None and detail_after != detail_before
                    detail = json.loads(step["details_report"].read_text(encoding="utf-8"))
                    digest = hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
                    valid = valid and record["fresh_details"] and detail["result_sha256"] == digest
                record["status"] = "passed" if valid else "failed"
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        record["error"] = str(exc)
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    for stream, content in (("stdout", stdout), ("stderr", stderr)):
        path = log_dir / f"{step['name']}.{stream}.txt"
        path.write_text(content, encoding="utf-8")
        record[stream] = str(path)
    return record


def suite_ids(suite):
    for test in suite:
        if hasattr(test, "id"):
            yield test.id()
        else:
            yield from suite_ids(test)


def write_operator_table(records, path):
    """One row per operator run; resource scopes and action units stay separate."""
    rows = []
    for record in records:
        if record["issue"] != "issue3" or record["kind"] != "simulation":
            continue
        row = {"architecture": record["architecture"], "operator": record["operation"],
               "status": record["status"], "report": record["report"]}
        if record["status"] == "passed" and record["fresh_report"]:
            report = json.loads(Path(record["report"]).read_text(encoding="utf-8"))
            row.update(inputs=json.dumps(report["operator"]["inputs"], ensure_ascii=False),
                       cycle_count=report["cycle_count"], latency_s=report["latency_s"],
                       scope=report["boundary"]["scope"],
                       memory=json.dumps(report["resource_requirements"], ensure_ascii=False),
                       actions=json.dumps({key: {field: a[field] for field in ("count", "unit", "role")}
                                           for key, a in report["action_counts"].items()}, ensure_ascii=False))
        rows.append(row)
    if not rows:
        return None
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=("architecture", "operator", "status", "inputs", "cycle_count",
                                                "latency_s", "scope", "memory", "actions", "report"))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def combined_coverage(records, root=ROOT):
    """Check the union against full discovery without executing tests twice."""
    selected, duplicates = set(), set()
    try:
        for record in records:
            if record["kind"] != "verification":
                continue
            if not record["fresh_report"]:
                return {"status": "failed", "error": "Missing fresh issue verification evidence"}
            evidence = Path(record["report"]).parent / "tests.json"
            tests = json.loads(evidence.read_text(encoding="utf-8"))["tests"]
            duplicates.update(selected & set(tests))
            selected.update(tests)
        discovered = set(suite_ids(discover(("tests",), root)))
        missing, extra = sorted(discovered - selected), sorted(selected - discovered)
        return {"status": "passed" if discovered and not missing and not extra and not duplicates else "failed",
                "discovered_tests": len(discovered), "recorded_tests": len(selected),
                "uncovered_tests": missing, "unexpected_tests": extra, "duplicate_tests": sorted(duplicates)}
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        return {"status": "failed", "error": str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("issue2", "issue3", "all"), default="all")
    parser.add_argument("--details", action="store_true", help="Also retain separate Issue #3 diagnostics")
    args = parser.parse_args(argv)
    root = ROOT
    plan = build_plan(args.scope, root, details=args.details)
    results = root / "results"
    logs = results / "run_all" / args.scope
    logs.mkdir(parents=True, exist_ok=True)
    before = source_snapshot(root)
    records = []
    for index, step in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] {step['name']}", flush=True)
        record = run_step(step, logs, root)
        records.append(record)
        print(f"  {record['status']}: {record['report']}", flush=True)
        if record.get("error"):
            print("  " + record["error"], flush=True)
    coverage = combined_coverage(records, root) if args.scope == "all" else {"status": "not_requested"}
    unchanged = source_snapshot(root) == before
    failed = not unchanged or any(r["status"] == "failed" for r in records) or coverage["status"] == "failed"
    pending = any(r["status"] == "needs_review" for r in records)
    status = "failed" if failed else "needs_review" if pending else "passed"
    exit_code = 1 if failed else 2 if pending else 0
    summary = {"schema_version": 1, "scope": args.scope, "status": status, "exit_code": exit_code,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(), "snapshot": before,
               "snapshot_unchanged_during_run": unchanged, "steps": records, "combined_test_coverage": coverage,
               "simulation_count": sum(r["kind"] == "simulation" for r in records),
               "simulation_passed": sum(r["kind"] == "simulation" and r["status"] == "passed" for r in records),
               "verification": {r["issue"]: {"status": r["status"], **r.get("verification", {})}
                                for r in records if r["kind"] == "verification"},
               "limits": ["all means all currently registered simulations and tests, not every future Issue #3 operator.",
                          "Individual issue criteria and evidence remain separate; no manual review is renewed."]}
    target = results / f"run_all_{args.scope}.json"
    summary["operator_table"] = write_operator_table(records, results / f"run_all_{args.scope}_operators.csv")
    save(target, summary)
    print(f"Summary: {status}; simulations={summary['simulation_passed']}/{summary['simulation_count']}; exit={exit_code}")
    print(f"Report: {target}")
    if pending:
        print("Issue #2 local review evidence needs reconciliation; this is separate from GitHub PR acceptance.")
        context = summary["verification"].get("issue2", {}).get("review_context")
        if context:
            print(f"Review context: {context['summary']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
