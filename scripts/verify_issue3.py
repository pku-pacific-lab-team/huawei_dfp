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
PROJECTION_OPS = ("rmsnorm_int8", "qkv_projection", "attention_output_projection")
POSITION_VOCAB_OPS = ("rope", "embedding", "vocabulary_projection")
MLA_OPS = ("mla_preprocess", "mla_cache", "mla_decode_attention")
OPERATIONS = ("gemm", *VECTOR_OPS, "rmsnorm", "softmax", "residual", "swiglu", "swiglu_ffn", "kv_cache", "decode_attention", *PROJECTION_OPS, *POSITION_VOCAB_OPS, *MLA_OPS, "moe_experts", "moe_router", "moe_combine", "moe")
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


def evaluate_delivery(manifest, coverage, conditions, records, root=ROOT):
    """Local Issue #3 delivery verdict, never a GitHub closure/review claim."""
    mapping = manifest.get("acceptance_mapping", {})
    required = {"independent_vector", "verifiable_gemm", "tested_model_blocks"}
    criteria = {name: bool(ids) and all(conditions.get(i, {}).get("status") == "passed" for i in ids)
                for name, ids in mapping.items()}
    mapping_valid = set(mapping) == required and all(criteria.values())
    blocks = {}
    models = coverage.get("models", [])
    inventory_valid = {m.get("name") for m in models} == {"LLaMA3-70B", "DeepSeek-R1"} and len(models) == 2
    for model in models:
        items = model.get("blocks", [])
        inventory_valid &= bool(items) and len({b.get("id") for b in items}) == len(items)
        for b in items:
            modules = b.get("tests", [])
            tests_pass = bool(modules) and all(
                any(n.startswith(m + ".") for n in records) and
                all(r["status"] == "passed" for n,r in records.items() if n.startswith(m + ".")) for m in modules)
            files = b.get("implementation", [])
            ok = b.get("status") == "implemented" and b.get("owner") == "issue3" and bool(files) and all((root / p).is_file() for p in files) and tests_pass
            blocks[model["name"] + "/" + b["id"]] = "passed" if ok else "failed"
    inventory_valid &= bool(blocks) and all(v == "passed" for v in blocks.values()) and coverage.get("full_target_model_coverage") is True
    all_conditions = bool(conditions) and manifest["requirements"].keys() <= conditions.keys() and all(v.get("status") == "passed" for v in conditions.values())
    passed = mapping_valid and inventory_valid and all_conditions and not manifest.get("pending_work", ["missing declaration"])
    return {"status": "passed" if passed else "failed", "acceptance_criteria": criteria,
            "model_blocks": blocks, "inventory_valid": bool(inventory_valid),
            "scope": "Local coarse-operator delivery; excludes full decoder/chip execution and numerical calibration.",
            "github_state_changed": False}


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
    dtype = "INT8" if kind in ("residual", "swiglu", "swiglu_ffn", "kv_cache", "decode_attention", "rope", "embedding", *PROJECTION_OPS, *MLA_OPS) else "INT32" if kind in ("gemm", "int32_add") else "FP32"
    checks["stored_output_dtype"] = ([t["dtype"] for t in op["outputs"]] == ["INT32", "FP32"] if kind == "moe_router" else
                                     all(t["dtype"] == ("INT8" if kind == "moe" else dtype) for t in op["outputs"]))
    if kind in {"moe_router", "moe_combine", "moe"}:
        logical = native["work_counts"]["logical"]
        checks["moe_hand_work"] = logical["operations"] == {"moe_router":330, "moe_combine":60, "moe":3414}[kind]
        checks["moe_hand_macs"] = logical["useful_macs"] == {"moe_router":90, "moe_combine":0, "moe":1350}[kind]
        checks["moe_global_lifetime"] = native["memory"]["lifetime_validation"]["status"] == "validated_symbolic_schedule"
        if arch == "tpu":
            checks["moe_hand_peak"] = report["resource_requirements"]["hbm"]["required_bytes"] == {"moe_router":477, "moe_combine":264, "moe":1455}[kind]
        if kind == "moe":
            checks["moe_scenario_explicit"] = native["routing"]["router_work_included"] and not native["routing"]["numeric_routing_verified"]
    if kind == "moe_experts":
        logical = native["work_counts"]["logical"]
        checks["moe_fixed_route"] = native["routing"]["counts"] == [3, 1, 2, 0, 0] and native["routing"]["total_assignments"] == 6
        checks["moe_outputs"] = [t["shape"] for t in op["outputs"]] == [[3, 2, 5], [3, 5]]
        checks["moe_hand_work"] = logical["useful_macs"] == 1260 and logical["operations"] == 3024
        checks["moe_hand_payloads"] = [logical[k] for k in ("route_index_read_bytes", "dispatch_payload_bytes", "publish_payload_bytes")] == [24, 30, 120]
        checks["moe_empty_experts"] = native["schedule"]["skipped_experts"] == [3, 4]
        checks["lifetime_validated"] = native["memory"]["lifetime_validation"]["status"] == "validated_symbolic_schedule"
        checks["single_resource"] = report["boundary"]["chip_execution_included"] is False
        checks["moe_scope"] = not op["attributes"]["routing_compute_included"] and not op["attributes"]["weighted_combine_included"]
    if kind in ("kv_cache", "decode_attention", "mla_cache", "mla_decode_attention"):
        checks["lifetime_validated"] = native["memory"]["lifetime_validation"]["status"] == "validated_symbolic_schedule"
        checks["single_resource"] = report["boundary"]["chip_execution_included"] is False
        state = native["cache_state"]
        checks["append_semantics"] = state["after"]["length"] == state["before"]["length"] + 1
        checks["state_aliases"] = details["io_contract"]["state_aliases"] == state["output_aliases"]
    if kind in ("residual", "swiglu", "swiglu_ffn", *PROJECTION_OPS, *POSITION_VOCAB_OPS, "mla_preprocess"):
        checks["lifetime_validated"] = native["memory"]["lifetime_validation"]["status"] == "validated_symbolic_schedule"
        checks["single_resource"] = report["boundary"]["chip_execution_included"] is False
        checks["block_plan"] = len(native["schedule"]["steps"]) == {
            "swiglu_ffn": 5, "rmsnorm_int8": 3, "qkv_projection": 6, "attention_output_projection": 2,
            "rope": 2, "embedding": 1, "vocabulary_projection": 2, "mla_preprocess": 14}.get(kind, 1)
    if kind == "mla_preprocess":
        checks["mla_branch_shapes"] = [t["shape"] for t in op["outputs"]] == [[2, 3, 1, 3], [2, 3, 1, 4], [2, 1, 1, 7], [2, 1, 1, 4]]
        checks["mla_hand_macs"] = native["work_counts"]["logical"]["useful_macs"] == 754
        checks["mla_core_excluded"] = op["attributes"]["attention_core_included"] is False
    if kind == "mla_cache":
        checks["mla_shared_storage"] = set(native["cache_state"]["allocations"]) == {"latent", "rope"}
        checks["mla_hand_traffic"] = native["work_counts"]["logical"]["appended_bytes"] == 22 and native["work_counts"]["logical"]["cache_read_bytes"] == 66
    if kind == "mla_decode_attention":
        checks["mla_head_plan"] = len(native["schedule"]["steps_per_head"]) == 14 and native["schedule"]["head_repetitions"] == 6
        checks["mla_output_projection"] = len(native["schedule"]["output_projection_steps"]) == 2 and op["outputs"][0]["shape"] == [2, 17]
        checks["mla_hand_macs"] = native["work_counts"]["logical"]["useful_macs"] == 1314
        checks["mla_cache_reads"] = native["work_counts"]["logical"]["cache_read_payload_bytes"] == 324
        checks["mla_core_included"] = op["attributes"]["attention_impl"] == "absorb"
    if kind == "rope":
        checks["rotation_outputs"] = [t["shape"] for t in op["outputs"]] == [t["shape"] for t in op["inputs"][:2]]
        checks["precomputed_tables"] = op["attributes"]["table_generation_included"] is False
        checks["position"] = op["attributes"]["position"] == 5
    if kind == "embedding":
        checks["index_dtype"] = op["inputs"][0]["dtype"] == "INT32"
        checks["memory_only"] = native["work_counts"]["logical"]["operations"] == 0
    if kind == "vocabulary_projection":
        checks["fp32_logits"] = op["attributes"]["output_semantics"] == "logits" and dtype == "FP32"
        checks["logits_conversion"] = native["work_counts"]["internal"]["boundary_elements"] == {
            "boundary_to_fp32": op["outputs"][0]["numel"]}
    if kind == "qkv_projection":
        attrs = op["attributes"]
        batch = op["inputs"][0]["shape"][0]
        checks["head_outputs"] = [t["shape"] for t in op["outputs"]] == [
            [batch, heads, 1, attrs["head_dim"]] for heads in (attrs["query_heads"], attrs["kv_heads"], attrs["kv_heads"])]
        checks["output_bytes"] = native["work_counts"]["logical"]["output_bytes"] == sum(t["logical_bytes"] for t in op["outputs"])
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
    coverage = json.loads((ROOT / "validation/issue3/model_coverage.json").read_text(encoding="utf-8"))
    delivery = evaluate_delivery(manifest, coverage, conditions, tests.records)
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
               "full_issue_complete": passed and delivery["status"] == "passed", "delivery_audit": delivery,
               "pending_work": manifest["pending_work"], "follow_up_outside_issue3": manifest.get("follow_up_outside_issue3", []),
               "limits": ["full_issue_complete is a local scoped delivery verdict; it does not close GitHub or claim review approval.",
                          "No Issue #2 review or acceptance result is imported, renewed, or overwritten.",
                          "Shared native dependencies are fingerprinted separately; analytical tests do not calibrate hardware."]}
    save(out / "acceptance.json", summary)
    print(f"Issue #3 implemented scope: {'passed' if passed else 'failed'}; local delivery complete: {summary['full_issue_complete']}")
    print(f"Unit tests: {counts}; CLI checks: {summary['cli_checks']['passed']}/{len(commands)}; unchanged snapshot: {unchanged}")
    for key, result in conditions.items():
        print(f"{key}: {result['status']}")
    print(f"Evidence: {out / 'acceptance.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
