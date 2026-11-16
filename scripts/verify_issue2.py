"""Reproduce issue #2 acceptance evidence; never publish or close the issue."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TESTS = {
    "A1": [
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_audit_covers_current_config_and_resolvable_evidence",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_tsp_stream_coordinates_and_cross_section_are_distinct",
        "tests.ascend.test_config.ConfigTest.test_every_parameter_has_explicit_scope_and_version",
        "tests.ascend.test_config.ConfigTest.test_units_scopes_status_and_numeric_values_are_checked",
        "tests.tpu.test_config.TPUConfigTest.test_every_parameter_has_provenance_and_assumptions_remain_visible",
        "tests.tpu.test_config.TPUConfigTest.test_bad_parameter_values_and_metadata_are_rejected",
        "tests.tsp.test_baseline.TSPReferenceTest.test_schema_and_value_failures",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_metadata_and_fixed_geometry_are_required",
        "tests.cim.test_model.MacroSpecificationTest.test_metadata_units_scope_and_fixed_shape_are_validated",
    ],
    "A2": [
        "tests.ascend.test_config.ConfigTest.test_derived_compute_bandwidth_and_area_inventory",
        "tests.ascend.test_single_core.SingleCoreTest.test_proxy_hand_calculation_and_count_fields_are_preserved",
        "tests.tsp.test_baseline.TSPReferenceTest.test_proxy_full_plane_loading_and_weight_reuse",
        "tests.tsp.test_baseline.TSPReferenceTest.test_proxy_cross_k_stream_budget_and_configured_precedence",
        "tests.tpu.test_config.TPUConfigTest.test_peak_units_and_physical_inventory",
        "tests.tpu.test_config.TPUConfigTest.test_hierarchy_peaks_and_local_inventory",
        "tests.tpu.test_config.TPUConfigTest.test_chip_inventory_scales_private_resources_only",
        "tests.tpu.test_count.TPUCountTest.test_chip_vmem_total_is_not_usable_by_one_tensorcore",
        "tests.tpu.test_count.TPUTimingTest.test_single_mxu_and_tiny_tail_match_hand_calculations",
        "tests.tpu.test_count.TPUTimingTest.test_four_mxus_run_one_wave_and_fifth_tile_needs_another",
        "tests.tpu.test_count.TPUTimingTest.test_k_accumulation_stays_in_vmem_and_final_export_is_once",
        "tests.tsp.test_baseline.TSPReferenceTest.test_geometry_capacity_and_unknown_physical_inventory",
        "tests.tsp_cim.test_topology.TSPCIMTopologyTest.test_peaks_and_bandwidth_scopes_are_hand_calculated",
        "tests.tsp_cim.test_topology.TSPCIMTopologyTest.test_sram_inventory_is_subtotal_not_complete_physical_area",
        "tests.tsp_cim.test_count.TSPCIMCountTest.test_documented_acc_occupancy_keeps_tails_and_is_not_elapsed_time",
        "tests.tsp_cim.test_count.TSPCIMTimingTest.test_full_domain_and_tail_have_hand_calculated_total_times",
        "tests.tsp_cim.test_count.TSPCIMTimingTest.test_cross_k_segments_charge_read_add_and_single_write",
        "tests.cim.test_model.MacroSpecificationTest.test_geometry_peak_and_resident_storage",
    ],
    "A3": [
        "tests.ascend.test_config.ConfigTest.test_missing_parameters_do_not_get_silent_defaults",
        "tests.ascend.test_single_core.SingleCoreTest.test_configured_service_wins_and_only_needed_proxy_bases_required",
        "tests.tsp.test_baseline.TSPReferenceTest.test_proxy_unknown_clock_and_missing_base",
        "tests.tpu.test_config.TPUConfigTest.test_missing_required_capacity_is_not_defaulted",
        "tests.tpu.test_config.TPUConfigTest.test_unknown_inputs_propagate_and_do_not_block_independent_counts",
        "tests.tpu.test_config.TPUConfigTest.test_hierarchy_unknowns_preserve_independent_levels",
        "tests.tpu.test_count.TPUTimingTest.test_required_unknowns_reject_timing_but_preserve_count_mode",
        "tests.tpu.test_count.TPUTimingTest.test_vmem_proxy_is_explicit_and_configured_service_is_respected",
        "tests.tsp.test_baseline.TSPReferenceTest.test_cli_count_and_missing_timing",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_unknown_clock_propagates_to_peak_and_provenance_is_copied",
        "tests.cim.test_model.MacroSpecificationTest.test_no_implicit_defaults_or_old_geometry",
        "tests.cim.test_model.MacroActionTest.test_program_is_full_context_and_unknown_timing_is_not_zero",
        "tests.tsp_cim.test_count.TSPCIMTimingTest.test_assumptions_do_not_fill_unknown_macro_timing_or_clock",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_nominal_chip_hbm_does_not_enable_core_timing",
    ],
    "A4": [
        "tests.tpu.test_count.TPUCountTest.test_chip_geometry_does_not_scale_single_tensorcore_counts",
    ],  # Also checks four real subprocess CLI invocations, recorded separately below.
    "A5": [
        "tests.ascend.test_single_core.SingleCoreTest.test_removed_experiment_and_scheduling_flags_fail_explicitly",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_legacy_overlay_and_inline_macro_are_not_accepted",
        "tests.common.test_workload.WorkloadTest.test_tsp_cim_entry_rejects_unsupported_precision_and_counts_explicit_int8",
    ],
}
TITLES = {"A1": "来源、单位与作用域可追溯", "A2": "派生算力、容量、带宽与面积库存测试",
          "A3": "缺参显式处理", "A4": "四入口 INT8 GEMM 冒烟", "A5": "迁移说明与四入口的 GEMM 边界"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot():
    """Text-normalized fingerprints cover uncommitted files and new modules too."""
    paths = {ROOT / name for name in ("README.md", "ARCHITECTURE.md", "ascend.py", "tpu.py", "tsp.py", "tsp_cim.py",
                                     "requirements.txt", "scripts/verify_issue2.py", "scripts/clear_results.ps1", "scripts/run_all.ps1",
                                     "calibration/parameter_source_audit.json")}
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("configs", "*.json")):
        paths.update((ROOT / folder).rglob(pattern))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in sorted(paths)}


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {}

    def startTest(self, test):
        self.records[test.id()] = {"status": "running"}
        super().startTest(test)

    def addSuccess(self, test):
        self.records[test.id()] = {"status": "passed"}
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.records[test.id()] = {"status": "failed"}
        super().addFailure(test, err)

    def addError(self, test, err):
        self.records[test.id()] = {"status": "error"}
        super().addError(test, err)

    def addSkip(self, test, reason):
        self.records[test.id()] = {"status": "skipped", "reason": reason}
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self.records[test.id()] = {"status": "expected_failure"}
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.records[test.id()] = {"status": "unexpected_success"}
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, err):
        if err:
            self.records[test.id()] = {"status": "failed", "subtest": str(subtest)}
        super().addSubTest(test, subtest, err)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation/issue2/run")
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ["PYTHONUTF8"] = "1"
    before = snapshot()
    review_path = ROOT / "validation/issue2/review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    reviewed = review.get("snapshot", {})
    changed = sorted(k for k in set(before) | set(reviewed) if before.get(k) != reviewed.get(k))
    review_valid = not changed and all(review.get("conditions", {}).get(k, {}).get("status") == "reviewed" for k in ("A1", "A5"))

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    with (out / "unittest.txt").open("w", encoding="utf-8") as log:
        tests = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(suite)
    counts = dict(Counter(r["status"] for r in tests.records.values()))
    suite_passed = tests.testsRun > 0 and counts == {"passed": tests.testsRun}
    save(out / "tests.json", {"tests_run": tests.testsRun, "status_counts": counts, "tests": tests.records})

    workload = {"operation": "gemm", "gemm": {"M": 16, "K": 32, "N": 16},
                "data_format": "INT8", "accumulation_dtype": "INT32", "output_dtype": "INT32"}
    save(out / "workload.json", workload)
    commands = {}

    def invoke(name, entry, options, expected=0, error_contains=None, check=None):
        # A unique temporary report avoids accidentally accepting stale JSON from a prior run.
        with tempfile.TemporaryDirectory(prefix="issue2-") as directory:
            target = Path(directory) / "report.json"
            arguments = [sys.executable, str(ROOT / entry), "--workload", str(out / "workload.json"),
                         *options, "--output", str(target)]
            proc = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60)
            (out / f"{name}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
            (out / f"{name}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
            checks = {"exit_code": proc.returncode == expected}
            report = None
            if expected == 0:
                checks["report_created"] = target.exists()
                if target.exists():
                    report = json.loads(target.read_text(encoding="utf-8"))
                    save(out / f"{name}.json", report)
                    checks.update(check(report) if check else {})
            else:
                checks["no_success_report"] = not target.exists()
                checks["explicit_error"] = error_contains in proc.stderr if error_contains else bool(proc.stderr.strip())
            command = {"entry": entry, "arguments": arguments, "exit_code": proc.returncode,
                       "expected_exit_code": expected, "checks": checks, "passed": all(checks.values()),
                       "stdout": f"{name}.stdout.txt", "stderr": f"{name}.stderr.txt"}
            if report is not None:
                command.update(report=f"{name}.json", useful_ops=report["counts"]["useful_ops"],
                               scope=report["boundary"]["scope"])
            commands[name] = command

    def smoke_check(scope):
        def check(r):
            w = r.get("workload", {})
            checks = {"shape": [w.get(k) for k in ("M", "K", "N")] == [16, 32, 16],
                    "precision": [w.get(k) for k in ("data_format", "accumulation_dtype", "output_dtype")] == ["INT8", "INT32", "INT32"],
                    "useful_ops": r.get("counts", {}).get("useful_ops") == 16384,
                    "scope": r.get("boundary", {}).get("scope") == scope, "count_mode": r.get("mode") == "count",
                    "actions": bool(r.get("action_counts")), "provenance": bool(r.get("parameter_provenance")),
                    "no_invented_time": r.get("timing", {}).get("total_cycles") is None}
            if scope == "single_tensorcore":
                boundary = r.get("boundary", {})
                inventory = r.get("hardware_inventory", {})
                hierarchy = r.get("hardware_hierarchy", {})
                checks.update({
                    "report_schema": r.get("schema_version") == 2,
                    "chip_hardware_scope": boundary.get("hardware_scope") == "chip",
                    "local_counting_scope": boundary.get("counting_scope") == "single_tensorcore",
                    "no_chip_execution": boundary.get("chip_execution_included") is False,
                    "one_counted_tensorcore": boundary.get("simulated_tensorcores") == 1,
                    "chip_hierarchy": hierarchy.get("chip", {}).get("tensorcores_per_chip") == 2
                                      and hierarchy.get("chip", {}).get("mxus_per_chip") == 8,
                    "local_and_chip_inventory": inventory.get("per_tensorcore", {}).get("mxus") == 4
                                                and inventory.get("chip", {}).get("mxus") == 8,
                })
            return checks
        return check

    for entry, scope in (("ascend", "single_core"), ("tpu", "single_tensorcore"),
                         ("tsp", "single_hem"), ("tsp_cim", "single_cluster")):
        invoke(entry, entry + ".py", ["--mode", "count"], check=smoke_check(scope))
    ascend_missing = json.loads((ROOT / "configs/hardware/ascend910_reference.json").read_text(encoding="utf-8"))
    ascend_missing["chip"]["hbm_bytes_per_cycle"].update(value=None, status="unknown")
    save(out / "ascend_missing_timing.hardware.json", ascend_missing)
    invoke("ascend_missing_timing", "ascend.py", ["--mode", "timing", "--hardware", str(out / "ascend_missing_timing.hardware.json")],
           2, "Ascend timing proxy requires chip.hbm_bytes_per_cycle")
    tsp_missing = json.loads((ROOT / "configs/hardware/tsp.json").read_text(encoding="utf-8"))
    tsp_missing["parameters"]["mem_read_bpc"].update(value=None, status="unknown")
    save(out / "tsp_missing_timing.hardware.json", tsp_missing)
    invoke("tsp_missing_timing", "tsp.py", ["--mode", "timing", "--hardware", str(out / "tsp_missing_timing.hardware.json")],
           2, "Missing timing base parameters: mem_read_bpc")

    def proxy_timing_check(scope, expected_cycles, clock, name):
        def check(r):
            checks = smoke_check(scope)(r)
            del checks["count_mode"], checks["no_invented_time"]
            t = r["timing"]
            checks.update(timing_mode=r["mode"] == "timing", analytical_estimate=t["status"] == "analytical_estimate",
                          expected_cycles=t["total_cycles"] == expected_cycles,
                          expected_ns=abs(t["time_ns"] - expected_cycles / clock) < 1e-9,
                          stage_sum=t["total_cycles"] == sum(t["stages_cycles"].values()),
                          explicit_assumptions=bool(t["assumptions"]) and name in t["proxy_parameters"]
                              and t["parameters"][name]["status"] == "assumed",
                          unknowns_preserved=name in t["missing_parameters"]
                              and r["parameter_provenance"][name]["value"] is None,
                          no_chip_time=t["chip_cycles"] is None)
            return checks
        return check

    invoke("ascend_system_timing", "ascend.py", [],
           check=proxy_timing_check("single_core", 60, 1, "cube.macs_per_cycle"))
    invoke("tsp_system_timing", "tsp.py", [],
           check=proxy_timing_check("single_hem", 58, 0.9, "mxm_row_ii"))
    tpu_missing = json.loads((ROOT / "configs/hardware/tpu_v4_reference.json").read_text(encoding="utf-8"))
    tpu_missing["parameters"]["hbm_usable_bytes"].update(value=None, status="unknown")
    save(out / "tpu_missing_capacity.hardware.json", tpu_missing)
    invoke("tpu_missing_capacity", "tpu.py", ["--hardware", str(out / "tpu_missing_capacity.hardware.json")], 2, "Counting parameters missing")
    tpu_missing = json.loads((ROOT / "configs/hardware/tpu_v4_reference.json").read_text(encoding="utf-8"))
    tpu_missing["parameters"]["clock_hz"].update(value=None, status="unknown")
    save(out / "tpu_missing_timing.hardware.json", tpu_missing)
    invoke("tpu_missing_timing", "tpu.py", ["--mode", "timing", "--hardware", str(out / "tpu_missing_timing.hardware.json")],
           2, "TPU timing parameters missing: clock_hz")

    def tpu_timing_check(r):
        checks = smoke_check("single_tensorcore")(r)
        del checks["count_mode"], checks["no_invented_time"]
        timing = r["timing"]
        checks.update(timing_mode=r["mode"] == "timing",
                      analytical_estimate=timing["status"] == "analytical_estimate",
                      expected_cycles=timing["total_cycles"] == 292,
                      expected_ns=abs(timing["time_ns"] - 292 / 1.05) < 1e-9,
                      stage_sum=timing["total_cycles"] == sum(timing["stages_cycles"].values()),
                      explicit_assumptions=bool(timing["assumptions"]) and all(
                          timing["parameters"][p]["status"] == "assumed"
                          for p in ("hbm_share_bytes_per_second", "vmem_service_bytes_per_cycle", "fill_drain_cycles_per_wave")),
                      unknowns_preserved="vmem_bandwidth_bytes_per_second" in r["missing_parameters"]
                          and r["parameter_provenance"]["vmem_bandwidth_bytes_per_second"]["value"] is None,
                      no_chip_time=timing["chip_cycles"] is None)
        return checks

    invoke("tpu_system_timing", "tpu.py", [], check=tpu_timing_check)

    def cim_timing_check(r):
        checks = smoke_check("single_cluster")(r)
        del checks["count_mode"], checks["no_invented_time"]
        timing = r["timing"]
        checks.update(timing_mode=r["mode"] == "timing",
                      analytical_estimate=timing["status"] == "analytical_estimate",
                      expected_cycles=timing["total_cycles"] == 306,
                      expected_ns=abs(timing["time_ns"] - 340) < 1e-9,
                      stage_sum=timing["total_cycles"] == sum(timing["stages_cycles"].values()),
                      explicit_assumptions=bool(timing["assumptions"]) and timing["parameters"]["writer_bpc"]["status"] == "assumed",
                      unknowns_preserved="macro.compute_first_cycles" in r["missing_parameters"],
                      no_chip_time=timing["chip_cycles"] is None)
        return checks

    invoke("cim_system_timing", "tsp_cim.py", [], check=cim_timing_check)
    invalid = {**workload, "operation": "transformer"}
    save(out / "non_gemm.workload.json", invalid)
    for entry in ("ascend", "tpu", "tsp", "tsp_cim"):
        invoke(entry + "_non_gemm", entry + ".py", ["--workload", str(out / "non_gemm.workload.json")], 2, "Only a GEMM workload")
    save(out / "commands.json", commands)

    extra = {"A3": ["ascend_missing_timing", "tsp_missing_timing", "tpu_missing_capacity", "tpu_missing_timing", "tpu_system_timing", "cim_system_timing"],
             "A4": ["ascend", "tpu", "tsp", "tsp_cim", "tpu_system_timing", "cim_system_timing"],
             "A5": ["ascend", "tpu", "tsp", "tsp_cim", "tpu_system_timing", "cim_system_timing", "ascend_non_gemm", "tpu_non_gemm", "tsp_non_gemm", "tsp_cim_non_gemm"]}
    for key in ("A3", "A4", "A5"):
        extra[key].extend(("ascend_system_timing", "tsp_system_timing"))
    conditions = {}
    for key in TITLES:
        matching = {name: tests.records.get(name, {"status": "not_found"})["status"] for name in TESTS[key]}
        passed = all(status == "passed" for status in matching.values())
        cli = {name: commands[name]["passed"] for name in extra.get(key, [])}
        passed = passed and all(cli.values())
        reviewed_condition = key in ("A1", "A5")
        status = "passed" if passed else "failed"
        if reviewed_condition and passed:
            status = "passed_with_recorded_review" if review_valid else "needs_review"
        conditions[key] = {"title": TITLES[key], "status": status, "tests": matching, "commands": cli,
                           "review_basis": "../review.json" if reviewed_condition else None}
    unchanged = snapshot() == before
    accepted = suite_passed and unchanged and all(c["status"].startswith("passed") for c in conditions.values())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    summary = {"issue": "https://github.com/pku-pacific-lab-team/huawei_dfp/issues/2",
               "generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
               "repository_head": git.stdout.strip(), "snapshot": before, "snapshot_unchanged_during_run": unchanged,
               "review_matches_current_files": review_valid, "review_changed_files": changed,
               "scope_adjustment": "User explicitly waived the historical 16-core / 128T target.",
               "unit_tests": {"total": tests.testsRun, "status_counts": counts, "passed": suite_passed, "details": "tests.json", "log": "unittest.txt"},
               "deliverables": {
                   "D1": {"title": "四架构配置与参数来源表", "files": [
                       "configs/hardware/ascend910_reference.json", "configs/hardware/tpu_v4_reference.json",
                       "configs/hardware/tsp.json", "configs/hardware/tsp_cim.json", "configs/hardware/cim_macro.json",
                       "calibration/parameter_source_audit.json"], "conditions": ["A1", "A3"]},
                   "D2": {"title": "架构设计与 MXM 替换说明", "files": ["ARCHITECTURE.md", "validation/issue2/review.json"], "conditions": ["A5"]},
                   "D3": {"title": "派生参数与库存单元测试", "evidence": ["tests.json", "unittest.txt"], "conditions": ["A2"]},
                   "D4": {"title": "四入口 INT8 GEMM 冒烟", "evidence": ["workload.json", "commands.json", "ascend.json", "tpu.json", "tsp.json", "tsp_cim.json"], "conditions": ["A4"]},
               },
               "conditions": conditions, "local_acceptance_passed": accepted,
               "limits": ["Local working-tree evidence, including uncommitted content; not a commit, publication or external sign-off.",
                          "A1/A5 reuse recorded source/code/document review only if fingerprints match. This command does not re-verify external publications.",
                          "Unknown and assumed parameters remain explicit; test success is not hardware calibration or chip-performance validation."]}
    save(out / "acceptance.json", summary)
    print(f"Unit tests: {counts}; unchanged snapshot: {unchanged}")
    for key, result in conditions.items():
        print(f"{key}: {result['status']} - {result['title']}")
    print(f"Evidence: {out / 'acceptance.json'}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
