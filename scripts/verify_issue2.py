"""Reproduce issue #2 acceptance evidence; never publish or close the issue."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verification import (ISSUE2_TEST_DIRS, discover, fingerprints, issue3_owned,
                                  output_directory, run_tests, save)

TESTS = {
    "A1": [
        "tests.common.test_hardware_archival.HardwareArchivalTest.test_archives_preserve_full_parameter_records_and_audit",
        "tests.common.test_config_documentation.ConfigDocumentationTest.test_classification_covers_each_real_root_and_execution_field_once",
        "tests.tpu.test_vector_service.VectorNonlinearTest.test_report_discloses_proxy_and_preserves_single_core_scope",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_audit_covers_current_config_and_resolvable_evidence",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_retired_ascend_parameters_keep_evidence_outside_active_schema",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_retired_tpu_parameters_keep_evidence_outside_active_schema",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_retired_tsp_parameters_keep_evidence_outside_active_schema",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_retired_cim_parameters_keep_evidence_outside_active_schema",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_tsp_stream_coordinates_and_cross_section_are_distinct",
        "tests.ascend.test_config.ConfigTest.test_every_parameter_has_explicit_scope_and_version",
        "tests.ascend.test_config.ConfigTest.test_units_scopes_status_and_numeric_values_are_checked",
        "tests.tpu.test_config.TPUConfigTest.test_every_parameter_has_provenance_and_assumptions_remain_visible",
        "tests.tpu.test_config.TPUConfigTest.test_cmem_proxy_scope_provenance_and_execution_guard",
        "tests.tpu.test_cmem.CMEMTest.test_reference_activates_tiled_with_no_new_hardware_parameters",
        "tests.tpu.test_config.TPUConfigTest.test_bad_parameter_values_and_metadata_are_rejected",
        "tests.tsp.test_baseline.TSPReferenceTest.test_schema_and_value_failures",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_metadata_and_fixed_geometry_are_required",
        "tests.cim.test_model.MacroSpecificationTest.test_metadata_units_scope_and_fixed_shape_are_validated",
    ],
    "A2": [
        "tests.common.test_timing_breakdowns.TimingBreakdownsTest.test_four_models_keep_service_and_elapsed_meanings_distinct",
        "tests.common.test_timing_breakdowns.TimingBreakdownsTest.test_tpu_serial_partition_and_pipeline_overlap",
        "tests.common.test_timing_breakdowns.TimingBreakdownsTest.test_background_load_is_not_charged_twice",
        "tests.tsp.test_weight_prefetch.WeightPrefetchTest.test_hand_schedule_cold_fill_overlap_and_iw_release",
        "tests.tsp.test_weight_prefetch.WeightPrefetchTest.test_reads_and_input_stream_contend_writes_are_independent",
        "tests.tsp.test_weight_prefetch.WeightPrefetchTest.test_partial_prefetch_and_exact_fractional_tail",
        "tests.tsp.test_weight_prefetch.WeightPrefetchTest.test_one_pending_buffer_spans_reused_m_tiles",
        "tests.tsp.test_weight_prefetch.WeightPrefetchTest.test_vector_padding_and_lw_do_not_depend_on_iw_port",
        "tests.tsp.test_weight_prefetch.WeightPrefetchTest.test_native_resource_dependencies_and_byte_conservation",
        "tests.ascend.test_vector_service.VectorServiceTest.test_shared_budget_and_hand_computed_work",
        "tests.ascend.test_vector_service.VectorServiceTest.test_configured_budget_costs_and_exact_rounding",
        "tests.ascend.test_vector_service.VectorServiceTest.test_inventory_and_clock_do_not_implicitly_scale_service",
        "tests.tpu.test_vector_service.VectorNonlinearTest.test_mixed_operations_share_one_budget_and_tail_rounding",
        "tests.tpu.test_vector_service.VectorNonlinearTest.test_fractional_costs_are_totals_not_multiplied_by_basic_cost",
        "tests.tsp.test_mem_concurrency.MEMConcurrencyTest.test_native_read_write_overlap_but_reads_still_contend",
        "tests.tsp.test_mem_concurrency.MEMConcurrencyTest.test_cim_wave_uses_separate_directional_budgets",
        "tests.tsp.test_mem_concurrency.MEMConcurrencyTest.test_background_load_competes_with_reads_but_not_writes",
        "tests.tsp.test_mem_concurrency.MEMConcurrencyTest.test_reports_disclose_assumption_and_keep_inventory",
        "tests.tsp.test_vxm_service.VXMServiceTest.test_defaults_and_shared_weighted_occupancy",
        "tests.tsp.test_vxm_service.VXMServiceTest.test_fractional_costs_round_exactly_once",
        "tests.tsp.test_vxm_service.VXMServiceTest.test_add_cost_controls_native_and_cim_without_changing_actions",
        "tests.tsp.test_vxm_service.VXMServiceTest.test_report_discloses_algorithmic_scope_and_no_inventory_fallback",
        "tests.common.test_execution_contract.ExecutionContractTest.test_tpu_two_k_blocks_hand_schedule_and_capacity_fallback",
        "tests.common.test_execution_contract.ExecutionContractTest.test_default_tpu_uses_four_mxus_and_counts_array_tail",
        "tests.common.test_execution_contract.ExecutionContractTest.test_cim_matched_baseline_and_background_sensitivity",
        "tests.ascend.test_config.ConfigTest.test_derived_compute_bandwidth_and_area_inventory",
        "tests.ascend.test_single_core.SingleCoreTest.test_proxy_hand_calculation_and_count_fields_are_preserved",
        "tests.ascend.test_pipeline.PipelineTest.test_hand_calculated_pipeline_and_double_buffer",
        "tests.ascend.test_pipeline.PipelineTest.test_ready_prefetch_does_not_wait_for_future_output",
        "tests.ascend.test_pipeline.PipelineTest.test_reference_counts_unchanged_and_stage_sum_is_not_makespan",
        "tests.tsp.test_baseline.TSPReferenceTest.test_proxy_full_plane_loading_and_weight_reuse",
        "tests.tsp.test_baseline.TSPReferenceTest.test_proxy_cross_k_stream_budget_and_configured_precedence",
        "tests.tsp.test_baseline.TSPReferenceTest.test_fixed_parameters_and_separate_allocated_services",
        "tests.tsp.test_baseline.TSPReferenceTest.test_fixed_install_ceiling_and_propagation_are_independent",
        "tests.tsp.test_baseline.TSPReferenceTest.test_vxm_tail_service_and_fixed_pipeline_latency",
        "tests.tpu.test_config.TPUConfigTest.test_peak_units_and_physical_inventory",
        "tests.tpu.test_config.TPUConfigTest.test_hierarchy_peaks_and_local_inventory",
        "tests.tpu.test_config.TPUConfigTest.test_chip_inventory_scales_private_resources_only",
        "tests.tpu.test_config.TPUConfigTest.test_vector_work_shares_one_service_and_rounds_once",
        "tests.tpu.test_count.TPUCountTest.test_chip_vmem_total_is_not_usable_by_one_tensorcore",
        "tests.tpu.test_count.TPUTimingTest.test_single_mxu_and_tiny_tail_match_hand_calculations",
        "tests.tpu.test_count.TPUTimingTest.test_four_mxus_run_one_wave_and_fifth_tile_needs_another",
        "tests.tpu.test_count.TPUTimingTest.test_k_accumulation_stays_in_vmem_and_final_export_is_once",
        "tests.tpu.test_cmem.CMEMTest.test_hand_calculated_cold_fill_reuse_and_shared_endpoint_timing",
        "tests.tpu.test_cmem.CMEMTest.test_tail_panels_and_outputs_conserve_bytes_and_compute",
        "tests.tpu.test_cmem.CMEMTest.test_directional_rates_endpoint_bottlenecks_and_core_shares",
        "tests.tpu.test_cmem.CMEMTest.test_cold_one_use_panel_pays_fill_and_startup_without_claiming_speedup",
        "tests.tpu.test_cmem_overlap.CMEMOverlapTest.test_double_buffer_hand_timeline_and_shared_hbm",
        "tests.tpu.test_cmem_overlap.CMEMOverlapTest.test_single_slot_released_after_last_read_before_compute_finishes",
        "tests.tsp.test_baseline.TSPReferenceTest.test_geometry_capacity_and_modeled_inventory",
        "tests.tsp_cim.test_topology.TSPCIMTopologyTest.test_peaks_and_bandwidth_scopes_are_hand_calculated",
        "tests.tsp_cim.test_topology.TSPCIMTopologyTest.test_sram_inventory_is_subtotal_not_complete_physical_area",
        "tests.cim.test_model.MacroSpecificationTest.test_geometry_peak_and_resident_storage",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_full_hemisphere_and_short_waves_do_not_scale_single_domain_time",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_tails_cross_k_reduction_and_all_orders_conserve_work",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_background_lw_overlaps_and_respects_context_release",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_four_contexts_are_per_domain_and_m_reuse_does_not_move_weights",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_restricted_services_slow_execution_without_changing_work",
        "tests.tsp_cim.test_hemisphere.CalendarTest.test_background_transfer_uses_residual_bandwidth_and_waits_when_full",
    ],
    "A3": [
        "tests.common.test_hardware_archival.HardwareArchivalTest.test_old_schemas_and_reintroduced_archived_inputs_are_rejected",
        "tests.common.test_hardware_archival.HardwareArchivalTest.test_gemm_and_vector_services_do_not_load_missing_or_invalid_archives",
        "tests.common.test_config_documentation.ConfigDocumentationTest.test_annotation_exception_does_not_hide_invalid_model_fields",
        "tests.common.test_timing_breakdowns.TimingBreakdownsTest.test_count_mode_does_not_invent_zero_cycles",
        "tests.tsp.test_weight_prefetch.WeightPrefetchTest.test_capacity_is_finite_and_old_serial_policy_is_rejected",
        "tests.ascend.test_vector_service.VectorServiceTest.test_only_requested_operations_require_parameters",
        "tests.ascend.test_vector_service.VectorServiceTest.test_invalid_counts_parameters_and_missing_keys_fail",
        "tests.tpu.test_vector_service.VectorNonlinearTest.test_only_used_operation_dependencies_are_required",
        "tests.tpu.test_vector_service.VectorNonlinearTest.test_gemm_counts_and_time_do_not_consume_vector_or_nonlinear",
        "tests.tsp.test_mem_concurrency.MEMConcurrencyTest.test_missing_directional_services_are_not_inferred",
        "tests.tsp.test_vxm_service.VXMServiceTest.test_only_active_operations_require_parameters",
        "tests.tsp.test_vxm_service.VXMServiceTest.test_invalid_workloads_and_parameters_fail_explicitly",
        "tests.tsp.test_vxm_service.VXMServiceTest.test_missing_add_cost_is_required_only_for_cross_k",
        "tests.common.test_execution_contract.ExecutionContractTest.test_policy_and_fixed_latency_changes_are_consumed",
        "tests.ascend.test_config.ConfigTest.test_missing_parameters_do_not_get_silent_defaults",
        "tests.ascend.test_l2.L2Test.test_missing_inputs_and_unsupported_policies_fail_explicitly",
        "tests.ascend.test_l2.L2Test.test_cold_reuse_eviction_and_bypass_conservation",
        "tests.ascend.test_l2.L2Test.test_reference_shares_hits_and_rate_sensitivity",
        "tests.ascend.test_l2.L2Test.test_miss_fill_read_order_and_reuse_are_time_consistent",
        "tests.ascend.test_single_core.SingleCoreTest.test_hbm_padded_capacity_is_checked_and_l2_rate_is_consumed",
        "tests.ascend.test_single_core.SingleCoreTest.test_hbm_bandwidth_and_core_share_drive_both_transfers",
        "tests.ascend.test_single_core.SingleCoreTest.test_hbm_clock_conversion_and_missing_timing_allow_counts",
        "tests.ascend.test_pipeline.PipelineTest.test_capacity_fallback_keeps_tiles_and_counts",
        "tests.ascend.test_pipeline.PipelineTest.test_pipeline_uses_configured_services_and_trace_is_optional",
        "tests.ascend.test_pipeline.PipelineTest.test_k_panels_tail_and_segmented_output_lifetimes",
        "tests.ascend.test_pipeline.PipelineTest.test_random_small_schedules_preserve_dependencies_resources_and_storage",
        "tests.tsp.test_baseline.TSPReferenceTest.test_proxy_unknown_clock_and_missing_base",
        "tests.tpu.test_config.TPUConfigTest.test_missing_required_capacity_is_not_defaulted",
        "tests.tpu.test_cmem.CMEMTest.test_equal_capacity_share_shrinks_groups_then_reports_bypass",
        "tests.tpu.test_cmem.CMEMTest.test_missing_capacity_and_only_used_timing_services_are_required",
        "tests.tpu.test_cmem.CMEMTest.test_large_k_is_aggregated_and_preserves_prefetch_savings",
        "tests.tpu.test_cmem_overlap.CMEMOverlapTest.test_trace_dependencies_and_aggregated_schedule_agree_with_tails",
        "tests.tpu.test_cmem_overlap.CMEMOverlapTest.test_capacity_preserves_reuse_and_counts_while_reducing_slots",
        "tests.tpu.test_cmem_overlap.CMEMOverlapTest.test_billion_panels_and_two_dimensional_tail_grid_do_not_expand_events",
        "tests.tpu.test_config.TPUConfigTest.test_unknown_inputs_propagate_and_do_not_block_independent_counts",
        "tests.tpu.test_config.TPUConfigTest.test_hierarchy_unknowns_preserve_independent_levels",
        "tests.tpu.test_config.TPUConfigTest.test_vector_fixed_service_has_no_geometry_fallback",
        "tests.tpu.test_config.TPUConfigTest.test_vector_costs_are_required_only_for_used_operations",
        "tests.tpu.test_count.TPUTimingTest.test_required_unknowns_reject_timing_but_preserve_count_mode",
        "tests.tpu.test_count.TPUTimingTest.test_vmem_proxy_is_explicit_and_configured_service_is_respected",
        "tests.tsp.test_baseline.TSPReferenceTest.test_cli_count_and_missing_timing",
        "tests.tsp.test_baseline.TSPReferenceTest.test_legacy_dynamic_fields_and_invalid_fixed_latencies_fail",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_unknown_clock_propagates_to_peak_and_provenance_is_copied",
        "tests.cim.test_model.MacroSpecificationTest.test_no_implicit_defaults_or_old_geometry",
        "tests.cim.test_model.MacroActionTest.test_program_is_full_context_and_local_timing_api_is_retired",
        "tests.common.test_parameter_source_audit.ParameterSourceAuditTest.test_hbm_bandwidth_does_not_replace_required_local_services",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_capacity_unknown_services_and_physical_half_selection",
    ],
    "A4": [
        "tests.common.test_execution_contract.ExecutionContractTest.test_four_reports_share_semantic_layers_without_changing_boundaries",
        "tests.tpu.test_count.TPUCountTest.test_chip_geometry_does_not_scale_single_tensorcore_counts",
    ],  # Also checks four real subprocess CLI invocations, recorded separately below.
    "A5": [
        "tests.common.test_config_documentation.ConfigDocumentationTest.test_notes_are_optional_and_do_not_change_reports_or_reference_ids",
        "tests.ascend.test_vector_service.VectorServiceTest.test_gemm_does_not_invoke_vector_or_require_unused_services",
        "tests.tpu.test_vector_service.VectorNonlinearTest.test_schema_migration_and_invalid_nonlinear_inputs",
        "tests.tsp.test_mem_concurrency.MEMConcurrencyTest.test_migration_requires_explicit_conflict_free_policy",
        "tests.ascend.test_single_core.SingleCoreTest.test_removed_experiment_and_scheduling_flags_fail_explicitly",
        "tests.ascend.test_pipeline.PipelineTest.test_cli_trace_and_removed_policy_arguments",
        "tests.ascend.test_config.ConfigTest.test_old_event_schema_is_rejected",
        "tests.ascend.test_config.ConfigTest.test_retired_fields_are_rejected_in_current_schema",
        "tests.tpu.test_config.TPUConfigTest.test_retired_fields_rejected_and_reserved_vector_services_do_not_execute",
        "tests.tpu.test_cmem.CMEMTest.test_invalid_policies_and_cli_errors_preserve_output",
        "tests.tpu.test_cmem_overlap.CMEMOverlapTest.test_invalid_slot_configuration_and_override_rejected",
        "tests.tsp.test_baseline.TSPReferenceTest.test_retired_parameters_rejected_and_fp32_costs_not_used_by_gemm",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_legacy_overlay_and_inline_macro_are_not_accepted",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_retired_inventory_fields_and_old_schema_are_rejected",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_fp32_service_is_visible_but_not_a_gemm_missing_parameter",
        "tests.tsp_cim.test_config.TSPCIMConfigTest.test_writer_scope_migration_is_explicit",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_native_comparison_inherits_actual_allocation_and_mode",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_cli_defaults_and_removed_scope_flags",
        "tests.tsp_cim.test_hemisphere.HemisphereTest.test_invalid_inputs_and_removed_selection_arguments_are_rejected",
        "tests.cim.test_model.MacroSpecificationTest.test_old_schema_and_retired_timing_fields_are_rejected",
        "tests.common.test_workload.WorkloadTest.test_tsp_cim_entry_rejects_unsupported_precision_and_counts_explicit_int8",
    ],
}
TITLES = {"A1": "来源、单位与作用域可追溯", "A2": "派生算力、容量、带宽与面积库存测试",
          "A3": "缺参显式处理", "A4": "四入口 INT8 GEMM 冒烟", "A5": "迁移说明与四入口的 GEMM 边界"}


def timing_breakdowns_valid(report):
    timing = report["timing"]
    if any(key in timing for key in ("stages_cycles", "service_cycles_sum", "stages_cycles_basis", "stage_semantics")):
        return False
    evaluated = timing["total_cycles"] is not None
    scope = report["boundary"]["scope"]
    available = {"service_cycles": scope in ("single_core", "single_tensorcore"),
                 "elapsed_cycles": scope == "single_hem" or
                     (scope == "single_tensorcore" and report["execution_contract"]["schedule"] == "serial")}
    for name, modeled in available.items():
        view = timing[name]
        expected = "not_evaluated" if not evaluated else "available" if modeled else "not_decomposed"
        if view["status"] != expected:
            return False
        if expected == "available":
            if not view["by_stage"] or any(type(v) is not int or v < 0 for v in view["by_stage"].values()):
                return False
            if sum(view["by_stage"].values()) != view["total_cycles"]:
                return False
        elif view["by_stage"] is not None or (name == "service_cycles" and view["total_cycles"] is not None):
            return False
    return timing["elapsed_cycles"]["total_cycles"] == timing["total_cycles"]


def snapshot():
    """Issue #2 native-model review scope; operator work has its own evidence."""
    paths = {ROOT / name for name in ("README.md", "ARCHITECTURE.md", "ascend.py", "tpu.py", "tsp.py", "tsp_cim.py",
                                     "requirements.txt", "scripts/verify_issue2.py", "scripts/verification.py",
                                     "scripts/clear_results.ps1", "scripts/run_all.ps1", "scripts/run_all.py",
                                     "calibration/parameter_source_audit.json")}
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("configs", "*.json")):
        paths.update(p for p in (ROOT / folder).rglob(pattern)
                     if not issue3_owned(p.relative_to(ROOT).as_posix())
                     and not p.relative_to(ROOT).as_posix().startswith("tests/validation/"))
    return fingerprints(paths)


def dependency_snapshot():
    # Shared CLI imports the optional operator dispatcher. Record its code for
    # reproducibility without treating operator tests/policy as A1/A5 criteria.
    return fingerprints(p for p in (ROOT / "src").rglob("*.py")
                        if issue3_owned(p.relative_to(ROOT).as_posix()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "validation/issue2/run")
    args = parser.parse_args(argv)
    try:
        out = output_directory(args.output_dir, "issue2")
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    os.environ["PYTHONUTF8"] = "1"
    before = snapshot()
    dependencies_before = dependency_snapshot()
    review_path = ROOT / "validation/issue2/review.json"
    review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {}
    context_path = ROOT / "validation/issue2/review_context.json"
    review_context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else None
    reviewed = review.get("snapshot", {})
    changed = sorted(k for k in set(before) | set(reviewed)
                     if not issue3_owned(k) and not k.startswith("tests/validation/")
                     and before.get(k) != reviewed.get(k))
    review_valid = not changed and all(review.get("conditions", {}).get(k, {}).get("status") == "reviewed" for k in ("A1", "A5"))

    tests, counts, suite_passed = run_tests(discover(ISSUE2_TEST_DIRS), out)

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
                    "no_invented_time": r.get("timing", {}).get("total_cycles") is None,
                    "timing_breakdowns": timing_breakdowns_valid(r)}
            if scope == "single_core":
                derived = r.get("hardware_derived", {})
                provenance = r.get("parameter_provenance", {})
                retired = {"vector.fp32_ops_per_cycle", "sfu.units_per_core", "sfu.fp32_ops_per_cycle",
                           "chip.noc_bytes_per_cycle", "chip.interconnect_bytes_per_second"}
                checks.update(
                    report_schema=r.get("schema_version") == 12,
                    shared_vector=derived["vector_service"]["base_ops_per_cycle"] == 64
                        and derived["vector_service"]["operation_costs"]["fp32_exp"] == 15
                        and derived["vector_service"]["operation_costs"]["fp32_rsqrt"] == 12
                        and derived["vector_service"]["gemm_operations"] == [],
                    no_retired_parameters=retired.isdisjoint(provenance),
                    no_generic_vector_sfu_peaks=all(k not in derived for k in
                        ("vector_fp32_peak_gops_per_core", "sfu_fp32_peak_gops_per_core")),
                    compute_inventory_archived="compute_units" not in derived["area_inventory"]
                        and {"cube.units_per_core", "vector.units_per_core"}.isdisjoint(provenance),
                    no_unknown_hardware_parameters=all(p["value"] is not None for p in provenance.values()),
                    explicit_l2_assumption=provenance["chip.l2_bytes_per_cycle"]["value"] == 3000
                        and provenance["chip.l2_bytes_per_cycle"]["status"] == "assumed"
                        and provenance["chip.l2_bytes_per_cycle"]["scope"] == "chip",
                    l2_cold=r["cache"]["hits"] == 0 and r["cache"]["misses"] == 2
                        and r["cache"]["fill_bytes"] == r["cache"]["read_bytes"] == 1024
                        and r["memory"]["l2"]["capacity_bytes"] == 1048576,
                    hbm_boundary=r["boundary"]["start"] == "prepacked_padded_A_B_in_HBM"
                        and r["boundary"]["end"] == "packed_padded_INT32_C_in_HBM",
                    hbm_capacity=r["memory"]["hbm"]["padded_footprint_bytes"] == 2048
                        and r["memory"]["hbm"]["usable_bytes"] == 32000000000,
                    hbm_actions=r["action_counts"]["hbm_read_bytes"]["count"] == 1024
                        and r["action_counts"]["hbm_write_bytes"]["count"] == 1024)
            if scope == "single_tensorcore":
                boundary = r.get("boundary", {})
                inventory = r.get("hardware_inventory", {})
                hierarchy = r.get("hardware_hierarchy", {})
                checks.update({
                    "report_schema": r.get("schema_version") == 13,
                    "vpu_inventory_archived": all("vpus" not in inventory[level] and "vpu_alus" not in inventory[level]
                        for level in ("chip", "per_tensorcore")) and "nominal_chip_int8_tops" not in r["hardware_derived"],
                    "scalar_inventory_retired": "scalar_units" not in inventory.get("chip", {})
                        and "scalar_units" not in inventory.get("per_tensorcore", {})
                        and "scalar_units_per_tensorcore" not in r["parameter_provenance"],
                    "sparse_inventory_retired": not {"sparsecores", "sparse_spmem_bytes"} & inventory.get("chip", {}).keys(),
                    "sparse_parameters_retired": not {"sparsecores_per_chip", "spmem_physical_bytes"} & r.get("parameter_provenance", {}).keys(),
                    "archived_inventory_removed": not {"register_file_bytes", "ici_links", "die_area_upper_bound_mm2", "die_area_bound_operator"} & inventory.get("chip", {}).keys()
                        and "interconnect_specification" not in r,
                    "reserved_shared_vector": r.get("hardware_derived", {}).get("vector_service", {}).get("included_in_gemm") is False,
                    "shared_nonlinear_proxy": r["hardware_derived"]["vector_service"]["operation_costs"]["fp32_exp"] == 15
                        and r["hardware_derived"]["vector_service"]["operation_costs"]["fp32_rsqrt"] == 12
                        and r["hardware_derived"]["vector_service"]["gemm_operations"] == []
                        and r["hardware_derived"]["vector_service"]["special_function_model"] == "algorithmic_work_proxy_not_native_instruction_timing",
                    "no_unknown_hardware_parameters": r.get("missing_parameters") == [],
                    "cmem_tiled": r.get("execution_contract", {}).get("effective_cmem_mode") == "tiled",
                    "cmem_cold_fill": r["counts"]["cmem_fill_bytes"] == 1024
                        and r["action_counts"]["cmem_write_bytes"]["count"] == 1024
                        and r["action_counts"]["cmem_read_bytes"]["count"] == 1024,
                    "cmem_proxy": all(
                        r["parameter_provenance"][f"cmem_{direction}_bandwidth_bytes_per_second"]["value"] == value
                        and r["parameter_provenance"][f"cmem_{direction}_bandwidth_bytes_per_second"]["status"] == "assumed"
                        and r["parameter_provenance"][f"cmem_{direction}_bandwidth_bytes_per_second"]["scope"] == "chip"
                        for direction, value in (("read", 2150704873472), ("write", 1075352436736))),
                    "chip_hardware_scope": boundary.get("hardware_scope") == "chip",
                    "local_counting_scope": boundary.get("counting_scope") == "single_tensorcore",
                    "no_chip_execution": boundary.get("chip_execution_included") is False,
                    "one_counted_tensorcore": boundary.get("simulated_tensorcores") == 1,
                    "chip_hierarchy": hierarchy.get("chip", {}).get("tensorcores_per_chip") == 2
                                      and hierarchy.get("chip", {}).get("mxus_per_chip") == 8,
                    "local_and_chip_inventory": inventory.get("per_tensorcore", {}).get("mxus") == 4
                                                and inventory.get("chip", {}).get("mxus") == 8,
                })
            if r.get("model") == "tsp_cim_hemisphere_reference":
                inventory = r.get("hardware_inventory", {}).get("chip", {})
                retired = {"physical_sram_bytes", "additional_local_sram_bytes", "collector_buffer_bytes_per_plane",
                           "sxm_physical_units", "stream_physical_endpoints", "clock_domains", "control_domains"}
                checks.update(
                    report_schema=r.get("schema_version") == 10,
                    nim_inventory_archived={"nim_units", "nim_tiles", "nim_logical_channels_per_tile", "nim_conversion_lanes"}.isdisjoint(inventory),
                    retired_inventory_absent=retired.isdisjoint(inventory),
                    known_storage_subtotal=inventory.get("known_mem_plus_macro_bytes") == 243793920,
                    no_gemm_missing_parameters=r.get("missing_parameters") == [],
                    vxm_costs_preserved=r["parameter_provenance"]["shared_system"]["vxm_fp32_exp_cost"]["value"] == 15
                        and r["parameter_provenance"]["shared_system"]["vxm_fp32_rsqrt_cost"]["value"] == 12
                        and r["hardware_derived"]["vxm_service"]["nonlinear_included_in_gemm"] is False,
                    resident_capacity_checked=r["boundary"]["mem_allocation_included"] is True,
                    eight_domains=r["hardware_inventory"]["allocated_cluster_domains"] == 8,
                    local_capacity=r["memory"]["capacity_bytes"] == 110*2**20,
                    collector_limit_explicit=r["boundary"]["collector_capacity_validated"] is False)
            return checks
        return check

    for entry, scope in (("ascend", "single_core"), ("tpu", "single_tensorcore"),
                         ("tsp", "single_hem"), ("tsp_cim", "single_hem")):
        invoke(entry, entry + ".py", ["--mode", "count"], check=smoke_check(scope))
    ascend_missing = json.loads((ROOT / "configs/hardware/ascend910_reference.json").read_text(encoding="utf-8"))
    ascend_missing["chip"]["hbm_bandwidth_bytes_per_second"].update(value=None, status="unknown")
    save(out / "ascend_missing_timing.hardware.json", ascend_missing)
    invoke("ascend_missing_timing", "ascend.py", ["--mode", "timing", "--hardware", str(out / "ascend_missing_timing.hardware.json")],
           2, "Timing parameters missing: chip.hbm_bandwidth_bytes_per_second")
    ascend_capacity = json.loads((ROOT / "configs/hardware/ascend910_reference.json").read_text(encoding="utf-8"))
    ascend_capacity["chip"]["hbm_usable_bytes"].update(value=None, status="unknown")
    save(out / "ascend_missing_capacity.hardware.json", ascend_capacity)
    invoke("ascend_missing_capacity", "ascend.py", ["--mode", "count", "--hardware", str(out / "ascend_missing_capacity.hardware.json")],
           2, "Counting parameters missing: chip.hbm_usable_bytes")
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
                          explicit_assumptions=bool(t["assumptions"]) and name in t["proxy_parameters"]
                              and t["parameters"][name]["status"] == "assumed",
                          explicit_config_service=name not in t["missing_parameters"]
                              and r["parameter_provenance"][name]["value"] == t["parameters"][name]["value"]
                              and r["parameter_provenance"][name]["status"] == "assumed",
                          no_chip_time=t["chip_cycles"] is None)
            if scope == "single_core":
                checks.update(hbm_derived_service=t["derived_services"]["hbm"]["service_bytes_per_cycle"] == 37.5,
                              hbm_source=t["parameters"]["chip.hbm_bandwidth_bytes_per_second"]["value"] == 1.2e12,
                              pipeline_policy=t["policy"] == "resource_pipeline",
                              service_work=t["service_cycles"]["total_cycles"] == sum(t["service_cycles"]["by_stage"].values()) == 84,
                              scheduled_time=t["total_cycles"] == t["schedule"]["total_cycles"],
                              l2_service=t["derived_services"]["l2"]["service_bytes_per_cycle"] == 93.75
                                  and t["schedule"]["resource_busy_cycles"]["l2"] == 24,
                              shared_hbm=t["schedule"]["resource_busy_cycles"]["hbm"] == 56,
                              bounded_overlap=t["total_cycles"] <= t["service_cycles"]["total_cycles"])
            if scope == "single_hem":
                checks.update(
                    fixed_service_schema=r["schema_version"] == 10 and t["model"] == "single_hem_weight_prefetch_v1",
                    fixed_install_parameters=t["parameters"]["install_bpc"]["value"] == 10240
                        and t["parameters"]["install_propagation_cycles"]["value"] == 19,
                    separate_derived_service=t["derived_services"]["install_bytes_per_cycle"] == 10240
                        and t["derived_services"]["weight_buffer_bytes"] == 204800,
                    no_dynamic_parameters=all(p["selection"] == "configured" and "derivation" not in p
                                              for p in t["parameters"].values()))
            return checks
        return check

    invoke("ascend_system_timing", "ascend.py", [],
           check=proxy_timing_check("single_core", 83, 1, "cube.macs_per_cycle"))
    invoke("tsp_system_timing", "tsp.py", [],
           check=proxy_timing_check("single_hem", 59, 0.9, "mxm_row_ii"))
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
                      expected_cycles=timing["total_cycles"] == 293,
                      expected_ns=abs(timing["time_ns"] - 293 / 1.05) < 1e-9,
                      explicit_assumptions=bool(timing["assumptions"]) and all(
                          timing["parameters"][p]["status"] == "assumed"
                          for p in ("hbm_share_bytes_per_second", "vmem_service_bytes_per_cycle", "fill_drain_cycles_per_wave")),
                      explicit_config_service="vmem_bandwidth_bytes_per_second" not in r["missing_parameters"]
                          and r["parameter_provenance"]["vmem_bandwidth_bytes_per_second"]["value"] == 1075200000000
                          and r["parameter_provenance"]["vmem_bandwidth_bytes_per_second"]["status"] == "assumed",
                      no_chip_time=timing["chip_cycles"] is None)
        return checks

    invoke("tpu_system_timing", "tpu.py", [], check=tpu_timing_check)

    def cim_timing_check(r):
        checks = smoke_check("single_hem")(r)
        del checks["count_mode"], checks["no_invented_time"]
        timing = r["timing"]
        checks.update(timing_mode=r["mode"] == "timing",
                      analytical_estimate=timing["status"] == "analytical_estimate",
                      expected_cycles=timing["total_cycles"] == 296,
                      expected_ns=abs(timing["time_ns"] - 296 / 0.9) < 1e-9,
                      explicit_assumptions=bool(timing["assumptions"]) and timing["parameters"]["writer_bpc_per_superlane_plane"]["status"] == "assumed",
                      no_standalone_macro_timing=not any(k in r["parameter_provenance"]["macro"]["parameters"]
                          for k in ("compute_first_cycles", "write_first_cycles", "write_wordline_ii_cycles", "context_switch_cycles")),
                      no_chip_time=timing["chip_cycles"] is None)
        return checks

    invoke("cim_system_timing", "tsp_cim.py", [], check=cim_timing_check)
    invalid = {**workload, "operation": "transformer"}
    save(out / "non_gemm.workload.json", invalid)
    for entry in ("ascend", "tpu", "tsp", "tsp_cim"):
        invoke(entry + "_non_gemm", entry + ".py", ["--workload", str(out / "non_gemm.workload.json")], 2, "Only a GEMM workload")
    save(out / "commands.json", commands)

    extra = {"A3": ["ascend_missing_capacity", "ascend_missing_timing", "tsp_missing_timing", "tpu_missing_capacity", "tpu_missing_timing", "tpu_system_timing", "cim_system_timing"],
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
                           "review_basis": review_path.as_posix() if reviewed_condition else None}
    unchanged = snapshot() == before and dependency_snapshot() == dependencies_before
    accepted = suite_passed and unchanged and all(c["status"].startswith("passed") for c in conditions.values())
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    summary = {"issue": "https://github.com/pku-pacific-lab-team/huawei_dfp/issues/2",
               "generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
               "repository_head": git.stdout.strip(), "snapshot": before, "snapshot_unchanged_during_run": unchanged,
               "dependency_snapshot": dependencies_before, "test_directories": list(ISSUE2_TEST_DIRS),
               "review_matches_current_files": review_valid, "review_changed_files": changed,
               "review_context": review_context,
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
               "limits": ["Issue #2 criteria only; operator acceptance is recorded separately by verify_issue3.py.",
                          "Optional operator code imported by shared CLI is fingerprinted as a dependency, not an A1/A5 review condition.",
                          "Local working-tree evidence, including uncommitted content; not a commit, publication or external sign-off.",
                          "A1/A5 reuse recorded source/code/document review only if fingerprints match. This command does not re-verify external publications.",
                          "Unknown and assumed parameters remain explicit; test success is not hardware calibration or chip-performance validation."]}
    save(out / "acceptance.json", summary)
    print(f"Unit tests: {counts}; unchanged snapshot: {unchanged}")
    for key, result in conditions.items():
        print(f"{key}: {result['status']} - {result['title']}")
    if review_context:
        print(f"Review context: {review_context['summary']}")
    print(f"Evidence: {out / 'acceptance.json'}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
