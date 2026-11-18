"""Shared Vector/Nonlinear occupancy and separation from the MXU GEMM path."""
import unittest

from src.core.tpu import TPUSimulator
from src.hardware.tpu import TPUSpec, VECTOR_SERVICE_PARAMETERS
from tests.tpu.support import hardware_config, spec, workload


class VectorNonlinearTest(unittest.TestCase):
    def test_mixed_operations_share_one_budget_and_tail_rounding(self):
        s = spec()
        self.assertEqual(s.vector_service_cycles({"fp32_exp": 2048}), 15)
        self.assertEqual(s.vector_service_cycles({"fp32_rsqrt": 2048}), 12)
        # 512 + 64*15 + 48*12 = 2048; rounding per operation would overcharge.
        self.assertEqual(s.vector_service_cycles(
            {"int32_add": 512, "fp32_exp": 64, "fp32_rsqrt": 48}), 1)
        self.assertEqual(s.vector_service_cycles({"fp32_exp": 2048, "fp32_mul": 2048}), 16)
        for op, boundary in (("fp32_exp", 136), ("fp32_rsqrt", 170)):
            self.assertEqual(s.vector_service_cycles({op: boundary}), 1)
            self.assertEqual(s.vector_service_cycles({op: boundary + 1}), 2)

    def test_fractional_costs_are_totals_not_multiplied_by_basic_cost(self):
        c = hardware_config()
        c["parameters"]["vector_fp32_basic_op_cost"]["value"] = 3
        self.assertEqual(TPUSpec(c).vector_service_cycles({"fp32_exp": 2048}), 15)
        for name in ("vector_fp32_exp_cost", "vector_fp32_rsqrt_cost"):
            c["parameters"][name]["value"] = 0.1
        c["parameters"]["vector_base_ops_per_cycle"]["value"] = 0.3
        s = TPUSpec(c)
        self.assertEqual(s.vector_service_cycles({"fp32_exp": 1, "fp32_rsqrt": 2}), 1)
        self.assertEqual(s.vector_service_cycles({"fp32_exp": 1, "fp32_rsqrt": 3}), 2)

    def test_only_used_operation_dependencies_are_required(self):
        c = hardware_config()
        for name in VECTOR_SERVICE_PARAMETERS:
            c["parameters"][name].update(value=None, status="unknown")
        s = TPUSpec(c)
        self.assertEqual(s.vector_service_cycles({}), 0)
        self.assertEqual(s.vector_service_cycles({"fp32_exp": 0}), 0)
        with self.assertRaisesRegex(ValueError, "vector_base_ops_per_cycle.*vector_fp32_exp_cost"):
            s.vector_service_cycles({"fp32_exp": 1})
        c["parameters"]["vector_base_ops_per_cycle"].update(value=2048, status="assumed")
        c["parameters"]["vector_fp32_exp_cost"].update(value=15, status="assumed")
        s = TPUSpec(c)
        self.assertEqual(s.vector_service_cycles({"fp32_exp": 2048, "fp32_rsqrt": 0}), 15)
        self.assertIsNone(s.vector_service()["operation_rates_per_cycle"]["fp32_rsqrt"])
        with self.assertRaisesRegex(ValueError, "vector_fp32_rsqrt_cost"):
            s.vector_service_cycles({"fp32_rsqrt": 1})

    def test_gemm_counts_and_time_do_not_consume_vector_or_nonlinear(self):
        for cmem in ("tiled", "bypass"):
            c = hardware_config(cmem_mode=cmem)
            baseline = TPUSimulator(TPUSpec(c)).simulate(workload(17, 257, 15), mode="timing")
            for name in VECTOR_SERVICE_PARAMETERS:
                c["parameters"][name].update(value=None, status="unknown")
            changed = TPUSimulator(TPUSpec(c)).simulate(workload(17, 257, 15), mode="timing")
            for key in ("counts", "action_counts", "memory", "timing"):
                with self.subTest(cmem=cmem, key=key):
                    self.assertEqual(changed[key], baseline[key])
            self.assertFalse(set(VECTOR_SERVICE_PARAMETERS) & changed["timing"]["parameters"].keys())

    def test_report_discloses_proxy_and_preserves_single_core_scope(self):
        c = hardware_config()
        service = TPUSpec(c).vector_service()
        self.assertEqual(service["scope"], "tensorcore")
        self.assertEqual(service["resource_policy"], "one_shared_vector_service")
        self.assertEqual(service["status"], "analytical_compute_service")
        self.assertEqual(service["gemm_operations"], [])
        self.assertFalse(service["nonlinear_included_in_gemm"])
        self.assertFalse(service["included_in_gemm"])
        self.assertIn("not_native_instruction", service["special_function_model"])
        self.assertEqual(set(service["assumed_inputs"]), set(VECTOR_SERVICE_PARAMETERS))
        c["parameters"]["tensorcores_per_chip"]["value"] = 4
        self.assertEqual(TPUSpec(c).vector_service(), service)
        service["operation_costs"]["fp32_exp"] = 999
        self.assertEqual(TPUSpec(c).vector_service_cycles({"fp32_exp": 2048}), 15)

    def test_schema_migration_and_invalid_nonlinear_inputs(self):
        c = hardware_config()
        c["schema_version"] = 10
        for name in ("vector_fp32_exp_cost", "vector_fp32_rsqrt_cost"):
            del c["parameters"][name]
        with self.assertRaisesRegex(ValueError, "Expected schema 12.*archive nominal_int8_tops"):
            TPUSpec(c)
        c["schema_version"] = 12
        with self.assertRaisesRegex(ValueError, "parameters: expected fields"):
            TPUSpec(c)
        for name in ("vector_fp32_exp_cost", "vector_fp32_rsqrt_cost"):
            for value in (0, -1, True, float("nan"), float("inf")):
                c = hardware_config()
                c["parameters"][name]["value"] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TPUSpec(c)
        for counts in ({"fp32_exp": True}, {"fp32_rsqrt": -1}, {"fp32_exp": 1.5},
                       {"softmax": 0}, {"fp32_div": 1}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                spec().vector_service_cycles(counts)
