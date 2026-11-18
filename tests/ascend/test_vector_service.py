import copy
import unittest

from src.core.ascend import AscendSimulator
from src.hardware.ascend import AscendSpec, VECTOR_SERVICE_PARAMETERS
from tests.ascend.support import hardware_config, workload


class VectorServiceTest(unittest.TestCase):
    def setUp(self):
        self.config = hardware_config(False)
        self.spec = AscendSpec(self.config)

    def test_shared_budget_and_hand_computed_work(self):
        for op, cycles in (("int32_add", 1), ("fp32_add", 1), ("fp32_mul", 1),
                           ("fp32_exp", 15), ("fp32_rsqrt", 12)):
            with self.subTest(op=op):
                self.assertEqual(self.spec.vector_service_cycles({op: 64}), cycles)
        # 32 + 15 + 12 + 5 = 64 equivalent operations, one shared cycle.
        self.assertEqual(self.spec.vector_service_cycles(
            {"int32_add": 32, "fp32_exp": 1, "fp32_rsqrt": 1, "fp32_mul": 5}), 1)
        self.assertEqual(self.spec.vector_service_cycles({"fp32_add": 64, "fp32_exp": 64}), 16)
        self.assertEqual(self.spec.vector_service_cycles({"fp32_exp": 4}), 1)
        self.assertEqual(self.spec.vector_service_cycles({"fp32_exp": 5}), 2)

    def test_configured_budget_costs_and_exact_rounding(self):
        self.config["vector"]["base_ops_per_cycle"]["value"] = 0.3
        self.config["vector"]["fp32_basic_op_cost"]["value"] = 0.1
        spec = AscendSpec(self.config)
        self.assertEqual(spec.vector_service_cycles({"fp32_add": 1, "fp32_mul": 2}), 1)
        self.assertEqual(spec.vector_service_cycles({"fp32_add": 4}), 2)
        self.config["vector"]["base_ops_per_cycle"]["value"] = 32
        self.config["vector"]["fp32_exp_cost"]["value"] = 30
        self.assertEqual(AscendSpec(self.config).vector_service_cycles({"fp32_exp": 64}), 60)

    def test_only_requested_operations_require_parameters(self):
        for name in VECTOR_SERVICE_PARAMETERS:
            self.config["vector"][name].update(value=None, status="unknown")
        spec = AscendSpec(self.config)
        self.assertEqual(spec.vector_service_cycles({}), 0)
        self.assertEqual(spec.vector_service_cycles({"fp32_exp": 0}), 0)
        with self.assertRaisesRegex(ValueError, "vector.base_ops_per_cycle.*vector.fp32_exp_cost"):
            spec.vector_service_cycles({"fp32_exp": 1})
        self.config["vector"]["base_ops_per_cycle"].update(value=64, status="assumed")
        self.config["vector"]["int32_add_cost"].update(value=1, status="assumed")
        spec = AscendSpec(self.config)
        self.assertEqual(spec.vector_service_cycles({"int32_add": 64, "fp32_exp": 0}), 1)
        with self.assertRaisesRegex(ValueError, "vector.fp32_rsqrt_cost"):
            spec.vector_service_cycles({"fp32_rsqrt": 1})

    def test_gemm_does_not_invoke_vector_or_require_unused_services(self):
        base = AscendSimulator(self.spec).simulate(workload(33, 289, 97), "timing")
        for name in VECTOR_SERVICE_PARAMETERS:
            self.config["vector"][name].update(value=None, status="unknown")
        report = AscendSimulator(AscendSpec(self.config)).simulate(workload(33, 289, 97), "timing")
        for key in ("counts", "action_counts", "memory", "cache", "stages", "timing"):
            self.assertEqual(base[key], report[key], key)
        service = report["hardware_derived"]["vector_service"]
        self.assertEqual(len(service["missing_parameters"]), 5)
        self.assertEqual(report["timing"]["missing_parameters"], [])
        self.assertFalse(service["nonlinear_included_in_gemm"])
        self.assertEqual(service["gemm_operations"], [])
        self.assertNotIn("compute_units", report["hardware_derived"]["area_inventory"])

    def test_inventory_and_clock_do_not_implicitly_scale_service(self):
        self.assertNotIn("units_per_core", self.config["vector"])
        self.config["chip"]["architectural_cores"]["value"] = 64
        self.config["clock_freq_ghz"].update(value=None, status="unknown")
        spec = AscendSpec(self.config)
        self.assertEqual(spec.vector_service_cycles({"fp32_exp": 64}), 15)
        self.assertEqual(spec.vector_service(), self.spec.vector_service())
        report = spec.derived()["vector_service"]
        self.assertEqual(report["operation_rates_per_cycle"]["fp32_exp"], 64 / 15)
        self.assertEqual(report["resource_policy"], "one_shared_vector_service")
        self.assertIn("not_native_instruction", report["special_function_model"])
        report["operation_costs"]["fp32_exp"] = 999
        self.assertEqual(spec.vector_service_cycles({"fp32_exp": 64}), 15)
        self.assertEqual(len(self.spec.provenance()), 29)
        self.assertTrue(all(p["value"] is not None for p in self.spec.provenance().values()))

    def test_invalid_counts_parameters_and_missing_keys_fail(self):
        for counts in (None, [], {"fp32_exp": -1}, {"fp32_exp": True}, {"fp32_exp": 1.0},
                       {"fp32_exp": float("nan")}, {"softmax": 0}, {"fp32_div": 1}, {1: 2}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.spec.vector_service_cycles(counts)
        for name in VECTOR_SERVICE_PARAMETERS:
            for value in (0, -1, True, float("nan"), float("inf")):
                c = copy.deepcopy(self.config)
                c["vector"][name]["value"] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    AscendSpec(c)
            c = copy.deepcopy(self.config)
            del c["vector"][name]
            with self.assertRaises(ValueError):
                AscendSpec(c)
        for field, value in (("unit", "cycle"), ("scope", "chip"), ("status", "unknown")):
            c = copy.deepcopy(self.config)
            c["vector"]["fp32_exp_cost"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                AscendSpec(c)
