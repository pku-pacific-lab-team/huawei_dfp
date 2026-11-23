"""Shared VXM occupancy, operation-specific dependencies and GEMM integration."""
import copy
import unittest

from src.core.tsp.simulator import TSPSimulator
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.hardware.tsp import TSPSpec, VXM_SERVICE_PARAMETERS
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader


class VXMServiceTest(unittest.TestCase):
    def setUp(self):
        self.config = TSPConfigLoader().load_hardware_config()
        self.spec = TSPSpec(self.config)

    def test_defaults_and_shared_weighted_occupancy(self):
        self.assertFalse(any(value is None for value in self.spec.values.values()))
        for op, cycles in (("int32_add", 1), ("fp32_add", 1), ("fp32_mul", 1),
                           ("fp32_exp", 15), ("fp32_rsqrt", 12)):
            with self.subTest(op=op):
                self.assertEqual(self.spec.vxm_service_cycles({op: 2560}), cycles)
        # Exactly 2560 equivalent operations, rounded once across operation types.
        self.assertEqual(self.spec.vxm_service_cycles(
            {"int32_add": 1280, "fp32_exp": 64, "fp32_rsqrt": 24, "fp32_mul": 32}), 1)
        self.assertEqual(self.spec.vxm_service_cycles({"int32_add": 2560, "fp32_add": 2560}), 2)
        self.assertEqual(self.spec.vxm_service_cycles({"fp32_exp": 170}), 1)
        self.assertEqual(self.spec.vxm_service_cycles({"fp32_exp": 171}), 2)

    def test_fractional_costs_round_exactly_once(self):
        self.config["parameters"]["vxm_base_ops_per_cycle"]["value"] = 0.3
        self.config["parameters"]["vxm_fp32_basic_op_cost"]["value"] = 0.1
        spec = TSPSpec(self.config)
        self.assertEqual(spec.vxm_service_cycles({"fp32_add": 1, "fp32_mul": 2}), 1)
        self.assertEqual(spec.vxm_service_cycles({"fp32_add": 4}), 2)

    def test_only_active_operations_require_parameters(self):
        for name in VXM_SERVICE_PARAMETERS:
            self.config["parameters"][name].update(value=None, status="unknown")
        spec = TSPSpec(self.config)
        self.assertEqual(spec.vxm_service_cycles({}), 0)
        self.assertEqual(spec.vxm_service_cycles({"fp32_exp": 0}), 0)
        with self.assertRaisesRegex(ValueError, "vxm_base_ops_per_cycle.*vxm_fp32_exp_cost"):
            spec.vxm_service_cycles({"fp32_exp": 1})
        self.config["parameters"]["vxm_base_ops_per_cycle"].update(value=320, status="assumed")
        self.config["parameters"]["vxm_int32_add_cost"].update(value=1, status="assumed")
        spec = TSPSpec(self.config)
        self.assertEqual(spec.vxm_service_cycles({"int32_add": 320, "fp32_exp": 0}), 1)
        self.assertEqual(TSPSimulator(spec).simulate(2, 321, 641, mode="timing")["timing"]["total_cycles"], 221)
        with self.assertRaisesRegex(ValueError, "vxm_fp32_rsqrt_cost"):
            spec.vxm_service_cycles({"fp32_rsqrt": 1})

    def test_invalid_workloads_and_parameters_fail_explicitly(self):
        for workload in (None, [], {"fp32_exp": -1}, {"fp32_exp": True}, {"fp32_exp": 1.0},
                         {"fp32_exp": float("nan")}, {"fp32_div": 1}, {"softmax": 0}, {1: 2}):
            with self.subTest(workload=workload), self.assertRaises(ValueError):
                self.spec.vxm_service_cycles(workload)
        for name in VXM_SERVICE_PARAMETERS:
            for value in (0, -1, True, float("nan"), float("inf")):
                config = copy.deepcopy(self.config)
                config["parameters"][name]["value"] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TSPSpec(config)
        for field, value in (("unit", "cycle"), ("scope", "chip"), ("status", "unknown")):
            config = copy.deepcopy(self.config)
            config["parameters"]["vxm_fp32_exp_cost"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                TSPSpec(config)
        for name, value in (("vxm_base_ops_per_cycle", 10000), ("vxm_int32_add_cost", 0.01)):
            config = copy.deepcopy(self.config)
            config["parameters"][name]["value"] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "budget"):
                TSPSpec(config)

    def test_report_discloses_algorithmic_scope_and_no_inventory_fallback(self):
        service = self.spec.derived()["vxm_service"]
        self.assertEqual(service["resource_policy"], "one_shared_vxm_service")
        self.assertEqual(service["operation_costs"]["fp32_exp"], 15)
        self.assertAlmostEqual(service["operation_rates_per_cycle"]["fp32_exp"], 2560/15)
        self.assertEqual(service["chip_peak_ops_per_cycle"], 5120)
        self.assertEqual(service["base_ops_per_cycle"], 2560)
        self.assertEqual(service["allocated_peak_fraction"], 0.5)
        self.assertEqual(self.spec.n_hemispheres * service["base_ops_per_cycle"], service["chip_peak_ops_per_cycle"])
        self.assertEqual(service["gemm_operations"], ["int32_add"])
        self.assertFalse(service["nonlinear_included_in_gemm"])
        self.assertEqual(service["missing_parameters"], [])
        self.assertEqual(set(service["assumed_inputs"]), set(VXM_SERVICE_PARAMETERS))
        self.assertIn("not_native_instruction", service["special_function_model"])
        service["operation_costs"]["fp32_exp"] = 999
        self.assertEqual(self.spec.vxm_service_cycles({"fp32_exp": 2560}), 15)
        self.config["parameters"]["vxm_alus_per_lane"]["value"] = 32
        self.config["parameters"]["hemispheres"]["value"] = 1
        self.config["parameters"]["usable_mem_bytes"]["value"] = 110*2**20
        changed = TSPSpec(self.config).vxm_service()
        self.assertEqual(changed["operation_rates_per_cycle"], self.spec.vxm_service()["operation_rates_per_cycle"])
        self.assertEqual(changed["base_ops_per_cycle"], 2560)
        self.assertEqual(changed["chip_peak_ops_per_cycle"], 10240)
        self.assertEqual(changed["allocated_peak_fraction"], 0.25)

    def test_add_cost_controls_native_and_cim_without_changing_actions(self):
        loader = TSPCIMConfigLoader()
        for kind in ("native", "cim"):
            reports = []
            for rate, cost in ((320, 1), (320, 4), (80, 1)):
                config = copy.deepcopy(self.config)
                config["parameters"]["vxm_base_ops_per_cycle"]["value"] = rate
                config["parameters"]["vxm_int32_add_cost"]["value"] = cost
                if kind == "native":
                    sim = TSPSimulator(TSPSpec(config))
                else:
                    combined = loader.load_hardware_config()
                    combined["_native_reference_config"] = config
                    sim = TSPCIMSimulator(loader.create_tsp_spec(combined))
                reports.append(sim.simulate(3, 321, 640, mode="timing"))
            with self.subTest(kind=kind):
                base, costly, slower = reports
                self.assertEqual(base["counts"], costly["counts"])
                self.assertEqual(base["action_counts"], costly["action_counts"])
                self.assertGreater(costly["timing"]["total_cycles"], base["timing"]["total_cycles"])
                self.assertEqual(costly["timing"]["total_cycles"], slower["timing"]["total_cycles"])
                self.assertIn("vxm_int32_add_cost", costly["timing"]["parameters"])
                self.assertNotIn("vxm_fp32_exp_cost", costly["timing"]["parameters"])

    def test_missing_add_cost_is_required_only_for_cross_k(self):
        self.config["parameters"]["vxm_int32_add_cost"].update(value=None, status="unknown")
        sim = TSPSimulator(TSPSpec(self.config))
        self.assertEqual(sim.simulate(1, 16, 16, mode="timing")["timing"]["total_cycles"], 44)
        self.assertIn("vxm_int32_add_cost", sim.simulate(1, 321, 16)["timing"]["missing_parameters"])
        with self.assertRaisesRegex(ValueError, "vxm_int32_add_cost"):
            sim.simulate(1, 321, 16, mode="timing")


if __name__ == "__main__":
    unittest.main()
