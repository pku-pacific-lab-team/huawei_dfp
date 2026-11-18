import unittest

from src.core.tpu import TPUSimulator
from src.hardware.tpu import PARAMETERS, TPUSpec
from tests.tpu.support import hardware_config, spec, workload


class TPUConfigTest(unittest.TestCase):
    def test_peak_units_and_physical_inventory(self):
        s = spec()
        d, inv = s.derived(), s.inventory()
        self.assertEqual(d["core_int8_macs_per_cycle"], 65536)
        self.assertAlmostEqual(d["core_int8_peak_tops"], 137.6256)
        self.assertAlmostEqual(d["chip_int8_peak_tops"], 275.2512)
        self.assertNotIn("nominal_chip_int8_tops", d)
        self.assertEqual(d["bandwidth_gbit_per_second"]["hbm_bandwidth_bytes_per_second"], 9600)
        self.assertNotIn("ici_link_one_way_bytes_per_second", d["bandwidth_gbit_per_second"])
        for key, expected in {"tensorcores": 2, "mxus": 8,
                              "private_vmem_bytes": 32 * 2**20, "shared_cmem_bytes": 128 * 2**20,
                              "external_hbm_bytes": 32 * 2**30}.items():
            self.assertEqual(inv[key], expected, key)
        self.assertIsNone(inv["total_area_mm2"])
        self.assertNotIn("sfu_units", inv)
        self.assertNotIn("scalar_units", inv)
        for field in ("register_file_bytes", "ici_links", "die_area_upper_bound_mm2", "die_area_bound_operator"):
            self.assertNotIn(field, inv)
        self.assertNotIn("scalar_units_per_tensorcore", s.provenance())
        self.assertNotIn("private_register_file_bytes", inv)
        self.assertNotIn("private_smem_bytes", inv)
        self.assertNotIn("sparsecores", inv)
        self.assertNotIn("sparse_spmem_bytes", inv)

    def test_hierarchy_peaks_and_local_inventory(self):
        s = spec()
        h, local, chip = s.hierarchy(), s.tensorcore_inventory(), s.inventory()
        self.assertEqual((h["mxu"]["rows"], h["mxu"]["columns"], h["mxu"]["cells"]), (128, 128, 16384))
        self.assertEqual(h["tensorcore"]["mxus_per_tensorcore"], 4)
        self.assertEqual(h["chip"]["tensorcores_per_chip"], 2)
        self.assertEqual(h["chip"]["mxus_per_chip"], chip["mxus"])
        for level, macs, tops in (("mxu", 16384, 34.4064), ("tensorcore", 65536, 137.6256),
                                  ("chip", 131072, 275.2512)):
            with self.subTest(level=level):
                self.assertEqual(h[level]["scope"], level)
                self.assertEqual(h[level]["int8_macs_per_cycle"], macs)
                self.assertAlmostEqual(h[level]["int8_peak_tops"], tops)
        self.assertEqual(h["peak_assumed_inputs"], ["int8_macs_per_cell_cycle"])
        self.assertIn("assumed_inputs_remain_assumed", h["peak_status"])
        self.assertEqual(local["scope"], "tensorcore")
        for name, expected in {"mxus": 4,
                               "private_vmem_bytes": 16 * 2**20}.items():
            self.assertEqual(local[name], expected, name)
        self.assertNotIn("sfu_units", local)
        self.assertNotIn("scalar_units", local)
        for name in ("private_smem_bytes", "private_register_file_bytes"):
            self.assertNotIn(name, local)
        for name in ("shared_cmem_bytes", "external_hbm_bytes", "ici_links", "register_file_bytes"):
            self.assertNotIn(name, local)

    def test_chip_inventory_scales_private_resources_only(self):
        config = hardware_config()
        base = TPUSpec(config)
        local, chip = base.tensorcore_inventory(), base.inventory()
        for cores in (1, 4):
            config["parameters"]["tensorcores_per_chip"].update(value=cores, status="assumed")
            changed = TPUSpec(config)
            inventory = changed.inventory()
            with self.subTest(cores=cores):
                self.assertEqual(changed.tensorcore_inventory(), local)
                for name in ("mxus", "private_vmem_bytes"):
                    self.assertEqual(inventory[name], cores * local[name], name)
                for name in ("shared_cmem_bytes", "external_hbm_bytes",
                             "total_area_mm2"):
                    self.assertEqual(inventory[name], chip[name], name)
                self.assertAlmostEqual(changed.hierarchy()["chip"]["int8_peak_tops"], cores * 137.6256)

    def test_hierarchy_unknowns_preserve_independent_levels(self):
        for parameter, unknown_levels in (("tensorcores_per_chip", ("chip",)),
                                           ("mxus_per_tensorcore", ("tensorcore", "chip")),
                                           ("mxu_rows", ("mxu", "tensorcore", "chip")),
                                           ("clock_hz", ("mxu", "tensorcore", "chip"))):
            config = hardware_config()
            config["parameters"][parameter].update(value=None, status="unknown")
            s = TPUSpec(config)
            h = s.hierarchy()
            with self.subTest(parameter=parameter):
                for level in ("mxu", "tensorcore", "chip"):
                    self.assertEqual(h[level]["int8_peak_tops"] is None, level in unknown_levels)
                    self.assertEqual(h[level]["int8_macs_per_cycle"] is None,
                                     parameter != "clock_hz" and level in unknown_levels)
                self.assertEqual(s.tensorcore_inventory()["private_vmem_bytes"], 16 * 2**20)
                self.assertEqual(s.inventory()["shared_cmem_bytes"], 128 * 2**20)
                if parameter == "tensorcores_per_chip":
                    self.assertIsNone(s.inventory()["private_vmem_bytes"])
                    self.assertIsNone(s.inventory()["mxus"])
                elif parameter != "clock_hz":
                    self.assertEqual(TPUSimulator(s).simulate(workload())["counts"]["useful_ops"], 16384)

    def test_every_parameter_has_provenance_and_assumptions_remain_visible(self):
        s = spec()
        self.assertEqual(set(s.provenance()), set(PARAMETERS))
        for name, p in s.provenance().items():
            self.assertEqual(set(p), {"value", "unit", "scope", "status", "source", "version", "conditions"})
            self.assertTrue(p["source"] and p["version"] and p["conditions"], name)
            self.assertEqual(p["value"] is None, p["status"] == "unknown", name)
        self.assertEqual(set(s.derived()["assumed_inputs"]),
                         {"int8_macs_per_cell_cycle", "vmem_usable_bytes", "hbm_usable_bytes", "vmem_bandwidth_bytes_per_second",
                          "vector_base_ops_per_cycle", "vector_int32_add_cost", "vector_fp32_basic_op_cost", "vector_fp32_exp_cost", "vector_fp32_rsqrt_cost", "cmem_usable_bytes",
                          "cmem_read_bandwidth_bytes_per_second", "cmem_write_bandwidth_bytes_per_second"})
        self.assertEqual(s.missing_parameters(), [])

    def test_cmem_proxy_scope_provenance_and_execution_guard(self):
        config = hardware_config()
        s = TPUSpec(config)
        self.assertEqual(len(s.parameters), 21)
        self.assertEqual(s.execution["cmem_mode"], "bypass")
        for direction, value in (("read", 2150704873472), ("write", 1075352436736)):
            name = f"cmem_{direction}_bandwidth_bytes_per_second"
            p = s.provenance()[name]
            self.assertEqual((p["value"], p["unit"], p["scope"], p["status"]),
                             (value, "byte/s", "chip", "assumed"))
            self.assertIn("TPUv4i", p["conditions"])
            self.assertIn("not validated TPU v4", p["conditions"])
            self.assertIn("libtpu 0.0.40", p["conditions"])
            self.assertIn("does not distinguish read and write", p["conditions"])
            self.assertEqual(s.derived()["bandwidth_gbit_per_second"][name], value * 8 / 1e9)
        for mode in ("count", "timing"):
            report = TPUSimulator(s).simulate(workload(), mode=mode)
            self.assertEqual(report["execution_contract"]["cmem_mode"], "bypass")
            self.assertEqual(report["missing_parameters"], [])
        config["parameters"]["tensorcores_per_chip"].update(value=4, status="assumed")
        changed = TPUSpec(config)
        for direction in ("read", "write"):
            name = f"cmem_{direction}_bandwidth_bytes_per_second"
            self.assertEqual(changed.values[name], s.values[name])
        config["execution"]["cmem_mode"] = "tiled"
        self.assertEqual(TPUSpec(config).execution["cmem_mode"], "tiled")
        for mode in ("auto", True, None, []):
            config["execution"]["cmem_mode"] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "must be bypass or tiled"):
                TPUSpec(config)
        config["execution"].pop("cmem_mode")
        with self.assertRaisesRegex(ValueError, "execution policy fields required"):
            TPUSpec(config)

    def test_bad_parameter_values_and_metadata_are_rejected(self):
        for field, value in (("value", True), ("value", -1), ("value", 0), ("value", float("nan")),
                             ("value", float("inf")), ("value", 2.0), ("value", None),
                             ("unit", "byte"), ("scope", "tensorcore"), ("status", "unknown"),
                             ("source", ""), ("version", None), ("conditions", " ")):
            config = hardware_config()
            config["parameters"]["tensorcores_per_chip"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                TPUSpec(config)

    def test_wrong_version_missing_and_unexpected_fields_fail(self):
        for mutation in (lambda c: c.update(schema_version=True), lambda c: c.update(schema_version=1),
                         lambda c: c.update(schema_version=2), lambda c: c.update(schema_version=3),
                         lambda c: c.update(schema_version=4), lambda c: c.update(schema_version=5),
                         lambda c: c.update(schema_version=6),
                         lambda c: c.update(schema_version=7),
                         lambda c: c.update(schema_version=8),
                         lambda c: c.update(schema_version=9),
                         lambda c: c.update(schema_version=10),
                         lambda c: c.update(schema_version=11),
                         lambda c: c.update(model="tpu_v4i"),
                         lambda c: c["parameters"]["cmem_physical_bytes"].update(scope="tensorcore"),
                         lambda c: c["parameters"].pop("vmem_bandwidth_bytes_per_second"),
                         lambda c: c["parameters"].update(unrecognized={}),
                         lambda c: c.update(interconnect={})):
            config = hardware_config()
            mutation(config)
            with self.assertRaises(ValueError):
                TPUSpec(config)

    def test_retired_fields_rejected_and_reserved_vector_services_do_not_execute(self):
        for name in ("register_file_chip_bytes", "ici_links_per_chip", "ici_link_reported_bytes_per_second",
                     "process_node_nm", "die_area_upper_bound_mm2",
                     "scalar_units_per_tensorcore", "scalar_ops_per_cycle", "sfu_units_per_tensorcore",
                     "sfu_fp32_ops_per_cycle", "ici_link_one_way_bytes_per_second",
                     "vector_int32_adds_per_cycle", "vector_fp32_ops_per_cycle",
                     "smem_physical_bytes", "register_file_physical_bytes", "cmem_bandwidth_bytes_per_second",
                     "sparsecores_per_chip", "spmem_physical_bytes"):
            config = hardware_config()
            config["parameters"][name] = {}
            with self.subTest(retired=name), self.assertRaisesRegex(ValueError, "parameters: expected fields"):
                TPUSpec(config)
        for mode in ("count", "timing"):
            task = workload(17, 257, 15)
            baseline = TPUSimulator(spec()).simulate(task, mode=mode)
            config = hardware_config()
            for name in ("vector_base_ops_per_cycle", "vector_int32_add_cost", "vector_fp32_basic_op_cost",
                         "cmem_usable_bytes", "cmem_read_bandwidth_bytes_per_second",
                         "cmem_write_bandwidth_bytes_per_second"):
                config["parameters"][name].update(value=1, status="assumed")
            result = TPUSimulator(TPUSpec(config)).simulate(task, mode=mode)
            for key in ("counts", "action_counts", "counting_policy", "timing"):
                with self.subTest(mode=mode, field=key):
                    self.assertEqual(result[key], baseline[key])
            for memory in ("vmem", "hbm"):
                self.assertEqual(result["memory"][memory], baseline["memory"][memory])
            self.assertEqual(result["memory"]["cmem"]["allocated_bytes"], 0)

    def test_vector_work_shares_one_service_and_rounds_once(self):
        s = spec()
        service = s.derived()["vector_service"]
        self.assertFalse(service["included_in_gemm"])
        self.assertEqual(service["resource_policy"], "one_shared_vector_service")
        self.assertEqual(service["operation_rates_per_cycle"], {"int32_add": 2048, "fp32_add": 2048, "fp32_mul": 2048,
                                                                   "fp32_exp": 2048 / 15, "fp32_rsqrt": 2048 / 12})
        # INT32 and FP32 cannot both claim the full rate in the same cycle.
        self.assertEqual(s.vector_service_cycles({"int32_add": 2048, "fp32_add": 2048}), 2)
        self.assertEqual(s.vector_service_cycles({"int32_add": 1, "fp32_add": 1, "fp32_mul": 1}), 1)
        self.assertEqual(s.vector_service_cycles({"fp32_mul": 2049}), 2)
        config = hardware_config()
        config["parameters"]["vector_fp32_basic_op_cost"]["value"] = 3
        s = TPUSpec(config)
        self.assertAlmostEqual(s.vector_service()["operation_rates_per_cycle"]["fp32_mul"], 2048 / 3)
        self.assertEqual(s.vector_service_cycles({"int32_add": 2048, "fp32_mul": 2048}), 4)
        # Exact decimal arithmetic must not introduce an extra service cycle.
        config["parameters"]["vector_base_ops_per_cycle"]["value"] = 0.3
        config["parameters"]["vector_int32_add_cost"]["value"] = 0.1
        self.assertEqual(TPUSpec(config).vector_service_cycles({"int32_add": 3}), 1)

    def test_vector_fixed_service_has_no_geometry_fallback(self):
        config = hardware_config()
        for name in ("vpus_per_tensorcore", "vpu_lanes", "vpu_alus_per_lane"):
            self.assertNotIn(name, config["parameters"])
        s = TPUSpec(config)
        self.assertNotIn("vpu_alus", s.tensorcore_inventory())
        self.assertNotIn("vpus", s.inventory())
        self.assertEqual(s.vector_service_cycles({"int32_add": 4096}), 2)
        config = hardware_config()
        config["parameters"]["vector_base_ops_per_cycle"].update(value=None, status="unknown")
        s = TPUSpec(config)
        self.assertIsNone(s.vector_service()["operation_rates_per_cycle"]["int32_add"])
        with self.assertRaisesRegex(ValueError, "Vector service parameters missing: vector_base_ops_per_cycle"):
            s.vector_service_cycles({"int32_add": 1})

    def test_vector_costs_are_required_only_for_used_operations(self):
        config = hardware_config()
        config["parameters"]["vector_fp32_basic_op_cost"].update(value=None, status="unknown")
        s = TPUSpec(config)
        self.assertEqual(s.vector_service_cycles({"int32_add": 2048, "fp32_mul": 0}), 1)
        self.assertIsNone(s.vector_service()["operation_rates_per_cycle"]["fp32_mul"])
        self.assertEqual(s.vector_service_cycles({}), 0)
        with self.assertRaisesRegex(ValueError, "vector_fp32_basic_op_cost"):
            s.vector_service_cycles({"fp32_add": 1})
        for op in ("fp32_fma", "fp32_div", "fp32_reduce_sum", "softmax"):
            with self.subTest(op=op), self.assertRaisesRegex(ValueError, "Unsupported Vector operations"):
                s.vector_service_cycles({op: 1})
        for counts in ([], {"int32_add": True}, {"int32_add": -1}, {"int32_add": 1.5}):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, "Vector operation counts"):
                s.vector_service_cycles(counts)

    def test_usable_memory_cannot_exceed_physical(self):
        for memory in ("vmem", "cmem", "hbm"):
            config = hardware_config()
            p = config["parameters"]
            p[f"{memory}_usable_bytes"].update(value=p[f"{memory}_physical_bytes"]["value"] + 1, status="assumed")
            with self.subTest(memory=memory), self.assertRaisesRegex(ValueError, "usable capacity"):
                TPUSpec(config)

    def test_unknown_inputs_propagate_and_do_not_block_independent_counts(self):
        config = hardware_config()
        for name in ("clock_hz", "mxus_per_tensorcore"):
            config["parameters"][name].update(value=None, status="unknown")
        s = TPUSpec(config)
        self.assertIsNone(s.derived()["chip_int8_peak_tops"])
        self.assertIsNone(s.inventory()["mxus"])
        self.assertEqual(TPUSimulator(s).simulate(workload())["counts"]["useful_ops"], 16384)

    def test_missing_required_capacity_is_not_defaulted(self):
        config = hardware_config()
        config["parameters"]["vmem_usable_bytes"].update(value=None, status="unknown")
        with self.assertRaisesRegex(ValueError, "Counting parameters missing: vmem_usable_bytes"):
            TPUSimulator(TPUSpec(config)).simulate(workload())
