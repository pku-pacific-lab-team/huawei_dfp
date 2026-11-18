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
        self.assertEqual(d["nominal_chip_int8_tops"], 275)
        self.assertLess(abs(d["chip_int8_peak_tops"] - d["nominal_chip_int8_tops"]), 0.5)
        self.assertEqual(d["bandwidth_gbit_per_second"]["hbm_bandwidth_bytes_per_second"], 9600)
        self.assertEqual(d["bandwidth_gbit_per_second"]["ici_link_reported_bytes_per_second"], 400)
        self.assertIsNone(d["bandwidth_gbit_per_second"]["ici_link_one_way_bytes_per_second"])
        for key, expected in {"tensorcores": 2, "mxus": 8, "vpus": 2, "vpu_alus": 4096,
                              "scalar_units": 2, "sparsecores": 4, "ici_links": 6,
                              "private_vmem_bytes": 32 * 2**20, "shared_cmem_bytes": 128 * 2**20,
                              "sparse_spmem_bytes": 10 * 2**20, "external_hbm_bytes": 32 * 2**30}.items():
            self.assertEqual(inv[key], expected, key)
        self.assertEqual(inv["die_area_upper_bound_mm2"], 600)
        self.assertEqual(inv["die_area_bound_operator"], "<")
        self.assertIsNone(inv["total_area_mm2"])
        self.assertIsNone(inv["sfu_units"])
        self.assertIsNone(inv["private_register_file_bytes"])

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
        for name, expected in {"mxus": 4, "vpus": 1, "vpu_alus": 2048, "scalar_units": 1,
                               "private_vmem_bytes": 16 * 2**20}.items():
            self.assertEqual(local[name], expected, name)
        for name in ("sfu_units", "private_smem_bytes", "private_register_file_bytes"):
            self.assertIsNone(local[name], name)
        for name in ("shared_cmem_bytes", "external_hbm_bytes", "sparse_spmem_bytes", "ici_links"):
            self.assertNotIn(name, local)

    def test_chip_inventory_scales_private_resources_only(self):
        config = hardware_config()
        # Synthetic known values also exercise resources unknown in the reference.
        for name, value in (("sfu_units_per_tensorcore", 2), ("smem_physical_bytes", 1024),
                            ("register_file_physical_bytes", 2048)):
            config["parameters"][name].update(value=value, status="assumed")
        base = TPUSpec(config)
        local, chip = base.tensorcore_inventory(), base.inventory()
        for cores in (1, 4):
            config["parameters"]["tensorcores_per_chip"].update(value=cores, status="assumed")
            changed = TPUSpec(config)
            inventory = changed.inventory()
            with self.subTest(cores=cores):
                self.assertEqual(changed.tensorcore_inventory(), local)
                for name in ("mxus", "vpus", "vpu_alus", "scalar_units", "sfu_units", "private_vmem_bytes",
                             "private_smem_bytes", "private_register_file_bytes"):
                    self.assertEqual(inventory[name], cores * local[name], name)
                for name in ("shared_cmem_bytes", "external_hbm_bytes", "sparse_spmem_bytes", "sparsecores",
                             "ici_links", "die_area_upper_bound_mm2", "total_area_mm2"):
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
                         {"int8_macs_per_cell_cycle", "vmem_usable_bytes", "hbm_usable_bytes"})
        self.assertIn("vector_fp32_ops_per_cycle", s.missing_parameters())

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
        for mutation in (lambda c: c.update(schema_version=True), lambda c: c.update(model="tpu_v4i"),
                         lambda c: c["parameters"].pop("vmem_bandwidth_bytes_per_second"),
                         lambda c: c["parameters"].update(unrecognized={}),
                         lambda c: c["interconnect"].update(topology="")):
            config = hardware_config()
            mutation(config)
            with self.assertRaises(ValueError):
                TPUSpec(config)

    def test_usable_memory_cannot_exceed_physical(self):
        for memory in ("vmem", "cmem", "hbm"):
            config = hardware_config()
            p = config["parameters"]
            p[f"{memory}_usable_bytes"].update(value=p[f"{memory}_physical_bytes"]["value"] + 1, status="assumed")
            with self.subTest(memory=memory), self.assertRaisesRegex(ValueError, "usable capacity"):
                TPUSpec(config)

    def test_unknown_inputs_propagate_and_do_not_block_independent_counts(self):
        config = hardware_config()
        for name in ("clock_hz", "mxus_per_tensorcore", "spmem_physical_bytes"):
            config["parameters"][name].update(value=None, status="unknown")
        s = TPUSpec(config)
        self.assertIsNone(s.derived()["chip_int8_peak_tops"])
        self.assertIsNone(s.inventory()["mxus"])
        self.assertIsNone(s.inventory()["sparse_spmem_bytes"])
        self.assertEqual(TPUSimulator(s).simulate(workload())["counts"]["useful_ops"], 16384)

    def test_missing_required_capacity_is_not_defaulted(self):
        config = hardware_config()
        config["parameters"]["vmem_usable_bytes"].update(value=None, status="unknown")
        with self.assertRaisesRegex(ValueError, "Counting parameters missing: vmem_usable_bytes"):
            TPUSimulator(TPUSpec(config)).simulate(workload())
