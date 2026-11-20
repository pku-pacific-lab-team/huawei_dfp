import copy
import unittest

from src.hardware.ascend import AscendSpec
from tests.ascend.support import hardware_config, spec


class ConfigTest(unittest.TestCase):
    def test_every_parameter_has_explicit_scope_and_version(self):
        for name, p in spec(False).provenance().items():
            with self.subTest(name=name):
                self.assertEqual(p["scope"], "chip" if name.startswith("chip.") else "core")
                self.assertTrue(p["source"])
                self.assertTrue(p["version"])
                self.assertEqual(p["value"] is None, p["status"] == "unknown")

    def test_units_scopes_status_and_numeric_values_are_checked(self):
        for key, value in (("unit", "GB/s"), ("scope", "chip"), ("status", "verified"),
                           ("value", True), ("value", -1), ("value", float("nan")),
                           ("source", ""), ("version", "")):
            config = hardware_config()
            config["transfers"]["l1_l0a"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                AscendSpec(config)

    def test_missing_parameters_do_not_get_silent_defaults(self):
        config = hardware_config()
        del config["transfers"]["l1_l0a"]
        with self.assertRaises(ValueError):
            AscendSpec(config)
        config = hardware_config(False)
        config["cube"]["macs_per_cycle"].update(value=None, status="unknown")
        reference = AscendSpec(config)
        self.assertEqual(reference.missing_timing(), ["cube.macs_per_cycle"])
        with self.assertRaisesRegex(ValueError, "cube.macs_per_cycle"):
            reference.require_timing()
        self.assertIsNone(reference.derived()["chip_int8_peak_tops"])

    def test_capacity_and_fractal_constraints(self):
        config = hardware_config()
        for mutate in (
            lambda c: c["memory"]["L0A"]["usable_bytes"].update(value=65537),
            lambda c: c["cube"]["k"].update(value=16),
            lambda c: c["chip"]["hbm_usable_bytes"].update(value=32000000001),
            lambda c: c["chip"]["hbm_physical_bytes"].update(value=None, status="unknown"),
            lambda c: c["chip"]["architectural_cores"].update(value=32.0),
        ):
            variant = copy.deepcopy(config)
            mutate(variant)
            with self.assertRaises(ValueError):
                AscendSpec(variant)

    def test_derived_compute_bandwidth_and_area_inventory(self):
        config = hardware_config()
        config["transfers"]["l1_l0a"]["value"] = 16
        config["memory"]["L1"]["usable_bytes"]["value"] = 524288
        derived = AscendSpec(config).derived()
        self.assertEqual(derived["int8_fractal_macs"], 8192)
        self.assertEqual(derived["core_int8_ops_per_cycle"], 16384)
        self.assertEqual(derived["core_int8_peak_tops"], 16.384)
        self.assertEqual(derived["chip_int8_peak_tops"], 524.288)
        self.assertEqual(derived["route_bandwidth_gbps"]["l1_l0a"], 128)
        inventory = derived["area_inventory"]
        self.assertNotIn("compute_units", inventory)
        self.assertNotIn("vector_fp32_peak_gops_per_core", derived)
        self.assertNotIn("sfu_fp32_peak_gops_per_core", derived)
        self.assertEqual(inventory["core_sram_bytes_by_level"]["L1"], 32 * 1048576)
        self.assertEqual(inventory["external_hbm_bytes"], 32000000000)
        self.assertIsNone(inventory["area_mm2"])

    def test_frequency_is_required_for_hbm_cycle_conversion(self):
        config = hardware_config()
        config["clock_freq_ghz"].update(value=None, status="unknown")
        hardware = AscendSpec(config)
        with self.assertRaisesRegex(ValueError, "clock_freq_ghz"):
            hardware.require_timing()
        self.assertIsNone(hardware.derived()["core_int8_peak_tops"])
        self.assertIsNone(hardware.derived()["route_bandwidth_gbps"]["l1_l0a"])

    def test_old_event_schema_is_rejected(self):
        for version in (2, 3, 4, 5, 6, 7):
            config = hardware_config()
            config["schema_version"] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "Expected schema 9"):
                AscendSpec(config)

    def test_retired_fields_are_rejected_in_current_schema(self):
        for path in ("sfu", "vector.fp32_ops_per_cycle", "chip.noc_bytes_per_cycle",
                     "chip.interconnect_bytes_per_second", "chip.hbm_bytes", "chip.hbm_bytes_per_cycle",
                     "transfers.gm_l1", "transfers.ub_gm"):
            config = hardware_config()
            parent = config
            keys = path.split(".")
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = {}
            with self.subTest(path=path), self.assertRaises(ValueError):
                AscendSpec(config)
