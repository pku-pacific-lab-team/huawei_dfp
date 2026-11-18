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
            config["transfers"]["gm_l1"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                AscendSpec(config)

    def test_missing_parameters_do_not_get_silent_defaults(self):
        config = hardware_config()
        del config["transfers"]["gm_l1"]
        with self.assertRaises(ValueError):
            AscendSpec(config)
        reference = spec(False)
        self.assertEqual(len(reference.missing_timing()), 6)
        with self.assertRaisesRegex(ValueError, "cube.macs_per_cycle"):
            reference.require_timing()
        self.assertIsNone(reference.derived()["chip_int8_peak_tops"])

    def test_capacity_and_fractal_constraints(self):
        config = hardware_config()
        for mutate in (
            lambda c: c["memory"]["L0A"]["usable_bytes"].update(value=65537),
            lambda c: c["cube"]["k"].update(value=16),
            lambda c: c["chip"]["architectural_cores"].update(value=32.0),
        ):
            variant = copy.deepcopy(config)
            mutate(variant)
            with self.assertRaises(ValueError):
                AscendSpec(variant)

    def test_derived_compute_bandwidth_and_area_inventory(self):
        config = hardware_config()
        config["transfers"]["gm_l1"]["value"] = 16
        config["memory"]["L1"]["usable_bytes"]["value"] = 524288
        derived = AscendSpec(config).derived()
        self.assertEqual(derived["int8_fractal_macs"], 8192)
        self.assertEqual(derived["core_int8_ops_per_cycle"], 16384)
        self.assertEqual(derived["core_int8_peak_tops"], 16.384)
        self.assertEqual(derived["chip_int8_peak_tops"], 524.288)
        self.assertEqual(derived["route_bandwidth_gbps"]["gm_l1"], 128)
        inventory = derived["area_inventory"]
        self.assertEqual(inventory["compute_units"]["cube"], 32)
        self.assertEqual(inventory["compute_units"]["vector"], 32)
        self.assertIsNone(inventory["compute_units"]["sfu"])
        self.assertEqual(inventory["core_sram_bytes_by_level"]["L1"], 32 * 1048576)
        self.assertIsNone(inventory["external_hbm_bytes"])
        self.assertIsNone(inventory["area_mm2"])

    def test_frequency_is_required_only_for_time_and_peak_conversion(self):
        config = hardware_config()
        config["clock_freq_ghz"].update(value=None, status="unknown")
        hardware = AscendSpec(config)
        hardware.require_timing()
        self.assertIsNone(hardware.derived()["core_int8_peak_tops"])
        self.assertIsNone(hardware.derived()["route_bandwidth_gbps"]["gm_l1"])

    def test_old_event_schema_is_rejected(self):
        config = hardware_config()
        config["schema_version"] = 2
        with self.assertRaises(ValueError):
            AscendSpec(config)
