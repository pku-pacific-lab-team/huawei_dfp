"""Guard the link between reviewed sources and the current reference profiles."""
import copy
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from src.hardware.ascend import AscendSpec
from src.hardware.tsp import TSPSpec


ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def parameters(data, prefix=""):
    found = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if {"value", "unit", "scope", "status"} <= value.keys():
                found[name] = value
            else:
                found.update(parameters(value, name))
    return found


class ParameterSourceAuditTest(unittest.TestCase):
    def test_audit_covers_current_config_and_resolvable_evidence(self):
        audit = read("calibration/parameter_source_audit.json")
        expected = {
            "configs/hardware/ascend910_reference.json",
            "configs/hardware/tpu_v4_reference.json",
            "configs/hardware/tsp.json",
            "configs/hardware/cim_macro.json",
            "configs/hardware/tsp_cim.json",
        }
        self.assertEqual(set(audit["configs"]), expected)
        count = 0
        statuses = Counter()
        for path, reviewed in audit["configs"].items():
            with self.subTest(config=path):
                config = read(path)
                digest = hashlib.sha256(json.dumps(
                    config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest()
                self.assertEqual(digest, reviewed["canonical_json_sha256"],
                                 "Reference changed: review and refresh the source audit.")
                actual = parameters(config)
                self.assertEqual(actual.keys(), reviewed["parameters"].keys())
                self.assertEqual(len(actual), reviewed["parameter_count"])
                for name, p in actual.items():
                    disposition = reviewed["parameters"][name]
                    self.assertEqual(p["status"], disposition["status"])
                    self.assertTrue(disposition["decision"])
                    self.assertTrue(disposition["evidence"])
                    for evidence in disposition["evidence"]:
                        self.assertIn(evidence["source_id"], audit["sources"])
                        self.assertTrue(evidence.get("locator") or evidence.get("section"))
                    self.assertNotIn("primary_document_pending", p.get("verification", ""))
                    statuses[p["status"]] += 1
                count += len(actual)
        self.assertEqual(count, audit["total_parameters_reviewed"])
        self.assertEqual(dict(statuses), audit["status_counts"])

    def test_tsp_stream_coordinates_and_cross_section_are_distinct(self):
        config = read("configs/hardware/tsp.json")
        spec = TSPSpec(config)
        self.assertEqual(spec.validate_sr_coordinate(44), 44)
        with self.assertRaises(ValueError):
            spec.validate_sr_coordinate(45)
        self.assertEqual(spec.values["stream_width_bytes"] // spec.n_superlanes, 16)
        self.assertEqual(spec.stream_bw_per_direction_bytes_per_cycle, 32 * 20 * 16)
        for name, old_scope in (("sr_coordinates", "chip"),
                                ("stream_width_bytes", "stream"),
                                ("streams_per_direction", "direction")):
            invalid = copy.deepcopy(config)
            invalid["parameters"][name]["scope"] = old_scope
            with self.subTest(parameter=name), self.assertRaises(ValueError):
                TSPSpec(invalid)

    def test_nominal_chip_hbm_does_not_enable_core_timing(self):
        config = read("configs/hardware/ascend910_reference.json")
        spec = AscendSpec(config)
        hbm = config["chip"]["hbm_bytes_per_cycle"]
        self.assertEqual(hbm["value"] * config["clock_freq_ghz"]["value"] * 1e9, 1.2e12)
        self.assertEqual(hbm["scope"], "chip")
        self.assertEqual(len(spec.missing_timing()), 6)
        self.assertTrue(all(p["value"] is None for p in config["transfers"].values()))
        with self.assertRaisesRegex(ValueError, "Timing parameters missing"):
            spec.require_timing()

