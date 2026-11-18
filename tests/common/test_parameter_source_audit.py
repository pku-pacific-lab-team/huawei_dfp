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
                self.assertEqual(dict(Counter(p["status"] for p in actual.values())), reviewed["status_counts"])
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

    def test_retired_ascend_parameters_keep_evidence_outside_active_schema(self):
        audit = read("calibration/parameter_source_audit.json")
        profile = audit["configs"]["configs/hardware/ascend910_reference.json"]
        retired = profile["retired_parameters"]
        self.assertEqual(set(retired), {"cube.units_per_core", "vector.units_per_core", "vector.fp32_ops_per_cycle", "sfu.units_per_core",
                                       "sfu.fp32_ops_per_cycle", "chip.noc_bytes_per_cycle",
                                       "chip.interconnect_bytes_per_second", "transfers.gm_l1", "transfers.ub_gm"})
        self.assertTrue(set(retired).isdisjoint(profile["parameters"]))
        for name, entry in retired.items():
            with self.subTest(parameter=name):
                self.assertTrue(entry["reason"])
                expected = "documented" if entry.get("archive_path") else "assumed" if name.startswith("transfers.") else "unknown"
                self.assertEqual(entry["last_configuration"]["status"], expected)
                self.assertEqual(entry["last_configuration"]["value"] is None, expected == "unknown")
                for evidence in entry["prior_audit"]["evidence"]:
                    self.assertIn(evidence["source_id"], audit["sources"])

    def test_retired_tsp_parameters_keep_evidence_outside_active_schema(self):
        audit = read("calibration/parameter_source_audit.json")
        profile = audit["configs"]["configs/hardware/tsp.json"]
        retired = profile["retired_parameters"]
        self.assertEqual(set(retired), {"parameters." + name for name in (
            "spare_superlanes", "physical_sram_bytes", "local_sram_bytes",
            "offchip_capacity_bytes", "offchip_bandwidth_bpc", "interchip_bandwidth_bpc",
            "vxm_add_ops_per_cycle", "nonlinear_ops_per_cycle", "mem_shared_bpc", "nominal_int8_tops")})
        self.assertTrue(set(retired).isdisjoint(profile["parameters"]))
        for name, entry in retired.items():
            with self.subTest(parameter=name):
                self.assertTrue(entry["reason"])
                expected = "documented" if entry.get("archive_path") else "assumed" if name in {"parameters.vxm_add_ops_per_cycle", "parameters.mem_shared_bpc"} else "unknown"
                self.assertEqual(entry["last_configuration"]["status"], expected)
                self.assertEqual(entry["last_configuration"]["value"] is None, expected == "unknown")
                self.assertTrue(entry["prior_audit"]["evidence"])
                for evidence in entry["prior_audit"]["evidence"]:
                    self.assertIn(evidence["source_id"], audit["sources"])

    def test_retired_tpu_parameters_keep_evidence_outside_active_schema(self):
        audit = read("calibration/parameter_source_audit.json")
        profile = audit["configs"]["configs/hardware/tpu_v4_reference.json"]
        retired = profile["retired_parameters"]
        self.assertEqual(set(retired), {"parameters.nominal_int8_tops", "parameters.vpus_per_tensorcore", "parameters.vpu_lanes",
                                       "parameters.vpu_alus_per_lane", "parameters.register_file_chip_bytes", "parameters.ici_links_per_chip",
                                       "parameters.ici_link_reported_bytes_per_second", "parameters.process_node_nm",
                                       "parameters.die_area_upper_bound_mm2", "parameters.scalar_units_per_tensorcore", "parameters.scalar_ops_per_cycle", "parameters.sfu_units_per_tensorcore",
                                       "parameters.sfu_fp32_ops_per_cycle", "parameters.ici_link_one_way_bytes_per_second",
                                       "parameters.vector_int32_adds_per_cycle", "parameters.vector_fp32_ops_per_cycle",
                                       "parameters.smem_physical_bytes", "parameters.register_file_physical_bytes",
                                       "parameters.cmem_bandwidth_bytes_per_second",
                                       "parameters.sparsecores_per_chip", "parameters.spmem_physical_bytes"})
        self.assertTrue(set(retired).isdisjoint(profile["parameters"]))
        documented = {"parameters.nominal_int8_tops": 275, "parameters.vpus_per_tensorcore": 1,
                      "parameters.vpu_lanes": 128, "parameters.vpu_alus_per_lane": 16,
                      "parameters.register_file_chip_bytes": 262144, "parameters.ici_links_per_chip": 6,
                      "parameters.ici_link_reported_bytes_per_second": 50000000000, "parameters.process_node_nm": 7,
                      "parameters.die_area_upper_bound_mm2": 600, "parameters.scalar_units_per_tensorcore": 1, "parameters.sparsecores_per_chip": 4, "parameters.spmem_physical_bytes": 10 * 2**20}
        for name, entry in retired.items():
            with self.subTest(parameter=name):
                self.assertTrue(entry["reason"])
                self.assertEqual(entry["last_configuration"]["status"], "documented" if name in documented else "unknown")
                self.assertEqual(entry["last_configuration"]["value"], documented.get(name))
                for evidence in entry["prior_audit"]["evidence"]:
                    self.assertIn(evidence["source_id"], audit["sources"])

        archive_path = "configs/archive/tpu_v4_reference.json"
        archive = read(archive_path)
        expected_archive = {"register_file_chip_bytes", "ici_links_per_chip",
                            "ici_link_reported_bytes_per_second", "process_node_nm", "die_area_upper_bound_mm2",
                            "nominal_int8_tops", "vpus_per_tensorcore", "vpu_lanes", "vpu_alus_per_lane"}
        self.assertEqual(set(archive["parameters"]), expected_archive)
        for name, parameter in archive["parameters"].items():
            self.assertEqual(parameter, retired["parameters." + name]["last_configuration"])
            self.assertEqual(retired["parameters." + name]["archive_path"], archive_path)
        digest = hashlib.sha256(json.dumps(archive, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(digest, audit["archives"][archive_path]["canonical_json_sha256"])
        self.assertFalse(audit["archives"][archive_path]["runtime_dependency"])
        self.assertIn("interconnect", archive)
        self.assertNotIn("interconnect", read("configs/hardware/tpu_v4_reference.json"))

    def test_retired_cim_parameters_keep_evidence_outside_active_schema(self):
        audit = read("calibration/parameter_source_audit.json")
        for filename, names in {
            "cim_macro": ("compute_first_cycles", "write_first_cycles",
                          "write_wordline_ii_cycles", "context_switch_cycles"),
            "tsp_cim": ("physical_sram_bytes", "additional_local_sram_bytes", "collector_buffer_bytes",
                        "nim_units_per_plane", "nim_tiles_per_plane", "nim_channels_per_tile", "nim_conversion_lanes"),
        }.items():
            profile = audit["configs"][f"configs/hardware/{filename}.json"]
            retired = profile["retired_parameters"]
            self.assertEqual(set(retired), {"parameters." + name for name in names})
            self.assertTrue(set(retired).isdisjoint(profile["parameters"]))
            for name, entry in retired.items():
                with self.subTest(profile=filename, parameter=name):
                    self.assertTrue(entry["reason"])
                    expected = "documented" if entry.get("archive_path") else "unknown"
                    self.assertEqual(entry["last_configuration"]["status"], expected)
                    self.assertEqual(entry["last_configuration"]["value"] is None, expected == "unknown")
                    self.assertTrue(entry["prior_audit"]["evidence"])
                    for evidence in entry["prior_audit"]["evidence"]:
                        self.assertIn(evidence["source_id"], audit["sources"])

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

    def test_hbm_bandwidth_does_not_replace_required_local_services(self):
        config = read("configs/hardware/ascend910_reference.json")
        spec = AscendSpec(config)
        hbm = config["chip"]["hbm_bandwidth_bytes_per_second"]
        self.assertEqual(hbm["value"], 1.2e12)
        self.assertEqual(hbm["scope"], "chip")
        self.assertEqual(spec.missing_timing(), [])
        self.assertTrue(all(p["status"] == "assumed" for p in config["transfers"].values()))
        spec.require_timing()
        config["transfers"]["l1_l0a"].update(value=None, status="unknown")
        with self.assertRaisesRegex(ValueError, "Timing parameters missing"):
            AscendSpec(config).require_timing()

