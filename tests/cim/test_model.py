"""Hand-calculated contracts for the Macro and its CIM-TSP integration."""
import json
from pathlib import Path
import unittest

from src.core.cim.macro_model import CIMMacroModel, MacroContexts
from src.core.cim.simulator import LOOP_ORDERS, count_macro_tiles
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.hardware.macro import CIMMacroSpec
from src.utils.tsp_cim_config import TSPCIMConfigLoader

ROOT = Path(__file__).resolve().parents[2]


def profile():
    return json.loads((ROOT / "configs/hardware/cim_macro.json").read_text(encoding="utf-8"))


def timed_spec():
    config = profile()
    for name, value in {"compute_first_cycles": 11, "write_first_cycles": 2,
                        "write_wordline_ii_cycles": 1, "context_switch_cycles": 0}.items():
        config["parameters"][name].update(
            value=value, status="assumed", source="synthetic test only",
            source_version="test-v1", verification="arithmetic fixture; not calibrated")
    return CIMMacroSpec(config)


class MacroSpecificationTest(unittest.TestCase):
    def test_geometry_peak_and_resident_storage(self):
        spec = CIMMacroSpec(profile())
        self.assertEqual((spec.block_k_dim, spec.block_n_dim), (16, 32))
        self.assertEqual(spec.context_bytes, 512)
        self.assertEqual(spec.resident_weight_bytes, 2048)
        self.assertEqual(spec.peak_mac_per_cycle_int8, 64)
        self.assertEqual(spec.derived()["output_bytes_per_launch"], 128)
        self.assertEqual(spec.derived()["wordlines_per_context"], 16)

    def test_no_implicit_defaults_or_old_geometry(self):
        for config in ({}, {"n_banks_internal": 64}, {"schema_version": True}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                CIMMacroSpec.from_dict(config)
        config = profile()
        del config["parameters"]["compute_first_cycles"]
        with self.assertRaises(ValueError):
            CIMMacroSpec(config)

    def test_metadata_units_scope_and_fixed_shape_are_validated(self):
        for key, value in (("value", 32), ("value", True), ("value", 16.0),
                           ("unit", "byte"), ("scope", "chip"), ("status", "unknown"),
                           ("source", ""), ("verification", "")):
            config = profile()
            config["parameters"]["banks_per_context"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                CIMMacroSpec(config)

    def test_reference_is_stable_and_configuration_is_defensively_copied(self):
        config = profile()
        spec = CIMMacroSpec(config)
        identity = spec.reference_id
        config["parameters"]["banks_per_context"]["value"] = 99
        spec.to_dict()["parameters"]["banks_per_context"]["value"] = 99
        self.assertEqual(spec.reference_id, identity)
        self.assertEqual(spec.block_n_dim, 32)
        with self.assertRaises(TypeError):
            spec.values["weight_bits"] = 4
        changed = profile()
        changed["parameters"]["weight_bits"]["source_version"] = "different source"
        self.assertNotEqual(identity, CIMMacroSpec(changed).reference_id)

    def test_invalid_and_inconsistent_timing(self):
        for name, value in (("compute_first_cycles", 7), ("compute_first_cycles", 8.5),
                            ("write_first_cycles", 0), ("context_switch_cycles", -1),
                            ("context_switch_cycles", True), ("write_wordline_ii_cycles", float("inf"))):
            config = timed_spec().to_dict()
            config["parameters"][name]["value"] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                CIMMacroSpec(config)
        config = timed_spec().to_dict()
        config["parameters"]["write_wordline_ii_cycles"]["value"] = 3
        with self.assertRaises(ValueError):
            CIMMacroSpec(config)


class MacroActionTest(unittest.TestCase):
    def setUp(self):
        self.spec = CIMMacroSpec(profile())
        self.model = CIMMacroModel(self.spec)

    def test_full_macro_actions_by_hand(self):
        result = self.model.compute(1)
        self.assertEqual(result["counts"], {
            "useful_mac": 512, "issued_mac": 512, "launches": 1,
            "bit_serial_cycles": 8, "activation_capture_bytes": 16, "result_bytes": 128,
            "weight_word_bit_products": 4096, "local_tree_adds": 3840,
            "shift_accumulates": 256})
        self.assertIsNone(result["cycles"])
        self.assertIsNone(result["energy_pj"])
        self.assertIsNone(result["area_mm2"])

    def test_tail_has_useful_and_full_physical_counts(self):
        result = self.model.compute(3, k=7, n=5)["counts"]
        self.assertEqual(result["useful_mac"], 105)
        self.assertEqual(result["issued_mac"], 1536)
        self.assertEqual(result["activation_capture_bytes"], 48)
        self.assertEqual(result["result_bytes"], 384)
        self.assertEqual(result["local_tree_adds"], 11520)

    def test_first_latency_is_not_ii(self):
        model = CIMMacroModel(timed_spec())
        self.assertEqual(model.compute(1, mode="timing")["cycles"], 11)
        self.assertEqual(model.compute(3, mode="timing")["cycles"], 27)
        self.assertEqual(model.compute(4, mode="timing")["cycles"] - model.compute(3, mode="timing")["cycles"], 8)
        self.assertEqual(model.program_context(mode="timing")["cycles"], 17)
        self.assertEqual(model.switch_context(mode="timing")["cycles"], 0)

    def test_program_is_full_context_and_unknown_timing_is_not_zero(self):
        self.assertEqual(self.model.program_context()["counts"],
                         {"program_bytes": 512, "program_wordlines": 16})
        for action in (lambda: self.model.compute(1, mode="timing"),
                       lambda: self.model.program_context(mode="timing"),
                       lambda: self.model.switch_context(mode="timing")):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                action()
        partial = timed_spec().to_dict()
        partial["parameters"]["write_first_cycles"].update(value=None, status="unknown")
        model = CIMMacroModel(CIMMacroSpec(partial))
        self.assertEqual(model.compute(1, mode="timing")["cycles"], 11)
        with self.assertRaises(ValueError):
            model.program_context(mode="timing")

    def test_invalid_operands_and_unsupported_formats(self):
        for rows, k, n in ((0, 16, 32), (True, 16, 32), (1, 17, 32),
                           (1, 16, 33), (1, 0, 32), (1, 16, 1.5)):
            with self.subTest(shape=(rows, k, n)), self.assertRaises(ValueError):
                self.model.compute(rows, k, n)
        for fmt in ("BF16", "FP16", "INT4", "INT16"):
            with self.assertRaisesRegex(ValueError, "INT8"):
                self.model.compute(1, data_format=fmt)
        with self.assertRaises(ValueError):
            self.model.compute(1, mode="approximate")

    def test_context_residency_switching_and_eviction(self):
        contexts = MacroContexts(self.spec)
        for tag in ("A", "B", "C", "D"):
            self.assertEqual(contexts.ensure(tag), (True, True))
        self.assertEqual(contexts.ensure("D"), (False, False))
        self.assertEqual(contexts.ensure("A"), (False, True))
        self.assertEqual(contexts.ensure("E"), (True, True))
        self.assertIn("A", contexts.tags)  # active A could not be overwritten
        self.assertNotIn("B", contexts.tags)
        self.assertEqual(contexts.ensure("A"), (False, True))

    def test_context_boundary_constraints(self):
        contexts = MacroContexts(self.spec)
        for index in (-1, 4, True, 0):
            with self.assertRaises(ValueError):
                contexts.activate(index)
        contexts.load(0, "A")
        contexts.activate(0)
        with self.assertRaisesRegex(ValueError, "active"):
            contexts.load(0, "B")
        for tag in (None, []):
            with self.assertRaises(ValueError):
                contexts.ensure(tag)


class MacroIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.spec = CIMMacroSpec(profile())

    def count(self, M, K, N, *, tm=1, tk=16, tn=32, macros=1, order="KNM"):
        return count_macro_tiles(self.spec, M, K, N, tm, tk, tn, macros, order, "test")

    def test_two_k_and_n_tails_and_external_reduction(self):
        result = self.count(1, 17, 33, tk=32, tn=64, macros=4)
        counts = result["counts"]
        self.assertEqual(counts["useful_mac"], 561)
        self.assertEqual(counts["issued_mac"], 2048)
        self.assertEqual(counts["program_bytes"], 2048)
        self.assertEqual(counts["result_bytes"], 512)
        self.assertEqual(counts["logical_output_bytes"], 132)
        self.assertEqual(counts["cross_macro_k_adds"], 33)
        self.assertEqual(counts["cross_tile_k_adds"], 0)

    def test_all_loop_orders_conserve_work_and_reduction(self):
        for order in LOOP_ORDERS:
            counts = self.count(3, 33, 65, tk=32, tn=64, macros=4, order=order)["counts"]
            self.assertEqual(counts["useful_mac"], 3*33*65)
            self.assertEqual(counts["issued_mac"], 3*3*3*512)
            self.assertEqual(counts["cross_macro_k_adds"]+counts["cross_tile_k_adds"], 3*65*2)

    def test_four_context_reuse_and_five_context_replacement(self):
        four = self.count(3, 64, 32, order="MKN")["counts"]
        self.assertEqual(four["context_programs"], 4)
        self.assertEqual(four["program_bytes"], 2048)
        five = self.count(3, 80, 32, order="MKN")["counts"]
        self.assertEqual(five["context_programs"], 15)
        knm = self.count(3, 80, 32, order="KNM")["counts"]
        self.assertEqual(knm["context_programs"], 5)

    def test_m_reuse_and_repeat_runs_start_empty(self):
        self.assertEqual(self.count(100, 16, 32)["counts"]["program_bytes"], 512)
        self.assertEqual(self.count(2, 16, 32), self.count(2, 16, 32))

    def test_impossible_macro_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.count(1, 16, 32, tk=32, macros=1)
        with self.assertRaisesRegex(ValueError, "align"):
            self.count(1, 16, 32, tk=17)

    def test_tsp_cim_uses_macro_profile_counts_and_unknown_costs(self):
        loader = TSPCIMConfigLoader()
        tsp_spec = loader.create_tsp_spec(loader.load_hardware_config())
        tsp = TSPCIMSimulator(tsp_spec)
        a, b = self.count(1, 16, 32), tsp.simulate(1, 16, 32)
        self.assertEqual(a["reference_id"], b["macro_reference_id"])
        self.assertEqual(a["counts"], {key: b["counts"][key] for key in a["counts"]})
        self.assertEqual(tsp_spec.macro_peak_mac_per_cycle_int8, 64)
        timed = tsp.simulate(1, 16, 32, mode="timing")
        self.assertEqual(timed["counts"], b["counts"])
        self.assertEqual(timed["timing"]["total_cycles"], 182)
        self.assertIn("macro.compute_first_cycles", timed["missing_parameters"])
        for result in (a, b):
            self.assertIsNone(result["timing"]["total_cycles"])
            self.assertIsNone(result["energy"]["total_pj"])
            self.assertIsNone(result["area"]["total_mm2"])
            self.assertEqual(result["hardware_inventory"]["used_macro_slots"], 1)


if __name__ == "__main__":
    unittest.main()
