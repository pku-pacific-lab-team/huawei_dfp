"""Hand-calculated contracts for the Macro actions and context state."""
import json
from pathlib import Path
import unittest

from src.core.cim.macro_model import CIMMacroModel, MacroContexts
from src.hardware.macro import CIMMacroSpec

ROOT = Path(__file__).resolve().parents[2]


def profile():
    return json.loads((ROOT / "configs/hardware/cim_macro.json").read_text(encoding="utf-8"))


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
        del config["parameters"]["input_ii_cycles"]
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

    def test_old_schema_and_retired_timing_fields_are_rejected(self):
        for name in ("compute_first_cycles", "write_first_cycles",
                     "write_wordline_ii_cycles", "context_switch_cycles"):
            config = profile()
            config["parameters"][name] = {"value": None}
            with self.subTest(name=name), self.assertRaises(ValueError):
                CIMMacroSpec(config)
        config = profile()
        config["schema_version"] = 1
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

    def test_program_is_full_context_and_local_timing_api_is_retired(self):
        self.assertEqual(self.model.program_context()["counts"],
                         {"program_bytes": 512, "program_wordlines": 16})
        self.assertEqual(self.model.switch_context()["counts"], {"context_switches": 1})
        # II remains an action/occupancy contract; it must not become elapsed latency.
        self.assertEqual(self.model.compute(3)["counts"]["bit_serial_cycles"], 24)
        for result in (self.model.compute(3), self.model.program_context(), self.model.switch_context()):
            self.assertIsNone(result["cycles"])
        for action in (lambda: self.model.compute(1, mode="timing"),
                       lambda: self.model.program_context(mode="timing"),
                       lambda: self.model.switch_context(mode="timing")):
            with self.assertRaises(TypeError):
                action()

    def test_invalid_operands_and_unsupported_formats(self):
        for rows, k, n in ((0, 16, 32), (True, 16, 32), (1, 17, 32),
                           (1, 16, 33), (1, 0, 32), (1, 16, 1.5)):
            with self.subTest(shape=(rows, k, n)), self.assertRaises(ValueError):
                self.model.compute(rows, k, n)
        for fmt in ("BF16", "FP16", "INT4", "INT16"):
            with self.assertRaisesRegex(ValueError, "INT8"):
                self.model.compute(1, data_format=fmt)
        with self.assertRaises(TypeError):
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


if __name__ == "__main__":
    unittest.main()
