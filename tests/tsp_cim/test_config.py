"""Strict topology profiles, reference resolution, and unknown-parameter behavior."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.utils.tsp_cim_config import TSPCIMConfigLoader


ROOT = Path(__file__).resolve().parents[2]


class TSPCIMConfigTest(unittest.TestCase):
    def setUp(self):
        self.loader = TSPCIMConfigLoader()
        self.config = self.loader.load_hardware_config()

    def test_references_resolve_from_owner_path_not_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = json.loads((ROOT / 'configs/hardware/tsp_cim.json').read_text(encoding='utf-8'))
            for key, name in (('macro_reference', 'cim_macro.json'), ('native_reference', 'tsp.json')):
                raw[key] = name
                (folder / name).write_text((ROOT / 'configs/hardware' / name).read_text(encoding='utf-8'), encoding='utf-8')
            path = folder / 'custom.json'
            path.write_text(json.dumps(raw), encoding='utf-8')
            reference = self.loader.create_tsp_spec(self.config)
            relocated = self.loader.create_tsp_spec(self.loader.load_hardware_config(str(path)))
            self.assertEqual(reference.reference_id, relocated.reference_id)
            self.assertEqual(reference.provenance(), relocated.provenance())

    def test_metadata_and_fixed_geometry_are_required(self):
        for field, value in (('value', 8), ('value', True), ('value', 4.0), ('value', 0),
                             ('unit', 'byte'), ('scope', 'chip'), ('status', 'unknown'),
                             ('source', ''), ('source_version', ''), ('verification', '')):
            config = copy.deepcopy(self.config)
            config['parameters']['macros_per_core'][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.loader.create_tsp_spec(config)
        config = copy.deepcopy(self.config)
        del config['parameters']['macros_per_core']
        with self.assertRaises(ValueError):
            self.loader.create_tsp_spec(config)

    def test_legacy_overlay_and_inline_macro_are_not_accepted(self):
        for field in ('superlane', 'cim_fabric', 'sram', 'clock_freq_ghz', 'streams'):
            config = copy.deepcopy(self.config)
            config[field] = {}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'legacy'):
                self.loader.create_tsp_spec(config)
        with self.assertRaises(ValueError):
            self.loader.create_tsp_spec({'macro': {}})

    def test_shared_geometry_cannot_drift_from_native_reference(self):
        shared = self.loader.create_tsp_spec(self.config).provenance()['shared_system']
        self.assertEqual(shared['vxm_fp32_exp_cost']['value'], 15)
        self.assertNotIn('nonlinear_ops_per_cycle', shared)
        for name in ('offchip_capacity_bytes', 'offchip_bandwidth_bpc', 'interchip_bandwidth_bpc'):
            self.assertNotIn(name, shared)
        config = copy.deepcopy(self.config)
        config['_native_reference_config']['parameters']['planes_per_hemisphere']['value'] = 3
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            self.loader.create_tsp_spec(config)
        config = copy.deepcopy(self.config)
        config['_native_reference_config']['parameters']['streams_per_direction']['value'] = 16
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.loader.create_tsp_spec(config)

    def test_unknown_clock_propagates_to_peak_and_provenance_is_copied(self):
        config = copy.deepcopy(self.config)
        config['_native_reference_config']['parameters']['clock_ghz'].update(value=None, status='unknown')
        spec = self.loader.create_tsp_spec(config)
        self.assertEqual(spec.derived()['int8_mac_per_cycle_chip']['value'], 409600)
        self.assertIsNone(spec.derived()['int8_tops_chip']['value'])
        self.assertIn('shared_system.clock_ghz', spec.missing_parameters())
        reference_id = spec.reference_id
        config['parameters']['cores_per_cluster']['value'] = 1
        spec.provenance()['topology']['cores_per_cluster']['value'] = 2
        self.assertEqual(spec.reference_id, reference_id)
        self.assertEqual(spec.cluster.cores, 5)

    def test_retired_inventory_fields_and_old_schema_are_rejected(self):
        for name in ('physical_sram_bytes', 'additional_local_sram_bytes', 'collector_buffer_bytes'):
            config = copy.deepcopy(self.config)
            config['parameters'][name] = {'value': None}
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.loader.create_tsp_spec(config)
        config = copy.deepcopy(self.config)
        config['schema_version'] = 2
        with self.assertRaises(ValueError):
            self.loader.create_tsp_spec(config)

    def test_fp32_service_is_visible_but_not_a_gemm_missing_parameter(self):
        spec = self.loader.create_tsp_spec(self.config)
        self.assertEqual(spec.provenance()['shared_system']['vxm_fp32_rsqrt_cost']['value'], 12)
        self.assertEqual(spec.derived()['vxm_service'], spec.native_reference_spec.vxm_service())
        self.config['_native_reference_config']['parameters']['vxm_fp32_exp_cost'].update(value=None, status='unknown')
        spec = self.loader.create_tsp_spec(self.config)
        self.assertIn('vxm_fp32_exp_cost', spec.derived()['vxm_service']['missing_parameters'])
        self.assertEqual(spec.missing_parameters(), [])

    def test_writer_scope_migration_is_explicit(self):
        for change in ('old_schema', 'old_scope'):
            config = copy.deepcopy(self.config)
            if change == 'old_schema':
                config['schema_version'] = 3
            else:
                config['parameters']['int32_writer_bpc']['scope'] = 'plane'
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.loader.create_tsp_spec(config)
        spec = self.loader.create_tsp_spec(self.config)
        self.assertEqual(spec.provenance()['topology']['int32_writer_bpc']['scope'], 'writer_superlane_plane')
        self.assertEqual(spec.derived()['nim_output_target_bpc_per_hemisphere']['value'], 10240)
