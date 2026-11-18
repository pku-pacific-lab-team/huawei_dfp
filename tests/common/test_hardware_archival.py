"""Reference-only archives preserve evidence without becoming runtime defaults."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from src.core.ascend.simulator import AscendSimulator
from src.core.tpu.simulator import TPUSimulator
from src.core.tsp.simulator import TSPSimulator
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.hardware.ascend import AscendSpec
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.hardware.tsp_cim import TSPCIMSpec
from src.utils.ascend_config import AscendConfigLoader
from src.utils.tpu_config import TPUConfigLoader
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.workload import parse_workloads


ROOT = Path(__file__).resolve().parents[2]
ARCHIVED = {
    'ascend910_reference': (AscendSpec, 8, ['cube.units_per_core', 'vector.units_per_core']),
    'tpu_v4_reference': (TPUSpec, 11, ['parameters.nominal_int8_tops', 'parameters.vpus_per_tensorcore',
                                     'parameters.vpu_lanes', 'parameters.vpu_alus_per_lane']),
    'tsp': (TSPSpec, 8, ['parameters.nominal_int8_tops']),
    'tsp_cim': (TSPCIMSpec, 6, ['parameters.nim_units_per_plane', 'parameters.nim_tiles_per_plane',
                              'parameters.nim_channels_per_tile', 'parameters.nim_conversion_lanes']),
}


class HardwareArchivalTest(unittest.TestCase):
    def test_archives_preserve_full_parameter_records_and_audit(self):
        audit = json.loads((ROOT/'calibration/parameter_source_audit.json').read_text(encoding='utf-8'))
        for name, (_, _, paths) in ARCHIVED.items():
            archive_path = f'configs/archive/{name}.json'
            archive = json.loads((ROOT/archive_path).read_text(encoding='utf-8'))
            active = json.loads((ROOT/'configs/hardware'/f'{name}.json').read_text(encoding='utf-8'))
            entry = audit['configs'][f'configs/hardware/{name}.json']
            with self.subTest(profile=name):
                self.assertFalse(audit['archives'][archive_path]['runtime_dependency'])
                digest = hashlib.sha256(json.dumps(archive, ensure_ascii=False, sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
                self.assertEqual(digest, audit['archives'][archive_path]['canonical_json_sha256'])
                for path in paths:
                    parent, key = path.split('.')
                    archive_key = key if parent == 'parameters' else path
                    retired = entry['retired_parameters'][path]
                    self.assertNotIn(key, active[parent])
                    self.assertEqual(archive['parameters'][archive_key], retired['last_configuration'])
                    self.assertEqual(retired['archive_path'], archive_path)
                    self.assertTrue(retired['prior_audit']['evidence'])
                    self.assertTrue(archive['archival_reasons'][archive_key])
                self.assertTrue(archive['archival_history'])
        # Earlier TPU archival data and interconnect evidence remain intact.
        tpu = json.loads((ROOT/'configs/archive/tpu_v4_reference.json').read_text(encoding='utf-8'))
        self.assertEqual(len(tpu['parameters']), 9)
        self.assertEqual(tpu['parameters']['register_file_chip_bytes']['value'], 262144)
        self.assertIn('interconnect', tpu)

    def test_old_schemas_and_reintroduced_archived_inputs_are_rejected(self):
        for name, (spec, old_schema, paths) in ARCHIVED.items():
            config = (TSPCIMConfigLoader().load_hardware_config() if name == 'tsp_cim' else
                      json.loads((ROOT/'configs/hardware'/f'{name}.json').read_text(encoding='utf-8')))
            archive = json.loads((ROOT/'configs/archive'/f'{name}.json').read_text(encoding='utf-8'))
            old = copy.deepcopy(config)
            old['schema_version'] = old_schema
            with self.subTest(profile=name, old_schema=old_schema), self.assertRaises(ValueError):
                spec(old)
            for path in paths:
                parent, key = path.split('.')
                changed = copy.deepcopy(config)
                changed[parent][key] = archive['parameters'][key if parent == 'parameters' else path]
                with self.subTest(profile=name, reintroduced=path), self.assertRaises(ValueError):
                    spec(changed)

    def test_gemm_and_vector_services_do_not_load_missing_or_invalid_archives(self):
        def run(config_dir):
            a = AscendConfigLoader(config_dir).load_hardware('ascend910_reference.json')
            t = TPUConfigLoader(config_dir).load_hardware('tpu_v4_reference.json')
            loader = TSPConfigLoader(config_dir)
            n = loader.create_tsp_spec(loader.load_hardware_config())
            loader = TSPCIMConfigLoader(config_dir)
            c = loader.create_tsp_spec(loader.load_hardware_config())
            work = parse_workloads({'gemm': {'M': 7, 'K': 321, 'N': 641}, 'data_format': 'INT8'})[0]
            reports = [sim.simulate(work, mode='timing') for sim in (AscendSimulator(a), TPUSimulator(t))]
            reports.extend(sim.simulate(*work.shape, mode='timing') for sim in (TSPSimulator(n), TSPCIMSimulator(c)))
            services = [a.vector_service_cycles({'fp32_exp': 65}),
                        t.vector_service_cycles({'fp32_exp': 2049}),
                        n.vxm_service_cycles({'fp32_exp': 321})]
            return reports, services

        expected = run(ROOT/'configs')
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)/'configs'
            shutil.copytree(ROOT/'configs/hardware', config_dir/'hardware')
            self.assertFalse((config_dir/'archive').exists())
            self.assertEqual(run(config_dir), expected)
            (config_dir/'archive').mkdir()
            for name in ARCHIVED:
                (config_dir/'archive'/f'{name}.json').write_text('deliberately invalid JSON', encoding='utf-8')
            self.assertEqual(run(config_dir), expected)
