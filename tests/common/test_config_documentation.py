"""Maintenance notes must stay separate from validated hardware model inputs."""
import copy
import json
from pathlib import Path
import unittest

from src.core.ascend.simulator import AscendSimulator
from src.core.tpu.simulator import TPUSimulator
from src.core.tsp.simulator import TSPSimulator
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.hardware.ascend import AscendSpec
from src.hardware.tpu import TPUSpec
from src.hardware.tsp import TSPSpec
from src.hardware.tsp_cim import TSPCIMSpec
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.workload import parse_workloads


ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    'ascend910_reference': (AscendSpec, AscendSimulator),
    'tpu_v4_reference': (TPUSpec, TPUSimulator),
    'tsp': (TSPSpec, TSPSimulator),
    'tsp_cim': (TSPCIMSpec, TSPCIMSimulator),
}


class ConfigDocumentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = {name: json.loads((ROOT/'configs/hardware'/f'{name}.json').read_text(encoding='utf-8'))
                   for name in PROFILES}
        cls.resolved = {**cls.raw, 'tsp_cim': TSPCIMConfigLoader().load_hardware_config()}

    def test_classification_covers_each_real_root_and_execution_field_once(self):
        for name, config in self.raw.items():
            with self.subTest(profile=name):
                notes = config['_documentation']
                fields = []
                for category in ('hardware_parameters', 'execution_policy', 'descriptive_metadata', 'configuration_control'):
                    fields.extend(notes[category])
                    self.assertTrue(all(isinstance(text, str) and text.strip() for text in notes[category].values()))
                self.assertEqual(len(fields), len(set(fields)))
                self.assertEqual(set(fields), set(config)-{'_documentation'})
                self.assertEqual(set(notes['execution_fields']), set(config['execution']))
                self.assertTrue(notes['parameter_fields'])
                self.assertTrue(notes['maintenance_rules'])
        self.assertIn('output_path', self.raw['ascend910_reference']['_documentation']['descriptive_metadata'])
        for key in ('native_reference', 'macro_reference'):
            self.assertIn(key, self.raw['tsp_cim']['_documentation']['configuration_control'])

    def test_notes_are_optional_and_do_not_change_reports_or_reference_ids(self):
        shape = (7, 321, 641)
        work = parse_workloads({'gemm': dict(zip('MKN', shape)), 'data_format': 'INT8'})[0]
        for name, (spec, simulator) in PROFILES.items():
            annotated = copy.deepcopy(self.resolved[name])
            plain = copy.deepcopy(annotated)
            plain.pop('_documentation')
            if name == 'tsp_cim':
                plain['_native_reference_config'].pop('_documentation')
            edited = copy.deepcopy(annotated)
            edited['_documentation'] = {'execution': {'schedule': 'not_a_real_policy'}, 'value': 99999}
            if name == 'tsp_cim':
                edited['_native_reference_config']['_documentation'] = {'arbitrary_notes': 'changed'}
            expected_input = copy.deepcopy(annotated)
            for mode in ('count', 'timing'):
                with self.subTest(profile=name, mode=mode):
                    reports = []
                    for config in (plain, annotated, edited):
                        sim = simulator(spec(config))
                        reports.append(sim.simulate(work, mode=mode) if name in ('ascend910_reference', 'tpu_v4_reference')
                                       else sim.simulate(*shape, mode=mode))
                    self.assertEqual(reports[0], reports[1])
                    self.assertEqual(reports[0], reports[2])
            self.assertEqual(annotated, expected_input, 'Constructing a Spec must not remove notes from the caller')

    def test_annotation_exception_does_not_hide_invalid_model_fields(self):
        for name, (spec, _) in PROFILES.items():
            for where in ('root', 'execution', 'parameter', 'missing_parameter', 'invalid_annotation'):
                with self.subTest(profile=name, mutation=where):
                    config = copy.deepcopy(self.resolved[name])
                    if where == 'root':
                        config['_documentations'] = {}
                    elif where == 'execution':
                        config['execution']['_documentation'] = {}
                    elif where == 'invalid_annotation':
                        config['_documentation'] = []
                    else:
                        parameter = config['clock_freq_ghz'] if name == 'ascend910_reference' else next(iter(config['parameters'].values()))
                        if where == 'parameter':
                            parameter['_documentation'] = {}
                        else:
                            parameter.pop('value')
                    with self.assertRaises(ValueError):
                        spec(config)
