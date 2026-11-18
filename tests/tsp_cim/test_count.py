"""Hand-calculated single Cluster-domain counts, tails and CLI scope contracts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.core.cim.simulator import LOOP_ORDERS
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.utils.tsp_cim_config import TSPCIMConfigLoader


ROOT = Path(__file__).resolve().parents[2]


class TSPCIMCountTest(unittest.TestCase):
    def setUp(self):
        loader = TSPCIMConfigLoader()
        self.sim = TSPCIMSimulator(loader.create_tsp_spec(loader.load_hardware_config()))

    def test_full_domain_uses_400_macros_with_independent_n_shards(self):
        result = self.sim.simulate(1, 320, 640)
        counts = result['counts']
        self.assertEqual(counts['useful_mac'], 204800)
        self.assertEqual(counts['issued_mac'], 204800)
        self.assertEqual(counts['context_programs'], 400)
        self.assertEqual(counts['program_bytes'], 204800)
        self.assertEqual(counts['logical_activation_broadcast_bytes'], 320)
        self.assertEqual(counts['activation_capture_bytes'], 6400)
        self.assertEqual(counts['result_bytes'], 51200)
        self.assertEqual(counts['logical_cluster_partial_bytes'], 2560)
        self.assertEqual(counts['logical_inter_superlane_adds'], 19 * 640)
        self.assertEqual(counts['logical_inter_k_tile_adds'], 0)
        self.assertEqual(result['hardware_inventory']['allocated_macros'], 400)
        self.assertEqual(result['hardware_inventory']['chip']['macros'], 6400)
        self.assertEqual(result['policy']['selected_tile_mkn'], [1, 320, 640])

    def test_n_shards_are_concatenated_not_reduced(self):
        result = self.sim.simulate(2, 16, 640)
        self.assertEqual(result['counts']['logical_inter_superlane_adds'], 0)
        self.assertEqual(result['counts']['logical_inter_k_tile_adds'], 0)
        self.assertEqual(result['hardware_inventory']['used_macro_slots'], 20)

    def test_two_dimensional_tails_preserve_work_and_separate_reduction_levels(self):
        counts = self.sim.simulate(3, 321, 641)['counts']
        expected = {'useful_mac': 617283, 'issued_mac': 677376, 'context_programs': 441,
                    'program_bytes': 225792, 'result_bytes': 169344,
                    'logical_inter_superlane_adds': 36537, 'logical_inter_k_tile_adds': 1923,
                    'logical_activation_broadcast_bytes': 2016, 'logical_cluster_partial_bytes': 15384,
                    'logical_output_bytes': 7692}
        for name, value in expected.items():
            with self.subTest(name=name):
                self.assertEqual(counts[name], value)

    def test_loop_orders_conserve_work_and_reduction(self):
        for order in LOOP_ORDERS:
            counts = self.sim.simulate(3, 321, 641, order, tile_m=1)['counts']
            self.assertEqual(counts['useful_mac'], 617283)
            self.assertEqual(counts['issued_mac'], 677376)
            self.assertEqual(counts['logical_inter_superlane_adds'] + counts['logical_inter_k_tile_adds'], 38460)

    def test_context_reuse_and_eviction_on_the_same_physical_domain(self):
        self.assertEqual(self.sim.simulate(10, 320, 640)['counts']['program_bytes'], 204800)
        four = self.sim.simulate(3, 16, 2560, 'MKN', tile_m=1)['counts']
        five = self.sim.simulate(3, 16, 3200, 'MKN', tile_m=1)['counts']
        reused = self.sim.simulate(3, 16, 3200, 'KNM', tile_m=1)['counts']
        self.assertEqual((four['context_programs'], five['context_programs'], reused['context_programs']), (80, 300, 100))

    def test_domains_remain_independent_and_runs_start_empty(self):
        a = self.sim.simulate(1, 320, 640)
        b = self.sim.simulate(1, 320, 640, hemisphere=1, plane=1, cluster=3)
        self.assertEqual(a['counts'], b['counts'])
        self.assertEqual(a['counts'], self.sim.simulate(1, 320, 640)['counts'])
        self.assertEqual(b['boundary']['selected_domain'], {'hemisphere': 1, 'plane': 1, 'cluster': 3})
        self.assertIsNone(b['timing']['total_cycles'])
        self.assertIsNone(b['energy']['total_pj'])
        self.assertIsNone(b['area']['total_mm2'])

    def test_documented_acc_occupancy_keeps_tails_and_is_not_elapsed_time(self):
        # 满域和极小尾块的一行都占 8 拍；K/N 双尾块则有四个任务。
        for shape, expected in (((1, 320, 640), 8), ((1, 1, 1), 8),
                                ((3, 321, 641), 96), ((3, 16, 3200), 120)):
            for order in LOOP_ORDERS:
                for tile_m in (None, 1):
                    with self.subTest(shape=shape, order=order, tile_m=tile_m):
                        result = self.sim.simulate(*shape, order, tile_m=tile_m)
                        occupancy = result['timing']['compute_occupancy']
                        self.assertEqual(occupancy['value'], expected)
                        self.assertEqual(occupancy['unit'], 'cycle')
                        self.assertEqual(occupancy['scope'], 'single_cluster')
                        self.assertEqual(occupancy['status'], 'documented_schedule')
                        self.assertTrue(occupancy['source'])
                        self.assertEqual(result['timing']['status'], 'not_computed')
                        for name in ('total_cycles', 'chip_cycles', 'time_ns', 'throughput_gops'):
                            self.assertIsNone(result['timing'][name])
                        self.assertIn('macro.compute_first_cycles', result['missing_parameters'])
        full = self.sim.simulate(1, 320, 640)
        self.assertEqual(full['counts']['bit_serial_cycles'], 3200)
        self.assertEqual(full['timing']['compute_occupancy']['value'], 8)

    def test_unsupported_execution_and_invalid_sizes_fail(self):
        for kwargs in ({'scope': 'chip'}, {'mode': 'invalid'}, {'tile_m': 0}, {'cluster': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sim.simulate(1, 16, 32, **kwargs)
        for shape in ((0, 16, 32), (1, True, 32), (1, 16, -1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.sim.simulate(*shape)

    def test_cli_scope_export_and_native_reference(self):
        common = [sys.executable, str(ROOT / 'tsp_cim.py'), '--workload', 'tiny_gemm.json', '--formats', 'INT8']
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            run = subprocess.run(common + ['--cluster', '1', '1', '3', '--compare-mxm', '--output', str(output)],
                                 cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['counts']['useful_mac'], 67108864)
            self.assertEqual(result['boundary']['scope'], 'single_cluster')
            self.assertEqual(result['hardware_inventory']['chip']['macros'], 6400)
            self.assertIsNone(result['comparison']['speedup'])
            self.assertEqual(result['comparison']['native_reference']['reference_id'], result['native_reference_id'])
        run = subprocess.run(common + ['--scope', 'chip'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(run.returncode, 2)
        self.assertIn('chip inventory is not chip execution', run.stderr)


class TSPCIMTimingTest(unittest.TestCase):
    def setUp(self):
        self.loader = TSPCIMConfigLoader()
        self.config = self.loader.load_hardware_config()
        self.sim = TSPCIMSimulator(self.loader.create_tsp_spec(self.config))

    def test_full_domain_and_tail_have_hand_calculated_total_times(self):
        full = self.sim.simulate(1, 320, 640, mode='timing')
        self.assertEqual(full['timing']['stages_cycles'], {
            'load': 160, 'switch': 1, 'compute': 8, 'drain': 31,
            'partial_read': 0, 'partial_add': 0, 'write': 10})
        self.assertEqual(full['timing']['total_cycles'], 210)
        self.assertAlmostEqual(full['timing']['time_ns'], 210 / 0.9)
        self.assertAlmostEqual(full['timing']['throughput_gops'], 409600 / (210 / 0.9))
        tail = self.sim.simulate(1, 1, 1, mode='timing')
        # 极小尾块仍预约一次 160 拍装载，但只写一个有效 INT32 结果。
        self.assertEqual(tail['timing']['total_cycles'], 182)
        self.assertEqual(tail['timing']['stages_cycles']['write'], 1)
        self.assertEqual(full['boundary']['end'], 'logical_INT32_C_in_local_MEM')
        self.assertIsNone(full['timing']['chip_cycles'])

    def test_cross_k_segments_charge_read_add_and_single_write(self):
        result = self.sim.simulate(1, 321, 32, mode='timing')
        self.assertEqual(result['timing']['stages_cycles'], {
            'load': 320, 'switch': 2, 'compute': 16, 'drain': 43,
            'partial_read': 1, 'partial_add': 1, 'write': 2})
        self.assertEqual(result['timing']['total_cycles'], 385)
        self.assertEqual(result['counts']['logical_inter_k_tile_adds'], 32)

    def test_residency_changes_load_time_and_each_run_starts_empty(self):
        retained = self.sim.simulate(3, 16, 2560, 'MKN', tile_m=1, mode='timing')
        evicted = self.sim.simulate(3, 16, 3200, 'MKN', tile_m=1, mode='timing')
        reused = self.sim.simulate(3, 16, 3200, 'KNM', tile_m=1, mode='timing')
        self.assertEqual([r['timing']['load_batches'] for r in (retained, evicted, reused)], [4, 15, 5])
        self.assertEqual(evicted['timing']['total_cycles'], 2865)
        self.assertEqual(reused['timing']['total_cycles'], 1255)
        self.assertEqual(reused, self.sim.simulate(3, 16, 3200, 'KNM', tile_m=1, mode='timing'))

    def test_timing_preserves_actions_for_all_loop_orders_and_row_groupings(self):
        for order in LOOP_ORDERS:
            for tile_m in (None, 1, 2):
                with self.subTest(order=order, tile_m=tile_m):
                    counted = self.sim.simulate(3, 321, 641, order, tile_m=tile_m)
                    timed = self.sim.simulate(3, 321, 641, order, tile_m=tile_m, mode='timing')
                    for field in ('counts', 'action_counts', 'hardware_inventory', 'parameter_provenance'):
                        self.assertEqual(counted[field], timed[field])
                    t = timed['timing']
                    self.assertEqual(t['total_cycles'], sum(t['stages_cycles'].values()))
                    self.assertEqual(t['stages_cycles']['compute'], 96)
                    self.assertEqual(t['scheduled_tiles'], timed['counts']['tiles'])
                    self.assertEqual(t['load_batches'] * 160, t['stages_cycles']['load'])

    def test_assumptions_do_not_fill_unknown_macro_timing_or_clock(self):
        from src.core.cim.macro_model import CIMMacroModel
        self.config['_native_reference_config']['parameters']['clock_ghz'].update(value=None, status='unknown')
        spec = self.loader.create_tsp_spec(self.config)
        result = TSPCIMSimulator(spec).simulate(1, 320, 640, mode='timing')
        self.assertEqual(result['timing']['total_cycles'], 210)
        self.assertIsNone(result['timing']['time_ns'])
        self.assertIsNone(result['timing']['throughput_gops'])
        self.assertEqual(result['timing']['clock']['status'], 'unknown')
        self.assertIn('macro.compute_first_cycles', result['missing_parameters'])
        with self.assertRaisesRegex(ValueError, 'Macro timing unavailable'):
            CIMMacroModel(spec.macro_spec).compute(1, mode='timing')
        params = result['timing']['parameters']
        self.assertEqual(params['writer_bpc']['value'], 256)
        self.assertEqual(params['writer_bpc']['status'], 'assumed')
        for item in params.values():
            self.assertTrue({'value', 'unit', 'scope', 'status', 'source', 'source_version', 'conditions'} <= item.keys())
        for field, key in (('energy', 'total_pj'), ('area', 'total_mm2')):
            self.assertIsNone(result[field][key])
