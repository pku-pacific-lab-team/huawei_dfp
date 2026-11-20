"""Hand schedules and conservation checks for the finite LW buffer."""
import copy
import unittest

from src.core.tsp.pipeline import schedule_prefetch
from src.core.tsp.simulator import TSPSimulator
from src.hardware.tsp import TSPSpec
from src.utils.tsp_config import TSPConfigLoader


CAPS = dict(mem_read=10, mem_write=10, stream_to_mxm=10, stream_to_vxm=10)


def tile(duration=20, weight=100, read=5, stream=5, write=5):
    return dict(rows=1, first_row_cycles=duration, row_interval_cycles=1,
                weight_bytes=weight, weight_stream_bytes=weight,
                row_payload_bytes=dict(mem_read=read, mem_write=write,
                                       stream_to_mxm=stream, stream_to_vxm=5))


class WeightPrefetchTest(unittest.TestCase):
    def test_hand_schedule_cold_fill_overlap_and_iw_release(self):
        r = schedule_prefetch([tile(), tile(), tile()], CAPS, 50, 1, 100)
        # Cold LW 10; each IW 3 + rows 20. Each next LW takes 20 residual cycles.
        self.assertEqual(r['total_cycles'], 79)
        self.assertEqual(r['elapsed_stages'], dict(weight_load_stall=10, install=9, row_service=60))
        self.assertEqual(r['weight_load_cycles'], 50)
        self.assertEqual(r['weight_load_overlap_cycles'], 40)
        t = r['schedule']['tiles']
        self.assertEqual([(x['install_start'], x['pipeline_start'], x['pipeline_end']) for x in t],
                         [(10, 13, 33), (33, 36, 56), (56, 59, 79)])
        self.assertEqual(t[1]['weight_load'], dict(start=13, end=33))
        self.assertEqual(t[2]['weight_load'], dict(start=36, end=56))

    def test_reads_and_input_stream_contend_writes_are_independent(self):
        for option, expected in ((dict(read=10), 66), (dict(stream=10), 66), (dict(write=10), 56)):
            r = schedule_prefetch([tile(**option), tile()], CAPS, 50, 1, 100)
            self.assertEqual(r['total_cycles'], expected)
            for resource in r['resources'].values():
                self.assertLessEqual(resource['peak_reserved_bpc'], resource['capacity_bpc'])

    def test_partial_prefetch_and_exact_fractional_tail(self):
        r = schedule_prefetch([tile(duration=5), tile(duration=5)], CAPS, 50, 1, 100)
        # Next LW: 25 bytes during rows, then 75 bytes / 10 B/c rounded to 8.
        self.assertEqual(r['total_cycles'], 34)
        self.assertEqual(r['weight_load_stall_cycles'], 18)
        self.assertEqual(r['weight_load_overlap_cycles'], 5)
        self.assertEqual([s['payload_bytes'] for s in r['schedule']['weight_load_segments']], [100, 25, 75])
        self.assertEqual(sum(r['elapsed_stages'].values()), r['total_cycles'])

    def test_one_pending_buffer_spans_reused_m_tiles(self):
        r = schedule_prefetch([tile(duration=5), tile(duration=15, weight=0), tile()], CAPS, 50, 1, 100)
        self.assertEqual(r['total_cycles'], 56)
        self.assertEqual(r['schedule']['tiles'][2]['weight_load'], dict(start=13, end=33))
        self.assertNotIn('weight_load', r['schedule']['tiles'][1])
        with self.assertRaisesRegex(ValueError, 'weight_buffer_bytes'):
            schedule_prefetch([tile()], CAPS, 50, 1, 99)

    def test_vector_padding_and_lw_do_not_depend_on_iw_port(self):
        a = tile(weight=16); a['weight_stream_bytes'] = 20
        r = schedule_prefetch([a], CAPS, 16, 0, 16)
        self.assertEqual(r['weight_load_cycles'], 2)
        self.assertEqual(r['install_cycles'], 1)
        self.assertEqual(r['schedule']['weight_load_segments'][0]['payload_bytes'], 16)
        slow = schedule_prefetch([a], CAPS, 8, 0, 16)
        self.assertEqual(slow['weight_load_cycles'], 2)
        self.assertEqual(slow['install_cycles'], 2)

    def test_native_resource_dependencies_and_byte_conservation(self):
        sim = TSPSimulator(TSPSpec(TSPConfigLoader().load_hardware_config()))
        for order in ('KNM', 'MKN', 'KMN', 'NKM'):
            for slices, streams in (([0], 1), ([0], 32), (None, 32)):
                r = sim.simulate(7, 321, 641, mode='timing', loop_order=order,
                                 tile=(3, 320, 640), mem_slices=slices, streams=streams)
                timing, schedule = r['timing'], r['schedule']
                self.assertEqual(sum(timing['elapsed_cycles']['by_stage'].values()), timing['total_cycles'])
                self.assertEqual(timing['weight_load_cycles'],
                                 timing['weight_load_stall_cycles'] + timing['weight_load_overlap_cycles'])
                self.assertAlmostEqual(sum(s['payload_bytes'] for s in schedule['weight_load_segments']),
                                       r['counts']['mem_read_weight_bytes'])
                self.assertAlmostEqual(sum(s['stream_bytes'] for s in schedule['weight_load_segments']),
                                       r['counts']['weight_stream_bytes'])
                previous_end = buffer_free = 0
                for t in schedule['tiles']:
                    self.assertGreaterEqual(t['install_start'], previous_end)
                    if t['weight_bytes']:
                        self.assertGreaterEqual(t['weight_load']['start'], buffer_free)
                        self.assertLessEqual(t['weight_load']['end'], t['install_start'])
                        buffer_free = t['install_end']
                    previous_end = t['pipeline_end']
                for resource in timing['resources'].values():
                    self.assertLessEqual(resource['peak_reserved_bpc'], resource['capacity_bpc'])
                self.assertEqual(r['counts'], sim.simulate(7, 321, 641, loop_order=order,
                    tile=(3, 320, 640), mem_slices=slices, streams=streams)['counts'])

    def test_capacity_is_finite_and_old_serial_policy_is_rejected(self):
        cfg = TSPConfigLoader().load_hardware_config()
        self.assertTrue(all(p['value'] is not None for p in cfg['parameters'].values()))
        for key, value in (('weight_load_overlap', 'serialized'), ('rules',
                dict(inter_tile_overlap='none', mem_read_write='independent_assumed_conflict_free'))):
            bad = copy.deepcopy(cfg); bad['execution'][key] = value
            with self.assertRaises(ValueError): TSPSpec(bad)
        cfg['parameters']['weight_buffer_bytes']['value'] = 256
        sim = TSPSimulator(TSPSpec(cfg))
        for mode in ('count', 'timing'):
            with self.assertRaisesRegex(ValueError, 'weight_buffer_bytes'):
                sim.simulate(1, 32, 16, mode=mode)
            self.assertEqual(sim.simulate(1, 32, 16, mode=mode, tile=(1, 16, 16))['counts']['weight_install_count'], 2)
