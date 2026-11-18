"""Hemisphere mapping, data dependencies, shared services and CLI migration."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.core.cim.simulator import LOOP_ORDERS
from src.core.tsp_cim.resources import ResourceCalendar
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.tsp_reference import simulate_reference
from src.utils.workload import parse_workloads


ROOT = Path(__file__).resolve().parents[2]


class HemisphereTest(unittest.TestCase):
    def setUp(self):
        self.loader = TSPCIMConfigLoader()
        self.config = self.loader.load_hardware_config()
        self.spec = self.loader.create_tsp_spec(self.config)
        self.sim = TSPCIMSimulator(self.spec)

    def test_full_hemisphere_and_short_waves_do_not_scale_single_domain_time(self):
        full = self.sim.simulate(1, 320, 5120, mode="timing")
        self.assertEqual(full["boundary"]["scope"], "single_hem")
        self.assertEqual(full["hardware_inventory"]["allocated_macros"], 3200)
        self.assertEqual(full["hardware_inventory"]["allocated_cluster_domains"], 8)
        self.assertEqual(full["counts"]["program_bytes"], 320*5120)
        self.assertEqual(full["counts"]["useful_mac"], 320*5120)
        self.assertEqual(full["timing"]["compute_occupancy"]["value"], 8)
        self.assertEqual(full["timing"]["compute_occupancy"]["domain_cycles_sum"], 64)
        wave = full["schedule"]["waves"][0]
        self.assertEqual({t["domain"] for t in wave["tasks"]}, set(range(8)))
        self.assertEqual((wave["load_start"], wave["load_end"]), (0, 160))
        # First row: switch1 + capture4 + compute8 + drain23 + stagger4 + output2.
        self.assertEqual(full["timing"]["total_cycles"], 202)
        slots = [(s["plane"], s["writer_superlane"]) for s in wave["writer_slots"]]
        self.assertEqual(len(set(slots)), 40)
        one = self.sim.simulate(1, 320, 640, mode="timing")
        self.assertEqual(one["schedule"]["waves"][0]["active_domains"], 1)
        self.assertEqual(one["timing"]["total_cycles"], 199)
        self.assertEqual(one["counts"]["activation_capture_bytes"], 6400)
        self.assertEqual(one["counts"]["result_bytes"], 51200)
        self.assertEqual(one["counts"]["logical_inter_superlane_adds"], 19*640)
        nine = self.sim.simulate(1, 320, 5760)
        self.assertEqual([w["active_domains"] for w in nine["schedule"]["waves"]], [8, 1])
        self.assertEqual(nine["counts"]["context_programs"], 3600)

    def test_n_shards_do_not_reduce_and_unknown_costs_stay_unset(self):
        report = self.sim.simulate(2, 16, 640)
        self.assertEqual(report["counts"]["logical_inter_superlane_adds"], 0)
        self.assertEqual(report["counts"]["logical_inter_k_tile_adds"], 0)
        self.assertEqual(report["hardware_inventory"]["used_macro_slots"], 20)
        self.assertEqual(report["macro_reference_id"], self.spec.macro_spec.reference_id)
        self.assertEqual(report["timing"]["compute_occupancy"]["value"], 16)
        self.assertIsNone(report["timing"]["total_cycles"])
        self.assertIsNone(report["energy"]["total_pj"])
        self.assertIsNone(report["area"]["total_mm2"])

    def test_invalid_inputs_and_removed_selection_arguments_are_rejected(self):
        for kwargs in ({"scope": "single_cluster"}, {"scope": "single_hem"},
                       {"scope": "chip"}, {"plane": 0}, {"cluster": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                self.sim.simulate(1, 16, 32, **kwargs)
        for kwargs in ({"mode": "invalid"}, {"loop_order": "MKK"}, {"tile_m": 0},
                       {"hemisphere": -1}, {"hemisphere": 2}, {"hemisphere": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sim.simulate(1, 16, 32, **kwargs)
        for shape in ((0, 16, 32), (1, True, 32), (1, 16, -1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.sim.simulate(*shape)

    def test_tails_cross_k_reduction_and_all_orders_conserve_work(self):
        for order in LOOP_ORDERS:
            for tm in (None, 1, 2):
                with self.subTest(order=order, tm=tm):
                    count = self.sim.simulate(3, 321, 641, order, tile_m=tm)
                    timed = self.sim.simulate(3, 321, 641, order, tile_m=tm, mode="timing")
                    self.assertEqual(count["counts"], timed["counts"])
                    self.assertEqual(count["action_counts"], timed["action_counts"])
                    c = count["counts"]
                    self.assertEqual(c["useful_mac"], 617283)
                    self.assertEqual(c["issued_mac"], 677376)
                    self.assertEqual(c["program_bytes"], 225792)
                    self.assertEqual(c["logical_inter_superlane_adds"], 36537)
                    self.assertEqual(c["logical_inter_k_tile_adds"], 1923)
                    self.assertEqual(c["mem_read_psum_bytes"], 7692)
                    self.assertEqual(c["mem_write_final_bytes"], 7692)
                    self.assertEqual(c["mem_write_psum_bytes"], 7692)
                    for wave in timed["schedule"]["waves"]:
                        for task in wave["tasks"]:
                            predecessor = task["psum_predecessor"]
                            if task["ks"]:
                                self.assertEqual(predecessor["k_stop"], task["ks"])
                                self.assertLessEqual(predecessor["wave"], wave["wave"])
                            else:
                                self.assertIsNone(predecessor)
                    self.assertEqual(sum(d["counts"].get("useful_mac",0) for d in count["schedule"]["domains"]), 617283)
                    self.assertIsNone(count["timing"]["total_cycles"])
                    self.assertEqual(timed["timing"]["total_cycles"], sum(timed["timing"]["elapsed_cycles"]["by_stage"].values()))

    def test_background_lw_overlaps_and_respects_context_release(self):
        result = self.sim.simulate(64, 16, 640*8*6, mode="timing", load_overlap="background")
        waves = result["schedule"]["waves"]
        self.assertLess(waves[1]["load_start"], waves[0]["pipeline_end"])
        self.assertLess(waves[1]["load_end"], waves[0]["pipeline_end"])
        self.assertTrue(waves[4]["release_after_waves"])
        for wave in waves:
            self.assertGreaterEqual(wave["pipeline_start"], wave["load_end"])
            for predecessor in wave["release_after_waves"]:
                self.assertGreaterEqual(wave["load_start"], waves[predecessor]["pipeline_end"])
        self.assertLess(result["timing"]["total_cycles"],
                        result["timing"]["load_elapsed_cycles_sum"]+result["timing"]["elapsed_cycles"]["by_stage"]["row_pipeline"])

    def test_four_contexts_are_per_domain_and_m_reuse_does_not_move_weights(self):
        four = self.sim.simulate(3, 16, 640*8*4, "MKN", tile_m=1)
        five = self.sim.simulate(3, 16, 640*8*5, "MKN", tile_m=1)
        retained = self.sim.simulate(3, 16, 640*8*5, "KNM", tile_m=1)
        self.assertEqual(four["counts"]["context_programs"], 640)
        self.assertEqual(five["counts"]["context_programs"], 2400)
        self.assertEqual(retained["counts"]["context_programs"], 800)
        self.assertEqual(retained, self.sim.simulate(3,16,640*8*5,"KNM",tile_m=1))

    def test_capacity_unknown_services_and_physical_half_selection(self):
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            self.sim.simulate(1, 20000, 20000)
        with self.assertRaisesRegex(ValueError, "unsupported_capacity"):
            self.sim.simulate(1, 16, 32, reserved_bytes=111*2**20)
        a = self.sim.simulate(2, 321, 641, hemisphere=0)
        b = self.sim.simulate(2, 321, 641, hemisphere=1)
        self.assertEqual(a["counts"],b["counts"])
        self.assertEqual(b["boundary"]["selected_hemisphere"],1)
        self.assertEqual(b["memory"]["capacity_bytes"],110*2**20)
        for name in ("mem_read_bpc", "mem_write_bpc", "vxm_base_ops_per_cycle", "vxm_int32_add_cost", "vxm_add_latency_cycles"):
            config = copy.deepcopy(self.config)
            config["_native_reference_config"]["parameters"][name].update(value=None,status="unknown")
            sim = TSPCIMSimulator(self.loader.create_tsp_spec(config))
            self.assertIsNone(sim.simulate(1,321,32)["timing"]["total_cycles"])
            with self.assertRaisesRegex(ValueError,"Missing CIM hemisphere timing"):
                sim.simulate(1,321,32,mode="timing")
        config = copy.deepcopy(self.config)
        config["_native_reference_config"]["parameters"]["clock_ghz"].update(value=None,status="unknown")
        result = TSPCIMSimulator(self.loader.create_tsp_spec(config)).simulate(1,320,640,mode="timing")
        self.assertEqual(result["timing"]["total_cycles"],199)
        self.assertIsNone(result["timing"]["time_ns"])

    def test_restricted_services_slow_execution_without_changing_work(self):
        fast = self.sim.simulate(8, 641, 5120, mode="timing")
        slow = self.sim.simulate(8, 641, 5120, mode="timing", mem_slices=[0,1], streams=1)
        self.assertEqual(fast["counts"],slow["counts"])
        self.assertGreater(slow["timing"]["total_cycles"],fast["timing"]["total_cycles"])
        for report in (fast,slow):
            for resource in report["timing"]["resources"].values():
                self.assertLessEqual(resource["peak_reserved_bpc"],resource["capacity_bpc"])
        config = copy.deepcopy(self.config)
        config["_native_reference_config"]["parameters"]["vxm_base_ops_per_cycle"]["value"] = 1
        slower = TSPCIMSimulator(self.loader.create_tsp_spec(config)).simulate(8,641,5120,mode="timing")
        self.assertGreater(slower["timing"]["total_cycles"],fast["timing"]["total_cycles"])

    def test_native_comparison_inherits_actual_allocation_and_mode(self):
        report = self.sim.simulate(4,321,641,mode="timing",tile_m=2,mem_slices=[0,1],streams=2,reserved_bytes=64)
        comparison = self.sim.compare_mxm(4,321,641,result=report)
        workload = parse_workloads({"gemm":{"M":4,"K":321,"N":641},"data_format":"INT8"})[0]
        expected = simulate_reference(self.spec.native_reference_spec,workload,mode="timing",
            tile=(2,320,640),mem_slices=[0,1],streams=2,reserved_bytes=64)
        self.assertEqual(comparison["native_reference"],expected)
        self.assertEqual(comparison["status"],"aligned_scope_analytical")
        self.assertEqual(comparison["analytical_cycle_ratio_native_over_cim"],expected["timing"]["total_cycles"]/report["timing"]["total_cycles"])
        self.assertIsNone(comparison["speedup"])
        with self.assertRaisesRegex(ValueError,"must match"):
            self.sim.compare_mxm(5,321,641,result=report)

    def test_cli_defaults_and_removed_scope_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"result.json"
            common = [sys.executable,str(ROOT/"tsp_cim.py"),"--compare-mxm","--output",str(output)]
            run = subprocess.run(common+['--hemisphere','1'],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(run.returncode,0,run.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"],10)
            self.assertEqual(report["boundary"]["scope"],"single_hem")
            self.assertEqual(report["policy"]["weight_load_overlap"],"background")
            self.assertEqual(report["comparison"]["native_load_policy"],"background")
            self.assertTrue(report["comparison"]["load_policy_matched"])
            self.assertEqual(report["timing"]["total_cycles"],4307)
            self.assertEqual(report["comparison"]["native_reference"]["timing"]["total_cycles"],601)
            before = output.read_bytes()
            for removed in (["--cluster", "0", "0", "1"], ["--scope", "single_cluster"],
                            ["--scope", "single_hem"], ["--scope", "chip"],
                            ["--energy", "old.json"], ["--area", "old.json"]):
                with self.subTest(args=removed):
                    bad = subprocess.run(common+removed,cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
                    self.assertEqual(bad.returncode,2)
                    self.assertIn("unrecognized arguments",bad.stderr)
                    self.assertEqual(output.read_bytes(),before)


class CalendarTest(unittest.TestCase):
    def test_background_transfer_uses_residual_bandwidth_and_waits_when_full(self):
        calendar = ResourceCalendar({'read':10,'shared':10})
        calendar.reserve(0,5,{'read':4,'shared':8},'pipeline')
        calendar.reserve(5,8,{'shared':10},'write')
        # First five cycles transfer 10 bytes; wait three cycles, then 30/10.
        self.assertEqual(calendar.transfer(40,0,('read','shared'),10,'load'),11)
        self.assertEqual(calendar.summary()['shared']['peak_reserved_bpc'],10)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            calendar.reserve(2,3,{'shared':1},'overbook')
