"""Cross-architecture policy visibility and numerical scheduling invariants."""
import copy
import unittest

from src.core.ascend.simulator import AscendSimulator
from src.core.tpu.simulator import TPUSimulator
from src.core.tsp.simulator import TSPSimulator
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.hardware.tpu import TPUSpec
from src.hardware.execution import read_execution
from src.utils.ascend_config import AscendConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from tests.tpu.support import hardware_config, spec, workload


class ExecutionContractTest(unittest.TestCase):
    def test_tpu_two_k_blocks_hand_schedule_and_capacity_fallback(self):
        sim = TPUSimulator(spec())
        options = dict(tile=(128,128,128), mode="timing")
        serial = sim.simulate(workload(128,256,128), schedule="serial", **options)
        pipe = sim.simulate(workload(128,256,128), **options)
        one = sim.simulate(workload(128,256,128), buffers=1, **options)
        self.assertEqual(serial["timing"]["total_cycles"], 1251)
        self.assertEqual(pipe["timing"]["total_cycles"], 1193)
        self.assertEqual(one["timing"]["total_cycles"], 1251)
        self.assertEqual(pipe["counts"], serial["counts"])
        self.assertEqual(pipe["timing"]["overlapped_service_cycles"], 58)
        for busy in pipe["timing"]["resource_busy_cycles"].values():
            self.assertLessEqual(busy, pipe["timing"]["total_cycles"])
        config = hardware_config()
        config["parameters"]["vmem_usable_bytes"]["value"] = 98304
        small = TPUSimulator(TPUSpec(config)).simulate(workload(128,256,128), **options)
        self.assertEqual(small["memory"]["vmem"]["input_buffer_slots"], 1)
        self.assertEqual(small["timing"]["total_cycles"], 1251)

    def test_default_tpu_uses_four_mxus_and_counts_array_tail(self):
        report = TPUSimulator(spec()).simulate(workload(512,256,512),mode="timing")
        self.assertEqual(report["timing"]["max_parallel_mxus_used"], 4)
        self.assertEqual(report["execution_contract"]["default_tile_mkn"], [256,128,256])
        tail = TPUSimulator(spec()).simulate(workload(1,1,1))
        self.assertEqual(tail["work_counts"]["logical"]["operations"], 2)
        self.assertEqual(tail["work_counts"]["issued"]["operations"], 32768)
        self.assertEqual(tail["work_counts"]["internal"]["status"], "not_modeled")

    def test_long_k_pipeline_acceleration_preserves_tail_and_busy_work(self):
        sim = TPUSimulator(spec())
        blocks = 1000000
        options = dict(tile=(128,128,128),mode="timing")
        task = workload(128,128*blocks+1,128)
        serial = sim.simulate(task,schedule="serial",**options)
        pipe = sim.simulate(task,**options)
        # All next-K DMA fits within 382 compute cycles: 58 per full next
        # block and 1 for the final K=1 tail. Initial DMA cannot be hidden.
        self.assertEqual(serial["timing"]["total_cycles"]-pipe["timing"]["total_cycles"],
                         (blocks-1)*58+1)
        self.assertEqual(pipe["timing"]["resource_busy_cycles"]["mxu"],
                         blocks*382+255)
        self.assertEqual(pipe["counts"],serial["counts"])

    def test_policy_and_fixed_latency_changes_are_consumed(self):
        cfg = hardware_config()
        cfg["execution"]["schedule"] = "serial"
        cfg["execution"]["default_tile_mkn"] = [128,128,128]
        cfg["execution"]["fixed_latency_cycles"]["transfer_startup"] = 2
        r = TPUSimulator(TPUSpec(cfg)).simulate(workload(1,1,1),mode="timing")
        self.assertEqual(r["timing"]["total_cycles"], 267)  # four transfers
        self.assertEqual(r["execution_contract"]["schedule"], "serial")
        cfg["execution"]["schedule"] = "resource_pipeline"
        pipe = TPUSimulator(TPUSpec(cfg)).simulate(workload(1,1,1),mode="timing")
        self.assertEqual(pipe["timing"]["total_cycles"], 267)
        for key, value in (("buffer_slots",True), ("schedule","magical"),
                           ("fixed_latency_cycles",{}), ("resident_boundary","local_MEM")):
            bad = copy.deepcopy(cfg["execution"])
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                read_execution(bad,"tpu")

    def test_cim_matched_baseline_and_background_sensitivity(self):
        loader = TSPCIMConfigLoader()
        config = loader.load_hardware_config()
        sim = TSPCIMSimulator(loader.create_tsp_spec(config))
        options = dict(mode="timing")
        shape = (64,16,640*16)
        default = sim.simulate(*shape, **options)
        serial = sim.simulate(*shape,load_overlap="serialized",**options)
        background = sim.simulate(*shape,load_overlap="background",**options)
        self.assertEqual(default,background)
        self.assertEqual(default["timing"]["total_cycles"],1214)
        self.assertEqual(serial["timing"]["total_cycles"],1374)
        self.assertTrue(sim.compare_mxm(*shape,result=default)["load_policy_matched"])
        self.assertEqual(serial["counts"],background["counts"])
        self.assertLess(background["timing"]["total_cycles"],serial["timing"]["total_cycles"])
        self.assertFalse(sim.compare_mxm(*shape,result=serial)["load_policy_matched"])
        self.assertTrue(sim.compare_mxm(*shape,result=background)["load_policy_matched"])
        comparison = sim.compare_load_policies(*shape)
        self.assertNotIn("native_serial_cycles", comparison)
        self.assertEqual(comparison["native_prefetch_cycles"],
                         sim.compare_mxm(*shape, result=background)["native_reference"]["timing"]["total_cycles"])
        self.assertTrue(comparison["counts_equal"])
        self.assertEqual(comparison["cim_cycles"]["serialized"],serial["timing"]["total_cycles"])
        base = sim.simulate(1,16,32,**options)
        config["execution"]["fixed_latency_cycles"]["local_pipeline"] += 7
        slower = TSPCIMSimulator(loader.create_tsp_spec(config)).simulate(1,16,32,**options)
        self.assertEqual(slower["timing"]["total_cycles"]-base["timing"]["total_cycles"],7)

    def test_four_reports_share_semantic_layers_without_changing_boundaries(self):
        loader = TSPCIMConfigLoader()
        cim = loader.create_tsp_spec(loader.load_hardware_config())
        reports = [AscendSimulator(AscendConfigLoader().load_hardware("ascend910_reference.json")).simulate(workload()),
                   TPUSimulator(spec()).simulate(workload()),
                   TSPSimulator(cim.native_reference_spec).simulate(16,32,16),
                   TSPCIMSimulator(cim).simulate(16,32,16)]
        for report in reports:
            self.assertEqual(report["work_counts"]["logical"]["operations"],16384)
            self.assertGreaterEqual(report["work_counts"]["issued"]["operations"],16384)
            self.assertEqual(report["execution_contract"]["scope"], report["boundary"]["scope"])
        self.assertEqual([r["execution_contract"]["resident_boundary"] for r in reports],
                         ["HBM","HBM","local_MEM","local_MEM"])
