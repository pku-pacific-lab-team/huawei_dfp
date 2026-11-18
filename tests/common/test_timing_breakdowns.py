"""Public timing views must not confuse overlapping work with elapsed time."""
import unittest

from src.core.ascend.simulator import AscendSimulator
from src.core.tpu.simulator import TPUSimulator
from src.core.tsp.simulator import TSPSimulator
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.utils.ascend_config import AscendConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from tests.tpu.support import spec, workload


class TimingBreakdownsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = TSPCIMConfigLoader()
        cim = loader.create_tsp_spec(loader.load_hardware_config())
        cls.simulators = {
            "ascend": AscendSimulator(AscendConfigLoader().load_hardware("ascend910_reference.json")),
            "tpu": TPUSimulator(spec(cmem_mode="tiled")),
            "tsp": TSPSimulator(cim.native_reference_spec),
            "tsp_cim": TSPCIMSimulator(cim),
        }

    def run_model(self, name, shape=(512, 256, 512), **options):
        sim = self.simulators[name]
        return (sim.simulate(workload(*shape), **options) if name in ("ascend", "tpu")
                else sim.simulate(*shape, **options))

    def assert_view(self, timing, name, status):
        self.assertTrue({"stages_cycles", "stages_cycles_basis", "stage_semantics", "service_cycles_sum"}.isdisjoint(timing))
        view = timing[name]
        self.assertEqual(view["status"], status)
        self.assertTrue(view["semantics"])
        if status == "available":
            self.assertTrue(view["by_stage"])
            self.assertTrue(all(type(v) is int and v >= 0 for v in view["by_stage"].values()))
            self.assertEqual(sum(view["by_stage"].values()), view["total_cycles"])
        else:
            self.assertIsNone(view["by_stage"])
            if status == "not_decomposed":
                self.assertTrue(view["reason"])
            if name == "service_cycles" or status == "not_evaluated":
                self.assertIsNone(view["total_cycles"])
        if name == "elapsed_cycles":
            self.assertEqual(view["total_cycles"], timing["total_cycles"])

    def test_four_models_keep_service_and_elapsed_meanings_distinct(self):
        for name, expected in (("ascend", 43588), ("tpu", 9244), ("tsp", 601), ("tsp_cim", 4307)):
            with self.subTest(model=name):
                report = self.run_model(name, mode="timing")
                timing = report["timing"]
                self.assertEqual(timing["total_cycles"], expected)
                service_available = name in ("ascend", "tpu")
                self.assert_view(timing, "service_cycles", "available" if service_available else "not_decomposed")
                self.assert_view(timing, "elapsed_cycles", "not_decomposed" if service_available else "available")
                if service_available:
                    self.assertGreater(timing["service_cycles"]["total_cycles"], expected)
                    self.assertEqual(timing["service_cycles"]["total_cycles"] - expected,
                                     timing["overlapped_service_cycles"])
                count = self.run_model(name, mode="count")
                for key in ("counts", "action_counts", "work_counts"):
                    self.assertEqual(report[key], count[key])

    def test_count_mode_does_not_invent_zero_cycles(self):
        for name in self.simulators:
            with self.subTest(model=name):
                timing = self.run_model(name, mode="count")["timing"]
                self.assertIsNone(timing["total_cycles"])
                for view in ("service_cycles", "elapsed_cycles"):
                    self.assert_view(timing, view, "not_evaluated")

    def test_tpu_serial_partition_and_pipeline_overlap(self):
        for cmem in ("tiled", "bypass"):
            for shape in ((512, 256, 512), (1, 16, 17)):
                with self.subTest(cmem=cmem, shape=shape):
                    options = dict(mode="timing", cmem_mode=cmem)
                    serial = self.run_model("tpu", shape, schedule="serial", **options)
                    pipe = self.run_model("tpu", shape, **options)
                    for view in ("service_cycles", "elapsed_cycles"):
                        self.assert_view(serial["timing"], view, "available")
                    self.assertEqual(serial["timing"]["elapsed_cycles"]["by_stage"],
                                     serial["timing"]["service_cycles"]["by_stage"])
                    self.assertEqual(serial["timing"]["service_cycles"], pipe["timing"]["service_cycles"])
                    self.assertEqual(serial["counts"], pipe["counts"])
                    # Even an accidentally equal service sum and makespan does not
                    # give the pipeline an attribution algorithm it does not have.
                    self.assert_view(pipe["timing"], "elapsed_cycles", "not_decomposed")
                    self.assertEqual("hbm_to_cmem" in serial["timing"]["elapsed_cycles"]["by_stage"], cmem == "tiled")

    def test_background_load_is_not_charged_twice(self):
        shape = (64, 16, 10240)
        native = self.run_model("tsp", shape, mode="timing")["timing"]
        self.assert_view(native, "service_cycles", "not_decomposed")
        self.assert_view(native, "elapsed_cycles", "available")
        stages = native["elapsed_cycles"]["by_stage"]
        self.assertEqual(stages, {"weight_load_stall": 1, "install": 320, "row_service": 1376})
        self.assertEqual(native["weight_load_cycles"] - stages["weight_load_stall"],
                         native["weight_load_overlap_cycles"])
        self.assertGreater(native["weight_load_overlap_cycles"], 0)
        for policy, expected, wait in (("background", 1214, 160), ("serialized", 1374, 320)):
            with self.subTest(policy=policy):
                timing = self.run_model("tsp_cim", shape, mode="timing", load_overlap=policy)["timing"]
                self.assert_view(timing, "service_cycles", "not_decomposed")
                self.assert_view(timing, "elapsed_cycles", "available")
                self.assertEqual(timing["total_cycles"], expected)
                self.assertEqual(timing["elapsed_cycles"]["by_stage"], {"load_wait": wait, "row_pipeline": 1054})
                self.assertEqual(timing["load_elapsed_cycles_sum"], 320)
