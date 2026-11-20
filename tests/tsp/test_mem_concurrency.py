"""Independent directional MEM budgets with an explicit bank-layout assumption."""
import copy
import unittest

from src.hardware.tsp import TSPSpec
from src.core.tsp.simulator import TSPSimulator
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.core.tsp_cim.hemisphere import HemisphereTiming
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader


POLICY = "independent_assumed_conflict_free"


class MEMConcurrencyTest(unittest.TestCase):
    def setUp(self):
        loader = TSPConfigLoader()
        self.config = loader.load_hardware_config()
        self.cim_loader = TSPCIMConfigLoader()
        self.cim_config = self.cim_loader.load_hardware_config()
        self.cim_spec = self.cim_loader.create_tsp_spec(self.cim_config)

    def test_native_read_write_overlap_but_reads_still_contend(self):
        sim = TSPSimulator(TSPSpec(self.config))
        opts = dict(mode="timing", mem_slices=[0], streams=32)
        first = sim.simulate(4, 320, 640, **opts)
        cross = sim.simulate(4, 640, 640, **opts)
        # First K: max(320/320, 2560/320) = 8, not (320+2560)/320 = 9.
        self.assertEqual(first["timing"]["max_row_interval_cycles"], 8)
        # Later K: activation + old partial sum share the read budget.
        self.assertEqual(cross["timing"]["max_row_interval_cycles"], 9)
        for field, expected in (("mem_read_bpc", 18), ("mem_write_bpc", 16)):
            cfg = copy.deepcopy(self.config)
            cfg["parameters"][field].update(value=160, status="assumed")
            other = TSPSimulator(TSPSpec(cfg)).simulate(4, 640, 640, **opts)
            self.assertEqual(other["timing"]["max_row_interval_cycles"], expected)
            self.assertEqual(other["action_counts"], cross["action_counts"])
        self.assertEqual(first["timing"]["derived_services"]["weight_load_peak_bytes_per_cycle"], 320)
        one = sim.simulate(1, 320, 640, **opts)
        self.assertEqual(first["timing"]["total_cycles"] - one["timing"]["total_cycles"], 3*8)

    def test_cim_wave_uses_separate_directional_budgets(self):
        opts = dict(mode="timing", mem_slices=[0], streams=32)
        result = TSPCIMSimulator(self.cim_spec).simulate(4, 640, 640, **opts)
        wave = result["schedule"]["waves"][0]
        self.assertEqual(wave["row_payload_bytes"]["mem_read"], 3200)
        self.assertEqual(wave["row_payload_bytes"]["mem_write"], 5120)
        # Two K tasks: max(10 read cycles, 16 write cycles, 8 compute cycles).
        self.assertEqual(wave["row_interval_cycles"], 16)
        self.assertNotIn("mem_shared", result["timing"]["resources"])
        for field, expected in (("mem_read_bpc", 20), ("mem_write_bpc", 32)):
            cfg = copy.deepcopy(self.cim_config)
            cfg["_native_reference_config"]["parameters"][field].update(value=160, status="assumed")
            other = TSPCIMSimulator(self.cim_loader.create_tsp_spec(cfg)).simulate(4, 640, 640, **opts)
            self.assertEqual(other["schedule"]["waves"][0]["row_interval_cycles"], expected)
            self.assertEqual(other["action_counts"], result["action_counts"])
            for resource in other["timing"]["resources"].values():
                self.assertLessEqual(resource["peak_reserved_bpc"], resource["capacity_bpc"])

    def test_background_load_competes_with_reads_but_not_writes(self):
        plan = {"mem_slices": [0], "streams_per_direction": 32}
        for resource, expected in (("mem_write", 10), ("mem_read", 20)):
            timing = HemisphereTiming(self.cim_spec, plan, set(), "background")
            timing.calendar.reserve(0, 10, {resource: 320}, "other_transfer")
            # These are the actual resources used by the CIM LW scheduler.
            end = timing.calendar.transfer(3200, 0, ("mem_read", "stream_in"), 320, "LW")
            self.assertEqual(end, expected)
            self.assertEqual(timing.calendar.summary()["mem_read"]["peak_reserved_bpc"], 320)

    def test_missing_directional_services_are_not_inferred(self):
        for name in ("mem_read_bpc", "mem_write_bpc"):
            cfg = copy.deepcopy(self.config)
            cfg["parameters"][name].update(value=None, status="unknown")
            sim = TSPSimulator(TSPSpec(cfg))
            self.assertIn(name, sim.simulate(1, 16, 16)["timing"]["missing_parameters"])
            with self.assertRaisesRegex(ValueError, name):
                sim.simulate(1, 16, 16, mode="timing")
            cfg = copy.deepcopy(self.cim_config)
            cfg["_native_reference_config"]["parameters"][name].update(value=None, status="unknown")
            sim = TSPCIMSimulator(self.cim_loader.create_tsp_spec(cfg))
            with self.assertRaisesRegex(ValueError, name):
                sim.simulate(1, 16, 16, mode="timing")

    def test_migration_requires_explicit_conflict_free_policy(self):
        for source, load, old_version in ((self.config, TSPSpec, 6),
                (self.cim_config, self.cim_loader.create_tsp_spec, 5)):
            for mutation in ("old_schema", "missing_policy", "other_policy"):
                cfg = copy.deepcopy(source)
                if mutation == "old_schema":
                    cfg["schema_version"] = old_version
                elif mutation == "missing_policy":
                    del cfg["execution"]["rules"]["mem_read_write"]
                else:
                    cfg["execution"]["rules"]["mem_read_write"] = "shared"
                with self.subTest(mutation=mutation, version=old_version), self.assertRaises(ValueError):
                    load(cfg)
        cfg = copy.deepcopy(self.config)
        cfg["parameters"]["mem_shared_bpc"] = copy.deepcopy(cfg["parameters"]["mem_read_bpc"])
        with self.assertRaisesRegex(ValueError, "Parameter keys mismatch"):
            TSPSpec(cfg)

    def test_reports_disclose_assumption_and_keep_inventory(self):
        spec = TSPSpec(self.config)
        self.assertEqual(len(spec.values), 30)
        self.assertTrue(all(v is not None for v in spec.values.values()))
        self.assertEqual(spec.mem_total_bytes, 220*2**20)
        self.assertEqual(spec.execution["rules"]["mem_read_write"], POLICY)
        self.assertEqual(self.cim_spec.execution["rules"]["mem_read_write"], POLICY)
        for sim in (TSPSimulator(spec), TSPCIMSimulator(self.cim_spec)):
            for mode in ("count", "timing"):
                report = sim.simulate(2, 16, 16, mode=mode)
                self.assertEqual(report["execution_contract"]["rules"]["mem_read_write"], POLICY)
                self.assertEqual(report["execution_contract"]["status"], "assumed")
                self.assertIn("conflict-free bank layout", " ".join(report["limitations"]))
                self.assertNotIn("mem_shared_bpc", report["timing"].get("parameters", {}))


if __name__ == "__main__":
    unittest.main()
