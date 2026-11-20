import copy
import unittest

from src.core.ascend import AscendSimulator
from src.core.ascend.cache import PanelCache
from src.hardware.ascend import AscendSpec
from tests.ascend.support import hardware_config, workload
from tests.ascend import test_pipeline


class L2Test(unittest.TestCase):
    def test_byte_capacity_lru_and_oversize_does_not_pollute(self):
        c = PanelCache(8)
        self.assertEqual([c.access(k, 4) for k in ("a", "b", "a", "c", "b")],
                         ["miss", "miss", "hit", "miss", "miss"])
        self.assertEqual(c.summary()["evictions"], 2)
        self.assertEqual(c.access("huge", 9), "bypass")
        self.assertEqual(c.access("b", 4), "hit")
        self.assertEqual(c.summary()["peak_resident_bytes"], 8)
        self.assertEqual(c.summary()["bypass_bytes"], 9)
        c = PanelCache(10)
        c.access("a", 6)
        c.access("b", 4)
        c.access("a", 6)
        c.access("c", 7)
        self.assertEqual(c.summary()["evictions"], 2)
        self.assertEqual(c.used, 7)

    def test_cold_reuse_eviction_and_bypass_conservation(self):
        reports = []
        for capacity, expected_hits, expected_hbm in ((1024, 1, 1536), (512, 0, 2048), (256, 0, 2048)):
            cfg = hardware_config()
            cfg["chip"]["l2_bytes"]["value"] = capacity * 32
            sim = AscendSimulator(AscendSpec(cfg))
            options = dict(tile=(16, 32, 16), l1_k=32)
            count = sim.simulate(workload(32, 32, 16), **options)
            r = sim.simulate(workload(32, 32, 16), "timing", trace=True, **options)
            again = sim.simulate(workload(32, 32, 16), "timing", **options)
            c, a = r["cache"], r["action_counts"]
            self.assertEqual(c, again["cache"])  # no warm state across calls
            self.assertEqual(c, count["cache"])
            self.assertEqual(a, count["action_counts"])
            self.assertEqual(c["hits"], expected_hits)
            self.assertEqual(a["hbm_read_bytes"]["count"], expected_hbm)
            self.assertEqual(c["requested_bytes"], 2048)
            self.assertEqual(c["hit_bytes"] + c["miss_bytes"], c["requested_bytes"])
            self.assertEqual(c["fill_bytes"] + c["bypass_bytes"], c["miss_bytes"])
            self.assertEqual(c["read_bytes"] + c["bypass_bytes"], c["requested_bytes"])
            self.assertEqual(a["hbm_write_bytes"]["count"], 2048)  # C always bypasses
            self.assertLessEqual(c["peak_resident_bytes"], capacity)
            test_pipeline.PipelineTest().check_trace(r)
            reports.append(r)
        self.assertEqual(reports[0]["counts"], reports[2]["counts"])
        self.assertEqual(reports[2]["cache"]["fill_bytes"], 0)
        self.assertEqual(reports[2]["cache"]["read_bytes"], 0)

    def test_miss_fill_read_order_and_reuse_are_time_consistent(self):
        cfg = hardware_config()
        cfg["chip"]["l2_bytes"]["value"] = 1024 * 32
        r = AscendSimulator(AscendSpec(cfg)).simulate(workload(32, 32, 16), "timing", (16,32,16),32,trace=True)
        events = r["timing"]["schedule"]["events"]
        resident = PanelCache(1024)
        last_read_end = 0
        for read in (e for e in events if e["stage"] == "l2_to_l1"):
            state = resident.access(tuple(read["cache_key"]), read["bytes"])
            self.assertEqual(state, read["cache_status"])
            path = [e for e in events if e.get("panel") == read["panel"] and e.get("operand") == read["operand"]
                    and e["stage"] in ("hbm_to_l2", "l2_fill", "l2_to_l1")]
            self.assertGreaterEqual(path[0]["start_cycle"], last_read_end)
            self.assertEqual([e["stage"] for e in path],
                             ["hbm_to_l2", "l2_fill", "l2_to_l1"] if state == "miss" else ["l2_to_l1"])
            for before, after in zip(path, path[1:]):
                self.assertLessEqual(before["end_cycle"], after["start_cycle"])
            last_read_end = read["end_cycle"]
        self.assertEqual(r["timing"]["schedule"]["resource_busy_cycles"]["l2"], 7)

    def test_reference_shares_hits_and_rate_sensitivity(self):
        cfg = hardware_config(False)
        sim = AscendSimulator(AscendSpec(cfg))
        r = sim.simulate(workload(512,256,512), "timing")
        self.assertEqual(r["memory"]["l2"]["capacity_bytes"], 1024**2)
        self.assertEqual(r["timing"]["derived_services"]["l2"]["service_bytes_per_cycle"], 93.75)
        self.assertEqual((r["cache"]["hits"], r["cache"]["misses"]), (112,16))
        self.assertEqual(r["cache"]["byte_hit_rate"], 0.875)
        self.assertEqual(r["cache"]["fill_bytes"], 256*1024)
        cfg["chip"]["l2_bytes_per_cycle"]["value"] /= 10
        slower = AscendSimulator(AscendSpec(cfg)).simulate(workload(512,256,512), "timing")
        self.assertEqual(r["action_counts"], slower["action_counts"])
        self.assertGreater(slower["timing"]["total_cycles"], r["timing"]["total_cycles"])
        self.assertTrue(all(p["value"] is not None for p in r["parameter_provenance"].values()))

    def test_missing_inputs_and_unsupported_policies_fail_explicitly(self):
        cfg = hardware_config(False)
        cfg["chip"]["l2_bytes_per_cycle"].update(value=None,status="unknown")
        sim = AscendSimulator(AscendSpec(cfg))
        sim.simulate(workload())
        with self.assertRaisesRegex(ValueError, "chip.l2_bytes_per_cycle"):
            sim.simulate(workload(), "timing")
        cfg["chip"]["l2_bytes"].update(value=None,status="unknown")
        with self.assertRaisesRegex(ValueError, "chip.l2_bytes"):
            AscendSimulator(AscendSpec(cfg)).simulate(workload())
        base = hardware_config(False)
        for key in base["execution"]["l2"]:
            cfg = copy.deepcopy(base)
            cfg["execution"]["l2"][key] = "unsupported"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "execution.l2"):
                AscendSpec(cfg)
