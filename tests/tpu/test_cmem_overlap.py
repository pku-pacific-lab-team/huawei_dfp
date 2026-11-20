"""CMEM overlap: hand timelines, resource conflicts, capacity and aggregation."""
import random
import unittest

from src.core.tpu import TPUSimulator
from src.core.tpu.cmem import make_cmem_plan, services
from src.core.tpu.cmem_pipeline import cmem_pipeline_cycles
from src.core.tpu.tiling import make_plan
from src.hardware.tpu import TPUSpec
from tests.tpu.support import hardware_config, spec, workload
from tests.tpu.test_cmem import small_spec


def configured_small(write_bpc=4, startup=0, capacity=None):
    base = small_spec()
    config = hardware_config(cmem_mode="tiled")
    for name, value in base.values.items():
        config["parameters"][name]["value"] = value
    config["execution"] = base.execution
    config["parameters"]["cmem_write_bandwidth_bytes_per_second"]["value"] = write_bpc*2e9
    if capacity is not None:
        config["parameters"]["cmem_usable_bytes"]["value"] = capacity
    config["execution"]["fixed_latency_cycles"]["transfer_startup"] = startup
    return TPUSpec(config)


def run_schedule(hardware, task, cmem_buffers=2, input_slots=2, events=None, accelerate=True):
    plan = make_plan(task, hardware.values["vmem_usable_bytes"], hardware.values["hbm_usable_bytes"], (2, 2, 2))
    cmem = make_cmem_plan(hardware, task, plan, "tiled", (2, 2), cmem_buffers)
    return cmem_pipeline_cycles(hardware, task, plan, input_slots, cmem, services(hardware),
                                events=events, accelerate=accelerate)


class CMEMOverlapTest(unittest.TestCase):
    def test_double_buffer_hand_timeline_and_shared_hbm(self):
        hardware, task = configured_small(write_bpc=1), workload(8, 3, 4)
        events = []
        self.assertEqual(run_schedule(hardware, task, events=events), 176)
        # Two 24-cycle fills; the second overlaps the first 16-cycle body,
        # but blocks its 2-cycle output export until cycle 48, not cycle 40.
        fills = [e for e in events if e["kind"] == "cmem_fill"]
        exports = [e for e in events if e["kind"] == "output_export"]
        self.assertEqual([(e["start"], e["end"]) for e in fills], [(0, 24), (24, 48)])
        self.assertEqual((exports[0]["start"], exports[0]["end"]), (48, 50))
        self.assertEqual(run_schedule(hardware, task, cmem_buffers=1), 192)
        r = TPUSimulator(hardware).simulate(task, mode="timing")
        self.assertEqual(r["timing"]["total_cycles"], 176)
        self.assertEqual(r["timing"]["serialized_cmem_outer_cycles"], 192)
        self.assertEqual(r["timing"]["cmem_outer_overlap_saved_cycles"], 16)
        self.assertEqual(r["timing"]["resource_busy_cycles"]["hbm"], 48+16)

    def test_single_slot_released_after_last_read_before_compute_finishes(self):
        events = []
        self.assertEqual(run_schedule(small_spec(), workload(8, 3, 4), cmem_buffers=1, events=events), 150)
        fills = [e for e in events if e["kind"] == "cmem_fill"]
        release = next(e for e in events if e["kind"] == "cmem_release")
        self.assertEqual(release["end"], 63)
        self.assertEqual((fills[1]["start"], fills[1]["end"]), (63, 69))
        # Last body runs 60..76; its input is no longer read after cycle 63.
        self.assertLess(fills[1]["end"], 76)

    def test_trace_dependencies_and_aggregated_schedule_agree_with_tails(self):
        rng = random.Random(73)
        for trial in range(24):
            hardware = configured_small(write_bpc=rng.choice((0.25, 1, 4, 8)), startup=rng.randrange(4))
            task = workload(rng.randrange(25, 46), rng.randrange(1, 9), rng.randrange(29, 50))
            slots, inputs, events = rng.choice((1, 2)), rng.choice((1, 2)), []
            with self.subTest(trial=trial, shape=task.shape, slots=slots, inputs=inputs):
                exact = run_schedule(hardware, task, slots, inputs, events=events)
                self.assertEqual(run_schedule(hardware, task, slots, inputs), exact)
                self.assertEqual(run_schedule(hardware, task, slots, inputs, accelerate=False), exact)
                hbm = sorted((e for e in events if e["kind"] in ("cmem_fill", "output_export")), key=lambda e:e["start"])
                self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(hbm, hbm[1:])))
                fills = {e["outer"]:e for e in events if e["kind"] == "cmem_fill"}
                releases = {e["outer"]:e for e in events if e["kind"] == "cmem_release"}
                for outer, fill in fills.items():
                    if outer >= slots:
                        self.assertGreaterEqual(fill["start"], releases[outer-slots]["end"])
                for e in events:
                    if e["kind"] == "inner_body":
                        self.assertGreaterEqual(e["start"], fills[e["outer"]]["end"])
                        self.assertLessEqual(e["last_input_read_end"], e["end"])
                report = TPUSimulator(hardware).simulate(task, mode="timing", cmem_buffers=slots, buffers=inputs)
                timing = report["timing"]
                self.assertEqual(timing["total_cycles"], exact)
                self.assertTrue(all(0 <= b <= exact for b in timing["resource_busy_cycles"].values()))
                self.assertGreaterEqual(timing["cmem_outer_overlap_saved_cycles"], 0)

    def test_capacity_preserves_reuse_and_counts_while_reducing_slots(self):
        task = workload(8, 3, 4)
        reports = [TPUSimulator(configured_small(capacity=c)).simulate(task, mode="timing")
                   for c in (95, 96)]
        for r, slots in zip(reports, (1, 2)):
            cmem = r["memory"]["cmem"]
            self.assertEqual(cmem["selected_groups_mn"], [2, 2])
            self.assertEqual(cmem["single_buffer_bytes"], 24)
            self.assertEqual(cmem["buffer_slots"], slots)
            self.assertEqual(cmem["allocated_bytes"], slots*24)
            self.assertLessEqual(cmem["allocated_bytes"], cmem["usable_bytes"])
        self.assertEqual(reports[0]["counts"], reports[1]["counts"])
        self.assertEqual(reports[0]["action_counts"], reports[1]["action_counts"])
        sim = TPUSimulator(configured_small())
        self.assertEqual(sim.simulate(task, mode="timing", schedule="serial")["timing"]["total_cycles"], 164)
        self.assertEqual(sim.simulate(workload(4, 3, 4))["memory"]["cmem"]["buffer_slots"], 1)

    def test_billion_panels_and_two_dimensional_tail_grid_do_not_expand_events(self):
        sim = TPUSimulator(spec(cmem_mode="tiled"))
        for m, n, groups in ((1, 10**9, (1, 1)), (10**9, 1, (1, 1)), (50001, 50003, (4, 4))):
            r = sim.simulate(workload(m, 1, n), tile=(1, 1, 1), cmem_groups_mn=groups, mode="timing")
            # 1 cold-fill cycle; each 1x1 output costs 259 cycles. All later
            # fills fit within compute. This also checks row/column tail skips.
            self.assertEqual(r["timing"]["total_cycles"], 1+259*m*n)
            self.assertNotIn("events", r["timing"])

    def test_invalid_slot_configuration_and_override_rejected(self):
        for value in (0, 3, True, 1.5, "2"):
            with self.subTest(value=value):
                config = hardware_config(cmem_mode="tiled")
                config["execution"]["cmem_buffer_slots"] = value
                with self.assertRaisesRegex(ValueError, "cmem_buffer_slots"):
                    TPUSpec(config)
                with self.assertRaisesRegex(ValueError, "cmem_buffers"):
                    TPUSimulator(spec()).simulate(workload(), cmem_buffers=value)
