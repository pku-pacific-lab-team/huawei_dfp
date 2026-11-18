from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import json
from pathlib import Path
import random
import tempfile
import unittest

from src.core.ascend import AscendSimulator
from src.core.ascend.schedule import Schedule
from src.hardware.ascend import AscendSpec
from tests.ascend.support import hardware_config, spec, workload
from tests.ascend.test_single_core import ascend


class PipelineTest(unittest.TestCase):
    def check_trace(self, report):
        timing = report["timing"]
        schedule = timing["schedule"]
        events = schedule["events"]
        resources, stages = defaultdict(list), defaultdict(int)
        inputs, local, computes, exports = {}, {}, defaultdict(list), {}
        for event in events:
            start, end = event["start_cycle"], event["end_cycle"]
            self.assertGreater(end, start)
            for predecessor in event["dependencies"]:
                self.assertLessEqual(events[predecessor]["end_cycle"], start)
            for resource in event["resources"]:
                resources[resource].append((start, end))
            stages[event["stage"]] += end - start
            if event["stage"] in {"hbm_to_l1", "l2_to_l1"}:
                inputs[event["panel"], event["operand"]] = event
            elif event["stage"] in {"l1_l0a", "l1_l0b"}:
                operand = "A" if event["stage"] == "l1_l0a" else "B"
                self.assertLessEqual(inputs[event["panel"], operand]["end_cycle"], start)
                local[event["block"], operand] = event
            elif event["stage"] == "compute":
                for operand in ("A", "B"):
                    self.assertLessEqual(local[event["block"], operand]["end_cycle"], start)
                computes[event["tile"]].append(event)
            elif event["stage"] == "l0c_ub":
                self.assertLessEqual(max(c["end_cycle"] for c in computes[event["tile"]]), start)
                exports[event["chunk"]] = event
            elif event["stage"] == "ub_to_hbm":
                self.assertLessEqual(exports[event["chunk"]]["end_cycle"], start)
        for resource, intervals in resources.items():
            intervals.sort()
            for left, right in zip(intervals, intervals[1:]):
                self.assertLessEqual(left[1], right[0], resource)
            self.assertEqual(sum(b - a for a, b in intervals), schedule["resource_busy_cycles"][resource])
        self.assertEqual(dict(stages), {k: v for k, v in timing["service_cycles"]["by_stage"].items() if v})
        self.assertEqual(max(e["end_cycle"] for e in events), timing["total_cycles"])
        self.assertLessEqual(timing["total_cycles"], timing["service_cycles"]["total_cycles"])
        self.assertGreaterEqual(timing["total_cycles"], max(schedule["resource_busy_cycles"].values()))
        slots = defaultdict(list)
        for lifetime in schedule["buffer_lifetimes"]:
            name, slot = lifetime["buffer"], lifetime["slot"]
            slots[name, slot].append((lifetime["start_cycle"], lifetime["end_cycle"]))
            self.assertLess(slot, report["memory"][name]["slots"])
            self.assertLessEqual(lifetime["bytes"], report["memory"][name]["slot_bytes"])
        for key, intervals in slots.items():
            intervals.sort()
            for left, right in zip(intervals, intervals[1:]):
                self.assertLessEqual(left[1], right[0], key)
        for name, peak in schedule["buffer_live_peak_bytes"].items():
            self.assertLessEqual(peak, report["memory"][name]["allocated_bytes"])
            self.assertLessEqual(report["memory"][name]["allocated_bytes"], report["memory"][name]["capacity_bytes"])

    def test_hand_calculated_pipeline_and_double_buffer(self):
        sim = AscendSimulator(spec())
        tiny = sim.simulate(workload(), "timing", trace=True)
        # Each cold 512-byte input: HBM 1 + fill 1 + L2 read 1 cycle.
        # A->L0 overlaps the B miss; C export/write each need 2 cycles.
        self.assertEqual(tiny["timing"]["total_cycles"], 12)
        self.assertEqual(tiny["timing"]["compute_cycles"], 1)
        self.assertEqual(tiny["timing"]["service_cycles"]["total_cycles"], 13)
        single = sim.simulate(workload(32, 32, 16), "timing", (16, 32, 16), 32, buffers=1, trace=True)
        double = sim.simulate(workload(32, 32, 16), "timing", (16, 32, 16), 32, buffers=2, trace=True)
        # Second tile reuses B from L2: its input path is 3 + 1 cycles.
        # One local slot inserts a one-cycle wait before the next panel.
        self.assertEqual([r["timing"]["total_cycles"] for r in (single, double)], [17, 16])
        for report in (tiny, single, double):
            self.check_trace(report)
        for key in ("counts", "action_counts", "stages"):
            self.assertEqual(single[key], double[key])

    def test_capacity_fallback_keeps_tiles_and_counts(self):
        cfg = hardware_config()
        for name, size in {"L1": 1024, "L0A": 512, "L0B": 512, "L0C": 1024, "UB": 1024}.items():
            cfg["memory"][name]["usable_bytes"]["value"] = size
        sim = AscendSimulator(AscendSpec(cfg))
        one = sim.simulate(workload(32, 32, 16), "timing", (16, 32, 16), 32, buffers=1)
        two = sim.simulate(workload(32, 32, 16), "timing", (16, 32, 16), 32, buffers=2, trace=True)
        self.assertEqual(two["timing"]["total_cycles"], 17)
        self.assertEqual(two["strategy"]["selected"], one["strategy"]["selected"])
        self.assertEqual(two["counts"], one["counts"])
        self.assertTrue(all(two["memory"][name]["slots"] == 1 for name in cfg["memory"]))
        self.check_trace(two)

    def test_k_panels_tail_and_segmented_output_lifetimes(self):
        cfg = hardware_config()
        cfg["memory"]["UB"]["usable_bytes"]["value"] = 1024
        r = AscendSimulator(AscendSpec(cfg)).simulate(workload(33, 97, 31), "timing", (32, 32, 16), 64, trace=True)
        events = r["timing"]["schedule"]["events"]
        self.assertEqual(r["counts"]["output_tiles"], 4)
        self.assertEqual(r["stages"]["ub_to_hbm"]["bytes"], 4 * 48 * 32)
        for tile in range(4):
            computes = [e for e in events if e["tile"] == tile and e["stage"] == "compute"]
            exports = [e for e in events if e["tile"] == tile and e["stage"] == "l0c_ub"]
            self.assertEqual(len(computes), 4)
            self.assertGreaterEqual(min(e["start_cycle"] for e in exports), max(e["end_cycle"] for e in computes))
        self.check_trace(r)

    def test_ready_prefetch_does_not_wait_for_future_output(self):
        q = Schedule()
        a = q.add("hbm_to_l1", 2)
        c = q.add("compute", 20, [a])
        q.add("ub_to_hbm", 2, [c])
        prefetch = q.add("hbm_to_l1", 2)
        result = q.run(trace=True)
        self.assertEqual(result["events"][prefetch]["start_cycle"], 2)
        self.assertEqual(result["total_cycles"], 24)

    def test_reference_counts_unchanged_and_stage_sum_is_not_makespan(self):
        sim = AscendSimulator(spec(False))
        count = sim.simulate(workload(512, 256, 512))
        pipe = sim.simulate(workload(512, 256, 512), "timing", trace=True)
        self.assertEqual(pipe["timing"]["total_cycles"], 43588)
        self.assertEqual(pipe["timing"]["service_cycles"]["total_cycles"], 73472)
        self.assertEqual(pipe["timing"]["schedule"]["resource_busy_cycles"]["hbm"], 34960)
        for key in ("counts", "action_counts", "strategy", "memory", "parameter_provenance"):
            self.assertEqual(count[key], pipe[key])
        self.check_trace(pipe)

    def test_pipeline_uses_configured_services_and_trace_is_optional(self):
        cfg = hardware_config(False)
        before = AscendSimulator(AscendSpec(cfg)).simulate(workload(32, 64, 16), "timing", (16, 32, 16), 32)
        self.assertNotIn("events", before["timing"]["schedule"])
        cfg["chip"]["hbm_bandwidth_bytes_per_second"]["value"] *= 2
        after = AscendSimulator(AscendSpec(cfg)).simulate(workload(32, 64, 16), "timing", (16, 32, 16), 32, trace=True)
        self.assertEqual(after["counts"], before["counts"])
        self.assertEqual(after["action_counts"], before["action_counts"])
        self.assertEqual(after["timing"]["schedule"]["resource_busy_cycles"]["hbm"] * 2,
                         before["timing"]["schedule"]["resource_busy_cycles"]["hbm"])
        self.assertLess(after["timing"]["total_cycles"], before["timing"]["total_cycles"])
        self.check_trace(after)

    def test_random_small_schedules_preserve_dependencies_resources_and_storage(self):
        rng = random.Random(910)
        for _ in range(24):
            cfg = hardware_config()
            for name in cfg["memory"]:
                minimum = 512 if name in ("L0A", "L0B") else 1024
                cfg["memory"][name]["usable_bytes"]["value"] = minimum * rng.choice((1, 2, 4, 8))
            cfg["cube"]["macs_per_cycle"]["value"] = rng.choice((1024, 8192, 16384))
            for route in cfg["transfers"].values():
                route["value"] = rng.choice((128, 512, 1024))
            shape = (rng.randint(1, 49), rng.randint(1, 129), rng.randint(1, 49))
            r = AscendSimulator(AscendSpec(cfg)).simulate(workload(*shape), "timing", (32, 64, 32), 96,
                                                       buffers=rng.choice((1, 2)), trace=True)
            with self.subTest(shape=shape):
                self.check_trace(r)

    def test_cli_trace_and_removed_policy_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(ascend.main(["--trace", "--output", str(target)]), 0)
            r = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(r["timing"]["total_cycles"], 43588)
            self.assertNotIn("serial_total_cycles", r["timing"])
            self.assertNotIn("speedup_vs_serial", r["timing"])
            self.check_trace(r)
            target.write_text("previous report", encoding="utf-8")
            with redirect_stderr(StringIO()):
                self.assertEqual(ascend.main(["--mode", "count", "--trace", "--output", str(target)]), 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "previous report")
        for policy in ("serial", "pipeline"):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                ascend.parse_args(["--timing-policy", policy])
            with self.assertRaises(TypeError):
                AscendSimulator(spec()).simulate(workload(), timing_policy=policy)
        for kw in ({"buffers": 0}, {"buffers": True}, {"buffers": 3}, {"trace": True}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                AscendSimulator(spec()).simulate(workload(), **kw)
