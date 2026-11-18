from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from src.core.ascend import AscendSimulator
from src.hardware.ascend import AscendSpec
from src.utils.workload import parse_workloads
from tests.ascend.support import ROOT, hardware_config, spec, workload

# unittest discovery imports tests/ascend as 'ascend'; load the CLI by its path.
entry_spec = importlib.util.spec_from_file_location("ascend_cli", ROOT / "ascend.py")
ascend = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(ascend)


class SingleCoreTest(unittest.TestCase):
    def test_k_reduction_keeps_output_local_and_exports_once(self):
        r = AscendSimulator(spec()).simulate(workload(16, 256, 16), "timing")
        self.assertEqual(r["counts"]["compute_blocks"], 2)
        self.assertEqual(r["counts"]["cube_mac"], 65536)
        self.assertEqual(r["stages"]["ub_to_hbm"]["bytes"], 1024)
        self.assertEqual(r["stages"]["ub_to_hbm"]["transfers"], 1)
        self.assertEqual(r["timing"]["compute_cycles"], 8)
        self.assertEqual(r["timing"]["total_cycles"], 68)

    def test_tail_padding_is_distinct_from_useful_work(self):
        r = AscendSimulator(spec(False)).simulate(workload(17, 33, 15))
        self.assertEqual(r["counts"]["useful_ops"], 2 * 17 * 33 * 15)
        self.assertEqual(r["counts"]["executed_padded_ops"], 2 * 32 * 64 * 16)
        self.assertEqual(r["stages"]["l2_to_l1"]["bytes"], 3072)
        self.assertEqual(r["stages"]["ub_to_hbm"]["bytes"], 2048)
        self.assertEqual(r["counts"]["logical_output_bytes"], 1020)

    def test_tiny_action_counts_and_missing_metrics(self):
        r = AscendSimulator(spec(False)).simulate(workload(512, 256, 512))
        self.assertEqual(r["counts"]["useful_ops"], 134217728)
        self.assertEqual(r["counts"]["int8_fractals"], 8192)
        self.assertEqual(r["stages"]["l2_to_l1"]["bytes"], 2097152)
        self.assertEqual(r["stages"]["ub_to_hbm"]["bytes"], 1048576)
        self.assertIsNone(r["timing"]["total_cycles"])
        self.assertIsNone(r["energy"]["total_pj"])
        self.assertNotIn("events", r)
        for action, count in r["action_counts"].items():
            self.assertEqual(count["scope"], "core")
            self.assertEqual(count["count"], r["counts"]["cube_mac"] if action == "cube_mac" else r["stages"][{"hbm_read_bytes": "hbm_to_l2", "hbm_write_bytes": "ub_to_hbm", "l2_read_bytes": "l2_to_l1", "l2_fill_bytes": "l2_fill"}.get(action, action)]["bytes"])
        for name, entry in r["memory"].items():
            if name == "hbm":
                self.assertLessEqual(entry["padded_footprint_bytes"], entry["usable_bytes"])
                continue
            self.assertLessEqual(entry["peak_bytes"], entry["capacity_bytes"])

    def test_bandwidth_changes_corresponding_service_and_pipeline_time(self):
        config = hardware_config()
        before = AscendSimulator(AscendSpec(config)).simulate(workload(), "timing")
        config["transfers"]["l1_l0b"]["value"] = 256
        after = AscendSimulator(AscendSpec(config)).simulate(workload(), "timing")
        self.assertEqual(after["stages"]["l1_l0b"]["cycles"], 2)
        self.assertEqual(after["timing"]["total_cycles"] - before["timing"]["total_cycles"], 1)
        self.assertEqual(after["counts"], before["counts"])

    def test_unknown_timing_and_unsupported_precision_fail(self):
        config = hardware_config(False)
        config["chip"]["hbm_bandwidth_bytes_per_second"].update(value=None, status="unknown")
        with self.assertRaisesRegex(ValueError, "Timing parameters missing: chip.hbm_bandwidth_bytes_per_second"):
            AscendSimulator(AscendSpec(config)).simulate(workload(), "timing")
        w = parse_workloads({"gemm": {"M": 16, "K": 32, "N": 16}, "data_format": "BF16"})[0]
        with self.assertRaisesRegex(ValueError, "INT8"):
            AscendSimulator(spec()).simulate(w)

    def test_cli_default_timing_and_explicit_format(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            with redirect_stdout(StringIO()):
                code = ascend.main(["--output", str(target)])
            self.assertEqual(code, 0)
            r = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(r["schema_version"], 12)
            self.assertFalse(r["workload"]["format_overridden"])
            self.assertEqual(r["workload"]["declared_data_format"], "INT8")
            self.assertEqual(r["counts"]["useful_ops"], 134217728)
            self.assertEqual(r["mode"], "timing")
            self.assertEqual(r["timing"]["total_cycles"], 43588)
            self.assertEqual(r["timing"]["policy"], "resource_pipeline")
            self.assertEqual(r["timing"]["service_cycles"]["total_cycles"], 73472)
            self.assertEqual(r["boundary"]["simulated_cores"], 1)
            with redirect_stderr(StringIO()):
                self.assertEqual(ascend.main(["--format", "BF16", "--output", str(target)]), 2)

    def test_cli_missing_timing_does_not_write_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            config = hardware_config(False)
            config["chip"]["hbm_bandwidth_bytes_per_second"].update(value=None, status="unknown")
            hardware = Path(directory) / "missing.json"
            hardware.write_text(json.dumps(config), encoding="utf-8")
            target.write_text("previous result", encoding="utf-8")
            with redirect_stderr(StringIO()):
                code = ascend.main(["--format", "INT8", "--mode", "timing", "--hardware", str(hardware), "--output", str(target)])
            self.assertEqual(code, 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "previous result")

    def test_proxy_hand_calculation_and_count_fields_are_preserved(self):
        sim = AscendSimulator(spec(False))
        count = sim.simulate(workload())
        timed = sim.simulate(workload(), "timing")
        t = timed["timing"]
        self.assertEqual(t["service_cycles"]["by_stage"], dict(compute=1, hbm_to_l2=28, l2_fill=12, l2_to_l1=12, hbm_to_l1=0, l1_l0a=1, l1_l0b=1, l0c_ub=1, ub_to_hbm=28))
        self.assertEqual(t["total_cycles"], 83)
        self.assertEqual(t["time_ns"], 83)
        self.assertEqual(len(t["proxy_parameters"]), 6)
        self.assertEqual(t["derived_services"]["hbm"]["service_bytes_per_cycle"], 37.5)
        for key in ("counts", "action_counts", "memory", "hardware_derived", "parameter_provenance", "strategy"):
            self.assertEqual(timed[key], count[key])
        self.assertEqual(t["missing_parameters"], count["timing"]["missing_parameters"])
        self.assertEqual(timed["parameter_provenance"]["cube.macs_per_cycle"]["value"], 8192)
        self.assertEqual(t["parameters"]["cube.macs_per_cycle"]["selection"], "configured")
        self.assertEqual(t["missing_parameters"], [])

    def test_hbm_bandwidth_and_core_share_drive_both_transfers(self):
        cfg = hardware_config(False)
        before = AscendSimulator(AscendSpec(cfg)).simulate(workload(), "timing")
        cfg["chip"]["hbm_bandwidth_bytes_per_second"]["value"] *= 2
        after = AscendSimulator(AscendSpec(cfg)).simulate(workload(), "timing")
        self.assertEqual(after["timing"]["total_cycles"], 55)
        self.assertEqual(after["timing"]["derived_services"]["hbm"]["service_bytes_per_cycle"], 75)
        self.assertEqual(after["counts"], before["counts"])
        self.assertEqual(after["action_counts"], before["action_counts"])
        cfg["chip"]["architectural_cores"]["value"] *= 2
        restored = AscendSimulator(AscendSpec(cfg)).simulate(workload(), "timing")
        for stage in ("hbm_to_l2", "ub_to_hbm"):
            self.assertEqual(restored["stages"][stage], before["stages"][stage])
        self.assertEqual(restored["timing"]["derived_services"]["l2"]["service_bytes_per_cycle"], 46.875)
        self.assertEqual(restored["memory"]["l2"]["capacity_bytes"], 512 * 1024)
        self.assertEqual(restored["memory"]["hbm"], before["memory"]["hbm"])

    def test_hbm_padded_capacity_is_checked_and_l2_rate_is_consumed(self):
        cfg = hardware_config(False)
        cfg["chip"]["hbm_usable_bytes"]["value"] = 5120
        baseline = AscendSimulator(AscendSpec(cfg)).simulate(workload(17, 33, 15), "timing")
        self.assertEqual(baseline["memory"]["hbm"]["logical_footprint_bytes"], 2076)
        self.assertEqual(baseline["memory"]["hbm"]["padded_footprint_bytes"], 5120)
        self.assertEqual(baseline["action_counts"]["hbm_read_bytes"]["count"], 3072)
        self.assertEqual(baseline["action_counts"]["hbm_write_bytes"]["count"], 2048)
        cfg["chip"]["l2_bytes_per_cycle"].update(value=1, status="assumed")
        changed = AscendSimulator(AscendSpec(cfg)).simulate(workload(17, 33, 15), "timing")
        self.assertGreater(changed["timing"]["total_cycles"], baseline["timing"]["total_cycles"])
        self.assertEqual(changed["action_counts"], baseline["action_counts"])
        cfg["chip"]["hbm_usable_bytes"]["value"] = 5119
        for mode in ("count", "timing"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Padded A/B/C footprint"):
                AscendSimulator(AscendSpec(cfg)).simulate(workload(17, 33, 15), mode)
        cfg["chip"]["hbm_usable_bytes"].update(value=None, status="unknown")
        with self.assertRaisesRegex(ValueError, "Counting parameters missing: chip.hbm_usable_bytes"):
            AscendSimulator(AscendSpec(cfg)).simulate(workload())

    def test_hbm_clock_conversion_and_missing_timing_allow_counts(self):
        cfg = hardware_config(False)
        before = AscendSimulator(AscendSpec(cfg)).simulate(workload(), "timing")
        cfg["clock_freq_ghz"]["value"] = 2
        after = AscendSimulator(AscendSpec(cfg)).simulate(workload(), "timing")
        self.assertEqual(after["timing"]["derived_services"]["hbm"]["service_bytes_per_cycle"], 18.75)
        self.assertEqual(after["timing"]["total_cycles"], 138)
        self.assertEqual(after["timing"]["time_ns"], 69)
        for section, key in ((None, "clock_freq_ghz"), ("chip", "hbm_bandwidth_bytes_per_second")):
            config = hardware_config(False)
            (config if section is None else config[section])[key].update(value=None, status="unknown")
            count = AscendSimulator(AscendSpec(config)).simulate(workload())
            self.assertEqual(count["counts"], before["counts"])
            with self.assertRaisesRegex(ValueError, key):
                AscendSimulator(AscendSpec(config)).simulate(workload(), "timing")

    def test_cli_proxy_timing_runs_reference_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "timing.json"
            with redirect_stdout(StringIO()):
                code = ascend.main(["--workload", "tiny_gemm.json", "--format", "INT8",
                                    "--mode", "timing", "--output", str(target)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["timing"]["total_cycles"], 43588)

    def test_removed_experiment_and_scheduling_flags_fail_explicitly(self):
        for args in (["--cores", "32"], ["--buffers", "3"], ["--memory-mode", "cached"],
                     ["--experiment", "ascend_scaling.json"], ["--energy", "old.json"]):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                ascend.parse_args(args)
            self.assertEqual(raised.exception.code, 2)
