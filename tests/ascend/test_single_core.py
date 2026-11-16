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
    def test_hand_calculated_minimum_tile_has_no_extra_accumulate_cycle(self):
        r = AscendSimulator(spec()).simulate(workload(), "timing")
        self.assertEqual(r["counts"]["cube_mac"], 8192)
        self.assertEqual(r["timing"]["compute_cycles"], 1)
        # GM A/B 2 + L1->L0A 1 + L1->L0B 1 + export 2 + store 2 + compute 1.
        self.assertEqual(r["timing"]["total_cycles"], 9)
        self.assertEqual(r["timing"]["time_ns"], 9)
        self.assertAlmostEqual(r["timing"]["throughput_gops"], 16384 / 9)
        self.assertEqual(r["boundary"]["scope"], "single_core")

    def test_k_reduction_keeps_output_local_and_exports_once(self):
        r = AscendSimulator(spec()).simulate(workload(16, 256, 16), "timing")
        self.assertEqual(r["counts"]["compute_blocks"], 2)
        self.assertEqual(r["counts"]["cube_mac"], 65536)
        self.assertEqual(r["stages"]["ub_gm"]["bytes"], 1024)
        self.assertEqual(r["stages"]["ub_gm"]["transfers"], 1)
        self.assertEqual(r["timing"]["compute_cycles"], 8)
        self.assertEqual(r["timing"]["total_cycles"], 44)

    def test_tail_padding_is_distinct_from_useful_work(self):
        r = AscendSimulator(spec(False)).simulate(workload(17, 33, 15))
        self.assertEqual(r["counts"]["useful_ops"], 2 * 17 * 33 * 15)
        self.assertEqual(r["counts"]["executed_padded_ops"], 2 * 32 * 64 * 16)
        self.assertEqual(r["stages"]["gm_l1"]["bytes"], 3072)
        self.assertEqual(r["stages"]["ub_gm"]["bytes"], 2048)
        self.assertEqual(r["counts"]["logical_output_bytes"], 1020)

    def test_tiny_action_counts_and_missing_metrics(self):
        r = AscendSimulator(spec(False)).simulate(workload(512, 256, 512))
        self.assertEqual(r["counts"]["useful_ops"], 134217728)
        self.assertEqual(r["counts"]["int8_fractals"], 8192)
        self.assertEqual(r["stages"]["gm_l1"]["bytes"], 2097152)
        self.assertEqual(r["stages"]["ub_gm"]["bytes"], 1048576)
        self.assertIsNone(r["timing"]["total_cycles"])
        self.assertIsNone(r["energy"]["total_pj"])
        self.assertNotIn("events", r)
        for action, count in r["action_counts"].items():
            self.assertEqual(count["scope"], "core")
            self.assertEqual(count["count"], r["counts"]["cube_mac"] if action == "cube_mac" else r["stages"][action]["bytes"])
        for entry in r["memory"].values():
            self.assertLessEqual(entry["peak_bytes"], entry["capacity_bytes"])

    def test_bandwidth_changes_only_the_corresponding_serial_stage(self):
        config = hardware_config()
        before = AscendSimulator(AscendSpec(config)).simulate(workload(), "timing")
        config["transfers"]["gm_l1"]["value"] = 256
        after = AscendSimulator(AscendSpec(config)).simulate(workload(), "timing")
        self.assertEqual(after["stages"]["gm_l1"]["cycles"], 4)
        self.assertEqual(after["timing"]["total_cycles"] - before["timing"]["total_cycles"], 2)
        self.assertEqual(after["counts"], before["counts"])

    def test_unknown_timing_and_unsupported_precision_fail(self):
        config = hardware_config(False)
        config["chip"]["hbm_bytes_per_cycle"].update(value=None, status="unknown")
        with self.assertRaisesRegex(ValueError, "timing proxy requires"):
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
            self.assertEqual(r["schema_version"], 3)
            self.assertFalse(r["workload"]["format_overridden"])
            self.assertEqual(r["workload"]["declared_data_format"], "INT8")
            self.assertEqual(r["counts"]["useful_ops"], 134217728)
            self.assertEqual(r["mode"], "timing")
            self.assertEqual(r["timing"]["total_cycles"], 97216)
            self.assertEqual(r["boundary"]["simulated_cores"], 1)
            with redirect_stderr(StringIO()):
                self.assertEqual(ascend.main(["--format", "BF16", "--output", str(target)]), 2)

    def test_cli_missing_timing_does_not_write_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            config = hardware_config(False)
            config["chip"]["hbm_bytes_per_cycle"].update(value=None, status="unknown")
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
        self.assertEqual(t["stages_cycles"], dict(compute=1, gm_l1=28, l1_l0a=1, l1_l0b=1, l0c_ub=1, ub_gm=28))
        self.assertEqual(t["total_cycles"], 60)
        self.assertEqual(t["time_ns"], 60)
        self.assertEqual(len(t["proxy_parameters"]), 6)
        self.assertEqual(t["parameters"]["transfers.gm_l1"]["value"], 37.5)
        for key in ("counts", "action_counts", "memory", "hardware_derived", "parameter_provenance", "strategy"):
            self.assertEqual(timed[key], count[key])
        self.assertEqual(t["missing_parameters"], count["timing"]["missing_parameters"])
        self.assertIsNone(timed["parameter_provenance"]["cube.macs_per_cycle"]["value"])

    def test_configured_service_wins_and_only_needed_proxy_bases_required(self):
        cfg = hardware_config(False)
        cfg["transfers"]["gm_l1"].update(value=512, status="assumed")
        r = AscendSimulator(AscendSpec(cfg)).simulate(workload(), "timing")
        self.assertEqual(r["timing"]["total_cycles"], 34)
        self.assertEqual(r["timing"]["parameters"]["transfers.gm_l1"]["selection"], "configured")
        cfg["transfers"]["ub_gm"].update(value=512, status="assumed")
        cfg["chip"]["hbm_bytes_per_cycle"].update(value=None, status="unknown")
        r = AscendSimulator(AscendSpec(cfg)).simulate(workload(), "timing")
        self.assertEqual(r["timing"]["total_cycles"], 8)

    def test_proxy_chip_share_tail_and_unknown_clock(self):
        cfg = hardware_config(False)
        cfg["chip"]["architectural_cores"]["value"] = 16
        cfg["clock_freq_ghz"].update(value=None, status="unknown")
        r = AscendSimulator(AscendSpec(cfg)).simulate(workload(1, 1, 1), "timing")
        self.assertEqual(r["timing"]["total_cycles"], 32)
        self.assertIsNone(r["timing"]["time_ns"])
        self.assertEqual(r["counts"]["useful_ops"], 2)
        self.assertEqual(r["boundary"]["simulated_cores"], 1)

    def test_cli_proxy_timing_runs_reference_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "timing.json"
            with redirect_stdout(StringIO()):
                code = ascend.main(["--workload", "tiny_gemm.json", "--format", "INT8",
                                    "--mode", "timing", "--output", str(target)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["timing"]["total_cycles"], 97216)

    def test_removed_experiment_and_scheduling_flags_fail_explicitly(self):
        for args in (["--cores", "32"], ["--buffers", "2"], ["--memory-mode", "cached"],
                     ["--experiment", "ascend_scaling.json"], ["--energy", "old.json"]):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                ascend.parse_args(args)
            self.assertEqual(raised.exception.code, 2)
