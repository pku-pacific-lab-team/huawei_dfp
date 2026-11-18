from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.core.tpu import TPUSimulator
from src.core.tpu.tiling import make_plan
from src.hardware.tpu import TPUSpec
from src.utils.workload import parse_workloads
from tests.tpu.support import ROOT, hardware_config, spec, workload

entry_spec = importlib.util.spec_from_file_location("tpu_cli", ROOT / "tpu.py")
tpu = importlib.util.module_from_spec(entry_spec)
entry_spec.loader.exec_module(tpu)


class TPUCountTest(unittest.TestCase):
    def test_hand_calculated_single_block(self):
        r = TPUSimulator(spec()).simulate(workload())
        self.assertEqual(r["counts"]["useful_macs"], 8192)
        self.assertEqual(r["counts"]["useful_ops"], 16384)
        self.assertEqual(r["counts"]["compute_blocks"], 1)
        self.assertEqual(r["counts"]["input_transfer_bytes"], 1024)
        self.assertEqual(r["counts"]["logical_output_bytes"], 1024)
        self.assertEqual(r["counts"]["psum_read_bytes"], 0)
        self.assertEqual(r["counts"]["psum_write_bytes"], 0)
        self.assertEqual(r["memory"]["vmem"]["peak_logical_bytes"], 2048)
        self.assertEqual(r["boundary"]["scope"], "single_tensorcore")
        self.assertEqual(r["schema_version"], 13)
        self.assertEqual(r["boundary"]["hardware_scope"], "chip")
        self.assertEqual(r["boundary"]["counting_scope"], "single_tensorcore")
        self.assertEqual(r["boundary"]["simulated_tensorcores"], 1)
        self.assertFalse(r["boundary"]["chip_execution_included"])
        self.assertEqual(r["hardware_inventory"]["chip"]["tensorcores"], 2)
        self.assertEqual(r["hardware_inventory"]["per_tensorcore"]["mxus"], 4)
        self.assertEqual(r["hardware_hierarchy"]["chip"]["mxus_per_chip"], 8)

    def test_hand_calculated_tails_and_k_accumulation(self):
        # 3x5 times 5x7, split into M=2+1, K=3+2, N=4+3.
        r = TPUSimulator(spec()).simulate(workload(3, 5, 7), tile=(2, 3, 4))
        c, a = r["counts"], r["action_counts"]
        self.assertEqual(c["useful_macs"], 105)
        self.assertEqual(c["compute_blocks"], 8)
        self.assertEqual(c["output_blocks"], 4)
        self.assertEqual(c["input_a_transfer_bytes"], 30)  # A is loaded for two N blocks.
        self.assertEqual(c["input_b_transfer_bytes"], 70)  # B is loaded for two M blocks.
        self.assertEqual(c["psum_read_bytes"], 84)
        self.assertEqual(c["psum_write_bytes"], 84)
        for name in ("vmem_psum_read_bytes", "vmem_psum_write_bytes", "vmem_final_write_bytes",
                     "vmem_export_read_bytes", "hbm_write_bytes"):
            self.assertEqual(a[name]["count"], 84, name)
        self.assertEqual(r["memory"]["vmem"]["peak_logical_bytes"], 50)
        self.assertEqual(r["memory"]["hbm"]["logical_footprint_bytes"], 134)
        self.assertEqual(c["executed_padded_ops"], 2 * 5 * 256 * 256)

    def test_k_blocking_changes_psum_actions_only(self):
        sim = TPUSimulator(spec())
        whole = sim.simulate(workload(3, 5, 7), tile=(3, 5, 7))
        split = sim.simulate(workload(3, 5, 7), tile=(3, 1, 7))
        for name in ("useful_macs", "useful_ops", "input_transfer_bytes", "logical_output_bytes"):
            self.assertEqual(whole["counts"][name], split["counts"][name])
        self.assertEqual(whole["counts"]["psum_read_bytes"], 0)
        self.assertEqual(split["counts"]["psum_read_bytes"], 336)
        self.assertEqual(split["action_counts"]["hbm_write_bytes"]["count"], 84)

    def test_output_blocking_changes_input_reloads_without_changing_arithmetic(self):
        sim = TPUSimulator(spec())
        whole = sim.simulate(workload(3, 5, 7), tile=(3, 5, 7))
        split = sim.simulate(workload(3, 5, 7), tile=(1, 5, 1))
        self.assertEqual(whole["counts"]["input_transfer_bytes"], 50)
        self.assertEqual(split["counts"]["input_transfer_bytes"], 210)
        self.assertEqual(split["counts"]["useful_ops"], whole["counts"]["useful_ops"])
        self.assertEqual(split["counts"]["logical_output_bytes"], whole["counts"]["logical_output_bytes"])

    def test_capacity_fallback_and_full_hbm_footprint(self):
        plan = make_plan(workload(3, 5, 7), 6, 134)
        self.assertEqual((plan.m, plan.k, plan.n), (1, 1, 1))
        self.assertEqual(plan.peak_working_bytes, 6)
        with self.assertRaisesRegex(ValueError, "minimum"):
            make_plan(workload(3, 5, 7), 5, 134)
        with self.assertRaisesRegex(ValueError, "HBM"):
            make_plan(workload(3, 5, 7), 6, 133)
        for tile in ((0, 1, 1), (-1, 2, 3), (True, 1, 1), (1.0, 2, 3), (1, 2)):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                make_plan(workload(), 2048, 2048, tile)

    def test_effective_capacity_changes_counts_but_not_inventory(self):
        config = hardware_config()
        original = TPUSimulator(TPUSpec(config)).simulate(workload(3, 5, 7))
        config["parameters"]["vmem_usable_bytes"]["value"] = 6
        small = TPUSimulator(TPUSpec(config)).simulate(workload(3, 5, 7))
        self.assertEqual(original["hardware_inventory"], small["hardware_inventory"])
        self.assertEqual(small["counts"]["useful_ops"], 210)
        self.assertEqual(small["memory"]["vmem"]["peak_logical_bytes"], 6)

    def test_no_timing_numeric_or_physical_execution_claims(self):
        r = TPUSimulator(spec()).simulate(workload())
        self.assertFalse(r["boundary"]["numeric_values_simulated"])
        for group, key in (("timing", "total_cycles"), ("timing", "time_ns"), ("timing", "throughput_gops"),
                           ("energy", "total_pj"), ("area", "total_mm2")):
            self.assertIsNone(r[group][key])
            self.assertEqual(r[group]["status"], "not_evaluated")
        self.assertNotIn("events", r)
        for action in r["action_counts"].values():
            self.assertEqual(action["scope"], "single_tensorcore")
            self.assertIn(action["basis"], ("logical_payload", "logical_arithmetic"))
        json.dumps(r, allow_nan=False)

    def test_chip_geometry_does_not_scale_single_tensorcore_counts(self):
        baseline = TPUSimulator(spec()).simulate(workload(3, 5, 7), tile=(2, 3, 4))
        for cores, mxus in ((1, 4), (4, 4), (2, 1), (2, 8)):
            config = hardware_config()
            config["parameters"]["tensorcores_per_chip"].update(value=cores, status="assumed")
            config["parameters"]["mxus_per_tensorcore"].update(value=mxus, status="assumed")
            report = TPUSimulator(TPUSpec(config)).simulate(workload(3, 5, 7), tile=(2, 3, 4))
            with self.subTest(cores=cores, mxus=mxus):
                for name in ("counts", "action_counts", "counting_policy", "boundary", "timing"):
                    self.assertEqual(report[name], baseline[name], name)
                for memory in ("vmem", "hbm"):
                    self.assertEqual(report["memory"][memory], baseline["memory"][memory])
                self.assertEqual(report["memory"]["cmem"]["allocated_bytes"], 0)
                self.assertEqual(report["hardware_inventory"]["chip"]["mxus"], cores * mxus)
                self.assertAlmostEqual(report["hardware_derived"]["chip_int8_peak_tops"], cores * mxus * 34.4064)

    def test_chip_vmem_total_is_not_usable_by_one_tensorcore(self):
        config = hardware_config()
        # The 11-byte full block fits the 12-byte chip inventory, but not either 6-byte VMEM.
        for name in ("vmem_physical_bytes", "vmem_usable_bytes"):
            config["parameters"][name].update(value=6, status="assumed")
        report = TPUSimulator(TPUSpec(config)).simulate(workload(1, 1, 2))
        self.assertEqual(report["hardware_inventory"]["chip"]["private_vmem_bytes"], 12)
        self.assertEqual(report["memory"]["vmem"]["usable_bytes"], 6)
        self.assertEqual(report["memory"]["vmem"]["peak_logical_bytes"], 6)
        self.assertEqual(report["counts"]["compute_blocks"], 2)
        self.assertEqual(report["memory"]["hbm"]["logical_footprint_bytes"], 11)
        # Chip HBM is not multiplied by two TensorCores to satisfy this footprint.
        config["parameters"]["hbm_usable_bytes"].update(value=10, status="assumed")
        with self.assertRaisesRegex(ValueError, "HBM"):
            TPUSimulator(TPUSpec(config)).simulate(workload(1, 1, 2))

    def test_unsupported_precision_rejected(self):
        w = parse_workloads({"gemm": {"M": 3, "K": 5, "N": 7}, "data_format": "BF16"})[0]
        with self.assertRaisesRegex(ValueError, "INT8"):
            TPUSimulator(spec()).simulate(w)

    def test_cli_public_workloads_and_format_override(self):
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory) / "custom.json"
            custom.write_text(json.dumps({"gemm": {"M": 1024, "K": 4096, "N": 1024},
                                          "data_format": "BF16"}), encoding="utf-8")
            for options, expected_ops, overridden in (([], 134217728, False),
                    (["--workload", str(custom), "--format", "INT8"], 8589934592, True)):
                target = Path(directory) / "report.json"
                output = StringIO()
                with self.subTest(options=options), redirect_stdout(output):
                    code = tpu.main([*options, "--output", str(target)])
                self.assertEqual(code, 0)
                r = json.loads(target.read_text(encoding="utf-8"))
                self.assertEqual(r["counts"]["useful_ops"], expected_ops)
                self.assertEqual(r["workload"]["format_overridden"], overridden)
                self.assertEqual(r["mode"], "timing")
                self.assertGreater(r["timing"]["total_cycles"], 0)
                self.assertEqual(r["schema_version"], 13)
                self.assertFalse(r["boundary"]["chip_execution_included"])
                self.assertEqual(r["hardware_hierarchy"]["chip"]["tensorcores_per_chip"], 2)
                self.assertEqual(r["hardware_inventory"]["chip"]["mxus"], 8)
                self.assertEqual(r["hardware_inventory"]["per_tensorcore"]["mxus"], 4)
                self.assertIn("Hardware: chip; counting: single TensorCore", output.getvalue())
                self.assertIn("[Inventory per TensorCore]", output.getvalue())
                self.assertIn("[Inventory chip]", output.getvalue())

    def test_cli_unsupported_precision_and_missing_parameters_fail_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            with redirect_stderr(StringIO()):
                self.assertEqual(tpu.main(["--format", "BF16", "--output", str(target)]), 2)
            self.assertFalse(target.exists())
            config = hardware_config()
            config["parameters"]["hbm_usable_bytes"].update(value=None, status="unknown")
            hardware = Path(directory) / "missing.json"
            hardware.write_text(json.dumps(config), encoding="utf-8")
            with redirect_stderr(StringIO()):
                code = tpu.main(["--hardware", str(hardware), "--format", "INT8", "--output", str(target)])
            self.assertEqual(code, 2)
            self.assertFalse(target.exists())

    def test_excluded_features_are_not_cli_options(self):
        for args in (["--mode", "invalid"], ["--scope", "chip"], ["--loop-order", "KNM"],
                     ["--buffers", "3"], ["--energy", "coefficients.json"], ["--format", "INT8", "BF16"]):
            with self.subTest(args=args), redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                tpu.parse_args(args)
            self.assertEqual(raised.exception.code, 2)

    def test_entry_works_from_another_directory_with_explicit_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps({"gemm": {"M": 1, "K": 1, "N": 1}, "data_format": "INT8"}), encoding="utf-8")
            target = Path(directory) / "report.json"
            result = subprocess.run([sys.executable, str(ROOT / "tpu.py"), "--workload", str(path),
                                     "--output", str(target)], cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            r = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(r["counts"]["useful_ops"], 2)
            self.assertFalse(r["workload"]["format_overridden"])


class TPUTimingTest(unittest.TestCase):
    def test_single_mxu_and_tiny_tail_match_hand_calculations(self):
        sim = TPUSimulator(spec())
        r = sim.simulate(workload(128, 128, 128), mode="timing")
        t = r["timing"]
        self.assertEqual(t["service_cycles"]["by_stage"], {
            "hbm_to_vmem": 58, "vmem_to_mxu": 32, "psum_read": 0,
            "matrix_compute": 128, "matrix_fill_drain": 254,
            "result_to_vmem": 64, "vmem_to_hbm": 115})
        self.assertEqual(t["total_cycles"], 651)
        self.assertAlmostEqual(t["time_ns"], 651 / 1.05)
        self.assertAlmostEqual(t["throughput_gops"], 2 * 128**3 / (651 / 1.05))
        self.assertEqual(t["max_parallel_mxus_used"], 1)
        tail = sim.simulate(workload(1, 1, 1), mode="timing")
        self.assertEqual(tail["timing"]["total_cycles"], 259)
        self.assertEqual(tail["timing"]["service_cycles"]["by_stage"]["matrix_fill_drain"], 254)
        self.assertIsNone(tail["timing"]["chip_cycles"])
        self.assertEqual(tail["counts"]["executed_padded_ops"], 2 * 128 * 128)

    def test_four_mxus_run_one_wave_and_fifth_tile_needs_another(self):
        sim = TPUSimulator(spec())
        four = sim.simulate(workload(256, 128, 256), tile=(256, 128, 256), mode="timing")["timing"]
        self.assertEqual((four["mxu_output_tiles"], four["matrix_waves"], four["max_parallel_mxus_used"]), (4, 1, 4))
        self.assertEqual(four["service_cycles"]["by_stage"]["matrix_compute"], 128)
        self.assertEqual(four["total_cycles"], 1276)
        five = sim.simulate(workload(513, 128, 128), tile=(1024, 128, 128), mode="timing")["timing"]
        self.assertEqual((five["mxu_output_tiles"], five["matrix_waves"], five["max_parallel_mxus_used"]), (5, 2, 4))
        self.assertEqual(five["service_cycles"]["by_stage"]["matrix_compute"], 256)
        self.assertEqual(five["service_cycles"]["by_stage"]["matrix_fill_drain"], 508)

    def test_k_accumulation_stays_in_vmem_and_final_export_is_once(self):
        sim = TPUSimulator(spec())
        counted = sim.simulate(workload(3, 5, 7), tile=(2, 3, 4))
        timed = sim.simulate(workload(3, 5, 7), tile=(2, 3, 4), mode="timing", schedule="serial")
        self.assertEqual(timed["timing"]["service_cycles"]["by_stage"], {
            "hbm_to_vmem": 8, "vmem_to_mxu": 8, "psum_read": 4,
            "matrix_compute": 20, "matrix_fill_drain": 2032,
            "result_to_vmem": 8, "vmem_to_hbm": 4})
        self.assertEqual(timed["timing"]["total_cycles"], 2084)
        for name in ("counts", "action_counts", "memory", "counting_policy", "hardware_inventory", "parameter_provenance"):
            self.assertEqual(counted[name], timed[name])
        self.assertEqual(timed["action_counts"]["hbm_write_bytes"]["count"], 84)

    def test_chip_core_count_changes_bandwidth_share_not_compute_resources(self):
        baseline = TPUSimulator(spec()).simulate(workload(), mode="timing")
        config = hardware_config()
        config["parameters"]["tensorcores_per_chip"].update(value=4, status="assumed")
        other = TPUSimulator(TPUSpec(config)).simulate(workload(), mode="timing")
        self.assertEqual(other["counts"], baseline["counts"])
        self.assertEqual(other["timing"]["service_cycles"]["by_stage"]["matrix_compute"], 32)
        self.assertEqual(other["timing"]["parameters"]["hbm_share_bytes_per_second"]["value"], 300e9)
        self.assertEqual(other["timing"]["total_cycles"], 296)
        self.assertFalse(other["boundary"]["chip_execution_included"])

    def test_required_unknowns_reject_timing_but_preserve_count_mode(self):
        for name in ("clock_hz", "hbm_bandwidth_bytes_per_second", "mxus_per_tensorcore",
                     "mxu_rows", "mxu_columns", "int8_macs_per_cell_cycle", "vmem_bandwidth_bytes_per_second"):
            config = hardware_config()
            config["parameters"][name].update(value=None, status="unknown")
            sim = TPUSimulator(TPUSpec(config))
            with self.subTest(parameter=name):
                self.assertIsNone(sim.simulate(workload())["timing"]["total_cycles"])
                with self.assertRaisesRegex(ValueError, "TPU timing parameters missing: " + name):
                    sim.simulate(workload(), mode="timing")
        with self.assertRaisesRegex(ValueError, "mode"):
            TPUSimulator(spec()).simulate(workload(), mode="invalid")

    def test_vmem_proxy_is_explicit_and_configured_service_is_respected(self):
        r = TPUSimulator(spec()).simulate(workload(), mode="timing")
        self.assertNotIn("vmem_bandwidth_bytes_per_second", r["missing_parameters"])
        self.assertEqual(r["parameter_provenance"]["vmem_bandwidth_bytes_per_second"]["value"], 1075200000000)
        params = r["timing"]["parameters"]
        self.assertEqual(params["vmem_service_bytes_per_cycle"]["value"], 1024)
        self.assertEqual(params["vmem_service_bytes_per_cycle"]["status"], "assumed")
        self.assertEqual(params["hbm_share_bytes_per_second"]["value"], 600e9)
        for item in params.values():
            self.assertTrue({"value", "unit", "scope", "status", "source", "conditions"} <= item.keys())
            self.assertTrue(item.get("source_version") or item.get("version"))
        config = hardware_config()
        config["parameters"]["vmem_bandwidth_bytes_per_second"].update(value=1050000000, status="assumed")
        slow = TPUSimulator(TPUSpec(config)).simulate(workload(1, 1, 1), mode="timing")
        self.assertEqual(slow["timing"]["parameters"]["dma_service_bytes_per_cycle"]["value"], 1)
        self.assertEqual(slow["timing"]["total_cycles"], 267)
        self.assertIsNone(slow["energy"]["total_pj"])
        self.assertIsNone(slow["area"]["total_mm2"])

    def test_large_block_count_is_aggregated_without_event_expansion(self):
        config = hardware_config()
        config["parameters"]["vmem_usable_bytes"].update(value=6, status="assumed")
        r = TPUSimulator(TPUSpec(config)).simulate(workload(1000, 1000, 1000), mode="timing")
        self.assertEqual(r["timing"]["scheduled_blocks"], 10**9)
        self.assertEqual(r["timing"]["total_cycles"], 259 * 10**9)
        self.assertEqual(r["memory"]["vmem"]["peak_logical_bytes"], 6)
        self.assertNotIn("events", r["timing"])

    def test_cli_timing_and_missing_clock_protect_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "report.json"
            output = StringIO()
            with redirect_stdout(output):
                code = tpu.main(["--workload", "tiny_gemm.json", "--format", "INT8",
                                 "--mode", "timing", "--output", str(target)])
            self.assertEqual(code, 0)
            before = target.read_bytes()
            r = json.loads(before)
            self.assertEqual(r["mode"], "timing")
            self.assertEqual(r["timing"]["total_cycles"], 9244)
            self.assertEqual(r["timing"]["max_parallel_mxus_used"], 4)
            self.assertIn("[Timing]", output.getvalue())
            config = hardware_config()
            config["parameters"]["clock_hz"].update(value=None, status="unknown")
            hardware = Path(directory) / "missing.json"
            hardware.write_text(json.dumps(config), encoding="utf-8")
            with redirect_stderr(StringIO()):
                code = tpu.main(["--hardware", str(hardware), "--format", "INT8",
                                 "--mode", "timing", "--output", str(target)])
            self.assertEqual(code, 2)
            self.assertEqual(target.read_bytes(), before)
