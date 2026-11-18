"""CMEM cold-fill, reuse, capacity and endpoint-service regressions."""
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import tpu
from src.core.tpu import TPUSimulator
from src.hardware.tpu import TPUSpec
from tests.tpu.support import ROOT, hardware_config, spec, workload


def small_spec():
    config = hardware_config(cmem_mode="tiled")
    values = {"clock_hz": 10**9, "mxu_rows": 2, "mxu_columns": 2, "mxus_per_tensorcore": 1,
              "hbm_bandwidth_bytes_per_second": 16 * 10**9,
              "vmem_bandwidth_bytes_per_second": 8 * 10**9,
              "cmem_read_bandwidth_bytes_per_second": 16 * 10**9,
              "cmem_write_bandwidth_bytes_per_second": 8 * 10**9}
    for name, value in values.items():
        config["parameters"][name].update(value=value, status="assumed")
    config["execution"]["default_tile_mkn"] = [2, 2, 2]
    config["execution"]["cmem_groups_mn"] = [2, 2]
    return TPUSpec(config)


class CMEMTest(unittest.TestCase):
    def test_reference_activates_tiled_with_no_new_hardware_parameters(self):
        config = json.loads((ROOT / "configs/hardware/tpu_v4_reference.json").read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], 12)
        self.assertEqual(config["execution"]["cmem_mode"], "tiled")
        self.assertEqual(config["execution"]["cmem_groups_mn"], [4, 4])
        self.assertEqual(len(config["parameters"]), 21)
        self.assertTrue(all(p["value"] is not None for p in config["parameters"].values()))
        r = TPUSimulator(TPUSpec(config)).simulate(workload(512, 256, 512), mode="timing")
        self.assertEqual(r["memory"]["cmem"]["mode"], "tiled")
        self.assertEqual(r["counts"]["hbm_input_read_bytes"], 262144)
        self.assertEqual(r["counts"]["input_transfer_bytes"], 524288)
        self.assertEqual(r["timing"]["total_cycles"], 9244)

    def test_hand_calculated_cold_fill_reuse_and_shared_endpoint_timing(self):
        sim = TPUSimulator(small_spec())
        r = sim.simulate(workload(4, 3, 4), mode="timing")
        c, a, t = r["counts"], r["action_counts"], r["timing"]
        self.assertEqual(c["input_transfer_bytes"], 48)
        self.assertEqual(c["hbm_input_read_bytes"], 24)
        self.assertEqual(c["hbm_read_bytes_saved_vs_bypass"], 24)
        self.assertEqual((a["cmem_write_bytes"]["count"], a["cmem_read_bytes"]["count"]), (24, 48))
        self.assertEqual(a["hbm_write_bytes"]["count"], 64)
        self.assertEqual(r["memory"]["cmem"]["allocated_bytes"], 24)
        self.assertEqual(t["service_cycles"]["by_stage"], {"hbm_to_cmem": 6, "cmem_to_vmem": 8,
            "vmem_to_mxu": 8, "psum_read": 8, "matrix_compute": 12,
            "matrix_fill_drain": 16, "result_to_vmem": 16, "vmem_to_hbm": 8})
        # Outer fill: 24/min(HBM=8, CMEM-write=4) = 6 cycles, charged once.
        # Four output blocks: each takes 18 cycles with one prefetch hidden.
        self.assertEqual(t["total_cycles"], 6 + 4 * 18)
        self.assertEqual(t["service_cycles"]["total_cycles"], 82)
        self.assertEqual(t["overlapped_service_cycles"], 4)
        self.assertEqual(t["resource_busy_cycles"], {"hbm": 14, "vmem": 48, "mxu": 28,
                                                     "cmem_read": 8, "cmem_write": 6})
        for options in ({"schedule": "serial"}, {"buffers": 1}):
            other = sim.simulate(workload(4, 3, 4), mode="timing", **options)
            self.assertEqual(other["timing"]["total_cycles"], 82)
            self.assertEqual(other["counts"], c)

    def test_tail_panels_and_outputs_conserve_bytes_and_compute(self):
        sim = TPUSimulator(spec(cmem_mode="tiled"))
        task = workload(5, 3, 7)
        r = sim.simulate(task, tile=(2, 2, 3), cmem_groups_mn=(2, 2))
        b = sim.simulate(task, tile=(2, 2, 3), cmem_mode="bypass")
        self.assertEqual(r["memory"]["cmem"]["outer_tile_mkn"], [4, 3, 6])
        self.assertEqual(r["memory"]["cmem"]["outer_tiles"], 4)
        self.assertEqual(r["memory"]["cmem"]["allocated_bytes"], 60)
        self.assertEqual(r["counts"]["hbm_input_a_read_bytes"], 30)
        self.assertEqual(r["counts"]["hbm_input_b_read_bytes"], 42)
        self.assertEqual(r["counts"]["input_transfer_bytes"], 108)
        self.assertEqual(r["counts"]["compute_blocks"], 18)
        for key in ("useful_ops", "executed_padded_ops", "psum_read_bytes", "psum_write_bytes", "logical_output_bytes"):
            self.assertEqual(r["counts"][key], b["counts"][key])
        self.assertEqual(r["memory"]["vmem"], b["memory"]["vmem"])

    def test_equal_capacity_share_shrinks_groups_then_reports_bypass(self):
        config = hardware_config(cmem_mode="tiled")
        task = workload(5, 3, 7)
        options = dict(tile=(2, 2, 3), cmem_groups_mn=(2, 2), mode="timing")
        for capacity, selected, allocated, reads in ((60, [2, 2], 30, 72),
                (59, [2, 1], 21, 87), (30, [1, 1], 15, 108)):
            config["parameters"]["cmem_usable_bytes"].update(value=capacity, status="assumed")
            r = TPUSimulator(TPUSpec(config)).simulate(task, **options)
            self.assertEqual(r["memory"]["cmem"]["usable_bytes"], capacity // 2)
            self.assertEqual(r["memory"]["cmem"]["selected_groups_mn"], selected)
            self.assertEqual(r["memory"]["cmem"]["allocated_bytes"], allocated)
            self.assertEqual(r["counts"]["hbm_input_read_bytes"], reads)
        config["parameters"]["cmem_usable_bytes"]["value"] = 28
        # A missing unused CMEM service must not block the explicit capacity fallback.
        config["parameters"]["cmem_read_bandwidth_bytes_per_second"].update(value=None, status="unknown")
        sim = TPUSimulator(TPUSpec(config))
        fallback = sim.simulate(task, **options)
        bypass = sim.simulate(task, cmem_mode="bypass", **options)
        self.assertEqual(fallback["memory"]["cmem"]["mode"], "bypass")
        self.assertEqual(fallback["memory"]["cmem"]["reason"], "insufficient_capacity_for_one_full_k_panel_pair")
        self.assertEqual(fallback["memory"]["cmem"]["allocated_bytes"], 0)
        self.assertEqual(fallback["counts"], bypass["counts"])
        self.assertEqual(fallback["timing"], bypass["timing"])
        self.assertEqual(fallback["execution_contract"]["cmem_mode"], "tiled")
        self.assertEqual(fallback["execution_contract"]["effective_cmem_mode"], "bypass")

    def test_missing_capacity_and_only_used_timing_services_are_required(self):
        config = hardware_config(cmem_mode="tiled")
        config["parameters"]["cmem_usable_bytes"].update(value=None, status="unknown")
        sim = TPUSimulator(TPUSpec(config))
        with self.assertRaisesRegex(ValueError, "CMEM counting parameters missing"):
            sim.simulate(workload())
        sim.simulate(workload(), cmem_mode="bypass")
        for name in ("cmem_read_bandwidth_bytes_per_second", "cmem_write_bandwidth_bytes_per_second"):
            config = hardware_config(cmem_mode="tiled")
            config["parameters"][name].update(value=None, status="unknown")
            sim = TPUSimulator(TPUSpec(config))
            self.assertEqual(sim.simulate(workload())["counts"]["cmem_fill_bytes"], 1024)
            with self.assertRaisesRegex(ValueError, "CMEM timing parameters missing: " + name):
                sim.simulate(workload(), mode="timing")
            sim.simulate(workload(), mode="timing", cmem_mode="bypass")

    def test_directional_rates_endpoint_bottlenecks_and_core_shares(self):
        task, options = workload(4, 3, 4), dict(tile=(2, 2, 2), cmem_groups_mn=(2, 2), mode="timing")
        base = small_spec()
        baseline = TPUSimulator(base).simulate(task, **options)
        for direction in ("read", "write"):
            config = hardware_config(cmem_mode="tiled")
            for name, value in base.values.items():
                config["parameters"][name]["value"] = value
            config["parameters"][f"cmem_{direction}_bandwidth_bytes_per_second"]["value"] /= 4
            r = TPUSimulator(TPUSpec(config)).simulate(task, **options)
            self.assertEqual(r["counts"], baseline["counts"])
            stages = r["timing"]["service_cycles"]["by_stage"]
            affected, unchanged = ("cmem_to_vmem", "hbm_to_cmem") if direction == "read" else ("hbm_to_cmem", "cmem_to_vmem")
            self.assertGreater(stages[affected], baseline["timing"]["service_cycles"]["by_stage"][affected])
            self.assertEqual(stages[unchanged], baseline["timing"]["service_cycles"]["by_stage"][unchanged])
            self.assertGreater(r["timing"]["total_cycles"], baseline["timing"]["total_cycles"])
        config = hardware_config(cmem_mode="tiled")
        config["parameters"]["tensorcores_per_chip"].update(value=4, status="assumed")
        r = TPUSimulator(TPUSpec(config)).simulate(workload(), mode="timing")
        self.assertEqual(r["memory"]["cmem"]["usable_bytes"], 32 * 2**20)
        self.assertAlmostEqual(r["timing"]["parameters"]["cmem_read_service_bytes_per_cycle"]["value"], 2150704873472 / 4 / 1.05e9)
        self.assertAlmostEqual(r["timing"]["cmem_transfer_services_bytes_per_cycle"]["hbm_to_cmem"], 1075352436736 / 4 / 1.05e9)

    def test_cold_one_use_panel_pays_fill_and_startup_without_claiming_speedup(self):
        sim = TPUSimulator(spec(cmem_mode="tiled"))
        tiled = sim.simulate(workload(1, 1, 1), mode="timing")
        bypass = sim.simulate(workload(1, 1, 1), mode="timing", cmem_mode="bypass")
        self.assertEqual(tiled["counts"]["hbm_read_bytes_saved_vs_bypass"], 0)
        self.assertEqual(tiled["timing"]["total_cycles"], bypass["timing"]["total_cycles"] + 1)
        self.assertEqual(sim.simulate(workload(1, 1, 1), mode="timing"), tiled)
        config = hardware_config(cmem_mode="tiled")
        config["execution"]["fixed_latency_cycles"]["transfer_startup"] = 3
        delayed = TPUSimulator(TPUSpec(config)).simulate(workload(1, 1, 1), mode="timing")
        # Five transfers: HBM->CMEM, CMEM->VMEM, input->MXU, result->VMEM, VMEM->HBM.
        self.assertEqual(delayed["timing"]["total_cycles"], 260 + 5 * 3)

    def test_large_k_is_aggregated_and_preserves_prefetch_savings(self):
        blocks = 1000000
        r = TPUSimulator(spec(cmem_mode="tiled")).simulate(workload(1, blocks, 1),
                    tile=(1, 1, 1), mode="timing")
        self.assertEqual(r["counts"]["compute_blocks"], blocks)
        self.assertEqual(r["counts"]["cmem_fill_bytes"], 2 * blocks)
        self.assertEqual(r["timing"]["overlapped_service_cycles"], blocks - 1)
        self.assertNotIn("events", r["timing"])
        self.assertTrue(all(busy <= r["timing"]["total_cycles"] for busy in r["timing"]["resource_busy_cycles"].values()))

    def test_invalid_policies_and_cli_errors_preserve_output(self):
        sim = TPUSimulator(spec(cmem_mode="tiled"))
        for groups in ([], [1], [1, 0], [True, 1], [1.5, 1], "22"):
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, "cmem_groups_mn"):
                sim.simulate(workload(), cmem_groups_mn=groups)
            config = hardware_config(cmem_mode="tiled")
            config["execution"]["cmem_groups_mn"] = groups
            with self.assertRaisesRegex(ValueError, "cmem_groups_mn"):
                TPUSpec(config)
        for mode in ("auto", [], True):
            with self.assertRaisesRegex(ValueError, "cmem_mode"):
                sim.simulate(workload(), cmem_mode=mode)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "result.json"
            reports = {}
            for mode in ("tiled", "bypass"):
                with redirect_stdout(StringIO()):
                    self.assertEqual(tpu.main(["--cmem-mode", mode, "--cmem-buffers", "1", "--output", str(out)]), 0)
                reports[mode] = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(reports[mode]["execution_contract"]["effective_cmem_mode"], mode)
                self.assertEqual(reports[mode]["execution_contract"]["cmem_buffer_slots"], 1)
            self.assertLess(reports["tiled"]["counts"]["hbm_input_read_bytes"], reports["bypass"]["counts"]["hbm_input_read_bytes"])
            before = out.read_bytes()
            with redirect_stderr(StringIO()):
                self.assertEqual(tpu.main(["--cmem-groups", "0", "1", "--output", str(out)]), 2)
            self.assertEqual(out.read_bytes(), before)
