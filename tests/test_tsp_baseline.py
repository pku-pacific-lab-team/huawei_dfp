import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import tsp
from src.core.tsp_simulator import TSPSimulator
from src.utils.tsp_config import TSPConfigLoader


class TSPBaselineTest(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        loader = TSPConfigLoader(str(config_dir))
        self.energy_params = loader.load_energy_config("7nm_tsp.json")
        self.tsp_spec = loader.create_tsp_spec(
            loader.load_hardware_config("tsp.json")
        )

    def test_public_peak_ranges_are_preserved(self) -> None:
        self.assertGreater(self.tsp_spec.peak_tops("INT8"), 730.0)
        self.assertLess(self.tsp_spec.peak_tops("INT8"), 760.0)
        self.assertGreater(self.tsp_spec.peak_tops("BF16"), 180.0)
        self.assertLess(self.tsp_spec.peak_tops("BF16"), 190.0)

    def test_sram_partition_and_area_inventory_are_exposed(self) -> None:
        mib = 1024 * 1024
        self.assertEqual(self.tsp_spec.physical_total_sram_bytes, 352 * mib)
        self.assertEqual(self.tsp_spec.global_mem_sram_bytes, 320 * mib)
        self.assertEqual(self.tsp_spec.reserved_local_sram_bytes, 32 * mib)
        self.assertEqual(self.tsp_spec.mem_total_bytes, self.tsp_spec.global_mem_sram_bytes)

        inventory = self.tsp_spec.area_inventory()
        self.assertEqual(inventory["global_mem_sram_mib"], 320.0)
        self.assertEqual(inventory["local_sram_mib"], 32.0)
        self.assertEqual(inventory["mxm_planes"], self.tsp_spec.total_mxm_planes_chip)
        self.assertEqual(inventory["clock_domains"], self.tsp_spec.total_mxm_planes_chip)
        self.assertGreater(inventory["stream_endpoints"], 0)

    def test_explicit_supercell_mesh_is_exposed(self) -> None:
        usage = self.tsp_spec.describe_supercell_usage(
            k_actual=320,
            n_actual=640,
            data_format="INT8",
            hemisphere=0,
        )

        self.assertEqual(usage["active_supercell_rows"], 20)
        self.assertEqual(usage["active_supercell_cols_total"], 40)
        self.assertEqual(len(usage["planes"]), 2)

        first_plane = usage["planes"][0]
        first_row = first_plane["rows"][0]
        first_supercell = first_row["supercells"][0]

        self.assertEqual(first_row["superlane"], 0)
        self.assertEqual(first_supercell["lane_start"], 0)
        self.assertEqual(first_supercell["lane_end"], 15)
        self.assertEqual(first_supercell["k_start"], 0)
        self.assertEqual(first_supercell["k_end"], 15)
        self.assertEqual(first_supercell["n_start"], 0)
        self.assertEqual(first_supercell["n_end"], 15)
        self.assertTrue(first_supercell["active"])

    def test_summary_reports_full_supercell_mesh(self) -> None:
        summary = self.tsp_spec.summarize_topology(
            data_format="INT8",
            k_actual=320,
            n_actual=640,
        )

        self.assertEqual(summary["supercells_per_plane"], 400)
        self.assertEqual(summary["supercells_per_hemisphere"], 800)
        self.assertEqual(summary["total_supercells_chip"], 1600)
        self.assertEqual(summary["active_supercell_rows"], 20)

    def test_summary_tracks_partial_k_row_utilization(self) -> None:
        summary = self.tsp_spec.summarize_topology(
            data_format="INT8",
            k_actual=256,
            n_actual=512,
        )

        self.assertEqual(summary["active_supercell_rows"], 16)
        self.assertEqual(summary["active_supercell_cols_total"], 32)

    def test_simulation_reports_expected_sections_and_keys(self) -> None:
        sim = TSPSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="BF16",
            verbose=False,
        )

        results = sim.simulate(M=512, K=640, N=640, loop_order="KNM")

        expected_result_keys = {
            "spatial_flow",
            "latencies",
            "latency_detail",
            "energies_pj",
            "total_latency_cycles",
            "single_hem_latency_cycles",
            "total_energy_pj",
            "throughput_gops",
            "energy_efficiency_tops_per_watt",
            "bottleneck",
        }
        self.assertTrue(expected_result_keys.issubset(results.keys()))
        self.assertGreater(results["total_latency_cycles"], 0)
        self.assertGreater(results["total_energy_pj"], 0.0)

        output = StringIO()
        with redirect_stdout(output):
            tsp.print_spatial_flow(results)
            tsp.print_pipeline_stages(results)
            tsp.print_latency_breakdown(results)
            tsp.print_energy_breakdown(results)
            tsp.print_performance_summary(results)
        report = output.getvalue()

        self.assertIn("[Spatial Data Flow Analysis]", report)
        self.assertIn("[Pipeline Stages]", report)
        self.assertIn("[Latency Breakdown (single hemisphere)]", report)
        self.assertIn("[Energy Breakdown]", report)
        self.assertIn("[Performance Summary]", report)
        self.assertIn("MXM Compute", report)
        self.assertIn("Weight Install", report)

    def test_default_bf16_efficiency_is_in_elevated_7nm_range(self) -> None:
        sim = TSPSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="BF16",
            verbose=False,
        )

        results = sim.simulate(M=1024, K=4096, N=1024, loop_order="KNM")
        energies = results["energies_pj"]

        self.assertIn("mxm_compute", energies)
        self.assertIn("weight_install", energies)
        self.assertIn("global_sram_standby", energies)
        self.assertIn("local_sram_standby", energies)
        self.assertIn("mxm_array_logic_leak", energies)
        self.assertIn("clock_backbone", energies)
        self.assertIn("control_backbone", energies)
        self.assertGreater(energies["mxm_compute"], 0.0)
        self.assertGreater(energies["weight_install"], 0.0)
        self.assertGreater(energies["global_sram_standby"], 0.0)
        self.assertGreater(energies["local_sram_standby"], 0.0)
        self.assertGreater(energies["clock_backbone"], 0.0)
        self.assertGreater(energies["control_backbone"], 0.0)
        self.assertGreater(results["energy_efficiency_tops_per_watt"], 1.55)
        self.assertLess(results["energy_efficiency_tops_per_watt"], 1.85)

    def test_default_int8_efficiency_is_above_14nm_tdp_band(self) -> None:
        sim = TSPSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="INT8",
            verbose=False,
        )

        results = sim.simulate(M=1024, K=4096, N=1024, loop_order="KNM")

        self.assertGreater(results["energy_efficiency_tops_per_watt"], 3.55)
        self.assertLess(results["energy_efficiency_tops_per_watt"], 4.60)


if __name__ == "__main__":
    unittest.main()
