import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.core.tsp_simulator import TSPSimulator
import tsp_cim
from src.core.tsp_cim_simulator import TSPCIMSimulator
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader


class TSPCIMEnergyModelTest(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        base_loader = TSPConfigLoader(str(config_dir))
        self.base_energy_params = base_loader.load_energy_config("7nm_tsp.json")
        self.base_tsp_spec = base_loader.create_tsp_spec(
            base_loader.load_hardware_config("tsp.json")
        )
        loader = TSPCIMConfigLoader(str(config_dir))
        self.energy_params = loader.load_energy_config("7nm_tsp_cim.json")
        self.tsp_spec = loader.create_tsp_spec(
            loader.load_hardware_config("tsp_cim.json")
        )

    def test_energy_corrections_are_reported_and_nonzero(self) -> None:
        sim = TSPCIMSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="BF16",
            verbose=False,
        )

        results = sim.simulate(M=512, K=256, N=512)
        energies = results["energies_pj"]

        expected_keys = {
            "global_sram_standby",
            "local_sram_standby",
            "macro_logic_leak",
            "fabric_retention",
            "fabric_logic_leak",
            "vxm_logic_leak",
            "clock_backbone",
            "clock_domains",
            "control_backbone",
            "control_domains",
            "sxm_broadcast",
            "reduce_collect",
            "reduce_buffer",
        }
        self.assertTrue(expected_keys.issubset(energies.keys()))

        for key in expected_keys:
            self.assertGreater(energies[key], 0.0, key)

    def test_energy_totals_and_breakdown_output_stay_consistent(self) -> None:
        sim = TSPCIMSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="BF16",
            verbose=False,
        )

        results = sim.simulate(M=512, K=256, N=512)
        energies = results["energies_pj"]
        stream_hop_keys = {
            "stream_hop_act_external",
            "stream_hop_act_internal",
            "stream_hop_wt_external",
            "stream_hop_wt_internal",
            "stream_hop_result_external",
            "stream_hop_result_internal",
        }

        self.assertAlmostEqual(
            results["total_energy_pj"],
            sum(energies.values()),
        )
        self.assertAlmostEqual(
            results["stream_hop_energy_pj"],
            sum(energies[key] for key in stream_hop_keys),
        )

        output = StringIO()
        with redirect_stdout(output):
            tsp_cim.print_energy_breakdown(results)
        report = output.getvalue()
        self.assertIn("Leakage & Static", report)
        self.assertIn("Clock/Control", report)
        self.assertIn("reduce_buffer", report)

    def test_default_bf16_uses_parallel_k_overlay_not_full_k_serialization(self) -> None:
        sim = TSPCIMSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="BF16",
            verbose=False,
        )

        results = sim.simulate(M=1024, K=4096, N=1024)
        energies = results["energies_pj"]

        self.assertIn("sram_psum_buffer_read", energies)
        self.assertIn("sram_psum_buffer_write", energies)
        self.assertIn("vxm_operand_io", energies)
        self.assertGreater(energies["sram_psum_buffer_read"], 0.0)
        self.assertGreater(energies["sram_psum_buffer_write"], 0.0)
        self.assertGreater(energies["vxm_operand_io"], 0.0)
        self.assertEqual(results["spatial_flow"]["serial_k_depth"], 1)
        self.assertGreater(
            results["spatial_flow"]["active_k_parallel_groups"],
            results["spatial_flow"]["active_k_groups"] - 1,
        )
        self.assertLess(results["latency_detail"]["cim_compute_total"], 100000)
        self.assertLess(results["total_latency_cycles"], 100000)
        self.assertGreater(results["energy_efficiency_tops_per_watt"], 10.0)
        self.assertLess(results["energy_efficiency_tops_per_watt"], 12.5)

    def test_default_bf16_compute_energy_slightly_exceeds_static_energy(self) -> None:
        cim = TSPCIMSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="BF16",
            verbose=False,
        ).simulate(M=1024, K=4096, N=1024)

        compute_energy = (
            cim["energies_pj"]["cim_local_readout"] +
            cim["energies_pj"]["cim_compute"]
        )
        static_energy = (
            cim["energies_pj"]["global_sram_standby"] +
            cim["energies_pj"]["local_sram_standby"] +
            cim["energies_pj"]["macro_logic_leak"] +
            cim["energies_pj"]["fabric_retention"] +
            cim["energies_pj"]["fabric_logic_leak"] +
            cim["energies_pj"]["sxm_logic_leak"] +
            cim["energies_pj"]["vxm_logic_leak"]
        )

        self.assertGreater(compute_energy, static_energy)
        self.assertLess(compute_energy / static_energy, 1.35)

    def test_default_bf16_noncompute_and_static_gaps_are_not_too_extreme(self) -> None:
        base = TSPSimulator(
            tsp_spec=self.base_tsp_spec,
            energy_params=self.base_energy_params,
            data_format="BF16",
            verbose=False,
        ).simulate(M=1024, K=4096, N=1024, loop_order="KNM")
        cim = TSPCIMSimulator(
            tsp_spec=self.tsp_spec,
            energy_params=self.energy_params,
            data_format="BF16",
            verbose=False,
        ).simulate(M=1024, K=4096, N=1024)

        base_noncompute = (
            base["single_hem_latency_cycles"] -
            base["latency_detail"]["mxm_compute_total"]
        )
        cim_noncompute = (
            cim["single_hem_latency_cycles"] -
            cim["latency_detail"]["cim_compute_total"]
        )
        base_static = (
            base["energies_pj"]["global_sram_standby"] +
            base["energies_pj"]["local_sram_standby"] +
            base["energies_pj"]["mxm_array_logic_leak"] +
            base["energies_pj"]["vxm_logic_leak"] +
            base["energies_pj"]["clock_backbone"] +
            base["energies_pj"]["clock_domains"] +
            base["energies_pj"]["control_backbone"] +
            base["energies_pj"]["control_domains"]
        )
        cim_static = (
            cim["energies_pj"]["global_sram_standby"] +
            cim["energies_pj"]["local_sram_standby"] +
            cim["energies_pj"]["macro_logic_leak"] +
            cim["energies_pj"]["fabric_retention"] +
            cim["energies_pj"]["fabric_logic_leak"] +
            cim["energies_pj"]["vxm_logic_leak"] +
            cim["energies_pj"]["clock_backbone"] +
            cim["energies_pj"]["clock_domains"] +
            cim["energies_pj"]["control_backbone"] +
            cim["energies_pj"]["control_domains"]
        )

        self.assertIn("fabric_retention", cim["energies_pj"])
        self.assertIn("fabric_logic_leak", cim["energies_pj"])
        self.assertIn("clock_domains", cim["energies_pj"])
        self.assertGreater(cim["energies_pj"]["fabric_retention"], 0.0)
        self.assertGreater(cim["energies_pj"]["fabric_logic_leak"], 0.0)
        self.assertGreater(cim["energies_pj"]["clock_domains"], 0.0)
        self.assertGreater(cim_noncompute / base_noncompute, 0.5)
        self.assertGreater(cim_static / base_static, 0.15)
        self.assertLess(cim["total_latency_cycles"], base["total_latency_cycles"])
        self.assertGreater(cim["energy_efficiency_tops_per_watt"], 5.0)


if __name__ == "__main__":
    unittest.main()
