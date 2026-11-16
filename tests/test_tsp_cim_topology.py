import unittest
from pathlib import Path

from src.utils.tsp_cim_config import TSPCIMConfigLoader


class TSPCIMTopologyTest(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        loader = TSPCIMConfigLoader(str(config_dir))
        self.tsp_spec = loader.create_tsp_spec(
            loader.load_hardware_config("tsp_cim.json")
        )

    def test_sram_partition_and_area_inventory_are_exposed(self) -> None:
        mib = 1024 * 1024
        self.assertEqual(self.tsp_spec.physical_total_sram_bytes, 352 * mib)
        self.assertEqual(self.tsp_spec.global_mem_sram_bytes, 320 * mib)
        self.assertEqual(self.tsp_spec.reserved_local_sram_bytes, 32 * mib)
        self.assertEqual(self.tsp_spec.mem_total_bytes, self.tsp_spec.global_mem_sram_bytes)

        inventory = self.tsp_spec.area_inventory()
        self.assertEqual(inventory["global_mem_sram_mib"], 320.0)
        self.assertEqual(inventory["local_sram_mib"], 32.0)
        self.assertEqual(inventory["cim_macros"], self.tsp_spec.total_macros_chip)
        self.assertGreater(inventory["stream_endpoints"], 0)

    def test_macro_overlay_maps_to_supercell_blocks(self) -> None:
        usage = self.tsp_spec.describe_macro_usage(
            k_actual=320,
            n_actual=640,
            hemisphere=0,
        )

        self.assertIn("sheets", usage)
        self.assertGreaterEqual(usage["macro_supercell_k_span"], 1)
        self.assertGreaterEqual(usage["macro_supercell_n_span"], 1)
        self.assertGreaterEqual(usage["macro_sheet_stack_k"], 1)
        self.assertGreaterEqual(usage["macro_sheet_stack_n"], 1)

        first_sheet = usage["sheets"][0]
        first_plane = first_sheet["planes"][0]
        first_row = first_plane["macro_rows"][0]
        first_macro = first_row["macros"][0]

        self.assertEqual(first_macro["supercell_k_start"], 0)
        self.assertEqual(
            first_macro["supercell_k_end"],
            usage["macro_supercell_k_span"] - 1,
        )
        self.assertEqual(first_macro["supercell_n_start"], 0)
        self.assertEqual(
            first_macro["supercell_n_end"],
            usage["macro_supercell_n_span"] - 1,
        )
        self.assertEqual(first_macro["superlane_start"], 0)
        self.assertEqual(first_macro["n_start"], 0)
        self.assertTrue(first_macro["active"])

    def test_mixed_kn_overlay_exposes_parallel_k_capacity(self) -> None:
        usage = self.tsp_spec.describe_macro_usage(
            k_actual=4096,
            n_actual=1024,
            hemisphere=0,
        )

        self.assertGreater(usage["active_k_parallel_groups"], 0)
        self.assertGreaterEqual(
            usage["active_k_parallel_groups"],
            usage["active_k_groups"],
        )
        self.assertEqual(usage["serial_k_depth"], 1)
        self.assertGreaterEqual(
            usage["active_n_parallel_groups"],
            usage["active_n_groups"],
        )


if __name__ == "__main__":
    unittest.main()
