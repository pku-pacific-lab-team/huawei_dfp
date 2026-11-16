import unittest
from pathlib import Path

from src.core.area_model import AreaModel
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader


class AreaModelTest(unittest.TestCase):
    def setUp(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        base_loader = TSPConfigLoader(str(config_dir))
        self.base_spec = base_loader.create_tsp_spec(
            base_loader.load_hardware_config("tsp.json")
        )
        self.base_area = base_loader.load_area_config("7nm_tsp.json")

        cim_loader = TSPCIMConfigLoader(str(config_dir))
        self.cim_spec = cim_loader.create_tsp_spec(
            cim_loader.load_hardware_config("tsp_cim.json")
        )
        self.cim_area = cim_loader.load_area_config("7nm_tsp_cim.json")

    def test_baseline_area_breakdown_is_positive(self) -> None:
        report = AreaModel(self.base_area).evaluate(
            self.base_spec.area_inventory(),
            throughput_gops=140000.0,
        )

        self.assertGreater(report["global_sram_mm2"], 0.0)
        self.assertGreater(report["local_sram_mm2"], 0.0)
        self.assertGreater(report["compute_area_mm2"], 0.0)
        self.assertGreater(report["total_area_mm2"], report["global_sram_mm2"])
        self.assertGreater(report["tops_per_mm2"], 0.0)

    def test_cim_area_breakdown_keeps_sram_capacity_no_smaller_than_baseline(self) -> None:
        base_report = AreaModel(self.base_area).evaluate(
            self.base_spec.area_inventory(),
            throughput_gops=140000.0,
        )
        cim_report = AreaModel(self.cim_area).evaluate(
            self.cim_spec.area_inventory(),
            throughput_gops=220000.0,
        )

        self.assertGreaterEqual(
            self.cim_spec.physical_total_sram_mib,
            self.base_spec.physical_total_sram_mib,
        )
        self.assertGreaterEqual(
            self.cim_spec.global_mem_sram_mib,
            self.base_spec.global_mem_sram_mib,
        )
        self.assertGreater(cim_report["fabric_area_mm2"], 0.0)
        self.assertGreater(cim_report["total_area_mm2"], cim_report["global_sram_mm2"])
        self.assertGreater(cim_report["tops_per_mm2"], base_report["tops_per_mm2"])


if __name__ == "__main__":
    unittest.main()
