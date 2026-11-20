"""Hardware inventory remains available without an area cost model."""
import unittest
from src.core.tsp.simulator import TSPSimulator
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader


class HardwareInventoryTest(unittest.TestCase):
    def test_reference_outputs_inventory_without_area_estimate(self):
        loader=TSPConfigLoader()
        spec=loader.create_tsp_spec(loader.load_hardware_config())
        result=TSPSimulator(spec).simulate(1,16,16)
        self.assertEqual(result["area"]["status"],"not_evaluated")
        self.assertIsNone(result["area"]["total_mm2"])
        self.assertEqual(result["hardware_inventory"]["global_mem_sram_mib"],220)
        self.assertNotIn("physical_sram_mib", result["hardware_inventory"])
        cim_loader=TSPCIMConfigLoader()
        cim=cim_loader.create_tsp_spec(cim_loader.load_hardware_config())
        self.assertEqual(cim.mem_total_bytes,spec.mem_total_bytes)
