"""Reviewed topology arithmetic and inventory contracts."""
import unittest

from src.utils.tsp_cim_config import TSPCIMConfigLoader


class TSPCIMTopologyTest(unittest.TestCase):
    def setUp(self):
        loader = TSPCIMConfigLoader()
        self.spec = loader.create_tsp_spec(loader.load_hardware_config())

    def test_core_slice_domain_plane_and_chip_have_distinct_counts(self):
        h = self.spec.hierarchy()
        self.assertEqual((h["core"]["k"], h["core"]["n"], h["core"]["macros"]), (16, 128, 4))
        self.assertEqual((h["cluster_slice"]["cores"], h["cluster_slice"]["macros"], h["cluster_slice"]["n"]), (5, 20, 640))
        self.assertEqual((h["cluster_domain"]["k"], h["cluster_domain"]["n"], h["cluster_domain"]["macros"]), (320, 640, 400))
        self.assertEqual((h["plane"]["cores"], h["plane"]["macros"], h["plane"]["cluster_domains"]), (400, 1600, 4))
        self.assertEqual((h["chip"]["planes"], h["chip"]["cores"], h["chip"]["macros"], h["chip"]["cluster_domains"]), (4, 1600, 6400, 16))

    def test_sram_inventory_is_subtotal_not_complete_physical_area(self):
        i = self.spec.inventory()
        self.assertEqual(i["macro_weight_active_context_bytes"], 6400 * 512)
        self.assertEqual(i["macro_weight_resident_bytes"], 6400 * 2048)
        self.assertEqual(i["known_mem_plus_macro_bytes"], int(232.5 * 2**20))
        for name in ("physical_sram_bytes", "additional_local_sram_bytes", "collector_buffer_bytes_per_plane",
                     "sxm_physical_units", "stream_physical_endpoints", "clock_domains", "control_domains"):
            self.assertNotIn(name, i)
        for name in ("nim_units", "nim_tiles", "nim_logical_channels_per_tile", "nim_conversion_lanes"):
            self.assertNotIn(name, i)
        self.assertEqual((i["vxm_alus"], i["mem_slices"], i["memory_tiles"]), (5120, 88, 1760))

    def test_peaks_and_bandwidth_scopes_are_hand_calculated(self):
        d = self.spec.derived()
        for scope, peak in (("core", 256), ("cluster_slice", 1280), ("cluster_domain", 25600),
                             ("plane", 102400), ("hemisphere", 204800), ("chip", 409600)):
            self.assertEqual(d["int8_mac_per_cycle_" + scope]["value"], peak)
        self.assertAlmostEqual(d["int8_tops_chip"]["value"], 737.28)
        self.assertEqual(d["stream_bpc_per_direction"]["value"], 10240)
        self.assertEqual(d["lw_capture_bpc_per_superlane_plane"]["value"], 256)
        self.assertEqual(d["lw_capture_bpc_per_plane"]["value"], 5120)
        self.assertEqual(d["lw_capture_bpc_per_chip"]["value"], 20480)
        self.assertEqual(d["lw_capture_bpc_per_chip"]["scope"], "chip")
        self.assertEqual(d["lw_scan_cycles_per_wordline"]["value"], 10)
        self.assertEqual(d["lw_full_context_scan_cycles"]["value"], 160)
        self.assertEqual(d["lw_full_context_scan_cycles"]["scope"], "superlane_plane")
        # 整片装满一套权重的容量/分布式捕获宽度也应得到 160 拍；不是单域尾块耗时。
        self.assertEqual(self.spec.inventory()["macro_weight_active_context_bytes"] //
                         d["lw_capture_bpc_per_chip"]["value"], 160)
        self.assertEqual(d["int32_writer_bpc_per_writer_superlane_plane"]["value"], 256)
        self.assertEqual(d["lw_capture_bpc_per_plane"]["value"] * 2, d["stream_bpc_per_direction"]["value"])
