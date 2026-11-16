"""Reviewed topology arithmetic and physical placement contracts."""
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

    def test_all_physical_macro_ids_are_unique_and_n_shards_are_disjoint(self):
        ids = set()
        for hem in range(2):
            for plane in range(2):
                for cluster in range(4):
                    for sl in range(20):
                        for core in range(5):
                            for macro in range(4):
                                p = self.spec.placement(sl, core, macro, hemisphere=hem, plane=plane, cluster=cluster)
                                ids.add(p.chip_macro_id)
                                self.assertEqual(p.slot, sl * 20 + core * 4 + macro)
                                self.assertEqual((p.k_start, p.k_stop), (sl * 16, (sl + 1) * 16))
                                self.assertEqual((p.n_start, p.n_stop), ((core * 4 + macro) * 32, (core * 4 + macro + 1) * 32))
        self.assertEqual(ids, set(range(6400)))
        self.assertEqual(self.spec.placement(0, 0, 0).n_start, self.spec.placement(19, 0, 0).n_start)

    def test_bad_coordinates_are_rejected(self):
        for coordinates in ((-1, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 4), (True, 0, 0)):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                self.spec.validate_domain(*coordinates)
        for coordinates in ((20, 0, 0), (0, 5, 0), (0, 0, 4), (0, 0, False)):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                self.spec.placement(*coordinates)

    def test_sram_inventory_is_subtotal_not_complete_physical_area(self):
        i = self.spec.inventory()
        self.assertEqual(i["macro_weight_active_context_bytes"], 6400 * 512)
        self.assertEqual(i["macro_weight_resident_bytes"], 6400 * 2048)
        self.assertEqual(i["known_mem_plus_macro_bytes"], int(232.5 * 2**20))
        self.assertIsNone(i["physical_sram_bytes"])
        self.assertIsNone(i["additional_local_sram_bytes"])
        self.assertIsNone(i["collector_buffer_bytes_per_plane"])
        self.assertEqual((i["nim_units"], i["nim_tiles"], i["nim_conversion_lanes"]), (4, 80, 256))
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
        self.assertEqual(d["nim_output_bpc_per_plane"]["value"], 256)
        self.assertEqual(d["lw_capture_bpc_per_plane"]["value"] * 2, d["stream_bpc_per_direction"]["value"])
