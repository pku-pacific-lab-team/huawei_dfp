import unittest

from src.core.ascend.tiling import make_plan
from tests.ascend.support import spec, workload


class TilingTest(unittest.TestCase):
    def test_baseline_footprint(self):
        plan = make_plan(workload(64, 256, 64), spec().capacities)
        self.assertEqual((plan.m, plan.k, plan.n, plan.l1_k), (64, 128, 64, 256))
        self.assertEqual(plan.m * plan.k, 8192)
        self.assertEqual((plan.m + plan.n) * plan.l1_k, 32768)
        self.assertEqual(4 * plan.m * plan.n, 16384)

    def test_each_capacity_is_enforced(self):
        for name, limit in (("L0A", 512), ("L0B", 512), ("L0C", 1024), ("L1", 1024), ("UB", 1024)):
            capacities = spec().capacities.copy()
            capacities[name] = limit
            with self.subTest(buffer=name):
                p = make_plan(workload(127, 513, 99), capacities)
                self.assertLessEqual(p.m * p.k, capacities["L0A"])
                self.assertLessEqual(p.k * p.n, capacities["L0B"])
                self.assertLessEqual(4 * p.m * p.n, capacities["L0C"])
                self.assertLessEqual((p.m + p.n) * p.l1_k, capacities["L1"])
                self.assertLessEqual(p.ub_chunk_bytes, capacities["UB"])

    def test_unaligned_tail_is_allocated_as_fractals(self):
        p = make_plan(workload(17, 33, 15), spec().capacities)
        self.assertEqual((p.m, p.k, p.n), (32, 64, 16))

    def test_invalid_tiles_and_too_small_buffers(self):
        for tile in ((15, 32, 16), (16, 16, 16), (0, 32, 16)):
            with self.assertRaises(ValueError):
                make_plan(workload(), spec().capacities, tile)
        for name in spec().capacities:
            capacities = spec().capacities.copy()
            capacities[name] = 1
            with self.assertRaises(ValueError):
                make_plan(workload(), capacities)
