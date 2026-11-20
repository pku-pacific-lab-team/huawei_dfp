from tests.operators.inspection import inspect_result
from collections import Counter
from copy import deepcopy
import unittest

from src.operators import simulate_composite
from src.hardware.vector import VectorExecution
from src.core.composite_storage import validate_storage_schedule
from src.core.ascend.composite import make_path as ascend_path
from src.core.tpu.composite import make_path as tpu_path
from src.core.tsp.composite import make_path as tsp_path
from tests.operators.test_composite import task
from tests.operators.test_vector import synthetic, execution


PATHS = {"ascend": ascend_path, "tpu": tpu_path, "tsp": tsp_path}


class CompositeStorageTest(unittest.TestCase):
    def run_case(self, arch, kind, *, width=7, tile=4, profile=None):
        op, spec = task(kind, width), synthetic(arch)
        profile = execution(arch) if profile is None else profile
        kwargs = {"mem_slices": [0], "streams": 1} if arch == "tsp" else {}
        path = PATHS[arch](op, spec, profile, mode="timing", tile_elements=tile, **kwargs)
        report = inspect_result(simulate_composite(op, arch, spec, mode="timing", execution=profile, tile_elements=tile, **kwargs, include_details=True))
        return op, path, report

    def test_hand_odd_tail_cross_tile_cycles_and_bytes(self):
        # Independent hand ledger: D=7, blocks 4+3, base=4 op/cycle,
        # HBM=16 B/cycle, UB/VMEM=32, single MEM slice=16, stream=320.
        # Each block has two tree levels. Each level takes 1 compute cycle;
        # the second block merges once. EXP costs 15/12 cycles for 4/3 values.
        phases = {"rmsnorm": {"sum_squares": 25, "normalizer": 11, "apply_scale_and_weight": 18},
                  "softmax": {"row_max": 19, "exponentials_and_sum": 58, "reciprocal": 4, "normalize": 10}}
        resident_phases = {"rmsnorm": {"sum_squares": 24, "normalizer": 11, "apply_scale_and_weight": 16},
                           "softmax": {"row_max": 18, "exponentials_and_sum": 57, "reciprocal": 4, "normalize": 8}}
        for arch in PATHS:
            for kind in ("rmsnorm", "softmax"):
                with self.subTest(arch=arch, kind=kind):
                    _, _, r = self.run_case(arch, kind)
                    n = r["native_report"]
                    rms = kind == "rmsnorm"
                    read, write, compute = (184, 124, 16) if rms else (208, 144, 43)
                    times = {"local_read": 15 if rms else 19, "local_write": 15 if rms else 19,
                             "compute": compute, "reduction_dependency": 0, "dma": 8 if rms else 10}
                    if arch == "tsp":
                        times.pop("dma")
                        times["local_read"] = 20 if rms else 25
                        expected = {"mem_read_bytes": read, "mem_write_bytes": write,
                                    "input_stream_bytes": (20 if rms else 25) * 320,
                                    "output_stream_bytes": (15 if rms else 19) * 320}
                    else:
                        local = "ub" if arch == "ascend" else "vmem"
                        expected = {"hbm_read_bytes": 84, "hbm_write_bytes": 28 if rms else 56,
                                    f"{local}_dma_write_bytes": 84, f"{local}_dma_read_bytes": 28 if rms else 56,
                                    f"{local}_read_bytes": read, f"{local}_write_bytes": write}
                    expected.update({"fp32_mul": 22, "fp32_add": 7, "fp32_rsqrt": 1} if rms else
                                    {"fp32_max": 6, "fp32_sub": 7, "fp32_exp": 7, "fp32_add": 6, "fp32_rcp": 1, "fp32_mul": 7})
                    self.assertEqual(n["timing"]["elapsed_cycles"]["by_stage"], times)
                    self.assertEqual(r["latency"]["cycle_count"], sum(times.values()))
                    self.assertEqual({k: v["count"] for k, v in n["action_counts"].items()}, expected)
                    by_phase, from_stages = Counter(), Counter()
                    for g in n["schedule"]["groups_per_row"]:
                        for stage in g["stages"]:
                            by_phase[g["phase"]] += sum(stage["cycles_per_execution"].values()) * g["repetitions_per_row"]
                            from_stages.update({k: v * g["repetitions_per_row"] for k, v in stage["counts_per_execution"].items()})
                    self.assertEqual(dict(by_phase), (resident_phases if arch == "tsp" else phases)[kind])
                    self.assertEqual(dict(from_stages), expected)

    def test_hand_startups_count_actual_transfers_and_compute_stages(self):
        for arch in PATHS:
            cfg = execution(arch).config
            cfg["policies"][arch]["transfer_startup_cycles"] = 2
            cfg["policies"][arch]["compute_startup_cycles"] = 3
            profile = VectorExecution(cfg, arch)
            for kind in ("rmsnorm", "softmax"):
                _, _, r = self.run_case(arch, kind, profile=profile)
                expected = ({"rmsnorm": 163, "softmax": 226} if arch == "tsp" else {"rmsnorm": 112, "softmax": 162})[kind]
                self.assertEqual(r["latency"]["cycle_count"], expected)

    def test_live_storage_covers_first_steady_tail_and_scalar_aliases(self):
        for arch in PATHS:
            for kind in ("rmsnorm", "softmax"):
                for width, tile in ((1, 1), (7, 4), (23, 5)):
                    op, path, r = self.run_case(arch, kind, width=width, tile=tile)
                    check = r["native_report"]["memory"]["lifetime_validation"]
                    self.assertEqual(check["status"], "validated_symbolic_schedule")
                    self.assertLessEqual(check["peak_live_temporary_bytes"], check["allocated_temporary_bytes"])
                    if width == 23:
                        self.assertLess(check["representative_tile_iterations"], 20)
                    path.memory = deepcopy(path.memory)
                    if arch == "tsp":
                        path.memory["scratch_tile_bytes"] = 0
                    else:
                        path.memory[path.local]["allocated_bytes"] = 0
                    with self.assertRaisesRegex(ValueError, "live temporary bytes"):
                        validate_storage_schedule(op, path, r["native_report"]["schedule"]["groups_per_row"])

    def test_invalid_schedule_lifetimes_are_rejected(self):
        for arch in PATHS:
            op, path, r = self.run_case(arch, "softmax")
            original = r["native_report"]["schedule"]["groups_per_row"]
            variants = []
            bad = deepcopy(original)
            # Normalize before exponentials have been produced for the full row.
            bad.insert(0, bad.pop(-1))
            variants.append(bad)
            bad = deepcopy(original)
            # Destroy MAX initialization; the next merge must fail.
            bad[0]["stages"] = [s for s in bad[0]["stages"] if s["name"] != "initialize_row_max"]
            variants.append(bad)
            bad = deepcopy(original)
            # The 3-element tail must carry its third value to the next level.
            tail = next(g for g in bad if g["phase"] == "row_max" and g["elements"] == 3)
            level = next(s for s in tail["stages"] if s["name"].startswith("reduce_"))
            level["writes"][0]["elements"] = 2
            variants.append(bad)
            if arch != "tsp":
                bad = deepcopy(original)
                group = next(g for g in bad if g["phase"] == "exponentials_and_sum")
                index = next(i for i, s in enumerate(group["stages"]) if s["name"] == "export_exp_workspace")
                export = group["stages"].pop(index)
                group["stages"].append(export)  # Local exp was overwritten by reduction.
                variants.append(bad)
                bad = deepcopy(original)
                bad[0]["stages"] = [s for s in bad[0]["stages"] if s["name"] != "load_x"]
                variants.append(bad)
            for bad in variants:
                with self.subTest(arch=arch), self.assertRaisesRegex(ValueError, "storage lifetime"):
                    validate_storage_schedule(op, path, bad)
