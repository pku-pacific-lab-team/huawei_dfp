import unittest

from src.models.block_graph import BlockCapacityError
from src.models.chip import load_hardware
from src.models.compute_path import compute_path_cycles
from src.models.config import DecodeRequest
from src.models.decoder_kernels import DecoderKernels
from src.models.deepseek_model import _run_deepseek
from src.models.llama_block import RankOperators
from src.models.llama_model import _run_model
from tests.models.test_llama_model import TINY
from tests.models import test_deepseek_model as moe


class SoftwareTileTest(unittest.TestCase):
    def test_shared_chip_serializes_while_remote_expert_overlaps(self):
        chips = [dict(chip_id=i, steps=[{"runs": {"work": str(i)}}]) for i in (0, 1)]
        schedule, profiles = [], {str(i): {"nodes": []} for i in (0, 1)}
        def op(name, deps, chip, cycles):
            profiles[str(chip)]["nodes"].append(dict(id=name, local_compute_cycles=cycles))
            schedule.append(dict(id=f"step.0/{name}", depends_on=deps, profile_node=name,
                                 invocation="work", chips=[chip]))
            return schedule[-1]["id"]

        root = op("router", [], 0, 2)
        a = op("expert.a", [root], 0, 10)
        b = op("expert.b", [root], 0, 3)
        c = op("expert.c", [root], 1, 18)
        combined = op("combine", [a, b, c], 0, 2)
        schedule.append(dict(id="model.logits", depends_on=[combined], chips=[0]))
        self.assertEqual(compute_path_cycles(schedule, [], chips, [], profiles), 2 + max(10 + 3, 18) + 2)

    def test_pp_delivery_by_source_and_tp_barrier(self):
        chips = [dict(chip_id=i, steps=[{"block": str(i)}]) for i in range(4)]
        stages = [{"chips": [0, 1]}, {"chips": [2, 3]}]
        profiles = {str(i): {"nodes": [dict(id="op", local_compute_cycles=n),
                                       dict(id="sync", local_compute_cycles=0)]}
                    for i, n in enumerate((9, 1, 1, 9))}
        left = dict(id="step.0/left", depends_on=[], stage=0, profile_slot="block", profile_node="op")
        sync = dict(id="step.0/sync", depends_on=[left["id"]], stage=0, profile_slot="block", profile_node="sync")
        pp = dict(id="step.0/pp", depends_on=[sync["id"]])
        right = dict(id="step.0/right", depends_on=[pp["id"]], stage=1, profile_slot="block", profile_node="op")
        gather = dict(id="step.0/gather", depends_on=[right["id"]], stage=1, profile_slot="block", profile_node="sync")
        schedule = [left, sync, pp, right, gather, dict(id="model.logits", depends_on=[gather["id"]], owner_chip=2)]
        events = [dict(id=pp["id"], deliveries=[dict(source=0, destination=2), dict(source=1, destination=3)]),
                  dict(id=gather["id"], kind="tp_gather", destinations=[2])]
        self.assertEqual(compute_path_cycles(schedule, events, chips, stages, profiles), 10)
        events.append(dict(id=sync["id"], kind="tp_all_reduce_sum"))
        self.assertEqual(compute_path_cycles(schedule, events, chips, stages, profiles), 18)

    def test_default_tie_and_timing_requirement(self):
        stages = [dict(layers=[0, 3], chips=[0])]
        args = TINY, DecodeRequest(1, 3, 1), "tsp_cim", load_hardware("tsp_cim"), stages
        report = _run_model(*args, "timing", auto_tile=True)
        search = report["tile_search"]
        self.assertEqual(search["selected_policy"], "default")
        self.assertEqual(search["saved_cycles"], 0)
        self.assertEqual(search["baseline"], search["selected"])
        self.assertEqual(report["chips"][0]["steps"][0]["local_compute_cycles"], search["selected"]["critical_path_cycles"])
        with self.assertRaisesRegex(ValueError, "requires timing"):
            _run_model(*args, "count", auto_tile=True)

    def test_tsp_weight_buffer_peak_excludes_cumulative_installs(self):
        spec = load_hardware("tsp")
        for mode in ("count", "timing"):
            kernels = DecoderKernels("tsp", spec, mode)
            for k in (320, 640):
                cost = kernels.resolve(kernels.kernel("gemm", 1, k, 320))
                buf = cost.private["weight_buffer"]
                self.assertEqual(buf["required_bytes"], 320 * 320)
                self.assertEqual(buf["capacity_bytes"], spec.values["weight_buffer_bytes"])
                self.assertEqual(cost.actions["weight_install_bytes"]["count"], k * 320)

    def test_smaller_gemm_when_default_weight_buffer_overflows(self):
        config = dict(TINY, num_hidden_layers=1, hidden_size=320, intermediate_size=1280, vocab_size=1280)
        stages = [dict(layers=[0, 1], chips=[0])]
        spec = load_hardware("tsp")
        spec.values["weight_buffer_bytes"] = 320 * 320
        args = config, DecodeRequest(1, 3, 1), "tsp", spec, stages, "timing"
        with self.assertRaises(BlockCapacityError):
            _run_model(*args)
        report = _run_model(*args, auto_tile=True)
        search = report["tile_search"]
        self.assertEqual(search["baseline"]["capacity_error"]["level"], "weight_buffer")
        self.assertIsNone(search["latency_reduction_fraction"])
        self.assertEqual(search["selected_policy"], "per_node")
        spec.values["weight_buffer_bytes"] = 1
        with self.assertRaises(BlockCapacityError):
            _run_model(*args, auto_tile=True)

    def test_custom_moe_routes_and_expert_tensor_parallelism(self):
        args = moe.C, moe.REQUEST, "tpu", load_hardware("tpu"), moe.PLAN, moe.ROUTES, "timing"
        baseline = _run_deepseek(*args)
        tuned = _run_deepseek(*args, auto_tile=True)
        self.assertEqual(baseline["routing"], tuned["routing"])
        self.assertEqual(baseline["communication_events"], tuned["communication_events"])
        self.assertEqual(baseline["step_totals"], tuned["step_totals"])
        def replay(r):
            return compute_path_cycles(r["schedule"], r["communication_events"], r["chips"], r["stages"], r["profiles"])
        search = tuned["tile_search"]
        self.assertEqual(replay(baseline), search["baseline"]["critical_path_cycles"])
        self.assertEqual(replay(tuned), search["selected"]["critical_path_cycles"])
        self.assertLessEqual(replay(tuned), replay(baseline))

    def test_larger_vector_rejected_with_other_resident_tensors(self):
        kernels = DecoderKernels("tsp", load_hardware("tsp"), "timing")
        g = RankOperators(kernels, 0)
        g.u = kernels.units
        inputs = [g.tensor(t, (8192,), "INT32", [16384] * 2, "input") for t in ("gate", "up")]
        g.element("act", inputs, "INT8", "swiglu")
        default = g.execute()
        g.reserved = [kernels.capacity - n for n in default["capacity"]["unit_peak_bytes"]]
        tuned = g.execute(search_tiles=True)
        record = tuned["nodes"][0]["tile_search"]
        large = next(c for c in record["candidates"] if c["requested"] == 2048)
        self.assertIn("capacity_error", large)
        self.assertGreater(large["capacity_error"]["required_bytes"], kernels.capacity)
        self.assertLessEqual(max(tuned["capacity"]["unit_peak_bytes"]), kernels.capacity)
