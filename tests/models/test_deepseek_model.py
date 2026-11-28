import unittest
from unittest.mock import patch

from src.models.block_graph import BlockCapacityError
from src.models.chip import load_hardware
from src.models.config import DecodeRequest, load_model
from src.models.deepseek_model import _run_deepseek
from src.models.deepseek_experts import ExpertOperators, R1Kernels
from src.models.deepseek_placement import uniform_stages


C = {
    **load_model("deepseek_r1"),
    "num_hidden_layers": 5,
    "first_k_dense_replace": 3,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_attention_heads": 4,
    "q_lora_rank": 12,
    "kv_lora_rank": 8,
    "qk_rope_head_dim": 4,
    "qk_nope_head_dim": 8,
    "v_head_dim": 8,
    "n_routed_experts": 8,
    "n_group": 2,
    "topk_group": 2,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 12,
    "vocab_size": 35,
}
REQUEST = DecodeRequest(2, 3, 2)
PLAN = [
    {"layers": [0, 3], "chips": [13, 15]},
    {
        "layers": [3, 5],
        "chips": [70, 61, 80, 93],
        "attention_chips": [70, 80],
        "expert_groups": [[70, 61], [80, 93]],
    },
]
ROUTES = {
    "assignments": [[0, 1], [0, 1]],
    "overrides": [{"step": 1, "layer": 3, "assignments": [[0, 4], [1, 5]]}],
}


def run(spec=None, plan=PLAN, routing=ROUTES, mode="timing", config=C, request=REQUEST):
    return _run_deepseek(config, request, "tpu", spec or load_hardware("tpu"), plan, routing, mode)


class R1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run()

    def test_three_dense_layers_absorbed_mla(self):
        r = self.report
        self.assertEqual(r["effective_lengths"], [4, 5])
        for chip in r["chips"][:2]:
            for s, length in enumerate((4, 5)):
                runs = chip["steps"][s]["runs"]
                for layer in range(3):
                    p = r["profiles"][runs[f"step.{s}/layer.{layer}/block"]]
                    nodes = {n["id"]: n for n in p["nodes"]}
                    self.assertEqual(nodes["kv.append"]["token_interval"], [length - 1, length])
                    self.assertEqual(nodes["kv.append"]["appended_bytes"], 2 * (8 + 4))
                    for name, width in (("latent", 8), ("position", 4)):
                        self.assertEqual(p["tensors"][name + ".cache"]["shape"], [2, 1, 5, width])
                    self.assertEqual(nodes["content.score"]["useful_macs"], 2 * 2 * 8 * length)
                    self.assertEqual(nodes["pv"]["useful_macs"], 2 * 2 * length * 8)
                    self.assertEqual(nodes["position.score"]["useful_macs"], 2 * 2 * 4 * length)
                    self.assertEqual(nodes["scores"]["action_counts"]["fp32_add"]["count"], 2 * 2 * length)
                    self.assertIn("kv.append", nodes["content.score"]["depends_on"])
                    self.assertIn("ffn.reduce", nodes)
        for i, t in enumerate(r["step_totals"]):
            self.assertEqual(t["logical_kv_append_bytes"], 2 * 12 * 5)
            self.assertEqual(t["physical_kv_append_bytes"], 2 * 12 * 5 * 2)
            self.assertEqual(t["routed_assignments"], 2 * 2 * 2)
            self.assertEqual(t["executed_useful_macs"], sum(c["steps"][i]["useful_macs"] for c in r["chips"]))

    def test_owners_routes_etp_sparse_work_and_shared_combine(self):
        r = self.report
        stage = r["stages"][1]
        self.assertEqual(stage["expert_owners"], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(stage["shared_group"], 1)
        self.assertEqual(r["routing"]["step.0/layer.3"]["assignments_per_group"], [4, 0])
        self.assertEqual(r["routing"]["step.1/layer.3"]["assignments_per_group"], [2, 2])
        self.assertEqual(r["routing"]["step.0/layer.3"]["active"][0]["slots"], [[0, 0], [1, 0]])
        for c in r["chips"][2:]:
            stored = sum(c["resident_bytes_per_unit"]["expert_weights"])
            self.assertEqual(stored, 2 * (4 + int(c["owns_shared_expert"])) * 3 * 32 * 6)
            if c["expert_group"] == 1:
                ids = list(c["steps"][0]["runs"])
                self.assertFalse(any("/expert.0" in k for k in ids))
                self.assertTrue(any("/expert.shared" in k for k in ids))
        root = r["chips"][2]
        step = root["steps"][0]["runs"]
        combine = r["profiles"][step["step.0/layer.3/combine"]]["nodes"][0]
        self.assertEqual(combine["action_counts"]["fp32_mul"]["count"], 2 * 2 * 32)
        self.assertEqual(combine["action_counts"]["fp32_add"]["count"], 2 * 2 * 32)
        router = r["profiles"][step["step.0/layer.3/router"]]["nodes"][-1]
        self.assertEqual(router["action_counts"]["fp32_add"]["count"], 2 * (8 + 8 + 2 + 1))
        self.assertEqual(router["action_counts"]["fp32_max"]["count"], 2 * (2 * (3 + 2) + 1 + 7 + 6))
        e = {e["id"]: e for e in r["communication_events"]}
        reduce = e["step.0/layer.3/expert.0/expert.reduce"]
        self.assertEqual((reduce["participants"], reduce["input_bytes_per_rank"]), ([70, 61], 2 * 32 * 4))
        self.assertEqual(e["step.1/layer.3/expert.4/return"]["logical_tensor_bytes"], 32 * 4)
        self.assertEqual(e["step.0/layer.3/expert.0/dispatch"]["deliveries"], [dict(source=70, destination=61)])

    def test_fragmented_router_private_capacity(self):
        spec = load_hardware("tpu")
        spec.values["vmem_usable_bytes"] = 64
        report = run(spec, mode="count")
        self.assertEqual(report["step_totals"], self.report["step_totals"])
        for profile, ref in self.report["profiles"].items():
            if ref["nodes"][-1]["id"] == "router.select":
                node = report["profiles"][profile]["nodes"][-1]
                self.assertEqual(node["private_memory"]["vmem"]["required_bytes"], 48)
                self.assertEqual(node["action_counts"], ref["nodes"][-1]["action_counts"])

        spec.values["vmem_usable_bytes"] = 47
        with self.assertRaises(BlockCapacityError) as failure:
            ExpertOperators(R1Kernels("tpu", spec, "count", C), C, 1).router().execute()
        self.assertEqual((failure.exception.node, failure.exception.required_bytes), ("router.select", 48))
        config = load_model("deepseek_r1")
        kernels = R1Kernels("ascend", load_hardware("ascend"), "count", config)
        node = ExpertOperators(kernels, config, 1).router().execute()["nodes"][-1]
        self.assertEqual(node["private_memory"]["ub"]["required_bytes"], 96)

    def test_complete_dag(self):
        r = self.report
        nodes = {}
        for n in r["schedule"]:
            self.assertNotIn(n["id"], nodes)
            self.assertTrue(set(n["depends_on"]) <= nodes.keys())
            nodes[n["id"]] = n["depends_on"]
        visited = {"model.logits"}
        for n in reversed(nodes):
            if n in visited:
                visited.update(nodes[n])
        self.assertEqual(visited, set(nodes))
        self.assertIn("step.0/layer.3/block/kv.append", nodes["step.1/layer.3/block/kv.append"])
        self.assertIn("step.0/head/select_token", nodes["step.1/token_ids"])
        for e in r["communication_events"]:
            self.assertIsNone(e["communication_cycles"])
            self.assertTrue(e["consumers"])
            self.assertTrue(all(e["id"] in nodes[n] for n in e["consumers"]))

    def test_inactive_expert_capacity_before_kernels(self):
        plan = [dict(layers=[0, 5], chips=[1, 2], attention_chips=[1], expert_groups=[[1], [2]])]
        c = {**C, "moe_intermediate_size": 4096, "first_k_dense_replace": 0}
        plan[0]["expert_owners"] = [0, 0, 1, 1, 1, 1, 1, 1]
        spec = load_hardware("tpu")
        spec.values["hbm_usable_bytes"] = 8_000_000
        with patch(
            "src.models.decoder_kernels.DecoderKernels.resolve", side_effect=AssertionError("early native call")
        ):
            with self.assertRaises(BlockCapacityError) as fail:
                run(spec, plan, "concentrated", config=c)
        self.assertEqual(fail.exception.chip, 2)
        self.assertGreater(fail.exception.tensors["expert_weights"], 8_000_000)

    def test_routes_and_plan_validation(self):
        for rows in ([[0, 0], [1, 2]], [[8, 0], [1, 2]], [[1, 2]], [[0], [1]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                run(routing={"assignments": rows})
        with self.assertRaises(ValueError):
            run(config={**C, "topk_group": 1}, routing={"assignments": [[0, 4], [1, 5]]})
        with self.assertRaises(ValueError):
            run(plan=[dict(layers=[0, 5], chips=[3, 4], expert_groups=[[3, 4], [4]])])

    def test_decode_count_timing_and_tp_ep_identity(self):
        plan = uniform_stages(C, "tpu", 1, 1, 1, 1)
        count, timed = [run(plan=plan, mode=m) for m in ("count", "timing")]
        self.assertFalse(count["communication_events"])
        self.assertEqual(count["step_totals"], timed["step_totals"])
        for key, p in timed["profiles"].items():
            other = count["profiles"][key]
            self.assertEqual(p["capacity"], other["capacity"])
            self.assertEqual(
                [n["action_counts"] for n in p["nodes"]], [n["action_counts"] for n in other["nodes"]]
            )
        self.assertIsNone(count["chips"][0]["steps"][0]["local_compute_cycles"])
