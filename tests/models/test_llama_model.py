import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.models.block_graph import BlockCapacityError
from src.models.chip import load_hardware
from src.models.config import DecodeRequest, load_model
from src.models.llama_model import _run_model, run_llama_model
from src.models.llama_placement import uniform_plan, validate_plan


ROOT = Path(__file__).resolve().parents[2]
TINY = {**load_model("llama3_70b"), "num_hidden_layers": 3, "hidden_size": 32, "vocab_size": 19,
        "intermediate_size": 48, "num_attention_heads": 4, "num_key_value_heads": 2}
PLAN = [{"layers": [0,1], "chips": [13,17]}, {"layers": [1,3], "chips": [54,66,83]}]
REQUEST = DecodeRequest(2, 3, 2)


class LlamaModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _run_model(TINY, REQUEST, "tpu", load_hardware("tpu"), PLAN, "timing")

    def test_invalid_stage_plans(self):
        for stages in ([], [{"layers": [0,2], "chips": [0]}],
                       [PLAN[0], {"layers": [2,3], "chips": [8]}],
                       [PLAN[0], {"layers": [0,3], "chips": [8]}],
                       [PLAN[0], {"layers": [1,3], "chips": [13]}],
                       *([{"layers": [0,3], "chips": ids}] for ids in ([], [3,3], [True], list(range(5))))):
            with self.subTest(stages=stages), self.assertRaises(ValueError):
                validate_plan(stages, TINY)

    def test_residency_and_layer_instances(self):
        r = self.report
        self.assertEqual(r["effective_lengths"], [4,5])
        profiles = r["profiles"]
        for c in r["chips"]:
            run = c["steps"][0]
            tensors = profiles[run["block"]]["tensors"]
            layers = c["layers"][1]-c["layers"][0]
            sizes = c["resident_bytes_per_unit"]
            for part, role, repeat in (("layer_weights","weight",layers), ("layer_kv_cache","kv_cache",layers),
                                       ("shared_rope_tables","rope_table",1)):
                per_unit = [sum(t["bytes_per_unit"][i] for t in tensors.values() if t["role"] == role)
                            for i in range(2)]
                self.assertEqual(sizes[part], [v*repeat for v in per_unit])
            kv = c["kv_heads"][1]-c["kv_heads"][0]
            self.assertEqual(sum(sizes["layer_kv_cache"]), layers*2*2*kv*5*8)
            persistent = sum(sum(v) for v in sizes.values())
            self.assertEqual(persistent, c["resident_bytes"])
            self.assertGreater(c["peak"]["required_bytes"], persistent)
            if c["stage"] == 0:
                self.assertEqual(sum(sizes["embedding_weight"]), 19*16)
                self.assertNotIn("head", run)
            else:
                lo, hi = c["logit_vocab_range"]
                self.assertEqual(sum(sizes["final_norm_and_lm_head"]), 32*(hi-lo)+4*32)
                self.assertNotIn("embedding", run)
            for step in c["steps"]:
                elapsed = layers*profiles[step["block"]]["local_compute_cycles"]
                for slot in ("head","embedding"):
                    if slot in step:
                        elapsed += profiles[step[slot]]["local_compute_cycles"]
                self.assertEqual(step["local_compute_cycles"], elapsed)
        self.assertEqual(r["chips"][0]["steps"], r["chips"][1]["steps"])
        for i, total in enumerate(r["step_totals"]):
            length = 4+i
            logical = 2*(32*19 + 3*(2*32**2+2*32*2*8+3*32*48+2*32*length))
            self.assertEqual(total["logical_useful_macs"], logical)
            self.assertEqual(total["executed_useful_macs"], logical+2*2*2*32*8)
            self.assertEqual(total["physical_kv_append_bytes"], 2*2*8*(2+2*3))
            self.assertEqual(total["executed_useful_macs"], sum(c["steps"][i]["useful_macs"] for c in r["chips"]))

    def test_network_and_decode_dag(self):
        r = self.report
        nodes = {}
        for n in r["schedule"]:
            self.assertNotIn(n["id"], nodes)
            self.assertTrue(all(p in nodes for p in n["depends_on"]))
            nodes[n["id"]] = n
        ancestors = {"model.logits"}
        for name in reversed(nodes):
            if name in ancestors:
                ancestors.update(nodes[name]["depends_on"])
        self.assertEqual(ancestors, set(nodes))
        events = {e["id"]: e for e in r["communication_events"]}
        self.assertTrue(all(e["communication_cycles"] is None and e["consumers"] for e in events.values()))
        self.assertEqual(events["step.1/token_ids"]["producers"], ["step.0/head/select_token"])
        self.assertEqual(events["step.1/token_ids"]["consumers"], ["step.1/embedding/lookup"])
        self.assertEqual(events["step.1/token_ids"]["logical_tensor_bytes"], 8)
        pp = events["step.0/pp.0.1"]
        self.assertEqual(pp["deliveries"], [dict(source=13,destination=54), dict(source=17,destination=66),
                                           dict(source=13,destination=83)])
        self.assertEqual((pp["shape"], pp["dtype"], pp["logical_tensor_bytes"]), ([2,32],"INT8",64))
        reduce = events["step.1/layer.2/attention.reduce"]
        self.assertEqual((reduce["participants"], reduce["input_bytes_per_rank"], reduce["int32_adds"]),
                         ([54,66,83], 256, 128))
        self.assertIn("step.0/layer.2/kv.append", nodes["step.1/layer.2/kv.append"]["depends_on"])
        gather = events["step.1/head/logits.gather"]
        self.assertEqual(gather["destinations"], [54])
        self.assertEqual([s["columns"] for s in gather["input_shards"]], [[0,7],[7,13],[13,19]])
        self.assertEqual(sum(s["payload_bytes"] for s in gather["input_shards"]), 2*19*4)
        self.assertEqual(gather["consumers"], ["model.logits"])

    def test_single_chip_endpoint_identities(self):
        r = _run_model(TINY, REQUEST, "tpu", load_hardware("tpu"), uniform_plan(3,1,1), "count")
        self.assertFalse(r["communication_events"])
        self.assertIsNone(r["chips"][0]["steps"][0]["local_compute_cycles"])
        deps = {n["id"]: n["depends_on"] for n in r["schedule"]}
        self.assertEqual(deps["step.1/embedding/lookup"], ["step.0/head/select_token"])
        self.assertEqual(deps["model.logits"], ["step.1/head/logits.fp32"])

    def test_last_stage_global_preflight_and_exact_capacity(self):
        spec = load_hardware("tpu")
        plan = [{"layers": [0,1], "chips": [7,8,9,10]}, {"layers": [1,3], "chips": [50]}]
        def run():
            return _run_model(TINY, REQUEST, "tpu", spec, plan, "count")
        peak = max(c["peak"]["required_bytes"] for c in run()["chips"])
        spec.values["hbm_usable_bytes"] = peak
        self.assertEqual(run()["chips"][-1]["peak"]["required_bytes"], peak)
        spec.values["hbm_usable_bytes"] -= 1
        with patch("src.models.decoder_kernels.DecoderKernels.resolve", side_effect=AssertionError("early execution")):
            with self.assertRaises(BlockCapacityError) as fail:
                run()
        error = fail.exception
        self.assertEqual((error.chip,error.level,error.required_bytes,error.capacity_bytes), (50,"HBM",peak,peak-1))
        self.assertTrue(error.node.startswith("stage.1/"))
        self.assertEqual(sum(error.tensors.values()), peak)

    def test_real_model_all_architectures(self):
        for arch in ("ascend","tpu","tsp","tsp_cim"):
            with self.subTest(arch=arch):
                r = run_llama_model(DecodeRequest(1,8190,2), arch, load_hardware(arch), mode="timing")
                layers = [layer for stage in r["stages"] for layer in range(*stage["layers"])]
                self.assertEqual(layers, list(range(80)))
                small = arch in ("tsp", "tsp_cim")
                self.assertEqual(len(r["chips"]), 336 if small else 3)
                self.assertEqual([len(s["chips"]) for s in r["stages"]], [16] + [4]*76 + [16] if small else [1]*3)
                layer_counts = [s["layers"][1] - s["layers"][0] for s in r["stages"]]
                self.assertEqual(layer_counts, [2] + [1]*76 + [2] if small else [27, 27, 26])
                # A TP16 endpoint stores each of the eight KV heads on two chips.
                kv_bytes = (4*16 + 76*8) * 2*128 if small else 80*8 * 2*128
                self.assertEqual([step["physical_kv_append_bytes"] for step in r["step_totals"]], [kv_bytes]*2)
                reductions = [e for e in r["communication_events"] if e["kind"] == "tp_all_reduce_sum"]
                self.assertEqual(len(reductions), 320 if small else 0)
                for c in r["chips"]:
                    p = r["profiles"][c["peak"]["profile"]]["capacity"]
                    self.assertLessEqual(c["peak"]["required_bytes"], p["capacity_bytes"])
                    if p["unit_capacity_bytes"]:
                        self.assertLessEqual(max(p["unit_peak_bytes"]), p["unit_capacity_bytes"])
                block_macs = 857735168 + (8191 - 128) * 2 * 8192
                self.assertEqual(r["step_totals"][0]["logical_useful_macs"], 80*block_macs + 8192*128256)

    def test_cli_exports_and_capacity_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, comm, plan = (Path(tmp)/name for name in ("model.json","model.communication.json","plan.json"))
            cmd = [sys.executable, str(ROOT/"scripts/run_llama_model.py"), "--architecture", "tsp", "--mode", "count",
                   "--output", str(output), "--decode-steps", "1"]
            subprocess.run(cmd, cwd=tmp, check=True, capture_output=True)
            original = [p.read_bytes() for p in (output,comm)]
            compute, communication = [json.loads(s) for s in original]
            self.assertEqual(compute["effective_lengths"], [8191])
            self.assertEqual(communication["kind"], "llama_communication")
            self.assertEqual(compute["stages"], communication["stages"])
            self.assertTrue(communication["events"])
            plan.write_text(json.dumps(uniform_plan(80,20,16)))
            failure = subprocess.run([*cmd, "--plan", str(plan)], cwd=tmp, capture_output=True)
            self.assertEqual(failure.returncode, 2)
            error = json.loads(failure.stderr)
            self.assertEqual((error["chip"], error["level"]), (0,"MEM"))
            uniform = subprocess.run([*cmd, "--pp", "20", "--tp", "16"], cwd=tmp, capture_output=True)
            self.assertEqual(json.loads(uniform.stderr), error)
            self.assertEqual([p.read_bytes() for p in (output,comm)], original)
