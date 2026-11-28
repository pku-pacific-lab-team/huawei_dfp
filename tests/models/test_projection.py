import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_projection import load_hardware
from src.models.config import DecodeRequest, load_model
from src.models.projection import run_up_projection
from src.operators.primitives import Gemm
from src.runtime.simulation import simulate_gemm


ROOT = Path(__file__).resolve().parents[2]
REQUEST = DecodeRequest(1, 127, 2)


class ProjectionTest(unittest.TestCase):
    def test_pinned_model_dimensions_and_decode_bounds(self):
        llama = load_model("llama3_70b")
        self.assertEqual([llama[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size",
                         "num_attention_heads", "num_key_value_heads", "vocab_size")],
                         [80, 8192, 28672, 64, 8, 128256])
        r1 = load_model("deepseek_r1")
        self.assertEqual([r1[k] for k in ("num_hidden_layers", "hidden_size", "first_k_dense_replace",
                         "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
                         "n_routed_experts", "num_experts_per_tok", "n_shared_experts", "vocab_size")],
                         [61, 7168, 3, 1536, 512, 128, 64, 128, 256, 8, 1, 129280])
        for config in (llama, r1):
            self.assertEqual(list(REQUEST.effective_lengths(config)), [128, 129])
            limit = config["max_position_embeddings"]
            self.assertEqual(list(DecodeRequest(1, limit-1, 1).effective_lengths(config)), [limit])
            with self.assertRaisesRegex(ValueError, "context"):
                DecodeRequest(1, limit-1, 2).effective_lengths(config)
        for args in ((0, 1, 1), (1, -1, 2), (1, 1, 0), (True, 1, 1), (1, 1.5, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                DecodeRequest(*args)

    def test_real_projection_matches_native_for_every_backend_and_mode(self):
        for arch, units in (("ascend", 32), ("tpu", 2), ("tsp", 2), ("tsp_cim", 2)):
            spec = load_hardware(arch)
            local_n = 7168 // units
            for mode in ("count", "timing"):
                with self.subTest(arch=arch, mode=mode):
                    native = simulate_gemm(Gemm(1, 8192, local_n), arch, spec, mode=mode, include_details=True)
                    with patch("src.models.projection.simulate_gemm", wraps=simulate_gemm) as run:
                        report = run_up_projection("llama3_70b", REQUEST, arch, spec, tp=4, mode=mode)
                        run.assert_called_once()
                    self.assertEqual(report["useful_macs_per_step"], 234881024)
                    self.assertEqual([c["global_output_columns"] for c in report["chips"]],
                                     [[0, 7168], [7168, 14336], [14336, 21504], [21504, 28672]])
                    resources = report["resources"]
                    for k, v in resources["bandwidth_per_unit"].items():
                        self.assertEqual(v * units, resources["chip_bandwidth"][k])
                    profile = report["chip_profiles"]["7168"]
                    self.assertEqual(profile["instances_per_local_run"], {str(local_n): units})
                    self.assertEqual(profile["useful_macs"], 58720256)
                    self.assertEqual([u["chip_columns"] for u in profile["unit_slices"]],
                                     [[i*local_n, (i+1)*local_n] for i in range(units)])
                    self.assertEqual(profile["cycle_count"], native.cycle_count)
                    self.assertEqual(profile["latency_s"], native.latency_s)
                    for action in native.action_ledger.entries:
                        aggregate = profile["action_counts"][action.id]
                        self.assertEqual((aggregate["count"], aggregate["unit"], aggregate["role"]),
                                         (action.count * units, action.unit, action.role))
                    if mode == "timing":
                        self.assertGreater(profile["cycle_count"], 0)
                        timing = native.native_report["timing"]
                        shares = resources["bandwidth_per_unit"]
                        if arch == "ascend":
                            self.assertEqual(shares["hbm_bytes_per_second"],
                                             timing["derived_services"]["hbm"]["share_bytes_per_second"])
                        elif arch == "tpu":
                            self.assertEqual(shares["hbm_bytes_per_second"],
                                             timing["parameters"]["hbm_share_bytes_per_second"]["value"])
                        else:
                            slices = native.native_report["policy"]["mem_slices"]
                            rate = timing["parameters"]["mem_read_bpc"]["value"]
                            self.assertEqual(shares["mem_read_bytes_per_cycle"], len(slices) * rate)
                    else:
                        self.assertIsNone(profile["cycle_count"])

    def test_uneven_partitions_preserve_offsets_and_max_elapsed(self):
        report = run_up_projection("llama3_70b", DecodeRequest(2, 127, 2), "ascend", load_hardware("ascend"),
                                   tp=3, layer=79, mode="timing")
        self.assertEqual(report["useful_macs_per_step"], 2 * 8192 * 28672)
        self.assertEqual(report["predecessors"], ["layers.79.ffn_norm"])
        previous = 0
        for chip in report["chips"]:
            lo, hi = chip["global_output_columns"]
            self.assertEqual(lo, previous)
            profile = report["chip_profiles"][chip["chip_profile"]]
            local_cycles = []
            for unit in profile["unit_slices"]:
                a, b = unit["chip_columns"]
                self.assertEqual(lo + a, previous)
                self.assertLess(a, b)
                previous = lo + b
                local_cycles.append(report["local_runs"][unit["local_run"]]["cycle_count"])
            self.assertEqual(hi, previous)
            self.assertEqual(profile["cycle_count"], max(local_cycles))
        self.assertEqual(previous, 28672)

    def test_projection_boundaries_and_shared_hbm(self):
        spec = load_hardware("ascend")
        for tp in (True, 0, -1, 28672, 28673):
            with self.subTest(tp=tp), self.assertRaises(ValueError):
                run_up_projection("llama3_70b", REQUEST, "ascend", spec, tp=tp)
        result = run_up_projection("deepseek_r1", REQUEST, "ascend", spec, tp=4, layer=2)
        self.assertEqual(result["useful_macs_per_step"], 7168 * 18432)
        with self.assertRaisesRegex(ValueError, "dense FFN"):
            run_up_projection("deepseek_r1", REQUEST, "ascend", spec, tp=4, layer=3)
        # Each native shard fits; the sum of all live allocations exceeds chip HBM.
        spec.chip["hbm_usable_bytes"] = 4_000_000
        with self.assertRaisesRegex(ValueError, "across units"):
            run_up_projection("llama3_70b", REQUEST, "ascend", spec, tp=4)

    def test_cli_from_another_directory_preserves_output_on_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "report.json"
            args = [sys.executable, str(ROOT / "scripts/run_projection.py"), "--architecture", "tpu",
                    "--mode", "count", "--output", str(target)]
            subprocess.run(args, cwd=folder, check=True, capture_output=True)
            previous = target.read_bytes()
            self.assertEqual(json.loads(previous)["effective_lengths"], [8191, 8192])
            failed = subprocess.run([*args, "--sequence-length", "8192"], cwd=folder, capture_output=True)
            self.assertEqual(failed.returncode, 2)
            self.assertEqual(target.read_bytes(), previous)
            subprocess.run([*args, "--model", "deepseek_r1"], cwd=folder, check=True, capture_output=True)
            self.assertEqual(json.loads(target.read_bytes())["effective_lengths"], [131073, 131074])
            subprocess.run([*args, "--sequence-length", "127"], cwd=folder, check=True, capture_output=True)
            self.assertEqual(json.loads(target.read_bytes())["effective_lengths"], [128, 129])
