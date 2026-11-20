"""MoE entry points, strict fixed-routing workloads and preserved output on error."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.blocks import load_block_execution
from tests.operators.test_mla import ROOT, config


class MoECLITest(unittest.TestCase):
    def invoke(self, arch, folder, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args], cwd=folder,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_four_entries_count_timing_profiles_and_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            out, details, profile = (Path(folder) / n for n in ("result.json", "details.json", "block.json"))
            block = copy.deepcopy(load_block_execution().config)
            block["parameters"]["to_fp32_cost"]["value"] = 30
            profile.write_text(json.dumps(block), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                opts = ["--mem-slices", "0", "1", "2", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    opts += ["--hemisphere", "1"]
                for mode in ("count", "timing"):
                    p = self.invoke(arch, folder, "--workload", "operators/moe_experts.json", "--mode", mode,
                        "--block-config", str(profile), "--vector-tile-elements", "3", "--output", str(out), "--details-output", str(details), *opts)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    result = json.loads(out.read_text(encoding="utf-8"))
                    native = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                    self.assertEqual(result["operator"]["kind"], "moe_experts")
                    self.assertEqual([t["shape"] for t in result["operator"]["outputs"]], [[3, 2, 5], [3, 5]])
                    self.assertEqual(native["routing"]["counts"], [3, 1, 2, 0, 0])
                    self.assertEqual(native["block_execution_profile"], block)
                    if mode == "count":
                        self.assertIsNone(result["cycle_count"])
                    else:
                        self.assertGreater(result["cycle_count"], 0)
                    if opts:
                        self.assertEqual(native["memory"]["lifetime_validation"]["global_allocations"]["reserved"], 18)
                    if arch == "tsp_cim":
                        self.assertEqual(result["boundary"]["selected_hemisphere"], 1)

    def test_routed_only_and_fp32_standalone_ffn_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            workload, out = (Path(folder) / n for n in ("workload.json", "out.json"))
            cfg = config("moe_experts")
            for key in ("shared_experts", "shared_gate_weight", "shared_up_weight", "shared_down_weight", "output_names"):
                cfg.pop(key)
            workload.write_text(json.dumps(cfg), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                p = self.invoke(arch, folder, "--workload", str(workload), "--mode", "count", "--output", str(out))
                self.assertEqual(p.returncode, 0, p.stderr)
                result = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(len(result["operator"]["outputs"]), 1)
                self.assertEqual(result["work_counts"]["logical"]["useful_macs"], 630)
            cfg = config("swiglu_ffn")
            cfg["output_dtype"] = "FP32"
            workload.write_text(json.dumps(cfg), encoding="utf-8")
            p = self.invoke("tpu", folder, "--workload", str(workload), "--output", str(out))
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["operator"]["outputs"][0]["dtype"], "FP32")

    def test_invalid_routes_fields_profiles_and_options_preserve_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            out, details, bad, blockfile = (Path(folder) / n for n in ("out.json", "detail.json", "bad.json", "block.json"))
            out.write_text("summary", encoding="utf-8")
            details.write_text("details", encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for change in ({"routing": {"expert_count": 5, "assignments": [[0, 0]] * 3}},
                               {"routing": {"expert_count": 5, "assignments": [[0, 5]] * 3}},
                               {"shared_experts": 0}, {"route_scale": 2.5}, {"output_names": ["tokens", "shared"]}):
                    cfg = config("moe_experts")
                    cfg.update(change)
                    bad.write_text(json.dumps(cfg), encoding="utf-8")
                    p = self.invoke(arch, folder, "--workload", str(bad), "--output", str(out), "--details-output", str(details))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(out.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")
                block = copy.deepcopy(load_block_execution().config)
                block["parameters"]["to_fp32_cost"].update(value=None, status="unknown")
                blockfile.write_text(json.dumps(block), encoding="utf-8")
                for opts in (["--vector-tile-elements", "0"], ["--report-format", "legacy"],
                             ["--attention-config", str(bad)], ["--mode", "timing", "--block-config", str(blockfile)]):
                    p = self.invoke(arch, folder, "--workload", "operators/moe_experts.json", *opts,
                        "--output", str(out), "--details-output", str(details))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(out.read_text(), "summary")
                    self.assertEqual(details.read_text(), "details")
            p = self.invoke("tsp", folder, "--workload", "operators/moe_experts.json", "--scope", "chip", "--output", "")
            self.assertEqual(p.returncode, 2)
            self.assertIn("chip execution is not modeled", p.stderr)


if __name__ == "__main__":
    unittest.main()
