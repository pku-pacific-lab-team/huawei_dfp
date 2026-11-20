"""Public router/combine/MoE entry points and non-destructive CLI errors."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.blocks import load_block_execution
from tests.operators.test_mla import ROOT, config


class MoEBlockCLITest(unittest.TestCase):
    def invoke(self, arch, folder, *args):
        return subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / f"{arch}.py"), *args], cwd=folder,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_four_entries_count_timing_and_custom_profile_reservations(self):
        with tempfile.TemporaryDirectory() as folder:
            out, details, profile = (Path(folder) / n for n in ("out.json", "details.json", "block.json"))
            block = copy.deepcopy(load_block_execution().config)
            block["parameters"]["fp32_rcp_cost"]["value"] = 70
            profile.write_text(json.dumps(block), encoding="utf-8")
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                opts = ["--mem-slices", "0", "1", "2", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                if arch == "tsp_cim":
                    opts += ["--hemisphere", "1"]
                for kind in ("moe_router", "moe_combine", "moe"):
                    for mode in ("count", "timing"):
                        p = self.invoke(arch, folder, "--workload", f"operators/{kind}.json", "--mode", mode,
                            "--block-config", str(profile), "--vector-tile-elements", "3", "--output", str(out), "--details-output", str(details), *opts)
                        self.assertEqual(p.returncode, 0, p.stderr)
                        result = json.loads(out.read_text(encoding="utf-8"))
                        native = json.loads(details.read_text(encoding="utf-8"))["native_report"]
                        self.assertEqual(result["operator"]["kind"], kind)
                        self.assertEqual(native["block_execution_profile"], block)
                        self.assertEqual(result["work_counts"]["logical"]["operations"], {"moe_router":330,"moe_combine":60,"moe":3414}[kind])
                        self.assertEqual(result["cycle_count"] is None, mode == "count")
                        if arch == "tsp_cim":
                            self.assertEqual(result["boundary"]["selected_hemisphere"], 1)

    def test_invalid_nested_scenarios_and_profiles_preserve_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            out, detail, workload, profile = (Path(folder) / n for n in ("out.json", "detail.json", "bad.json", "block.json"))
            out.write_text("old summary"); detail.write_text("old details")
            bad = []
            cfg = config("moe"); cfg["router"]["selected_groups"] = 1; bad.append(cfg)
            cfg = config("moe"); cfg["experts"]["routing"]["assignments"][0] = [1, 1]; bad.append(cfg)
            cfg = config("moe_router"); cfg["route_scale"] = 0; bad.append(cfg)
            cfg = config("moe_combine"); cfg["weights"]["shape"] = [3, 1]; bad.append(cfg)
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for cfg in bad:
                    workload.write_text(json.dumps(cfg), encoding="utf-8")
                    p = self.invoke(arch, folder, "--workload", str(workload), "--output", str(out), "--details-output", str(detail))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(out.read_text(), "old summary")
                    self.assertEqual(detail.read_text(), "old details")
                block = copy.deepcopy(load_block_execution().config)
                block["parameters"]["fp32_rcp_cost"].update(value=None, status="unknown")
                profile.write_text(json.dumps(block), encoding="utf-8")
                for extra in (["--block-config", str(profile), "--mode", "timing"], ["--vector-tile-elements", "0"], ["--report-format", "legacy"]):
                    p = self.invoke(arch, folder, "--workload", "operators/moe.json", "--output", str(out), "--details-output", str(detail), *extra)
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertEqual(out.read_text(), "old summary")
                    self.assertEqual(detail.read_text(), "old details")

    def test_chip_scope_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for kind in ("moe", "moe_router", "moe_combine"):
                p = self.invoke("tsp", folder, "--workload", f"operators/{kind}.json", "--scope", "chip", "--output", "")
                self.assertEqual(p.returncode, 2)
                self.assertIn("chip execution is not modeled", p.stderr)


if __name__ == "__main__":
    unittest.main()
