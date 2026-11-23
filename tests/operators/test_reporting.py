"""Compact results, opt-in diagnostics and the schema-2 export contract."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.operators import Gemm, TensorSpec, Vector, simulate_gemm, simulate_vector, plan_connection
from tests.operators.test_gemm import specs
from tests.operators.test_layering import canonical_hash


ROOT = Path(__file__).resolve().parents[2]


class ReportingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def task(self):
        return Vector("fp32_add", (TensorSpec("x0", (1025,), "FP32"), TensorSpec("x1", (1025,), "FP32")))

    def test_default_result_drops_diagnostics_and_exposes_scalar_results(self):
        for arch, spec in self.hardware.items():
            with self.subTest(architecture=arch):
                result = simulate_vector(self.task(), arch, spec, mode="timing")
                detailed = simulate_vector(self.task(), arch, spec, mode="timing", include_details=True)
                summary = result.to_dict()
                self.assertEqual(summary, detailed.to_dict())
                self.assertIsNone(result._native_report)
                with self.assertRaisesRegex(ValueError, "include_details"):
                    result.to_detailed_dict()
                self.assertEqual(summary["schema_version"], 2)
                self.assertEqual(summary["cycle_count"], result.cycle_count)
                self.assertEqual(summary["latency_s"], result.latency_s)
                self.assertFalse({"native_report", "io_contract", "action_ledger", "execution_contract"} & summary.keys())
                self.assertEqual(summary["action_counts"]["fp32_add"]["count"], 1025)
                self.assertEqual(summary["energy"], {"status": "not_evaluated", "total_pj": None})
                self.assertEqual(detailed.to_detailed_dict()["result_sha256"], canonical_hash(summary))
                self.assertLess(len(json.dumps(summary)), 8000)
                if arch == "ascend":
                    self.assertEqual((result.cycle_count, result.latency_s), (400, 4e-7))
                # Connection checks work without retaining native reports.
                self.assertIn("placement_and_allocation_validation", plan_connection(result, result)["requirements"])

    def test_memory_scopes_and_allocation_bases_survive_compaction(self):
        for arch, spec in self.hardware.items():
            vector = simulate_vector(self.task(), arch, spec).to_dict()["resource_requirements"]
            if arch in {"ascend", "tpu"}:
                local = "ub" if arch == "ascend" else "vmem"
                self.assertEqual(vector["hbm"]["required_bytes"], 12300)
                self.assertEqual(vector["hbm"]["requirement_basis"], "logical_footprint_bytes")
                self.assertEqual(vector[local]["required_bytes"], 12288)
                self.assertEqual(vector[local]["requirement_basis"], "allocated_bytes")
                if arch == "tpu":
                    self.assertEqual(vector["cmem"]["mode"], "bypass")
                    self.assertIsNone(vector["cmem"]["capacity_bytes"])
            else:
                self.assertEqual(vector["mem"]["required_bytes"], 12408)
                self.assertEqual(vector["mem"]["requirement_basis"], "resident_bytes")
                self.assertEqual(vector["mem"]["scope"], "single_hem")
            gemm = simulate_gemm(Gemm(2, 17, 7), arch, spec, include_details=True)
            native = gemm.native_report["memory"]
            for key, memory in gemm.resource_requirements.items():
                original = native if key == "mem" else native[key]
                self.assertEqual(memory["required_bytes"], original[memory["requirement_basis"]])
                self.assertLessEqual(memory["required_bytes"], memory["capacity_bytes"])

    def test_actions_preserve_roles_units_and_provenance_in_separate_view(self):
        result = simulate_gemm(Gemm(1, 17, 33), "tsp_cim", self.hardware["tsp_cim"], include_details=True)
        summary, details = result.to_dict(), result.to_detailed_dict()
        actions = summary["action_counts"]
        self.assertEqual(set(actions), {a.id for a in result.action_ledger.entries})
        self.assertTrue(any(a["role"] == "summary" for a in actions.values()))
        self.assertTrue(any(a["role"] == "reference" for a in actions.values()))
        for action in result.action_ledger.entries:
            for key in ("count", "unit", "dtype", "scope", "basis", "role"):
                self.assertEqual(actions[action.id][key], getattr(action, key))
            self.assertEqual(details["action_metadata"][action.id]["source"], action.source)
            self.assertNotIn("source", actions[action.id])
        summary["action_counts"].clear()
        details["native_report"]["counts"].clear()
        self.assertTrue(result.to_dict()["action_counts"])
        self.assertTrue(result.native_report["counts"])

    def test_cli_details_are_explicit_and_bound_to_the_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for arch in self.hardware:
                output, details = folder / f"{arch}.result.json", folder / f"{arch}.details.json"
                args = [sys.executable, str(ROOT / f"{arch}.py"), "--workload", "operators/fp32_add.json",
                        "--output", str(output)]
                p = subprocess.run(args, cwd=folder, capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertFalse(details.exists())
                compact = output.read_bytes()
                self.assertNotIn("parameter_provenance", p.stdout)
                self.assertLess(len(p.stdout.splitlines()), 25)
                p = subprocess.run(args + ["--details-output", str(details)], cwd=folder,
                                   capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(output.read_bytes(), compact)
                detail = json.loads(details.read_text(encoding="utf-8"))
                self.assertEqual(detail["result_sha256"], canonical_hash(json.loads(compact)))
                self.assertIn("parameter_provenance", detail["native_report"])
                before = output.read_bytes()
                p = subprocess.run(args + ["--details-output", str(output)], cwd=folder, capture_output=True, timeout=30)
                self.assertEqual(p.returncode, 2)
                self.assertEqual(output.read_bytes(), before)

    def test_count_and_missing_clock_keep_unknown_time(self):
        counted = simulate_vector(self.task(), "tsp", self.hardware["tsp"]).to_dict()
        self.assertIsNone(counted["cycle_count"])
        self.assertIsNone(counted["latency_s"])
        self.assertEqual(counted["timing_status"], "not_evaluated")
        from src.hardware.tsp import TSPSpec
        from src.utils.tsp_config import TSPConfigLoader
        config = TSPConfigLoader().load_hardware_config()
        config["parameters"]["clock_ghz"].update(value=None, status="unknown")
        timed = simulate_vector(self.task(), "tsp", TSPSpec(config), mode="timing")
        self.assertGreater(timed.cycle_count, 0)
        self.assertIsNone(timed.latency_s)
        self.assertEqual(timed.timing_status, "cycles_only")

    def test_diagnostics_only_output_and_explicit_gemm_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target, details = folder / "summary.json", folder / "details.json"
            target.write_text("preserve", encoding="utf-8")
            base = [sys.executable, str(ROOT / "ascend.py")]
            for options in (["--details-output", str(details)], ["--report-format", "operator", "--trace"]):
                p = subprocess.run(base + ["--output", str(target), *options], cwd=folder,
                                   capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(p.returncode, 2)
                self.assertEqual(target.read_text(encoding="utf-8"), "preserve")
                self.assertFalse(details.exists())
            p = subprocess.run(base + ["--workload", "operators/fp32_add.json", "--output", "",
                                       "--details-output", str(details)], cwd=folder,
                               capture_output=True, text=True, encoding="utf-8", timeout=30)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertNotIn("[Report]", p.stdout)
            self.assertEqual(json.loads(details.read_text(encoding="utf-8"))["kind"], "operator_details")
            self.assertEqual(target.read_text(encoding="utf-8"), "preserve")
