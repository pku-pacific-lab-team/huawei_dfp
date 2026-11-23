from tests.operators.inspection import inspect_cli_report
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.hardware.vector import VectorExecution, load_vector_execution
from src.operators import TensorSpec, Softmax, Vector, simulate_composite, simulate_vector
from tests.operators.test_gemm import specs


ROOT = Path(__file__).resolve().parents[2]


class CompositeCLITest(unittest.TestCase):
    def run_entry(self, arch, folder, *args):
        return subprocess.run([sys.executable, str(ROOT / f"{arch}.py"), *args, "--details-output", str(Path(folder) / "diagnostics.json")], cwd=folder,
                              text=True, encoding="utf-8", capture_output=True, timeout=30)

    def test_four_entries_both_modes_from_another_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "nested/result.json"
            for arch in ("ascend", "tpu", "tsp", "tsp_cim"):
                for kind in ("rmsnorm", "softmax"):
                    for mode in ("count", "timing"):
                        with self.subTest(arch=arch, kind=kind, mode=mode):
                            options = ["--mem-slices", "0", "1", "--streams", "2", "--reserved-bytes", "17"] if arch in {"tsp", "tsp_cim"} else []
                            if arch == "tsp_cim":
                                options += ["--hemisphere", "1"]
                            p = self.run_entry(arch, folder, "--workload", f"operators/{kind}.json", "--mode", mode,
                                               "--vector-tile-elements", "80", "--output", str(output), *options)
                            self.assertEqual(p.returncode, 0, p.stderr)
                            r = inspect_cli_report(output, Path(folder) / "diagnostics.json")
                            self.assertEqual(r["operator"]["kind"], kind)
                            self.assertEqual(r["execution_contract"]["selected_tile_elements"], 80)
                            self.assertEqual(r["execution_contract"]["schedule"], "serial_rows_passes_tiles")
                            self.assertEqual(r["latency"]["status"], "not_evaluated" if mode == "count" else "analytical_estimate")
                            if arch == "tsp_cim":
                                self.assertEqual(r["boundary"]["selected_hemisphere"], 1)
                                self.assertEqual(r["native_report"]["memory"]["reserved_per_slice_bytes"], 9)

    def test_explicit_profile_and_failures_preserve_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output, profile, bad = (Path(folder) / n for n in ("result.json", "profile.json", "bad.json"))
            config = load_vector_execution("tpu").config
            config["composite"]["parameters"]["reduction_round_cycles"]["value"] = 5
            profile.write_text(json.dumps(config), encoding="utf-8")
            p = self.run_entry("tpu", folder, "--workload", "operators/softmax.json", "--vector-config", str(profile), "--output", str(output))
            self.assertEqual(p.returncode, 0, p.stderr)
            saved = output.read_bytes()
            result = inspect_cli_report(output, Path(folder) / "diagnostics.json")
            self.assertGreater(result["native_report"]["timing"]["elapsed_cycles"]["by_stage"]["reduction_dependency"], 0)
            bad.write_text(json.dumps(dict(schema_version=1, operation="rmsnorm", inputs=[dict(name="x", shape=[2, 4], dtype="FP32")])), encoding="utf-8")
            for arch, args in (("ascend", ["--buf", "1"]), ("tpu", ["--cmem-mode", "bypass"]),
                               ("tsp", ["--scope", "chip"]), ("tsp_cim", ["--compare-mxm"])):
                p = self.run_entry(arch, folder, "--workload", "operators/softmax.json", "--output", str(output), *args)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertEqual(output.read_bytes(), saved)
                p = self.run_entry(arch, folder, "--workload", str(bad), "--output", str(output))
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertEqual(output.read_bytes(), saved)
            p = self.run_entry("tpu", folder, "--workload", "operators/softmax.json", "--report-format", "legacy", "--output", str(output))
            self.assertEqual(p.returncode, 2)
            self.assertEqual(output.read_bytes(), saved)

    def test_profile_v1_still_runs_vector_but_rejects_composites(self):
        for arch, spec in specs().items():
            v2 = load_vector_execution(arch)
            config = copy.deepcopy(v2.config)
            config["schema_version"] = 1
            config.pop("composite")
            old = VectorExecution(config, arch)
            op = Vector("fp32_exp", (TensorSpec("x", (17,), "FP32"),))
            a = simulate_vector(op, arch, spec, mode="timing", execution=old).to_dict()
            b = simulate_vector(op, arch, spec, mode="timing", execution=v2).to_dict()
            self.assertEqual((a["cycle_count"], a["latency_s"]), (b["cycle_count"], b["latency_s"]))
            self.assertEqual(a["action_counts"], b["action_counts"])
            with self.assertRaisesRegex(ValueError, "schema 2"):
                simulate_composite(Softmax(op.inputs[0]), arch, spec, execution=old)

    def test_composite_profile_sources_hash_and_validation(self):
        cfg = load_vector_execution("ascend").config
        audit = json.loads((ROOT / "calibration/operator_execution_audit.json").read_text(encoding="utf-8"))
        record = audit["vector_execution_profiles"]["configs/execution/vector.json"]
        digest = hashlib.sha256(json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(record["canonical_json_sha256"], digest)
        self.assertEqual(record["schema_version"], 2)
        self.assertEqual(set(record["composite_parameters"]), set(cfg["composite"]["parameters"]))
        for parameter in record["composite_parameters"].values():
            self.assertEqual(parameter["status"], "assumed")
            for evidence in parameter["evidence"]:
                source = audit["sources"][evidence["source_id"]]
                self.assertEqual(source["classification"], "modeling_assumption")
                self.assertIn('id="' + evidence["section"] + '"', (ROOT / source["path"]).read_text(encoding="utf-8"))
        for path, value in ((["schedule"], "parallel"), (["storage"], "free"), (["parameters", "fp32_rcp_cost", "value"], 0),
                            (["parameters", "reduction_round_cycles", "value"], 1.5)):
            bad = copy.deepcopy(cfg)
            location = bad["composite"]
            for key in path[:-1]:
                location = location[key]
            location[path[-1]] = value
            with self.assertRaises(ValueError):
                VectorExecution(bad, "tpu")
        broken = copy.deepcopy(cfg)
        broken.pop("composite")
        with self.assertRaises(ValueError):
            VectorExecution(broken, "tpu")
