from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.workload import load_workload_config, parse_workloads


# 按文件加载 CLI，避免 unittest 将同名的 tests/tsp、tests/tsp_cim 包当成入口。
ROOT = Path(__file__).resolve().parents[2]
tsp_entry = importlib.util.spec_from_file_location("tsp_cli", ROOT / "tsp.py")
tsp = importlib.util.module_from_spec(tsp_entry)
tsp_entry.loader.exec_module(tsp)
tsp_cim_entry = importlib.util.spec_from_file_location("tsp_cim_cli", ROOT / "tsp_cim.py")
tsp_cim = importlib.util.module_from_spec(tsp_cim_entry)
tsp_cim_entry.loader.exec_module(tsp_cim)


class WorkloadTest(unittest.TestCase):
    def config(self, fmt="BF16"):
        return {"gemm": {"M": 16, "K": 32, "N": 16}, "data_format": fmt}

    def test_declaration_and_override(self):
        self.assertEqual(parse_workloads(self.config())[0].data_format, "BF16")
        workloads = parse_workloads(self.config(), ["INT8", "FP16"])
        self.assertEqual([w.data_format for w in workloads], ["INT8", "FP16"])
        self.assertTrue(all(w.format_overridden for w in workloads))
        self.assertEqual(workloads[0].output_dtype, "INT32")
        self.assertEqual(workloads[1].output_dtype, "FP32")

    def test_invalid_or_unsupported_semantics_are_rejected(self):
        for value in (0, -1, True, 1.5):
            config = self.config()
            config["gemm"]["M"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_workloads(config)
        for field, value in (("data_format", "bad"), ("output_dtype", "FP16"),
                             ("custom_overheads", {"x": 1}), ("transpose_a", True)):
            config = self.config()
            config[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_workloads(config)
        config = self.config()
        del config["gemm"]["K"]
        with self.assertRaises(ValueError):
            parse_workloads(config)

    def test_all_loaders_share_shape_and_validation(self):
        for loader in (TSPConfigLoader(), TSPCIMConfigLoader()):
            self.assertEqual(loader.get_workload(self.config()), (16, 32, 16))
            with self.assertRaises(ValueError):
                loader.get_workload({"gemm": {}, "data_format": "INT8"})

    def test_single_int8_preset_and_loader_defaults(self):
        root = Path(__file__).resolve().parents[2] / "configs"
        self.assertEqual(sorted(p.name for p in (root / "workload").glob("*.json")), ["tiny_gemm.json"])
        config = load_workload_config("tiny_gemm.json", root)
        task = parse_workloads(config)[0]
        self.assertEqual(task.shape, (512, 256, 512))
        self.assertEqual((task.data_format, task.accumulation_dtype, task.output_dtype), ("INT8", "INT32", "INT32"))
        self.assertFalse(task.format_overridden)
        for loader in (TSPConfigLoader(), TSPCIMConfigLoader()):
            self.assertEqual(loader.load_workload_config(), config)

    def test_tsp_entries_use_shared_precision_and_print_effective_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(self.config("FP16")), encoding="utf-8")
            for module in (tsp, tsp_cim):
                for override in ([], ["--formats", "INT8"]):
                    out, err = StringIO(), StringIO()
                    with self.subTest(entry=module.__name__, override=override), patch("sys.argv", [module.__name__, "--workload", str(path), "--output", str(Path(directory) / "report.json")] + override), redirect_stdout(out), redirect_stderr(err):
                        status = module.main()
                        if module is tsp:
                            self.assertEqual(status, 0 if override else 2, err.getvalue())
                            if not override:
                                self.assertIn("Native reference", err.getvalue())
                    line = next(line for line in out.getvalue().splitlines() if line.startswith("[Effective workload]"))
                    metadata = json.loads(line.partition("] ")[2])
                    self.assertEqual(metadata["data_format"], "INT8" if override else "FP16")
                    self.assertEqual(metadata["format_overridden"], bool(override))

    def test_tsp_cim_entry_rejects_unsupported_precision_and_counts_explicit_int8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(self.config("FP16")), encoding="utf-8")
            for override, expected_status in (([], 2), (["--formats", "INT8"], 0),
                                               (["--formats", "INT8", "BF16"], 2)):
                out, err = StringIO(), StringIO()
                with patch("sys.argv", ["tsp_cim.py", "--workload", str(path), "--output", str(Path(directory) / "report.json")] + override), redirect_stdout(out), redirect_stderr(err):
                    self.assertEqual(tsp_cim.main(), expected_status, err.getvalue())
                if expected_status == 2:
                    self.assertIn("INT8", err.getvalue())
                else:
                    self.assertIn('"issued_mac"', out.getvalue())
                    self.assertIn('"total_cycles": 296', out.getvalue())

    def test_override_is_not_reparsed_as_declared_precision(self):
        config = self.config("FP16")
        config.update(accumulation_dtype="INT32", output_dtype="INT32")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            for module in (tsp, tsp_cim):
                with patch("sys.argv", [module.__name__, "--workload", str(path), "--formats", "INT8", "--output", str(Path(directory) / "report.json")]), redirect_stdout(StringIO()):
                    module.main()
