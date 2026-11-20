from tests.operators.inspection import inspect_result
"""Regression contracts for the operator execution layering refactor."""
import ast
from contextlib import redirect_stdout
import hashlib
from importlib.util import resolve_name
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import ascend
from src.core.composite_memory import LocalMemoryPath, MemoryKind, MemoryPath, ResidentMemoryPath
from src.core.operator_result import NativeResult
from src.hardware.vector import load_vector_execution
from src.operators import Gemm, RMSNorm, Softmax, TensorSpec, Vector, simulate_composite, simulate_gemm, simulate_vector
from src.operators.adapters import adapt_native_result
from src.runtime.backends import get_backend
from tests.operators.test_gemm import specs


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "validation/issue3/baselines/report_format.json"


def canonical_hash(report):
    return hashlib.sha256(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def baseline_operator(operation):
    if operation == "gemm":
        return Gemm.from_tensors(TensorSpec("activation", (2, 17), "INT8"),
                                 TensorSpec("weight", (17, 7), "INT8"), output_name="projection"), simulate_gemm
    if operation == "rmsnorm":
        return RMSNorm(TensorSpec("x", (2, 7), "FP32"), TensorSpec("weight", (7,), "FP32"),
                       output_name="norm"), simulate_composite
    if operation == "softmax":
        return Softmax(TensorSpec("x", (2, 7), "FP32"), output_name="prob"), simulate_composite
    dtype = "INT32" if operation == "int32_add" else "FP32"
    arity = 1 if operation in {"fp32_exp", "fp32_rsqrt"} else 2
    return Vector(operation, tuple(TensorSpec(f"x{i}", (2, 7), dtype) for i in range(arity)),
                  output_name="y"), simulate_vector


class LayeringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = specs()

    def test_complete_api_reports_match_pre_refactor_baseline(self):
        cases = json.loads(BASELINE.read_text(encoding="utf-8"))["api_cases"]
        self.assertEqual(len(cases), 64)
        for case in cases:
            arch, operation, mode = case["architecture"], case["operation"], case["mode"]
            with self.subTest(architecture=arch, operation=operation, mode=mode):
                op, simulate = baseline_operator(operation)
                report = inspect_result(simulate(op, arch, self.specs[arch], mode=mode, **case["policy"], include_details=True))
                self.assertEqual(canonical_hash(report["native_report"]), case["native_sha256"])
                self.assertEqual(canonical_hash(report["operator"]), case["operator_sha256"])
                self.assertEqual(canonical_hash(report["action_ledger"]), case["ledger_sha256"])

    def test_core_has_no_dependency_on_public_operator_or_runtime_layers(self):
        for path in (ROOT / "src/core").rglob("*.py"):
            package = ".".join(path.relative_to(ROOT).parts[:-1])
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    module = resolve_name("." * node.level + module, package) if node.level else module
                    names = [module] + [module + "." + alias.name for alias in node.names]
                with self.subTest(path=path.relative_to(ROOT), line=getattr(node, "lineno", None)):
                    self.assertFalse(any(n == layer or n.startswith(layer + ".")
                                         for n in names for layer in ("src.operators", "src.runtime")))

    def test_native_results_adapt_without_recounting_or_aliasing(self):
        op, _ = baseline_operator("fp32_add")
        for arch, spec in self.specs.items():
            with self.subTest(architecture=arch):
                native = get_backend(arch, spec).vector_kernel(op, spec, load_vector_execution(arch),
                                                              mode="timing", tile_elements=4)
                self.assertIsInstance(native, NativeResult)
                result = adapt_native_result(op.spec, arch, native, include_details=True)
                exported = inspect_result(result)
                self.assertEqual(exported["native_report"], native.report)
                self.assertEqual(exported["latency"]["cycle_count"], native.report["timing"]["total_cycles"])
                for entry in exported["action_ledger"]["entries"]:
                    source = native.report["action_counts"][entry["id"]]
                    for field in ("count", "unit", "scope", "basis"):
                        self.assertEqual(entry[field], source[field])
                native.report["timing"]["total_cycles"] += 1
                self.assertEqual(inspect_result(result), exported)
                native.action_specs.pop(next(iter(native.action_specs)))
                with self.assertRaisesRegex(ValueError, "metadata"):
                    adapt_native_result(op.spec, arch, native, include_details=True)

    def test_memory_paths_require_complete_fields_and_expose_storage_kind(self):
        for path_type in (MemoryPath, LocalMemoryPath, ResidentMemoryPath):
            with self.assertRaises(TypeError):
                path_type()
        op, _ = baseline_operator("softmax")
        for arch, spec in self.specs.items():
            with self.subTest(architecture=arch):
                path = get_backend(arch, spec).composite_path(op, spec, load_vector_execution(arch),
                                                            mode="timing", tile_elements=4)
                self.assertIsInstance(path, MemoryPath)
                if arch in {"ascend", "tpu"}:
                    self.assertIs(path.kind, MemoryKind.LOCAL_DMA)
                    self.assertEqual(path.temporary_capacity_bytes, path.memory[path.local]["allocated_bytes"])
                else:
                    self.assertIs(path.kind, MemoryKind.RESIDENT)
                    self.assertEqual(path.temporary_capacity_bytes, path.memory["scratch_tile_bytes"] + path.memory["row_state_bytes"])
                counts, cycles = path.access([28], "read")
                self.assertGreater(cycles, 0)
                self.assertEqual(counts["mem_read_bytes" if path.kind is MemoryKind.RESIDENT else path.local + "_read_bytes"], 28)

    def test_ascend_vector_cli_does_not_construct_gemm_simulator(self):
        with patch.object(ascend, "AscendSimulator", side_effect=AssertionError("unexpected GEMM construction")), \
                patch.object(ascend, "write_operator_result") as output, redirect_stdout(io.StringIO()):
            self.assertEqual(ascend.main(["--workload", "operators/fp32_add.json", "--mode", "count", "--output", ""]), 0)
            self.assertEqual(output.call_count, 1)
            self.assertEqual(output.call_args.args[0].operator.kind, "fp32_add")
