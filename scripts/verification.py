"""Shared evidence mechanics; issue-specific acceptance lives in each runner."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ISSUE2_TEST_DIRS = ("tests/ascend", "tests/tpu", "tests/tsp", "tests/tsp_cim", "tests/cim", "tests/common")
ISSUE3_TEST_DIRS = ("tests/operators", "tests/validation")

# Acceptance contract, independent of the implementation/evidence manifest.
# Changing these IDs changes the agreed Issue #3 delivery scope.
ISSUE3_EXPECTED_BLOCKS = {
    "LLaMA3-70B": frozenset((
        "embedding", "normalization", "residual", "vocabulary_projection", "qkv_projections",
        "rope", "gqa_attention_core", "kv_cache", "attention_output_projection", "swiglu_ffn")),
    "DeepSeek-R1": frozenset((
        "embedding", "normalization", "residual", "vocabulary_projection", "mla_projections",
        "mla_rope", "mla_compressed_cache", "mla_attention_core", "attention_output_projection",
        "dense_swiglu_ffn", "shared_expert_ffn", "moe_routed_ffn", "moe_router_and_combine")),
}


def issue3_inventory(coverage):
    """Check required identities before evidence dictionaries can hide duplicates."""
    models = coverage.get("models", [])
    invalid_models = not isinstance(models, list)
    models = models if isinstance(models, list) else []
    names = [m.get("name") for m in models if isinstance(m, dict) and isinstance(m.get("name"), str)]
    invalid_models |= len(names) != len(models)
    counts = Counter(names)
    result = {"missing_models": sorted(ISSUE3_EXPECTED_BLOCKS.keys() - counts.keys()),
              "unexpected_models": sorted(counts.keys() - ISSUE3_EXPECTED_BLOCKS.keys()),
              "duplicate_models": sorted(n for n, count in counts.items() if count > 1),
              "invalid_models": invalid_models, "models": {}}
    for name, expected in ISSUE3_EXPECTED_BLOCKS.items():
        model = next((m for m in models if isinstance(m, dict) and m.get("name") == name), {})
        items = model.get("blocks", [])
        invalid_blocks = not isinstance(items, list)
        items = items if isinstance(items, list) else []
        invalid_indices = [i for i, b in enumerate(items)
                           if not isinstance(b, dict) or not isinstance(b.get("id"), str) or not b["id"]]
        ids = Counter(b["id"] for i, b in enumerate(items) if i not in invalid_indices)
        result["models"][name] = {
            "missing_ids": sorted(expected - ids.keys()), "unexpected_ids": sorted(ids.keys() - expected),
            "duplicate_ids": sorted(i for i, count in ids.items() if count > 1),
            "invalid_blocks": invalid_blocks, "invalid_id_indices": invalid_indices}
    valid = not any(result[k] for k in ("missing_models", "unexpected_models", "duplicate_models", "invalid_models"))
    valid &= all(not any(details.values()) for details in result["models"].values())
    result["status"] = "passed" if valid else "failed"
    return result


def issue3_owned(name):
    """Files owned by the operator work, distinct from native GEMM dependencies."""
    path = Path(name)
    return (name.startswith(("src/operators/", "src/runtime/", "tests/operators/", "configs/execution/",
                             "configs/workload/operators/", "validation/issue3/"))
            or name in {"scripts/verify_issue3.py", "calibration/operator_execution_audit.json",
                        "src/hardware/vector.py", "src/hardware/blocks.py", "src/hardware/attention.py",
                        "src/utils/operator_report.py", "src/utils/operator_workload.py"}
            or (name.startswith("src/core/") and
                (path.name in {"vector.py", "vector_resources.py", "operator_result.py", "kv_cache.py", "moe.py"}
                 or path.name.startswith(("composite", "block_")))))


def fingerprints(paths, root=ROOT):
    """Normalize text newlines so Windows and Unix produce the same digest."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
            for p in sorted(set(paths))}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def output_directory(path, issue, root=ROOT):
    """Reject the other issue's evidence tree before creating or writing files."""
    out = Path(path).resolve()
    for other in ("issue2", "issue3"):
        if other == issue:
            continue
        protected = (root / "validation" / other).resolve()
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError(f"{issue} output must not overlap {other} evidence: {out}")
    return out


def discover(folders, root=ROOT):
    loader = unittest.TestLoader()
    return unittest.TestSuite(loader.discover(str(root / folder), top_level_dir=str(root)) for folder in folders)


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = {}

    def startTest(self, test):
        self.records[test.id()] = {"status": "running"}
        super().startTest(test)

    def addSuccess(self, test):
        # A failing subtest must not be overwritten by an enclosing success.
        if self.records[test.id()]["status"] != "failed":
            self.records[test.id()] = {"status": "passed"}
        super().addSuccess(test)

    def addFailure(self, test, err):
        self.records[test.id()] = {"status": "failed"}
        super().addFailure(test, err)

    def addError(self, test, err):
        self.records[test.id()] = {"status": "error"}
        super().addError(test, err)

    def addSkip(self, test, reason):
        self.records[test.id()] = {"status": "skipped", "reason": reason}
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):
        self.records[test.id()] = {"status": "expected_failure"}
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self.records[test.id()] = {"status": "unexpected_success"}
        super().addUnexpectedSuccess(test)

    def addSubTest(self, test, subtest, err):
        if err:
            self.records[test.id()] = {"status": "failed", "subtest": str(subtest)}
        super().addSubTest(test, subtest, err)


def run_tests(suite, out):
    with (out / "unittest.txt").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=RecordedResult).run(suite)
    counts = dict(Counter(r["status"] for r in result.records.values()))
    passed = result.wasSuccessful() and result.testsRun > 0 and counts == {"passed": result.testsRun}
    save(out / "tests.json", {"tests_run": result.testsRun, "status_counts": counts, "tests": result.records})
    return result, counts, passed
