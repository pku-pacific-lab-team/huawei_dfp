"""Shape-only operator contracts; tensor dtype denotes stored payload."""
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite, prod


DTYPE_BYTES = {"INT8": 1, "INT32": 4, "FP16": 2, "BF16": 2, "FP32": 4}


def require_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def output_name(requested, inputs, default="output"):
    """Names identify tensors within a call; output storage must be disjoint."""
    names = {t.name for t in inputs}
    if requested is None:
        requested = default
        while requested in names:
            requested += "_"
    require_text(requested, "output name")
    if requested in names:
        raise ValueError("Input/output alias is not modeled")
    return requested


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple
    dtype: str

    def __post_init__(self):
        require_text(self.name, "tensor name")
        if not isinstance(self.shape, (tuple, list)) or any(type(d) is not int or d <= 0 for d in self.shape):
            raise ValueError("Tensor dimensions must be positive integers; () denotes a scalar")
        if not isinstance(self.dtype, str) or self.dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported tensor dtype: {self.dtype!r}")
        object.__setattr__(self, "shape", tuple(self.shape))

    @property
    def numel(self):
        return prod(self.shape)

    @property
    def nbytes(self):
        return self.numel * DTYPE_BYTES[self.dtype]

    def to_dict(self):
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype,
                "numel": self.numel, "logical_bytes": self.nbytes}


@dataclass(frozen=True)
class OperatorSpec:
    """A description, not a promise that every backend supports this operation."""
    kind: str
    inputs: tuple
    outputs: tuple
    accumulation_dtype: str | None = None
    attributes: tuple = ()

    def __post_init__(self):
        require_text(self.kind, "operator kind")
        for field in ("inputs", "outputs"):
            tensors = getattr(self, field)
            if not isinstance(tensors, (tuple, list)) or not tensors or any(not isinstance(t, TensorSpec) for t in tensors):
                raise ValueError(f"{field} must contain TensorSpec objects")
            if len({t.name for t in tensors}) != len(tensors):
                raise ValueError(f"Duplicate tensor names in {field}")
            object.__setattr__(self, field, tuple(tensors))
        if self.accumulation_dtype is not None and (not isinstance(self.accumulation_dtype, str) or self.accumulation_dtype not in DTYPE_BYTES):
            raise ValueError("Unsupported accumulation dtype")
        attrs = self.attributes.items() if isinstance(self.attributes, dict) else self.attributes
        if not isinstance(attrs, (tuple, list)) and not isinstance(self.attributes, dict):
            raise ValueError("Operator attributes must be key/value pairs or a dictionary")
        pairs = []
        for pair in attrs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Operator attributes must be key/value pairs")
            key, value = pair
            require_text(key, "attribute key")
            if type(value) not in (str, bool, int, float, type(None)) or (type(value) is float and not isfinite(value)):
                raise ValueError("Operator attributes must be finite JSON scalars")
            pairs.append((key, value))
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate operator attribute")
        object.__setattr__(self, "attributes", tuple(sorted(pairs)))

    def to_dict(self):
        return {"kind": self.kind, "inputs": [t.to_dict() for t in self.inputs],
                "outputs": [t.to_dict() for t in self.outputs],
                "accumulation_dtype": self.accumulation_dtype, "attributes": dict(self.attributes)}


@dataclass(frozen=True)
class OpResult:
    """Compact analytical result; native diagnostics are retained only on request."""
    operator: OperatorSpec
    architecture: str
    mode: str
    latency_s: float | None
    cycle_count: int | None
    boundary: dict
    work_counts: dict
    resource_requirements: dict
    action_ledger: object
    _native_report: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        from .actions import ActionLedger
        if not isinstance(self.operator, OperatorSpec) or not isinstance(self.action_ledger, ActionLedger):
            raise ValueError("OpResult requires an OperatorSpec and ActionLedger")
        require_text(self.architecture, "architecture")
        if any(a.architecture != self.architecture for a in self.action_ledger.entries):
            raise ValueError("Ledger architecture does not match result")
        for key in ("boundary", "work_counts", "resource_requirements", "_native_report"):
            object.__setattr__(self, key, deepcopy(getattr(self, key)))

    @classmethod
    def from_native(cls, operator, architecture, ledger, native, *, include_details=False):
        from .resources import memory_requirements
        timing = native["timing"]
        ns = timing["time_ns"]
        return cls(operator, architecture, native["mode"], None if ns is None else ns / 1e9,
                   timing["total_cycles"], native["boundary"], native["work_counts"],
                   memory_requirements(native["memory"]), ledger, native if include_details else None)

    @property
    def timing_status(self):
        return "not_evaluated" if self.cycle_count is None else "cycles_only" if self.latency_s is None else "analytical_estimate"

    @property
    def native_report(self):
        if self._native_report is None:
            raise ValueError("Native diagnostics were not retained; run with include_details=True")
        return deepcopy(self._native_report)

    def to_dict(self):
        return {
            "schema_version": 2, "kind": "operator_result", "status": "success",
            "architecture": self.architecture, "mode": self.mode,
            "operator": self.operator.to_dict(),
            "latency_s": self.latency_s, "cycle_count": self.cycle_count, "timing_status": self.timing_status,
            "boundary": deepcopy(self.boundary), "work_counts": deepcopy(self.work_counts),
            "resource_requirements": deepcopy(self.resource_requirements),
            "action_counts": self.action_ledger.counts_dict(),
            "coverage": {
                "offchip_memory": "native_payload_counts" if self.architecture in {"ascend", "tpu"} else "outside_native_boundary",
                "interconnect": "onchip_streams_only" if self.architecture in {"tsp", "tsp_cim"} else "not_modeled",
                "physical_memory_transactions": "not_modeled",
            },
            "energy": {"status": "not_evaluated", "total_pj": None},
        }

    def to_detailed_dict(self):
        import hashlib
        import json
        from .connections import io_contract
        summary = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        return {"schema_version": 1, "kind": "operator_details",
                "result_sha256": hashlib.sha256(summary.encode("utf-8")).hexdigest(),
                "native_report": self.native_report, "io_contract": io_contract(self),
                "action_metadata": {a.id: {"source": a.source, "source_basis": a.source_basis,
                                            "summarizes": list(a.summarizes)} for a in self.action_ledger.entries}}
