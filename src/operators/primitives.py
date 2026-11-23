"""First executable operator: INT8 GEMM with a materialized INT32 output."""
from dataclasses import dataclass

from .types import OperatorSpec, TensorSpec, output_name as select_output_name
from ..utils.workload import parse_workloads


@dataclass(frozen=True)
class Gemm:
    M: int
    K: int
    N: int
    inputs: tuple | None = None
    output_name: str | None = None

    def __post_init__(self):
        if any(type(d) is not int or d <= 0 for d in (self.M, self.K, self.N)):
            raise ValueError("GEMM M, K, N must be positive integers")
        inputs = (TensorSpec("A", (self.M, self.K), "INT8"), TensorSpec("B", (self.K, self.N), "INT8")) if self.inputs is None else self.inputs
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2 or any(not isinstance(t, TensorSpec) for t in inputs):
            raise ValueError("GEMM requires two TensorSpec inputs")
        if any(t.dtype != "INT8" or t.shape != shape for t, shape in zip(inputs, ((self.M, self.K), (self.K, self.N)))):
            raise ValueError("GEMM tensors must match M/K/N and INT8 storage")
        if inputs[0].name == inputs[1].name:
            raise ValueError("GEMM input alias is not modeled")
        object.__setattr__(self, "inputs", tuple(inputs))
        object.__setattr__(self, "output_name", select_output_name(self.output_name, inputs, "C"))

    @classmethod
    def from_tensors(cls, a, b, *, output_name=None):
        if not isinstance(a, TensorSpec) or not isinstance(b, TensorSpec):
            raise ValueError("GEMM inputs must be TensorSpec objects")
        if a.dtype != "INT8" or b.dtype != "INT8":
            raise ValueError("GEMM supports INT8 inputs only")
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("GEMM requires A[M,K] and B[K,N]")
        return cls(a.shape[0], a.shape[1], b.shape[1], (a, b), output_name)

    @property
    def spec(self):
        return OperatorSpec("gemm", self.inputs,
                            (TensorSpec(self.output_name, (self.M, self.N), "INT32"),), "INT32")

    @property
    def useful_macs(self):
        return self.M * self.K * self.N

    def to_workload(self):
        return parse_workloads({"gemm": {"M": self.M, "K": self.K, "N": self.N},
                                "data_format": "INT8"})[0]


def simulate_gemm(operator, architecture, spec, *, mode="count", include_details=False, **policy):
    """Compatibility entry; execution orchestration lives in src.runtime."""
    from ..runtime.simulation import simulate_gemm as run
    return run(operator, architecture, spec, mode=mode, include_details=include_details, **policy)
