"""Independent element-wise operations with explicit stored tensor types."""
from dataclasses import dataclass

from .types import OperatorSpec, TensorSpec, output_name as select_output_name


OPERATIONS = {"int32_add": ("INT32", 2, "vector"), "fp32_add": ("FP32", 2, "vector"),
              "fp32_mul": ("FP32", 2, "vector"), "fp32_exp": ("FP32", 1, "nonlinear"),
              "fp32_rsqrt": ("FP32", 1, "nonlinear")}


@dataclass(frozen=True)
class Vector:
    operation: str
    inputs: tuple
    output_name: str | None = None

    def __post_init__(self):
        if not isinstance(self.operation, str) or self.operation not in OPERATIONS:
            raise ValueError("Unsupported Vector operation")
        dtype, arity, _ = OPERATIONS[self.operation]
        if not isinstance(self.inputs, (tuple, list)) or len(self.inputs) != arity or any(not isinstance(t, TensorSpec) for t in self.inputs):
            raise ValueError(f"{self.operation} requires {arity} TensorSpec inputs")
        if len({t.name for t in self.inputs}) != arity:
            raise ValueError("Vector inputs must have distinct names; alias/in-place execution is not modeled")
        if any(t.dtype != dtype or t.shape != self.inputs[0].shape for t in self.inputs):
            raise ValueError(f"{self.operation} requires equal shapes and dtype={dtype}; broadcasting/conversion is not modeled")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def numel(self):
        return self.inputs[0].numel

    @property
    def dtype(self):
        return OPERATIONS[self.operation][0]

    @property
    def category(self):
        return OPERATIONS[self.operation][2]

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, (TensorSpec(self.output_name, self.inputs[0].shape, self.dtype),))


def simulate_vector(operator, architecture, spec, *, mode="count", execution=None, include_details=False, **policy):
    """Compatibility entry; execution orchestration lives in src.runtime."""
    from ..runtime.simulation import simulate_vector as run
    return run(operator, architecture, spec, mode=mode, execution=execution, include_details=include_details, **policy)
