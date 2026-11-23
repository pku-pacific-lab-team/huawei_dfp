"""Shape-only, last-axis FP32 normalization blocks."""
from dataclasses import dataclass
from math import isfinite, prod

from .types import OperatorSpec, TensorSpec, output_name as select_output_name


def validate_input(x, axis):
    if not isinstance(x, TensorSpec) or x.dtype != "FP32" or not x.shape:
        raise ValueError("Normalization requires a non-scalar FP32 TensorSpec")
    if type(axis) is not int or axis not in (-1, len(x.shape) - 1):
        raise ValueError("Only last-axis normalization is modeled")


class RowOperator:
    @property
    def width(self):
        return self.x.shape[-1]

    @property
    def rows(self):
        return prod(self.x.shape[:-1])

    @property
    def numel(self):
        return self.x.numel

    @property
    def spec(self):
        attributes = {"axis": self.axis}
        if self.operation == "rmsnorm":
            attributes["epsilon"] = self.epsilon
        return OperatorSpec(self.operation, self.inputs, (TensorSpec(self.output_name, self.x.shape, "FP32"),),
                            "FP32", attributes)


@dataclass(frozen=True)
class RMSNorm(RowOperator):
    x: TensorSpec
    weight: TensorSpec
    epsilon: float = 1e-5
    axis: int = -1
    output_name: str | None = None
    operation = "rmsnorm"

    def __post_init__(self):
        validate_input(self.x, self.axis)
        if (not isinstance(self.weight, TensorSpec) or self.weight.dtype != "FP32"
                or self.weight.shape != (self.width,) or self.weight.name == self.x.name):
            raise ValueError("RMSNorm weight must be a distinct FP32 tensor of shape [width]")
        if type(self.epsilon) not in (int, float) or not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("RMSNorm epsilon must be finite and positive")
        object.__setattr__(self, "axis", len(self.x.shape) - 1)
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.x, self.weight)


@dataclass(frozen=True)
class Softmax(RowOperator):
    x: TensorSpec
    axis: int = -1
    output_name: str | None = None
    operation = "softmax"

    def __post_init__(self):
        validate_input(self.x, self.axis)
        object.__setattr__(self, "axis", len(self.x.shape) - 1)
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.x,)


def simulate_composite(operator, architecture, spec, *, mode="count", execution=None, include_details=False, **policy):
    """Compatibility entry; execution orchestration lives in src.runtime."""
    from ..runtime.simulation import simulate_composite as run
    return run(operator, architecture, spec, mode=mode, execution=execution, include_details=include_details, **policy)
