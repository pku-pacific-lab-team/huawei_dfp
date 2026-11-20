"""Coarse Decode blocks; storage types and intermediate identities are explicit."""
from dataclasses import dataclass

from .primitives import Gemm
from .types import OperatorSpec, TensorSpec, output_name as select_output_name


def checked_inputs(inputs):
    if any(not isinstance(t, TensorSpec) or not t.shape for t in inputs):
        raise ValueError("Block inputs must be non-scalar TensorSpec objects")
    if len({t.name for t in inputs}) != len(inputs):
        raise ValueError("Block input alias is not modeled")


@dataclass(frozen=True)
class Residual:
    x: TensorSpec
    residual: TensorSpec
    output_name: str | None = None
    operation = "residual"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if self.x.shape != self.residual.shape or self.x.dtype != self.residual.dtype or self.x.dtype not in ("INT8", "FP32"):
            raise ValueError("Residual requires equal shapes and matching INT8 or FP32 storage")
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.x, self.residual)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs,
                            (TensorSpec(self.output_name, self.x.shape, self.x.dtype),), "FP32")


@dataclass(frozen=True)
class SwiGLU:
    gate: TensorSpec
    up: TensorSpec
    output_dtype: str = "INT8"
    output_name: str | None = None
    operation = "swiglu"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if self.gate.shape != self.up.shape or self.gate.dtype != self.up.dtype or self.gate.dtype not in ("INT8", "INT32", "FP32"):
            raise ValueError("SwiGLU requires equal shapes and matching INT8, INT32 or FP32 storage")
        if self.output_dtype not in ("INT8", "FP32"):
            raise ValueError("SwiGLU output storage must be INT8 or FP32")
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.gate, self.up)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs,
                            (TensorSpec(self.output_name, self.gate.shape, self.output_dtype),), "FP32",
                            {"formula": "silu(gate) * up"})


@dataclass(frozen=True)
class SwiGLUFFN:
    x: TensorSpec
    gate_weight: TensorSpec
    up_weight: TensorSpec
    down_weight: TensorSpec
    output_name: str | None = None
    operation = "swiglu_ffn"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if any(t.dtype != "INT8" or len(t.shape) != 2 for t in self.inputs):
            raise ValueError("SwiGLUFFN requires rank-2 INT8 activation and weights")
        hidden, intermediate = self.x.shape[1], self.gate_weight.shape[1]
        if self.gate_weight.shape != (hidden, intermediate) or self.up_weight.shape != (hidden, intermediate) or self.down_weight.shape != (intermediate, hidden):
            raise ValueError("FFN weights must have shapes [hidden, intermediate], [hidden, intermediate], [intermediate, hidden]")
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.x, self.gate_weight, self.up_weight, self.down_weight)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs,
                            (TensorSpec(self.output_name, self.x.shape, "INT8"),), "INT32",
                            {"activation_compute_dtype": "FP32", "intermediate_size": self.gate_weight.shape[1],
                             "bias": False})


@dataclass(frozen=True)
class _FFNOutput:
    """A block-internal storage boundary, not a public quantization instruction."""
    x: TensorSpec
    output_name: str
    operation = "ffn_output"

    @property
    def inputs(self):
        return (self.x,)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs,
                            (TensorSpec(self.output_name, self.x.shape, "INT8"),), "FP32")


def block_steps(operator):
    """Return a topological, serial plan; internal names cannot alias caller tensors."""
    from .embedding import Embedding
    from .rope import RoPE, rope_steps
    if isinstance(operator, RoPE):
        return rope_steps(operator)
    if isinstance(operator, Embedding):
        return [("lookup", operator)]
    if isinstance(operator, (Residual, SwiGLU)):
        return [(operator.operation, operator)]
    if not isinstance(operator, SwiGLUFFN):
        from .projections import projection_steps
        return projection_steps(operator)
    taken = {t.name for t in (*operator.inputs, *operator.spec.outputs)}
    def internal(label):
        name = "ffn." + label
        while name in taken:
            name += "_"
        taken.add(name)
        return name
    gate = Gemm.from_tensors(operator.x, operator.gate_weight, output_name=internal("gate"))
    up = Gemm.from_tensors(operator.x, operator.up_weight, output_name=internal("up"))
    activation = SwiGLU(gate.spec.outputs[0], up.spec.outputs[0], output_name=internal("activation"))
    down = Gemm.from_tensors(activation.spec.outputs[0], operator.down_weight, output_name=internal("down"))
    output = _FFNOutput(down.spec.outputs[0], operator.output_name)
    return [("gate", gate), ("up", up), ("activation", activation), ("down", down), ("output", output)]


def simulate_block(operator, architecture, spec, *, mode="count", include_details=False, **policy):
    from ..runtime.blocks import simulate_block as run
    return run(operator, architecture, spec, mode=mode, include_details=include_details, **policy)
