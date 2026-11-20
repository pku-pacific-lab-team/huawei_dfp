"""INT8 Decode normalization and attention projections, without positional encoding."""
from dataclasses import dataclass

from .blocks import checked_inputs
from .composite import RMSNorm
from .primitives import Gemm
from .types import OperatorSpec, TensorSpec, output_name as select_output_name


@dataclass(frozen=True)
class RMSNormINT8:
    x: TensorSpec
    weight: TensorSpec
    epsilon: float = 1e-5
    axis: int = -1
    output_name: str | None = None
    operation = "rmsnorm_int8"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if self.x.dtype != "INT8":
            raise ValueError("RMSNormINT8 requires INT8 activation storage")
        norm = RMSNorm(TensorSpec(self.x.name, self.x.shape, "FP32"), self.weight, self.epsilon, self.axis)
        object.__setattr__(self, "axis", norm.axis)
        object.__setattr__(self, "epsilon", norm.epsilon)
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.x, self.weight)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, (TensorSpec(self.output_name, self.x.shape, "INT8"),),
                            "FP32", {"axis": self.axis, "epsilon": self.epsilon, "weight_storage": "FP32"})


@dataclass(frozen=True)
class QKVProjection:
    x: TensorSpec
    query_weight: TensorSpec
    key_weight: TensorSpec
    value_weight: TensorSpec
    query_heads: int
    kv_heads: int
    head_dim: int
    output_names: tuple[str, str, str] | None = None
    operation = "qkv_projection"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if any(type(v) is not int or v <= 0 for v in (self.query_heads, self.kv_heads, self.head_dim)):
            raise ValueError("Head counts and head_dim must be positive integers")
        if self.query_heads % self.kv_heads:
            raise ValueError("Query heads must be divisible by KV heads")
        if any(t.dtype != "INT8" or len(t.shape) != 2 for t in self.inputs):
            raise ValueError("QKVProjection requires rank-2 INT8 activation and weights")
        hidden = self.x.shape[1]
        expected = ((hidden, self.query_heads * self.head_dim), (hidden, self.kv_heads * self.head_dim))
        if self.query_weight.shape != expected[0] or any(t.shape != expected[1] for t in (self.key_weight, self.value_weight)):
            raise ValueError("Q/K/V weights must match hidden size and head counts")
        names = self.output_names
        if names is None:
            names = tuple(select_output_name(None, self.inputs, default=label) for label in ("query", "key", "value"))
        if not isinstance(names, (tuple, list)) or len(names) != 3 or any(not isinstance(n, str) for n in names):
            raise ValueError("QKVProjection requires three distinct output names")
        names = tuple(select_output_name(n, self.inputs) for n in names)
        if len(set(names)) != 3:
            raise ValueError("Q/K/V outputs cannot alias")
        object.__setattr__(self, "output_names", names)

    @property
    def inputs(self):
        return (self.x, self.query_weight, self.key_weight, self.value_weight)

    @property
    def spec(self):
        outputs = tuple(TensorSpec(name, (self.x.shape[0], heads, 1, self.head_dim), "INT8")
                        for name, heads in zip(self.output_names, (self.query_heads, self.kv_heads, self.kv_heads)))
        return OperatorSpec(self.operation, self.inputs, outputs, "INT32",
                            {"query_heads": self.query_heads, "kv_heads": self.kv_heads, "head_dim": self.head_dim,
                             "bias": False, "positional_encoding_included": False, "head_order": "batch_head_token_channel"})

    @property
    def storage_views(self):
        return [{"tensor": t.name, "matrix_shape": [t.shape[0], t.shape[1] * t.shape[3]],
                 "head_shape": list(t.shape), "binding": "same_output_backing"} for t in self.spec.outputs]


@dataclass(frozen=True)
class AttentionOutputProjection:
    context: TensorSpec
    weight: TensorSpec
    output_name: str | None = None
    operation = "attention_output_projection"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if (self.context.dtype != "INT8" or len(self.context.shape) != 4 or self.context.shape[2] != 1
                or self.weight.dtype != "INT8" or len(self.weight.shape) != 2):
            raise ValueError("Attention output projection requires INT8 context [batch, heads, 1, head_dim] and rank-2 INT8 weight")
        if self.weight.shape[0] != self.context.shape[1] * self.context.shape[3]:
            raise ValueError("Output weight rows must equal heads * head_dim")
        object.__setattr__(self, "output_name", select_output_name(self.output_name, self.inputs))

    @property
    def inputs(self):
        return (self.context, self.weight)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs,
                            (TensorSpec(self.output_name, (self.context.shape[0], self.weight.shape[1]), "INT8"),),
                            "INT32", {"bias": False, "head_order": "batch_head_token_channel"})

    @property
    def storage_views(self):
        return [{"tensor": self.context.name, "matrix_shape": [self.context.shape[0], self.weight.shape[0]],
                 "head_shape": list(self.context.shape), "binding": "same_input_backing"}]


@dataclass(frozen=True)
class _StorageBoundary:
    """Internal work proxy; equal-element views do not allocate an extra reshape buffer."""
    x: TensorSpec
    output: TensorSpec
    operation = "storage_boundary"

    def __post_init__(self):
        if self.x.numel != self.output.numel or self.x.name == self.output.name:
            raise ValueError("Storage boundary requires equal elements in distinct buffers")
        if (self.x.dtype, self.output.dtype) not in {("INT8", "FP32"), ("INT32", "INT8"), ("FP32", "INT8")}:
            raise ValueError("Unsupported internal storage boundary")

    @property
    def inputs(self):
        return (self.x,)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, (self.output,), "FP32")


def projection_steps(operator):
    if not isinstance(operator, (RMSNormINT8, QKVProjection, AttentionOutputProjection)):
        raise ValueError("Expected a supported Decode block")
    taken = {t.name for t in (*operator.inputs, *operator.spec.outputs)}

    def internal(label):
        name = operator.operation + "." + label
        while name in taken:
            name += "_"
        taken.add(name)
        return name

    if isinstance(operator, RMSNormINT8):
        cast = _StorageBoundary(operator.x, TensorSpec(internal("input_fp32"), operator.x.shape, "FP32"))
        norm = RMSNorm(cast.output, operator.weight, operator.epsilon, operator.axis, internal("normalized"))
        out = _StorageBoundary(norm.spec.outputs[0], operator.spec.outputs[0])
        return [("input_storage", cast), ("rmsnorm", norm), ("output_storage", out)]
    if isinstance(operator, QKVProjection):
        nodes = []
        for label, weight, output in zip(("query", "key", "value"), operator.inputs[1:], operator.spec.outputs):
            gemm = Gemm.from_tensors(operator.x, weight, output_name=internal(label + "_accumulator"))
            nodes.extend([(label, gemm), (label + "_storage", _StorageBoundary(gemm.spec.outputs[0], output))])
        return nodes
    view = TensorSpec(operator.context.name, (operator.context.shape[0], operator.weight.shape[0]), "INT8")
    gemm = Gemm.from_tensors(view, operator.weight, output_name=internal("accumulator"))
    return [("projection", gemm), ("output_storage", _StorageBoundary(gemm.spec.outputs[0], operator.spec.outputs[0]))]
