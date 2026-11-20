"""Single-token, adjacent-pair RoPE with explicit precomputed table inputs."""
from dataclasses import dataclass

from .blocks import checked_inputs
from .types import OperatorSpec, TensorSpec, output_name as select_output_name


@dataclass(frozen=True)
class RoPE:
    query: TensorSpec
    key: TensorSpec
    cos: TensorSpec
    sin: TensorSpec
    position: int
    output_names: tuple[str, str] | None = None
    operation = "rope"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if any(t.dtype != "INT8" or len(t.shape) != 4 or t.shape[2] != 1 for t in (self.query, self.key)):
            raise ValueError("RoPE requires INT8 Q/K of shape [batch, heads, 1, head_dim]")
        b, hq, _, d = self.query.shape
        if self.key.shape[0] != b or self.key.shape[3] != d or hq % self.key.shape[1]:
            raise ValueError("RoPE Q/K batch and head_dim must match; query heads must be divisible by KV heads")
        if d % 2:
            raise ValueError("Adjacent-pair RoPE requires an even head_dim")
        if any(t.dtype != "FP32" or len(t.shape) != 2 or t.shape[1] != d // 2 for t in (self.cos, self.sin)) or self.cos.shape != self.sin.shape:
            raise ValueError("RoPE requires matching FP32 cos/sin tables [positions, head_dim/2]")
        if type(self.position) is not int or not 0 <= self.position < self.cos.shape[0]:
            raise ValueError("RoPE position must be an integer within the supplied tables")
        names = self.output_names
        if names is None:
            names = tuple(select_output_name(None, self.inputs, default=n) for n in ("rotated_query", "rotated_key"))
        if not isinstance(names, (tuple, list)) or len(names) != 2 or any(not isinstance(n, str) for n in names):
            raise ValueError("RoPE requires two distinct output names")
        names = tuple(select_output_name(n, self.inputs) for n in names)
        if len(set(names)) != 2:
            raise ValueError("RoPE outputs cannot alias")
        object.__setattr__(self, "output_names", names)

    @property
    def inputs(self):
        return (self.query, self.key, self.cos, self.sin)

    @property
    def spec(self):
        outputs = tuple(TensorSpec(n, t.shape, "INT8") for n, t in zip(self.output_names, (self.query, self.key)))
        return OperatorSpec(self.operation, self.inputs, outputs, "FP32",
                            {"position": self.position, "pairing": "adjacent_channels", "full_head_rotation": True,
                             "table_generation_included": False, "table_reuse": "none_across_heads",
                             "uniform_batch_position": True})


@dataclass(frozen=True)
class _RotaryTensor:
    x: TensorSpec
    cos: TensorSpec
    sin: TensorSpec
    position: int
    output_name: str
    operation = "rope_apply"

    @property
    def inputs(self):
        return (self.x, self.cos, self.sin)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, (TensorSpec(self.output_name, self.x.shape, "INT8"),),
                            "FP32", {"position": self.position, "pairing": "adjacent_channels"})


def rope_steps(op):
    return [(label, _RotaryTensor(t, op.cos, op.sin, op.position, name))
            for label, t, name in zip(("query_rope", "key_rope"), (op.query, op.key), op.output_names)]
