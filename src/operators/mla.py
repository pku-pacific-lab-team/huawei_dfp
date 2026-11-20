"""MLA preprocessing and shared compressed cache contracts (INT8 what-if)."""
from dataclasses import dataclass, asdict, replace

from .blocks import checked_inputs
from .composite import RMSNorm
from .primitives import Gemm
from .projections import _StorageBoundary
from .rope import RoPE, _RotaryTensor
from .types import OperatorSpec, TensorSpec, output_name, require_text


@dataclass(frozen=True)
class MLASpec:
    hidden_size: int
    query_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("MLA dimensions must be positive integers; direct Q (rank zero) is not implemented")
        if self.qk_rope_head_dim % 2:
            raise ValueError("MLA RoPE dimension must be even")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MLAPreprocess:
    x: TensorSpec
    q_down_weight: TensorSpec
    q_norm_weight: TensorSpec
    q_up_weight: TensorSpec
    kv_down_weight: TensorSpec
    kv_norm_weight: TensorSpec
    cos: TensorSpec
    sin: TensorSpec
    dimensions: MLASpec
    position: int
    epsilon: float = 1e-6
    output_names: tuple[str, str, str, str] | None = None
    operation = "mla_preprocess"

    def __post_init__(self):
        checked_inputs(self.inputs)
        if not isinstance(self.dimensions, MLASpec):
            raise ValueError("Expected MLASpec dimensions")
        d = self.dimensions
        if self.x.dtype != "INT8" or len(self.x.shape) != 2 or self.x.shape[1] != d.hidden_size:
            raise ValueError("MLA activation must be INT8 [batch, hidden_size]")
        shapes = ((d.hidden_size, d.q_lora_rank),
                  (d.q_lora_rank, d.query_heads * (d.qk_nope_head_dim + d.qk_rope_head_dim)),
                  (d.hidden_size, d.kv_lora_rank + d.qk_rope_head_dim))
        for t, shape in zip((self.q_down_weight, self.q_up_weight, self.kv_down_weight), shapes):
            if t.dtype != "INT8" or t.shape != shape:
                raise ValueError("MLA projection weight shape/dtype does not match dimensions")
        for weight, rank in ((self.q_norm_weight, d.q_lora_rank), (self.kv_norm_weight, d.kv_lora_rank)):
            if weight.dtype != "FP32" or weight.shape != (rank,):
                raise ValueError("MLA norm weight must be FP32 [compression_rank]")
        # Reuse normalization validation without constructing an execution node.
        norm = RMSNorm(TensorSpec("norm_check", (1, d.q_lora_rank), "FP32"),
                       TensorSpec("weight_check", (d.q_lora_rank,), "FP32"), self.epsilon)
        object.__setattr__(self, "epsilon", norm.epsilon)
        names = self.output_names
        if names is None:
            names = tuple(output_name(None, self.inputs, label) for label in
                          ("query_nope", "query_rope", "latent_kv", "key_rope"))
        if not isinstance(names, (list, tuple)) or len(names) != 4 or any(not isinstance(n, str) for n in names):
            raise ValueError("MLA preprocessing requires four distinct output names")
        names = tuple(output_name(n, self.inputs) for n in names)
        if len(set(names)) != 4:
            raise ValueError("MLA outputs cannot alias")
        object.__setattr__(self, "output_names", names)
        q, k = self.spec.outputs[1], self.spec.outputs[3]
        RoPE(q, k, self.cos, self.sin, self.position)

    @property
    def inputs(self):
        return (self.x, self.q_down_weight, self.q_norm_weight, self.q_up_weight,
                self.kv_down_weight, self.kv_norm_weight, self.cos, self.sin)

    @property
    def spec(self):
        d, b = self.dimensions, self.x.shape[0]
        shapes = ((b, d.query_heads, 1, d.qk_nope_head_dim), (b, d.query_heads, 1, d.qk_rope_head_dim),
                  (b, 1, 1, d.kv_lora_rank), (b, 1, 1, d.qk_rope_head_dim))
        return OperatorSpec(self.operation, self.inputs,
                            tuple(TensorSpec(n, s, "INT8") for n, s in zip(self.output_names, shapes)), "INT32",
                            {**d.to_dict(), "position": self.position, "epsilon": self.epsilon,
                             "normalization_dtype": "FP32", "shared_kv_heads": 1,
                             "split_policy": "materialized_selected_channels", "attention_core_included": False,
                             "table_generation_included": False, "uniform_batch_position": True})


@dataclass(frozen=True)
class _ChannelBoundary:
    """Select a contiguous interval in each logical row and materialize a cast.

    The complete source backing stays live until its last branch consumes it.
    Each row has its own tiled transfers; address/packing instructions are symbolic.
    """
    x: TensorSpec
    output: TensorSpec
    rows: int
    row_stride: int
    offset: int
    row_elements: int
    operation = "storage_boundary"

    def __post_init__(self):
        if (any(type(v) is not int or v <= 0 for v in (self.rows, self.row_stride, self.row_elements))
                or type(self.offset) is not int or self.offset < 0
                or self.offset + self.row_elements > self.row_stride
                or self.x.numel != self.rows * self.row_stride or self.output.numel != self.rows * self.row_elements
                or self.x.dtype != "INT32" or self.output.dtype not in {"INT8", "FP32"}
                or self.x.name == self.output.name):
            raise ValueError("Invalid materialized channel boundary")

    @property
    def inputs(self):
        return (self.x,)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, (self.output,), "FP32",
                            {"rows": self.rows, "row_stride": self.row_stride, "offset": self.offset,
                             "selected_channels": self.row_elements, "materialized": True})


def mla_steps(op):
    d, b = op.dimensions, op.x.shape[0]
    taken = {t.name for t in (*op.inputs, *op.spec.outputs)}

    def tensor(label, shape, dtype):
        name = op.operation + "." + label
        while name in taken:
            name += "_"
        taken.add(name)
        return TensorSpec(name, shape, dtype)

    def matrix(x, weight, label):
        out = tensor(label, (x.shape[0], weight.shape[1]), "INT32")
        return Gemm.from_tensors(x, weight, output_name=out.name)

    q_nope, q_rope, latent, k_rope = op.spec.outputs
    qa = matrix(op.x, op.q_down_weight, "q_accumulator")
    q_cast = _StorageBoundary(qa.spec.outputs[0], tensor("q_fp32", (b, d.q_lora_rank), "FP32"))
    q_norm = RMSNorm(q_cast.output, op.q_norm_weight, op.epsilon,
                     output_name=tensor("q_normalized", q_cast.output.shape, "FP32").name)
    q_store = _StorageBoundary(q_norm.spec.outputs[0], tensor("q_compressed", (b, d.q_lora_rank), "INT8"))
    qb = matrix(q_store.output, op.q_up_weight, "q_expanded")
    stride = d.qk_nope_head_dim + d.qk_rope_head_dim
    qc = _ChannelBoundary(qb.spec.outputs[0], q_nope, b * d.query_heads, stride, 0, d.qk_nope_head_dim)
    qp = _ChannelBoundary(qb.spec.outputs[0], tensor("q_positional", q_rope.shape, "INT8"),
                          b * d.query_heads, stride, d.qk_nope_head_dim, d.qk_rope_head_dim)
    qr = _RotaryTensor(qp.output, op.cos, op.sin, op.position, q_rope.name)
    kva = matrix(op.x, op.kv_down_weight, "kv_accumulator")
    kv = _ChannelBoundary(kva.spec.outputs[0], tensor("kv_fp32", (b, d.kv_lora_rank), "FP32"),
                          b, d.kv_lora_rank + d.qk_rope_head_dim, 0, d.kv_lora_rank)
    kv_norm = RMSNorm(kv.output, op.kv_norm_weight, op.epsilon,
                      output_name=tensor("kv_normalized", kv.output.shape, "FP32").name)
    kv_store = _StorageBoundary(kv_norm.spec.outputs[0], latent)
    kp = _ChannelBoundary(kva.spec.outputs[0], tensor("k_positional", k_rope.shape, "INT8"),
                          b, d.kv_lora_rank + d.qk_rope_head_dim, d.kv_lora_rank, d.qk_rope_head_dim)
    kr = _RotaryTensor(kp.output, op.cos, op.sin, op.position, k_rope.name)
    return list(zip(("q_down", "q_to_fp32", "q_norm", "q_to_int8", "q_up", "q_nope_split", "q_rope_split",
                     "q_rope", "kv_down", "kv_split", "kv_norm", "kv_to_int8", "k_rope_split", "k_rope"),
                    (qa, q_cast, q_norm, q_store, qb, qc, qp, qr, kva, kv, kv_norm, kv_store, kp, kr)))


@dataclass(frozen=True)
class MLACacheSpec:
    batch: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    length: int
    capacity: int
    name: str = "mla_cache"
    dtype: str = "INT8"

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in (self.batch, self.kv_lora_rank, self.qk_rope_head_dim, self.capacity)):
            raise ValueError("MLA cache dimensions must be positive integers")
        if self.qk_rope_head_dim % 2 or type(self.length) is not int or not 0 <= self.length <= self.capacity:
            raise ValueError("Invalid MLA cache length or RoPE dimension")
        if self.dtype != "INT8":
            raise ValueError("MLA cache uses INT8 what-if storage")
        require_text(self.name, "cache name")

    @property
    def widths(self):
        return (self.kv_lora_rank, self.qk_rope_head_dim)

    @property
    def buffers(self):
        return tuple(TensorSpec(self.name + "." + label, (self.batch, 1, self.capacity, width), self.dtype)
                     for label, width in zip(("latent", "rope"), self.widths))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MLACacheAccess:
    cache: MLACacheSpec
    append_tokens: int = 1
    read_cache: bool = True
    operation = "mla_cache"

    def __post_init__(self):
        if not isinstance(self.cache, MLACacheSpec):
            raise ValueError("Expected MLACacheSpec")
        if type(self.append_tokens) is not int or self.append_tokens not in (0, 1):
            raise ValueError("MLA Decode appends zero or one token per batch item")
        if type(self.read_cache) is not bool or not (self.read_cache or self.append_tokens):
            raise ValueError("MLA cache access must append or read")
        if self.cache.length + self.append_tokens > self.cache.capacity:
            raise ValueError("MLA cache append exceeds capacity")
        if self.read_cache and self.cache_after.length == 0:
            raise ValueError("Cannot read an empty MLA cache")

    @property
    def cache_after(self):
        return replace(self.cache, length=self.cache.length + self.append_tokens)

    @property
    def inputs(self):
        tokens = tuple(TensorSpec(self.cache.name + ".new_" + label, (self.cache.batch, 1, 1, width), "INT8")
                       for label, width in zip(("latent", "rope"), self.cache.widths)) if self.append_tokens else ()
        return self.cache.buffers + tokens

    @property
    def spec(self):
        outputs = tuple(TensorSpec(t.name + ".after", t.shape, t.dtype) for t in self.cache.buffers)
        return OperatorSpec(self.operation, self.inputs, outputs, None,
                            {"length_before": self.cache.length, "length_after": self.cache_after.length,
                             "capacity": self.cache.capacity, "append_tokens": self.append_tokens,
                             "read_cache": self.read_cache, "shared_across_query_heads": True,
                             "state_outputs_alias_inputs": True})


def simulate_mla_cache(operator, architecture, spec, **options):
    from ..runtime.mla import simulate_mla_cache as run
    return run(operator, architecture, spec, **options)
