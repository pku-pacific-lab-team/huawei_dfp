"""Single-token Decode contracts; cache state is metadata, never numerical data."""
from dataclasses import dataclass, replace
from math import sqrt

from .types import OperatorSpec, TensorSpec, output_name, require_text
from .primitives import Gemm
from .composite import Softmax


@dataclass(frozen=True)
class KVCacheSpec:
    batch: int
    kv_heads: int
    head_dim: int
    length: int
    capacity: int
    name: str = "kv_cache"
    dtype: str = "INT8"

    def __post_init__(self):
        for name in ("batch", "kv_heads", "head_dim", "capacity"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Cache {name} must be a positive integer")
        if type(self.length) is not int or not 0 <= self.length <= self.capacity:
            raise ValueError("Cache length must be between zero and capacity")
        if self.dtype != "INT8":
            raise ValueError("KV Cache uses INT8 what-if storage")
        require_text(self.name, "cache name")

    @property
    def buffers(self):
        return tuple(TensorSpec(self.name + "." + k, (self.batch, self.kv_heads, self.capacity, self.head_dim), self.dtype)
                     for k in ("key", "value"))

    def to_dict(self):
        return {k: getattr(self, k) for k in ("batch", "kv_heads", "head_dim", "length", "capacity", "name", "dtype")}


@dataclass(frozen=True)
class KVCacheAccess:
    cache: KVCacheSpec
    append_tokens: int = 1
    read_cache: bool = True
    operation = "kv_cache"

    def __post_init__(self):
        if not isinstance(self.cache, KVCacheSpec):
            raise ValueError("Expected KVCacheSpec")
        if type(self.append_tokens) is not int or self.append_tokens not in (0, 1):
            raise ValueError("Decode appends zero or one token per batch item")
        if type(self.read_cache) is not bool or not (self.read_cache or self.append_tokens):
            raise ValueError("Cache access must append or read")
        if self.cache.length + self.append_tokens > self.cache.capacity:
            raise ValueError("KV Cache append exceeds capacity; eviction and growth are not modeled")
        if self.read_cache and self.cache.length + self.append_tokens == 0:
            raise ValueError("Cannot read an empty cache")

    @property
    def cache_after(self):
        return replace(self.cache, length=self.cache.length + self.append_tokens)

    @property
    def inputs(self):
        tokens = tuple(TensorSpec(self.cache.name + ".new_" + k,
                                 (self.cache.batch, self.cache.kv_heads, 1, self.cache.head_dim), "INT8")
                       for k in ("key", "value")) if self.append_tokens else ()
        return self.cache.buffers + tokens

    @property
    def spec(self):
        outputs = tuple(TensorSpec(t.name + ".after", t.shape, t.dtype) for t in self.cache.buffers)
        return OperatorSpec(self.operation, self.inputs, outputs, None,
                            {"length_before": self.cache.length, "length_after": self.cache_after.length,
                             "capacity": self.cache.capacity, "append_tokens": self.append_tokens,
                             "read_cache": self.read_cache, "state_outputs_alias_inputs": True})


@dataclass(frozen=True)
class DecodeAttention:
    query: TensorSpec
    cache: KVCacheSpec
    append_tokens: int = 1
    output_name: str | None = None
    operation = "decode_attention"

    def __post_init__(self):
        if not isinstance(self.cache, KVCacheSpec) or not isinstance(self.query, TensorSpec):
            raise ValueError("DecodeAttention requires query TensorSpec and KVCacheSpec")
        q, c = self.query, self.cache
        if q.dtype != "INT8" or len(q.shape) != 4 or (q.shape[0], q.shape[2], q.shape[3]) != (c.batch, 1, c.head_dim):
            raise ValueError("Query must be INT8 [batch, query_heads, 1, head_dim]")
        if q.shape[1] % c.kv_heads:
            raise ValueError("Query heads must be an integer multiple of KV heads")
        KVCacheAccess(c, self.append_tokens, True)
        names = [t.name for t in self.inputs]
        state_names = {t.name for t in self.cache_access.spec.outputs}
        if len(set(names)) != len(names) or q.name in state_names:
            raise ValueError("Query must not alias cache inputs, state outputs or new-token inputs")
        # Reserve state output names too, so the result tensor cannot alias them.
        reserved = self.inputs + self.cache_access.spec.outputs
        object.__setattr__(self, "output_name", output_name(self.output_name, reserved, "context"))

    @property
    def query_heads(self):
        return self.query.shape[1]

    @property
    def cache_access(self):
        return KVCacheAccess(self.cache, self.append_tokens, not bool(self.append_tokens))

    @property
    def inputs(self):
        return (self.query,) + self.cache_access.inputs

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs,
                            (TensorSpec(self.output_name, self.query.shape, "INT8"),) + self.cache_access.spec.outputs,
                            "INT32", {"batch": self.cache.batch, "query_heads": self.query_heads,
                                      "kv_heads": self.cache.kv_heads, "head_dim": self.cache.head_dim,
                                      "length_before": self.cache.length,
                                      "length_after": self.cache.length + self.append_tokens,
                                      "cache_capacity": self.cache.capacity, "append_tokens": self.append_tokens,
                                      "scale": 1 / sqrt(self.cache.head_dim), "softmax_compute_dtype": "FP32",
                                      "projection_and_position_processing_included": False})


@dataclass(frozen=True)
class _AttentionStage:
    x: TensorSpec
    operation: str
    output_dtype: str
    output_name: str
    scale: float | None = None
    scale_label: str = "scale_by_inverse_sqrt_head_dim"

    @property
    def inputs(self):
        return (self.x,)

    @property
    def spec(self):
        return OperatorSpec(self.operation, self.inputs, (TensorSpec(self.output_name, self.x.shape, self.output_dtype),),
                            "FP32", {"scale": self.scale} if self.scale is not None else {})


def attention_steps(operator):
    """One serial query-head template; the runtime binds views to shared backing storage."""
    d, length = operator.cache.head_dim, operator.cache.length + operator.append_tokens
    q, k, v = (TensorSpec("head." + name, shape, "INT8") for name, shape in
               (("query", (1, d)), ("key_view", (d, length)), ("value", (length, d))))
    score = Gemm.from_tensors(q, k, output_name="head.scores")
    scale = _AttentionStage(score.spec.outputs[0], "attention_scale", "FP32", "head.scaled", 1 / sqrt(d))
    softmax = Softmax(scale.spec.outputs[0], output_name="head.probabilities_fp32")
    probabilities = _AttentionStage(softmax.spec.outputs[0], "storage_boundary", "INT8", "head.probabilities_int8")
    context = Gemm.from_tensors(probabilities.spec.outputs[0], v, output_name="head.context_int32")
    output = _AttentionStage(context.spec.outputs[0], "storage_boundary", "INT8", "head.context")
    return (q, k, v), [("qk", score), ("scale", scale), ("softmax", softmax),
                       ("probability_storage", probabilities), ("pv", context), ("output_storage", output)]


def simulate_attention(operator, architecture, spec, *, mode="count", include_details=False, **policy):
    from ..runtime.attention import simulate_attention as run
    return run(operator, architecture, spec, mode=mode, include_details=include_details, **policy)


def simulate_kv_cache(operator, architecture, spec, *, mode="count", include_details=False, **policy):
    from ..runtime.attention import simulate_kv_cache as run
    return run(operator, architecture, spec, mode=mode, include_details=include_details, **policy)
