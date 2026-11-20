"""Pure parsing of Vector and normalization workload descriptions."""
from ..operators import TensorSpec, Vector, RMSNorm, Softmax
from ..operators.blocks import Residual, SwiGLU, SwiGLUFFN
from ..operators.attention import KVCacheSpec, KVCacheAccess, DecodeAttention
from ..operators.projections import RMSNormINT8, QKVProjection, AttentionOutputProjection, VocabularyProjection
from ..operators.rope import RoPE
from ..operators.embedding import Embedding
from ..operators.mla import MLASpec, MLAPreprocess, MLACacheSpec, MLACacheAccess, MLADecodeAttention
from ..operators.moe import MoERoutingPlan, MoEExperts


def parse_moe_workload(config):
    required = {"schema_version", "operation", "x", "gate_weight", "up_weight", "down_weight", "routing"}
    optional = {"description", "shared_experts", "shared_gate_weight", "shared_up_weight", "shared_down_weight", "output_names"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError("MoE experts require activation, three routed weights and a fixed routing plan")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["operation"] != "moe_experts":
        raise ValueError("Expected MoE experts workload schema 1")
    if "description" in config and (not isinstance(config["description"], str) or not config["description"].strip()):
        raise ValueError("MoE description must be nonempty text")
    routing = config["routing"]
    if not isinstance(routing, dict) or not {"expert_count", "assignments"} <= set(routing) or set(routing) - {"expert_count", "assignments", "name"}:
        raise ValueError("MoE routing requires expert_count and assignments, with optional name")
    tensors = {}
    for key in ("x", "gate_weight", "up_weight", "down_weight", "shared_gate_weight", "shared_up_weight", "shared_down_weight"):
        if key in config:
            tensor = config[key]
            if not isinstance(tensor, dict) or set(tensor) != {"name", "shape", "dtype"}:
                raise ValueError("MoE tensor requires exactly name, shape and dtype")
            tensors[key] = TensorSpec(**tensor)
    return MoEExperts(**tensors, routing=MoERoutingPlan(**routing), shared_experts=config.get("shared_experts", 0),
                      output_names=config.get("output_names"))


def parse_mla_attention_workload(config):
    required = {"schema_version", "operation", "dimensions", "cache", "query_nope", "query_rope", "kv_up_weight"}
    optional = {"description", "append_tokens", "output_weight", "new_latent", "new_rope", "position",
                "scale_multiplier", "scale_source", "output_name"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - optional:
        raise ValueError("MLA Attention requires dimensions, cache, query branches and KV up weight")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["operation"] != "mla_decode_attention":
        raise ValueError("Expected MLA Attention workload schema 1")
    if "description" in config and (not isinstance(config["description"], str) or not config["description"].strip()):
        raise ValueError("MLA description must be nonempty text")
    dims, cache = config["dimensions"], config["cache"]
    if not isinstance(dims, dict) or set(dims) != set(MLASpec.__dataclass_fields__):
        raise ValueError("MLA requires all seven independent dimensions")
    fields = {"batch", "kv_lora_rank", "qk_rope_head_dim", "length", "capacity"}
    if not isinstance(cache, dict) or not fields <= set(cache) or set(cache) - fields - {"name", "dtype"}:
        raise ValueError("MLA cache requires dimensions, length and capacity")
    tensors = {}
    for key in ("query_nope", "query_rope", "kv_up_weight", "output_weight", "new_latent", "new_rope"):
        if key in config:
            t = config[key]
            if not isinstance(t, dict) or set(t) != {"name", "shape", "dtype"}:
                raise ValueError("MLA tensor requires exactly name, shape and dtype")
            tensors[key] = TensorSpec(**t)
    options = {k: config[k] for k in optional - {"description", *tensors} if k in config}
    return MLADecodeAttention(**tensors, dimensions=MLASpec(**dims), cache=MLACacheSpec(**cache), **options)


def parse_attention_workload(config):
    required = {"schema_version", "operation", "cache"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError("Attention/cache workload requires schema_version, operation, cache")
    kind = config["operation"]
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or kind not in ("kv_cache", "decode_attention", "mla_cache"):
        raise ValueError("Expected KV Cache/DecodeAttention workload schema 1")
    optional = {"description", "append_tokens"} | ({"query", "output_name"} if kind == "decode_attention" else {"read_cache"})
    if set(config) - required - optional:
        raise ValueError("Unknown attention/cache workload fields")
    if "description" in config and (not isinstance(config["description"], str) or not config["description"].strip()):
        raise ValueError("Attention description must be nonempty text")
    fields = {"batch", "length", "capacity"} | ({"kv_lora_rank", "qk_rope_head_dim"} if kind == "mla_cache" else {"kv_heads", "head_dim"})
    cache = config["cache"]
    if not isinstance(cache, dict) or not fields <= set(cache) or set(cache) - fields - {"name", "dtype"}:
        raise ValueError("Cache requires exactly its dimensions, length and capacity")
    cache = (MLACacheSpec if kind == "mla_cache" else KVCacheSpec)(**cache)
    append = config.get("append_tokens", 1)
    if kind == "mla_cache":
        return MLACacheAccess(cache, append, config.get("read_cache", True))
    if kind == "kv_cache":
        return KVCacheAccess(cache, append, config.get("read_cache", True))
    q = config.get("query")
    if not isinstance(q, dict) or set(q) != {"name", "shape", "dtype"}:
        raise ValueError("Attention query requires name, shape and dtype")
    return DecodeAttention(TensorSpec(**q), cache, append, config.get("output_name"))


def parse_block_workload(config):
    required = {"schema_version", "operation", "inputs"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError("Block workload requires schema_version, operation, inputs")
    operation = config["operation"]
    factories = {"residual": (Residual, 2), "swiglu": (SwiGLU, 2), "swiglu_ffn": (SwiGLUFFN, 4),
                 "rmsnorm_int8": (RMSNormINT8, 2), "qkv_projection": (QKVProjection, 4),
                 "attention_output_projection": (AttentionOutputProjection, 2), "rope": (RoPE, 4),
                 "embedding": (Embedding, 2), "vocabulary_projection": (VocabularyProjection, 2),
                 "mla_preprocess": (MLAPreprocess, 8)}
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or not isinstance(operation, str) or operation not in factories:
        raise ValueError("Expected a supported block workload schema 1")
    options_by_kind = {"swiglu": {"output_dtype"}, "swiglu_ffn": {"output_dtype"}, "rmsnorm_int8": {"axis", "epsilon"},
                       "qkv_projection": {"query_heads", "kv_heads", "head_dim", "output_names"},
                       "rope": {"position", "output_names"},
                       "mla_preprocess": {"dimensions", "position", "epsilon", "output_names"}}
    optional = {"description"} | options_by_kind.get(operation, set())
    if operation not in {"qkv_projection", "rope", "mla_preprocess"}:
        optional.add("output_name")
    if set(config) - required - optional:
        raise ValueError("Unknown block workload fields")
    if "description" in config and (not isinstance(config["description"], str) or not config["description"].strip()):
        raise ValueError("Block description must be nonempty text")
    factory, arity = factories[operation]
    if not isinstance(config["inputs"], list) or len(config["inputs"]) != arity:
        raise ValueError(f"{operation} requires {arity} inputs")
    tensors = []
    for tensor in config["inputs"]:
        if not isinstance(tensor, dict) or set(tensor) != {"name", "shape", "dtype"}:
            raise ValueError("Block input requires exactly name, shape, dtype")
        tensors.append(TensorSpec(**tensor))
    if operation == "qkv_projection" and not {"query_heads", "kv_heads", "head_dim"} <= set(config):
        raise ValueError("QKV workload requires query_heads, kv_heads and head_dim")
    if operation == "rope" and "position" not in config:
        raise ValueError("RoPE workload requires position")
    options = {k: config[k] for k in optional - {"description"} if k in config}
    if operation == "mla_preprocess":
        fields = set(MLASpec.__dataclass_fields__)
        if "position" not in options or not isinstance(options.get("dimensions"), dict) or set(options["dimensions"]) != fields:
            raise ValueError("MLA requires position and all seven independent dimensions")
        options["dimensions"] = MLASpec(**options["dimensions"])
    return factory(*tensors, **options)


def parse_vector_workload(config):
    required = {"schema_version", "operation", "op", "inputs"}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {"description", "output_name"}:
        raise ValueError("Vector workload requires schema_version, operation, op, inputs")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or config["operation"] != "vector":
        raise ValueError("Expected Vector workload schema 1")
    if "description" in config and (not isinstance(config["description"], str) or not config["description"].strip()):
        raise ValueError("Vector description must be nonempty text")
    if not isinstance(config["inputs"], list):
        raise ValueError("Vector inputs must be a list")
    tensors = []
    for tensor in config["inputs"]:
        if not isinstance(tensor, dict) or set(tensor) != {"name", "shape", "dtype"}:
            raise ValueError("Vector input requires exactly name, shape, dtype")
        tensors.append(TensorSpec(**tensor))
    return Vector(config["op"], tuple(tensors), output_name=config.get("output_name"))


def parse_composite_workload(config):
    required = {"schema_version", "operation", "inputs"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError("Composite workload requires schema_version, operation, inputs")
    operation = config["operation"]
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or operation not in ("rmsnorm", "softmax"):
        raise ValueError("Expected RMSNorm/Softmax workload schema 1")
    optional = {"axis", "description", "output_name"} | ({"epsilon"} if operation == "rmsnorm" else set())
    if set(config) - required - optional:
        raise ValueError("Unknown composite workload fields")
    if "description" in config and (not isinstance(config["description"], str) or not config["description"].strip()):
        raise ValueError("Composite description must be nonempty text")
    if not isinstance(config["inputs"], list) or len(config["inputs"]) != (2 if operation == "rmsnorm" else 1):
        raise ValueError("RMSNorm requires x/weight inputs; Softmax requires one x input")
    tensors = []
    for tensor in config["inputs"]:
        if not isinstance(tensor, dict) or set(tensor) != {"name", "shape", "dtype"}:
            raise ValueError("Composite input requires exactly name, shape, dtype")
        tensors.append(TensorSpec(**tensor))
    if operation == "rmsnorm":
        return RMSNorm(*tensors, epsilon=config.get("epsilon", 1e-5), axis=config.get("axis", -1), output_name=config.get("output_name"))
    return Softmax(*tensors, axis=config.get("axis", -1), output_name=config.get("output_name"))
