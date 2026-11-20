"""Pure parsing of Vector and normalization workload descriptions."""
from ..operators import TensorSpec, Vector, RMSNorm, Softmax
from ..operators.blocks import Residual, SwiGLU, SwiGLUFFN
from ..operators.attention import KVCacheSpec, KVCacheAccess, DecodeAttention


def parse_attention_workload(config):
    required = {"schema_version", "operation", "cache"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError("Attention/cache workload requires schema_version, operation, cache")
    kind = config["operation"]
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or kind not in ("kv_cache", "decode_attention"):
        raise ValueError("Expected KV Cache/DecodeAttention workload schema 1")
    optional = {"description", "append_tokens"} | ({"query", "output_name"} if kind == "decode_attention" else {"read_cache"})
    if set(config) - required - optional:
        raise ValueError("Unknown attention/cache workload fields")
    if "description" in config and (not isinstance(config["description"], str) or not config["description"].strip()):
        raise ValueError("Attention description must be nonempty text")
    fields = {"batch", "kv_heads", "head_dim", "length", "capacity"}
    cache = config["cache"]
    if not isinstance(cache, dict) or not fields <= set(cache) or set(cache) - fields - {"name", "dtype"}:
        raise ValueError("Cache requires batch, kv_heads, head_dim, length and capacity")
    cache = KVCacheSpec(**cache)
    append = config.get("append_tokens", 1)
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
    factories = {"residual": (Residual, 2), "swiglu": (SwiGLU, 2), "swiglu_ffn": (SwiGLUFFN, 4)}
    if type(config["schema_version"]) is not int or config["schema_version"] != 1 or not isinstance(operation, str) or operation not in factories:
        raise ValueError("Expected Residual/SwiGLU/FFN workload schema 1")
    optional = {"description", "output_name"} | ({"output_dtype"} if operation == "swiglu" else set())
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
    options = {k: config[k] for k in ("output_name", "output_dtype") if k in config}
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
