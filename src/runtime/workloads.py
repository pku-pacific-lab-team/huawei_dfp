"""Execute a parsed workload using explicit configuration and native backends."""
from ..hardware.vector import load_vector_execution
from ..utils.operator_workload import parse_vector_workload, parse_composite_workload
from .simulation import simulate_vector, simulate_composite
from ..hardware.blocks import load_block_execution
from ..utils.operator_workload import parse_block_workload
from .blocks import simulate_block
from ..hardware.attention import load_attention_execution
from ..utils.operator_workload import parse_attention_workload
from .attention import simulate_attention, simulate_kv_cache


def execute_workload(config, architecture, load_spec, *, mode, execution_file=None, block_execution_file=None,
                     attention_execution_file=None, config_dir=None, **policy):
    if isinstance(config, dict) and config.get("operation") in ("kv_cache", "decode_attention"):
        op = parse_attention_workload(config)
        if config["operation"] == "decode_attention":
            policy["block_execution"] = load_block_execution(block_execution_file, config_dir)
            simulate = simulate_attention
        else:
            if block_execution_file is not None:
                raise ValueError("KV Cache does not use a block compute profile")
            simulate = simulate_kv_cache
        return simulate(op, architecture, load_spec(), mode=mode,
                        execution=load_vector_execution(architecture, execution_file, config_dir),
                        attention_execution=load_attention_execution(attention_execution_file, config_dir), **policy)
    if attention_execution_file is not None:
        raise ValueError("Attention profile requires kv_cache/decode_attention")
    if isinstance(config, dict) and config.get("operation") in ("residual", "swiglu", "swiglu_ffn",
                                                              "rmsnorm_int8", "qkv_projection", "attention_output_projection"):
        return simulate_block(parse_block_workload(config), architecture, load_spec(), mode=mode,
                              execution=load_vector_execution(architecture, execution_file, config_dir),
                              block_execution=load_block_execution(block_execution_file, config_dir), **policy)
    if block_execution_file is not None:
        raise ValueError("Block profile requires a materialized block")
    vector = isinstance(config, dict) and config.get("operation") == "vector"
    operator = parse_vector_workload(config) if vector else parse_composite_workload(config)
    execution = load_vector_execution(architecture, execution_file, config_dir)
    simulate = simulate_vector if vector else simulate_composite
    return simulate(operator, architecture, load_spec(), mode=mode, execution=execution, **policy)
