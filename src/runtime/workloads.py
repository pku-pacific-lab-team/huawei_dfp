"""Execute a parsed workload using explicit configuration and native backends."""
from ..hardware.vector import load_vector_execution
from ..utils.operator_workload import parse_vector_workload, parse_composite_workload
from .simulation import simulate_vector, simulate_composite


def execute_workload(config, architecture, load_spec, *, mode, execution_file=None, config_dir=None, **policy):
    vector = isinstance(config, dict) and config.get("operation") == "vector"
    operator = parse_vector_workload(config) if vector else parse_composite_workload(config)
    execution = load_vector_execution(architecture, execution_file, config_dir)
    simulate = simulate_vector if vector else simulate_composite
    return simulate(operator, architecture, load_spec(), mode=mode, execution=execution, **policy)
