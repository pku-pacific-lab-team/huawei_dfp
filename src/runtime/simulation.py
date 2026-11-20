"""Select native kernels, then adapt their results without recalculating work."""
from ..hardware.vector import VectorExecution, load_vector_execution
from ..operators.primitives import Gemm
from ..operators.vector import Vector
from ..operators.composite import RMSNorm, Softmax
from ..operators.adapters import adapt_gemm_report, adapt_native_result
from ..core.composite import simulate as run_composite
from .backends import get_backend


def execution_profile(architecture, execution, mode, *, composite=False):
    if mode not in ("count", "timing"):
        raise ValueError("mode must be count or timing")
    execution = load_vector_execution(architecture) if execution is None else execution
    if not isinstance(execution, VectorExecution) or execution.architecture != architecture:
        raise ValueError("Vector execution profile does not match architecture")
    if composite and execution.composite is None:
        raise ValueError("Composite execution requires a matching Vector profile schema 2")
    return execution


def simulate_gemm(operator, architecture, spec, *, mode="count", include_details=False, **policy):
    if not isinstance(operator, Gemm):
        raise ValueError("GEMM execution requires Gemm")
    backend = get_backend(architecture, spec)
    work = operator.to_workload()
    simulator = backend.gemm_simulator(spec)
    if backend.gemm_uses_shape:
        report = simulator.simulate(*work.shape, mode=mode, **policy)
        report["workload"] = work.to_dict()
    else:
        report = simulator.simulate(work, mode=mode, **policy)
    return adapt_gemm_report(architecture, report, operator=operator, include_details=include_details)


def simulate_vector(operator, architecture, spec, *, mode="count", execution=None, include_details=False, **policy):
    if not isinstance(operator, Vector):
        raise ValueError("simulate_vector requires a Vector operator")
    backend = get_backend(architecture, spec)
    execution = execution_profile(architecture, execution, mode)
    native = backend.vector_kernel(operator, spec, execution, mode=mode, **policy)
    requested = policy.get("tile_elements")
    native.report["execution_contract"]["requested_tile_elements"] = execution.policy["tile_elements"] if requested is None else requested
    return adapt_native_result(operator.spec, architecture, native, include_details=include_details)


def simulate_composite(operator, architecture, spec, *, mode="count", execution=None, include_details=False, **policy):
    if not isinstance(operator, (RMSNorm, Softmax)):
        raise ValueError("Expected RMSNorm or Softmax")
    backend = get_backend(architecture, spec)
    execution = execution_profile(architecture, execution, mode, composite=True)
    path = backend.composite_path(operator, spec, execution, mode=mode, **policy)
    native = run_composite(operator, path, execution, mode)
    return adapt_native_result(operator.spec, architecture, native, include_details=include_details)
