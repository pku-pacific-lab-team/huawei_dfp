"""CIM-TSP independent Vector work uses the native shared VXM only."""
from ..tsp.vector import simulate as simulate_native


def simulate(op, spec, execution, *, mode, **policy):
    result = simulate_native(op, spec.native_reference_spec, execution, mode=mode, **policy)
    result.report["reference_ids"]["cim"] = spec.reference_id
    result.report["boundary"]["macro_execution_included"] = False
    return result
