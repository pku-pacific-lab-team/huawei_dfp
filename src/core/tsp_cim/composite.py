"""CIM normalization uses the native VXM and MEM, without Macro work."""
from dataclasses import replace

from ..tsp.composite import make_path as native_path


def make_path(op, spec, execution, **policy):
    path = native_path(op, spec.native_reference_spec, execution, **policy)
    return replace(path, references={**path.references, "cim": spec.reference_id},
                   boundary={**path.boundary, "macro_execution_included": False})
