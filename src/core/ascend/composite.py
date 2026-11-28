"""Single AI Core row blocks through HBM/UB; L2 bypass."""
from fractions import Fraction

from ..composite_memory import LocalMemoryPath
from ..capacity import CapacityExceeded
from ..vector import positive, tile_groups
from ..vector_resources import ascend_services


def make_path(op, spec, execution, *, mode, tile_elements=None):
    local_name, timed = "ub", mode == "timing"
    startup = execution.policy["transfer_startup_cycles"]
    footprint = sum(t.nbytes for t in op.inputs) + op.x.nbytes
    hbm = positive(spec.chip["hbm_usable_bytes"], "chip.hbm_usable_bytes")
    if footprint > hbm:
        raise ValueError("unsupported_capacity: complete normalization tensors exceed HBM")
    slots, scalars = (2, 4) if op.operation == "rmsnorm" else (1, 8)
    local = spec.capacities["UB"]
    if local < scalars + 4 * slots:
        raise CapacityExceeded("unsupported_capacity: local memory cannot hold one Vector element and its operands",
                               local_name, scalars + 4 * slots, local)
    tile, _ = tile_groups(op.width, tile_elements, execution, (local - scalars) // (4 * slots))
    clock = None if spec.clock_ghz is None else Fraction(str(spec.clock_ghz)) * 10**9
    resources = ascend_services(spec, execution, timed)
    hbm_bpc, read_bpc, write_bpc = resources.hbm_bpc, resources.read_bpc, resources.write_bpc
    memory = {"hbm": {"scope": "chip", "usable_bytes": int(hbm), "logical_footprint_bytes": footprint},
                   "ub": {"scope": "core", "usable_bytes": local, "tile_slots": slots,
                          "row_state_bytes": scalars, "allocated_bytes": 4 * tile * slots + scalars}}
    boundary = {"start": "tensors_in_HBM", "end": "output_in_HBM", "l2_policy": "bypass"}
    provenance, references = spec.provenance(), {}
    services = {"hbm_bpc": hbm_bpc, "ub_read_bpc": read_bpc, "ub_write_bpc": write_bpc}
    required = ["clock_freq_ghz", "chip.hbm_bandwidth_bytes_per_second", "chip.architectural_cores",
                     "ascend_ub_read_bpc", "ascend_ub_write_bpc"]
    compute = spec.vector_service_cycles
    budget = spec.vector_parameters["base_ops_per_cycle"]
    compute_prefix = "vector."
    requested_tile = execution.policy["tile_elements"] if tile_elements is None else tile_elements
    return LocalMemoryPath(
        spec=spec,
        timed=timed,
        tile=tile,
        requested_tile=requested_tile,
        clock=clock,
        startup=startup,
        compute=compute,
        budget=budget,
        compute_prefix=compute_prefix,
        memory=memory,
        boundary=boundary,
        provenance=provenance,
        references=references,
        services=services,
        required=required,
        read_bpc=read_bpc,
        write_bpc=write_bpc,
        local=local_name,
        hbm_bpc=hbm_bpc)
