"""Single-core HBM -> UB -> Vector -> UB -> HBM, with L2 bypass."""
from collections import Counter
from fractions import Fraction

from ..vector import build_result, cycles, dma_actions, dma_counts, positive, tile_groups
from ..vector_resources import ascend_services
from ...hardware.ascend import VECTOR_OPERATION_COSTS


def simulate(op, spec, execution, *, mode, tile_elements=None):
    footprint = (len(op.inputs) + 1) * op.numel * 4
    hbm = positive(spec.chip["hbm_usable_bytes"], "chip.hbm_usable_bytes")
    if footprint > hbm:
        raise ValueError("unsupported_capacity: Vector tensors exceed HBM capacity")
    local = spec.capacities["UB"]
    tile, groups = tile_groups(op.numel, tile_elements, execution, local // (4 * (len(op.inputs) + 1)))
    stages, services = Counter(), {}
    required = ["chip.hbm_bandwidth_bytes_per_second", "clock_freq_ghz", "ascend_ub_read_bpc", "ascend_ub_write_bpc",
                "vector.base_ops_per_cycle", "vector." + VECTOR_OPERATION_COSTS[op.operation]]
    if mode == "timing":
        resources = ascend_services(spec, execution, True)
        hbm_bpc, read, write = resources.hbm_bpc, resources.read_bpc, resources.write_bpc
        services = {"hbm_bpc": float(hbm_bpc), "ub_read_bpc": float(read), "ub_write_bpc": float(write)}
        for n, repeats in groups:
            stages["hbm_to_ub"] += repeats * len(op.inputs) * cycles(4*n, min(hbm_bpc, write), execution.policy["transfer_startup_cycles"])
            stages["ub_operand_read"] += repeats * cycles(4*n*len(op.inputs), read)
            stages["vector_compute"] += repeats * (spec.vector_service_cycles({op.operation: n}) + execution.policy["compute_startup_cycles"])
            stages["ub_result_write"] += repeats * cycles(4*n, write)
            stages["ub_to_hbm"] += repeats * cycles(4*n, min(hbm_bpc, read), execution.policy["transfer_startup_cycles"])
    memory = {"hbm": {"scope": "chip", "usable_bytes": int(hbm), "logical_footprint_bytes": footprint},
              "ub": {"scope": "core", "usable_bytes": local, "allocated_bytes": tile * 4 * (len(op.inputs)+1), "slots": 1},
              "layout": "Disjoint inputs/output; one UB tile working set; logical payload, no L2 allocation"}
    return build_result(op, spec, execution, mode, memory, groups, dma_counts(op, "ub"), stages, dma_actions("ub"),
                        None if spec.clock_ghz is None else float(Fraction(str(spec.clock_ghz))*10**9),
                        boundary={"start": "tensors_in_HBM", "end": "output_in_HBM", "l2_policy": "bypass"},
                        services=services, required_parameters=required)
