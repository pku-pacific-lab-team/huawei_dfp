"""Single-TensorCore HBM/VMEM Vector service; CMEM explicitly bypassed."""
from collections import Counter

from ..vector import build_result, cycles, dma_actions, dma_counts, positive, tile_groups
from ..vector_resources import tpu_services
from ...hardware.tpu import VECTOR_OPERATION_COSTS


def simulate(op, spec, execution, *, mode, tile_elements=None):
    v = spec.values
    footprint = (len(op.inputs)+1) * op.numel * 4
    hbm, vmem = positive(v["hbm_usable_bytes"], "hbm_usable_bytes"), positive(v["vmem_usable_bytes"], "vmem_usable_bytes")
    if footprint > hbm:
        raise ValueError("unsupported_capacity: Vector tensors exceed HBM capacity")
    tile, groups = tile_groups(op.numel, tile_elements, execution, int(vmem) // (4 * (len(op.inputs)+1)))
    stages, services = Counter(), {}
    required = ["clock_hz", "tensorcores_per_chip", "hbm_bandwidth_bytes_per_second", "vmem_bandwidth_bytes_per_second",
                "vector_base_ops_per_cycle", VECTOR_OPERATION_COSTS[op.operation]]
    if mode == "timing":
        resources = tpu_services(spec, True)
        hbm_bpc, local_bpc = resources.hbm_bpc, resources.read_bpc
        dma = min(hbm_bpc, local_bpc)
        services = {"hbm_bpc": float(hbm_bpc), "vmem_shared_bpc": float(local_bpc), "dma_bpc": float(dma)}
        for n, repeats in groups:
            stages["hbm_to_vmem"] += repeats * len(op.inputs) * cycles(4*n, dma, execution.policy["transfer_startup_cycles"])
            stages["vmem_operand_read"] += repeats * cycles(4*n*len(op.inputs), local_bpc)
            stages["vector_compute"] += repeats * (spec.vector_service_cycles({op.operation: n}) + execution.policy["compute_startup_cycles"])
            stages["vmem_result_write"] += repeats * cycles(4*n, local_bpc)
            stages["vmem_to_hbm"] += repeats * cycles(4*n, dma, execution.policy["transfer_startup_cycles"])
    memory = {"hbm": {"scope": "chip", "usable_bytes": int(hbm), "logical_footprint_bytes": footprint},
              "vmem": {"scope": "tensorcore", "usable_bytes": int(vmem), "allocated_bytes": tile*4*(len(op.inputs)+1), "slots": 1},
              "cmem": {"mode": "bypass", "allocated_bytes": 0},
              "layout": "Disjoint inputs/output; one VMEM tile; no cross-TensorCore capacity pooling"}
    return build_result(op, spec, execution, mode, memory, groups, dma_counts(op, "vmem"), stages, dma_actions("vmem"), v["clock_hz"],
                        boundary={"start": "tensors_in_HBM", "end": "output_in_HBM", "cmem_policy": "bypass"},
                        services=services, required_parameters=required)
