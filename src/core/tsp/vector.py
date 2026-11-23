"""Resident MEM -> shared VXM -> MEM; reused unchanged by CIM-TSP."""
from collections import Counter
from fractions import Fraction

from ..vector import build_result, cycles, tile_groups
from ..vector_resources import resident_allocation, resident_services
from ...hardware.tsp import VXM_OPERATION_COSTS


def simulate(op, spec, execution, *, mode, tile_elements=None, mem_slices=None, streams=None,
             reserved_bytes=0, hemisphere=0):
    allocation = resident_allocation(spec, mem_slices, streams, reserved_bytes, hemisphere, label="Vector")
    slices, streams = allocation.slices, allocation.streams
    per_tensor = (4*op.numel + len(slices)-1) // len(slices)
    reserved = (reserved_bytes + len(slices)-1) // len(slices)
    per_slice = (len(op.inputs)+1) * per_tensor + reserved
    if per_slice > spec.mem_slice_capacity_bytes:
        raise ValueError("unsupported_capacity: complete Vector inputs/output exceed local MEM; no off-chip supply")
    tile, groups = tile_groups(op.numel, tile_elements, execution)
    counts = Counter(mem_read_bytes=4*op.numel*len(op.inputs), mem_write_bytes=4*op.numel,
                     input_stream_bytes=0, output_stream_bytes=0)
    stages, services = Counter(), {}
    required = ["mem_read_bpc", "mem_write_bpc", "vxm_base_ops_per_cycle", VXM_OPERATION_COSTS[op.operation]]
    if mode == "timing":
        resources = resident_services(spec, allocation, True)
        read, write, stream = resources.read_bpc, resources.write_bpc, resources.stream_bpc
        services = {"mem_read_bpc": float(read), "mem_write_bpc": float(write), "stream_bpc_per_direction": float(stream)}
    for n, repeats in groups:
        padded = (4*n + spec.stream_width_bytes-1) // spec.stream_width_bytes * spec.stream_width_bytes
        counts["input_stream_bytes"] += padded * len(op.inputs) * repeats
        counts["output_stream_bytes"] += padded * repeats
        if mode == "timing":
            stages["mem_to_vxm"] += repeats*len(op.inputs)*(max(cycles(4*n, read), cycles(padded, stream))+execution.policy["transfer_startup_cycles"])
            stages["vector_compute"] += repeats*(spec.vxm_service_cycles({op.operation: n})+execution.policy["compute_startup_cycles"])
            stages["vxm_to_mem"] += repeats*(max(cycles(4*n, write), cycles(padded, stream))+execution.policy["transfer_startup_cycles"])
    actions = {"mem_read_bytes": ("mem", "read", "onchip_memory", "byte", "logical"),
               "mem_write_bytes": ("mem", "write", "onchip_memory", "byte", "logical"),
               "input_stream_bytes": ("stream_to_vxm", "transfer", "interconnect", "byte", "padded"),
               "output_stream_bytes": ("stream_from_vxm", "transfer", "interconnect", "byte", "padded")}
    memory = {"scope": "single_hem", "capacity_bytes": spec.mem_slice_capacity_bytes * len(slices),
              "resident_bytes": per_slice * len(slices), "logical_payload_bytes": (len(op.inputs)+1)*op.numel*4,
              "per_tensor_per_slice_bytes": per_tensor, "reserved_per_slice_bytes": reserved,
              "mem_slices": list(slices), "streams_per_direction": streams,
              "layout": "Disjoint full tensors uniformly striped; stream tail padding is not MEM padding; no extra full-tile allocation"}
    return build_result(op, spec, execution, mode, memory, groups, counts, stages, actions,
                        None if spec.clock_freq_ghz is None else float(Fraction(str(spec.clock_freq_ghz))*10**9),
                        boundary={"start": "tensors_in_local_MEM", "end": "output_in_local_MEM", "selected_hemisphere": hemisphere,
                                  "offchip_supply_included": False, "compute_resource": "one_shared_vxm_service"},
                        services=services, required_parameters=required, reference_ids={"native": spec.reference_id})
