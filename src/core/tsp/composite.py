"""Resident MEM row blocks with explicit scratch and directional streams."""
from copy import deepcopy
from fractions import Fraction

from ..vector import tile_groups
from ..composite_memory import ResidentMemoryPath
from ..vector_resources import resident_allocation, resident_services


def make_path(op, spec, execution, *, mode, tile_elements=None, mem_slices=None, streams=None,
              reserved_bytes=0, hemisphere=0):
    allocation = resident_allocation(spec, mem_slices, streams, reserved_bytes, hemisphere, label="Composite")
    slices, streams = allocation.slices, allocation.streams
    scalars = 4 if op.operation == "rmsnorm" else 8
    stripe = lambda size: (size + len(slices) - 1) // len(slices)
    tensors = [t.nbytes for t in op.inputs] + [op.x.nbytes]
    base = sum(stripe(size) for size in tensors) + stripe(scalars) + stripe(reserved_bytes)
    capacity = spec.mem_slice_capacity_bytes
    if base + stripe(4) > capacity:
        raise ValueError("unsupported_capacity: complete normalization tensors and scratch exceed local MEM")
    tile, _ = tile_groups(op.width, tile_elements, execution, ((capacity - base) * len(slices)) // 4)
    memory = {"scope": "single_hem", "capacity_bytes": capacity * len(slices),
                   "logical_payload_bytes": sum(tensors), "scratch_tile_bytes": 4 * tile,
                   "row_state_bytes": scalars, "reserved_per_slice_bytes": stripe(reserved_bytes),
                   "allocated_per_slice_bytes": base + stripe(4 * tile),
                   "resident_bytes": (base + stripe(4 * tile)) * len(slices),
                   "mem_slices": list(slices), "streams_per_direction": streams,
                   "layout": "Full tensors plus one reusable tile scratch and row scalars; separate byte-striped allocations; no offchip paging"}
    timed = mode == "timing"
    clock = None if spec.clock_freq_ghz is None else Fraction(str(spec.clock_freq_ghz)) * 10**9
    width, startup = spec.stream_width_bytes, execution.policy["transfer_startup_cycles"]
    resources = resident_services(spec, allocation, timed)
    read_bpc, write_bpc, stream_bpc = resources.read_bpc, resources.write_bpc, resources.stream_bpc
    boundary = {"start": "tensors_in_local_MEM", "end": "output_in_local_MEM", "selected_hemisphere": hemisphere,
                     "offchip_supply_included": False, "compute_resource": "one_shared_vxm_service"}
    provenance, references = deepcopy(spec.parameters), {"native": spec.reference_id}
    services = {"mem_read_bpc": read_bpc, "mem_write_bpc": write_bpc, "stream_bpc_per_direction": stream_bpc}
    required = ["mem_read_bpc", "mem_write_bpc", "stream_width_bytes", "streams_per_direction"]
    compute, budget = spec.vxm_service_cycles, spec.values["vxm_base_ops_per_cycle"]
    compute_prefix = "vxm_"
    requested_tile = execution.policy["tile_elements"] if tile_elements is None else tile_elements
    return ResidentMemoryPath(
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
        width=width,
        stream_bpc=stream_bpc)
