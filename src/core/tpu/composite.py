"""Single TensorCore row blocks through HBM/VMEM; CMEM bypass."""
from ..composite_memory import LocalMemoryPath
from ..capacity import CapacityExceeded
from ..vector import positive, tile_groups
from ..vector_resources import tpu_services


def make_path(op, spec, execution, *, mode, tile_elements=None):
    v = spec.values
    local_name, timed = "vmem", mode == "timing"
    startup = execution.policy["transfer_startup_cycles"]
    footprint = sum(t.nbytes for t in op.inputs) + op.x.nbytes
    hbm = positive(v["hbm_usable_bytes"], "hbm_usable_bytes")
    local = int(positive(v["vmem_usable_bytes"], "vmem_usable_bytes"))
    if footprint > hbm:
        raise ValueError("unsupported_capacity: complete normalization tensors exceed HBM")
    slots, scalars = (2, 4) if op.operation == "rmsnorm" else (1, 8)
    if local < scalars + 4 * slots:
        raise CapacityExceeded("unsupported_capacity: local memory cannot hold one Vector element and its operands",
                               local_name, scalars + 4 * slots, local)
    tile, _ = tile_groups(op.width, tile_elements, execution, (local - scalars) // (4 * slots))
    clock = v["clock_hz"]
    resources = tpu_services(spec, timed)
    hbm_bpc, read_bpc, write_bpc = resources.hbm_bpc, resources.read_bpc, resources.write_bpc
    memory = {"hbm": {"scope": "chip", "usable_bytes": int(hbm), "logical_footprint_bytes": footprint},
                   "vmem": {"scope": "tensorcore", "usable_bytes": local, "tile_slots": slots,
                            "row_state_bytes": scalars, "allocated_bytes": 4 * tile * slots + scalars},
                   "cmem": {"mode": "bypass", "allocated_bytes": 0}}
    boundary = {"start": "tensors_in_HBM", "end": "output_in_HBM", "cmem_policy": "bypass"}
    provenance, references = spec.provenance(), {}
    services = {"hbm_bpc": hbm_bpc, "vmem_shared_bpc": read_bpc}
    required = ["clock_hz", "tensorcores_per_chip", "hbm_bandwidth_bytes_per_second", "vmem_bandwidth_bytes_per_second"]
    compute, budget = spec.vector_service_cycles, v["vector_base_ops_per_cycle"]
    compute_prefix = "vector_"
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
