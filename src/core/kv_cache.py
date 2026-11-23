"""Cache append copies and streaming reads; no cache-wide rewrite or arithmetic."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction

from .composite_memory import LocalMemoryPath, ResidentMemoryPath
from .metrics import attach_timing_breakdowns
from .operator_result import ActionSpec, NativeResult
from .vector import tile_groups, positive
from .vector_resources import ascend_services, tpu_services, resident_services


def make_path(arch, spec, execution, mode, live_bytes, capacity, allocation, tile_elements, transfer_size):
    resident, timed = allocation is not None, mode == "timing"
    native = spec.native_reference_spec if arch == "tsp_cim" else spec
    if live_bytes > capacity:
        raise ValueError("unsupported_capacity: preallocated KV Cache and inputs exceed storage")
    if resident:
        services = resident_services(native, allocation, timed)
        clock = None if native.clock_freq_ghz is None else Fraction(str(native.clock_freq_ghz)) * 10**9
        tile, _ = tile_groups(transfer_size, tile_elements, execution)
        memory = {"scope": "single_hem", "capacity_bytes": capacity, "resident_bytes": live_bytes,
                  "scratch_tile_bytes": 0, "row_state_bytes": 0, "mem_slices": list(allocation.slices)}
        provenance = deepcopy(native.parameters)
        rates = {"mem_read_bpc": services.read_bpc, "mem_write_bpc": services.write_bpc, "stream_bpc": services.stream_bpc}
    else:
        local = "ub" if arch == "ascend" else "vmem"
        local_capacity = spec.capacities["UB"] if arch == "ascend" else int(positive(spec.values["vmem_usable_bytes"], "vmem_usable_bytes"))
        tile, _ = tile_groups(transfer_size, tile_elements, execution, local_capacity)
        services = ascend_services(spec, execution, timed) if arch == "ascend" else tpu_services(spec, timed)
        clock = (None if spec.clock_ghz is None else Fraction(str(spec.clock_ghz)) * 10**9) if arch == "ascend" else spec.values["clock_hz"]
        memory = {"hbm": {"scope": "chip", "capacity_bytes": capacity, "peak_bytes": live_bytes},
                  local: {"scope": "core" if arch == "ascend" else "tensorcore", "capacity_bytes": local_capacity, "allocated_bytes": tile}}
        provenance = spec.provenance()
        rates = {"hbm_bpc": services.hbm_bpc, "local_read_bpc": services.read_bpc, "local_write_bpc": services.write_bpc}
    kwargs = dict(spec=native, timed=timed, tile=tile,
                  requested_tile=execution.policy["tile_elements"] if tile_elements is None else tile_elements,
                  clock=clock, startup=execution.policy["transfer_startup_cycles"], compute=lambda _: 0,
                  budget=None, compute_prefix="", memory=memory, boundary={}, provenance=provenance,
                  references={}, services=rates, required=[], read_bpc=services.read_bpc, write_bpc=services.write_bpc)
    if resident:
        return ResidentMemoryPath(**kwargs, width=native.stream_width_bytes, stream_bpc=services.stream_bpc)
    return LocalMemoryPath(**kwargs, local=local, hbm_bpc=services.hbm_bpc)


def simulate(op, path, execution, mode):
    c = op.cache
    phases = []
    if op.append_tokens:
        phases.append(("append", c.head_dim, True, 2 * c.batch * c.kv_heads))
    if op.read_cache:
        phases.append(("read", op.cache_after.length * c.head_dim, False, 2 * c.batch * c.kv_heads))
    return simulate_payloads(op, path, execution, mode, phases=phases,
        appended_bytes=2 * c.batch * c.kv_heads * c.head_dim * op.append_tokens,
        cache_read_bytes=2 * c.batch * c.kv_heads * c.head_dim * op.cache_after.length if op.read_cache else 0,
        model="kv_cache_access",
        policy="Append new K/V only, then stream filled cache once per KV head; one reusable transfer buffer, no full read copy.")


def simulate_payloads(op, path, execution, mode, *, phases, appended_bytes, cache_read_bytes, model, policy):
    """Common memory service execution for independently sized cache components."""
    counts, times, groups = Counter(), Counter(), []
    for phase, size, copy, instances in phases:
        _, tiles = tile_groups(size, path.tile, execution)
        for n, repetitions in tiles:
            repetitions *= instances
            if isinstance(path, ResidentMemoryPath):
                read, rt = path.access([n], "read")
                write, wt = path.access([n], "write") if copy else (Counter(), 0)
            else:
                read, rt = path.load(n)
                write, wt = path.export(n) if copy else (Counter(), 0)
            counts.update({phase + "." + k: v * repetitions for k, v in (read + write).items()})
            times[phase + ".read"] += rt * repetitions
            if copy:
                times[phase + ".write"] += wt * repetitions
            groups.append({"phase": phase, "bytes": n, "repetitions": repetitions,
                           "cycles_per_repetition": rt + wt if path.timed else None})
    scope = execution.policy["scope"]
    metadata, actions = {}, {}
    for key, count in counts.items():
        component, name, category, basis = path.action_specs[key.split(".", 1)[1]]
        metadata[key] = ActionSpec(component, name, category, "INT8")
        actions[key] = {"count": count, "unit": "byte", "scope": scope, "basis": basis}
    total = sum(times.values()) if path.timed else None
    timing = {"status": "analytical_estimate" if path.timed else "not_evaluated", "total_cycles": total,
              "time_ns": None if total is None or path.clock is None else float(Fraction(total) * 10**9 / Fraction(str(path.clock))),
              "model": "serial_cache_copy_and_streaming_read",
              "derived_services": {k: None if v is None else float(v) for k, v in path.services.items()}}
    attach_timing_breakdowns(timing, service=dict(times) if path.timed else None, elapsed=dict(times) if path.timed else None)
    report = {"schema_version": 1, "model": model, "mode": mode, "workload": op.spec.to_dict(),
              "boundary": {"scope": scope}, "execution_contract": {"scope": scope, "selected_tile_elements": path.tile,
                                                                           "requested_tile_elements": path.requested_tile},
              "timing": timing, "memory": deepcopy(path.memory), "action_counts": actions,
              "work_counts": {"logical": {"operations": 0, "output_bytes": sum(t.nbytes for t in op.spec.outputs),
                                          "appended_bytes": appended_bytes,
                                          "cache_read_bytes": cache_read_bytes},
                              "issued": {"operations": None, "status": "not_applicable"},
                              "internal": {"status": "not_modeled"}},
              "schedule": {"tile_groups": groups, "policy": policy},
              "parameter_provenance": path.provenance,
              "limitations": ["Cache offsets and head layouts are symbolic; no gather/scatter or transpose instructions.",
                              "Resident copy counts MEM-to-stream and stream-to-MEM service; transient forwarding registers are not modeled.",
                              "Reads end in a reusable local tile buffer or stream; no second full cache allocation."]}
    return NativeResult(report, metadata, "INT8 cache append/read payload; allocated capacity is not traffic")
