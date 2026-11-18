"""INT8 hemisphere model: static Cluster waves and shared resource reservations.

The row pipeline is compressed, like the native reference, rather than a cycle
simulator. Wave barriers are deliberate; LW can overlap the preceding pipeline.
"""
from ..metrics import attach_contract, attach_timing_breakdowns
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from itertools import product
from math import ceil

from ..cim.simulator import LOOP_ORDERS, account_macro_tile
from ..tsp.tiling import TSPTilingEngine, align_up
from .resources import ResourceCalendar
from ...hardware.tsp import VXM_ADD_PARAMETERS


COMMIT = "c641844c1d213d02a7d961d889838cae9b9dc8aa"
SOURCE = f"https://github.com/pku-pacific-lab-team/LPU-HUAWEI-2012/blob/{COMMIT}/docs/legacy/patent-idea-review.md"


def _waves(spec, M, K, N, tm, order):
    """Stable K/N placement avoids moving weights when visiting another M group."""
    tk, tn = spec.plane.domain_k, spec.cluster.n
    domains = spec.values["planes_per_hemisphere"] * spec.plane.clusters_per_superlane
    nt = ceil(N / tn)
    axes = {key: range(0, size, step) for key, size, step in zip("MKN", (M, K, N), (tm, tk, tn))}
    pending, occupied, row_start = [], set(), None
    for indices in product(*(axes[key] for key in order)):
        pos = dict(zip(order, indices))
        ms, ks, ns = (pos[key] for key in "MKN")
        domain = (ks // tk * nt + ns // tn) % domains
        if pending and (domain in occupied or ms != row_start):
            yield pending
            pending, occupied = [], set()
        pending.append(dict(ms=ms, ks=ks, ns=ns, m=min(tm, M-ms),
                            k=min(tk, K-ks), n=min(tn, N-ns), domain=domain))
        occupied.add(domain)
        row_start = ms
    if pending:
        yield pending


def simulate_hemisphere(spec, M, K, N, order, *, mode, hemisphere, tile_m,
                        mem_slices, streams, reserved_bytes, load_overlap):
    if order not in LOOP_ORDERS:
        raise ValueError("loop_order must be a permutation of MKN")
    if type(hemisphere) is not int or not 0 <= hemisphere < spec.values["hemispheres"]:
        raise ValueError("hemisphere outside topology: expected 0 or 1")
    execution = spec.execution
    load_overlap = execution["weight_load_overlap"] if load_overlap is None else load_overlap
    if load_overlap not in {"serialized", "background"}:
        raise ValueError("load_overlap must be serialized or background")
    tm = min(M, execution["default_tile_mkn"][0]) if tile_m is None else tile_m
    # Reuse native allocation/validation, with CIM's N32 padded resident layout.
    # Padding N here changes only the memory plan, never the logical workload.
    plan = TSPTilingEngine(spec.native_reference_spec).derive_tile_sizes(
        M, K, align_up(N, spec.macro_spec.block_n_dim), tile=(tm, 320, 640),
        mem_slices=mem_slices, streams=streams, reserved_bytes=reserved_bytes)
    tm = plan["m"]
    native, v = spec.native_reference_spec, spec.native_reference_spec.values
    need = {"mem_read_bpc", "mem_write_bpc"}
    if K > spec.plane.domain_k:
        need.update(VXM_ADD_PARAMETERS)
    missing = sorted(name for name in need if v[name] is None)
    if mode == "timing" and missing:
        raise ValueError("Missing CIM hemisphere timing parameters: " + ", ".join(missing))
    domains = spec.values["planes_per_hemisphere"] * spec.plane.clusters_per_superlane
    contexts, caches = [{} for _ in range(domains)], [{} for _ in range(domains)]
    counts = Counter({name: 0 for name in (
        "tiles", "program_bytes", "context_programs", "context_switches", "program_wordlines",
        "cross_macro_k_adds", "cross_tile_k_adds", "mem_read_weight_bytes",
        "mem_read_activation_bytes", "mem_read_psum_bytes", "mem_write_psum_bytes",
        "mem_write_final_bytes", "activation_stream_bytes", "psum_stream_bytes", "write_stream_bytes")})
    switch_away, schedules, per_domain = {}, [], [Counter() for _ in range(domains)]
    psum_progress = {}
    timing = HemisphereTiming(spec, plan, need, load_overlap) if mode == "timing" else None
    for index, tasks in enumerate(_waves(spec, M, K, N, tm, order)):
        programmed, release_after = [], set()
        switches = 0
        for task in tasks:
            domain = task["domain"]
            output_block = (task["ms"], task["ns"])
            predecessor = psum_progress.get(output_block)
            if (predecessor["k_stop"] if predecessor else 0) != task["ks"]:
                raise ValueError("K reduction order is inconsistent")
            task["psum_predecessor"] = predecessor
            psum_progress[output_block] = {"wave": index, "domain": domain,
                                           "k_stop": task["ks"] + task["k"]}
            before = counts.copy()
            counts["tiles"] += 1
            programs, changed = account_macro_tile(spec.macro_spec, contexts[domain], caches[domain], counts,
                **{key: task[key] for key in ("ks", "ns", "m", "k", "n")},
                pn=spec.cluster.n // spec.macro_spec.block_n_dim)
            for slot, context in programs:
                key = (domain, slot, context)
                if key in switch_away:
                    release_after.add(switch_away[key])
                programmed.append((domain, slot))
            for slot, old in changed:
                if old is not None:
                    switch_away[domain, slot, old] = index
            switches += len(changed)
            m, k, n, ks = (task[key] for key in ("m", "k", "n", "ks"))
            counts["mem_read_weight_bytes"] += len(programs) * spec.macro_spec.context_bytes
            counts["mem_read_activation_bytes"] += m * align_up(k, spec.core.k)
            counts["mem_read_psum_bytes"] += 4*m*n if ks else 0
            counts["mem_write_final_bytes" if ks+k == K else "mem_write_psum_bytes"] += 4*m*n
            counts["activation_stream_bytes"] += m * align_up(k, native.stream_width_bytes)
            counts["psum_stream_bytes"] += m * align_up(4*n, native.stream_width_bytes) if ks else 0
            counts["write_stream_bytes"] += m * align_up(4*n, native.n_lanes_per_superlane)
            per_domain[domain].update(counts - before)
        record = {"wave": index, "tasks": tasks, "active_domains": len(tasks),
                  "programmed_macros": len(programmed), "context_switches": switches,
                  "release_after_waves": sorted(release_after)}
        if timing:
            record.update(timing.schedule(tasks, programmed, switches, release_after, schedules))
        schedules.append(record)

    counts.update(useful_ops=2*M*K*N, executed_padded_ops=2*counts["issued_mac"], logical_output_bytes=4*M*N,
        logical_activation_broadcast_bytes=counts["mem_read_activation_bytes"],
        logical_inter_superlane_adds=counts["cross_macro_k_adds"],
        logical_inter_k_tile_adds=counts["cross_tile_k_adds"],
        logical_cluster_partial_bytes=counts["mem_write_psum_bytes"]+counts["mem_write_final_bytes"])
    # Keep physical Macro endpoint actions distinct from shared MEM/stream payloads.
    actions = {}
    units = {"launches": "launch", "context_programs": "context", "context_switches": "switch",
             "program_wordlines": "wordline", "bit_serial_cycles": "macro_cycle",
             "weight_word_bit_products": "word_bit_product", "shift_accumulates": "shift_accumulate"}
    for name, count in counts.items():
        if name in {"tiles", "useful_ops", "executed_padded_ops", "logical_output_bytes"}:
            continue
        shared = name.startswith(("mem_", "activation_stream", "psum_stream", "write_stream"))
        logical = name.startswith(("logical_", "cross_"))
        actions[name] = {"count": count, "unit": "byte" if name.endswith("bytes") else
            "MAC" if name.endswith("mac") else "add" if name.endswith("adds") else units.get(name,"count"),
            "scope": "single_hem" if shared else "logical_gemm" if logical else "sum_of_macro_actions",
            "basis": "padded payload; not bank transactions" if shared else
                "logical requirement; aliases must not be charged twice" if logical else "action count; not switching energy"}
    occupied = sum(sum(task["m"] * spec.macro_spec.values["input_ii_cycles"] for task in wave["tasks"]) for wave in schedules)
    wave_compute = sum(max(task["m"] for task in wave["tasks"]) * spec.macro_spec.values["input_ii_cycles"] for wave in schedules)
    timing_report = timing.report(schedules, counts["useful_ops"]) if timing else {
        "status": "not_computed", "scope": "single_hem", "total_cycles": None, "chip_cycles": None,
        "time_ns": None, "throughput_gops": None}
    timing_report["compute_occupancy"] = {
        "value": wave_compute, "unit": "cycle", "scope": "single_hem_static_waves",
        "domain_cycles_sum": occupied, "status": "derived_schedule",
        "formula": "sum(max(rows in wave) * INT8 II); not sum of domain times",
        "conditions": "Compute occupancy only; excludes load, supply, drain and output constraints."}
    timing_report["missing_parameters"] = missing
    result = {
        "schema_version": 10, "model": "tsp_cim_hemisphere_reference", "mode": mode,
        "reference_id": spec.reference_id, "macro_reference_id": spec.macro_spec.reference_id,
        "native_reference_id": native.reference_id,
        "workload": {"operation": "gemm", "M": M, "K": K, "N": N, "data_format": "INT8",
                     "accumulation_dtype": "INT32", "output_dtype": "INT32"},
        "boundary": {"scope": "single_hem", "active_hemispheres": 1, "selected_hemisphere": hemisphere,
            "start": "packed_A_B_in_local_MEM", "end": "INT32_C_in_local_MEM",
            "simulated_cluster_domains": domains, "chip_execution_included": False,
            "mem_allocation_included": True, "collector_capacity_validated": False,
            "nim_conversion_included": False, "numeric_values_simulated": False,
            "host_transfer_included": False, "packing_included": False},
        "policy": {"loop_order": order, "selected_tile_mkn": [tm, spec.plane.domain_k, spec.cluster.n],
            "mem_slices": plan["mem_slices"], "streams_per_direction": plan["streams_per_direction"],
            "mapping": "domain=(K_tile*N_tiles+N_tile) modulo 8; independent K/N shards; no M replication",
            "wave_policy": "flush on repeated domain or new M group; no padding to eight tasks",
            "context_initial_state": "empty_per_simulate", "context_replacement": "round_robin_inactive",
            "row_pipeline": "compressed shared-resource pipeline; serialized wave boundaries",
            "lw_policy": "wordline outer/Core-pair inner; idle slots retained",
            "weight_load_overlap": load_overlap,
            "writer_mapping": "(plane, core) -> (plane, writer_SL=19-(5*cluster+core)); fixed slots",
            "tail_policy": "full K16/N32 Macro actions; valid INT32 output columns"},
        "memory": plan["memory"], "counts": dict(counts), "action_counts": actions,
        "schedule": {"wave_count": len(schedules), "waves": schedules,
            "domains": [{"hemisphere": hemisphere, "plane": d//4, "cluster": d%4,
                         "tasks": per_domain[d]["tiles"], "counts": dict(per_domain[d])} for d in range(domains)]},
        "timing": timing_report,
        "hardware_inventory": {"chip": spec.inventory(), "allocated_macros": domains*spec.plane.domain_macros,
            "allocated_cores": domains*spec.plane.domain_macros//spec.core.macros,
            "allocated_cluster_domains": domains, "used_macro_slots": sum(map(len, contexts)),
            "resident_macro_weight_bytes": domains*spec.plane.domain_macros*spec.macro_spec.resident_weight_bytes},
        "hardware_derived": {"macro": spec.macro_spec.derived(), **spec.derived()}, "topology": spec.hierarchy(),
        "parameter_provenance": spec.provenance(),
        "missing_parameters": spec.missing_parameters(),
        "energy": {"status": "not_evaluated", "total_pj": None, "tops_per_watt": None},
        "area": {"status": "not_evaluated", "total_mm2": None, "tops_per_mm2": None},
        "limitations": [
            "Static hemisphere waves, not asynchronous domain refill, chip execution or an ISA scheduler.",
            "Shared bandwidth reservations are fluid analytical service, not individual stream slots or bank conflicts.",
            "Independent MEM read/write services assume a conflict-free bank layout; all reads still share one read budget.",
            "Row pipeline and distributed INT32 writers are explicit assumptions, not measured hardware timing.",
            "K segments merge through MEM and one shared VXM service; no inter-Cluster PSUM link is invented.",
            "MEM resident capacity is checked; collector capacity and exact rolling per-Core IW timing are not validated.",
            "Known MEM plus Macro SRAM is a subtotal; energy, area and FP/BFP conversion are not evaluated."]}

    return attach_contract(result, {**execution, "default_tile_mkn": [tm,320,640],
                                    "weight_load_overlap": load_overlap})


class HemisphereTiming:
    def __init__(self, spec, plan, need, load_overlap):
        self.load_overlap = load_overlap
        self.spec = spec
        native, slices = spec.native_reference_spec, len(plan["mem_slices"])
        v = native.values
        self.calendar = ResourceCalendar({"mem_read": slices*v["mem_read_bpc"],
            "mem_write": slices*v["mem_write_bpc"],
            "stream_in": plan["streams_per_direction"]*native.stream_width_bytes,
            "stream_out": plan["streams_per_direction"]*native.stream_width_bytes})
        self.load_free = self.pipeline_free = 0
        self.parameters = {name: {**deepcopy(native.parameters[name]), "selection": "configured"} for name in sorted(need)}
        # §3.4.6 explicitly distinguishes distributed writer SLs from a Plane total.
        for name, value, unit, scope, conditions in (
            ("int8_input_ii", spec.macro_spec.values["input_ii_cycles"], "cycle", "cluster_domain", "INT8 ACC occupancy; §5.1."),
            ("local_pipeline_cycles", spec.execution["fixed_latency_cycles"]["local_pipeline"], "cycle", "core", "Analytical allowance inherited from source latency model, not physical calibration."),
            ("writer_bpc_per_superlane_plane", spec.values["int32_writer_bpc"], "byte/cycle", "writer_superlane_plane", "INT32 bypass uses §3.4.5-3.4.6 distributed writer target; 20 distinct writers/Plane, constrained by shared MEM/stream."),
            ("switch_cycles", spec.execution["fixed_latency_cycles"]["context_switch"], "cycle", "cluster_domain", "Conservative drained wave switch; exact rolling per-Core IW not simulated.")):
            self.parameters[name] = dict(value=value, unit=unit, scope=scope, status="assumed",
                source=SOURCE, source_version=COMMIT, conditions=conditions)
        self.parameters["int8_input_ii"]["status"] = "documented"

    def schedule(self, tasks, programmed, switches, release_after, previous):
        caps = self.calendar.capacities
        release = max((previous[index]["pipeline_end"] for index in release_after), default=0)
        load_start = max(self.load_free, release, self.pipeline_free if self.load_overlap == "serialized" else 0)
        load_end = load_start
        if programmed:
            # Each slot identifies a physical SL/Core/Macro; each active Macro
            # writes exactly 16 wordlines. Both planes share one input direction.
            beats = Counter()
            for domain, slot in programmed:
                local_core = (slot % self.spec.cluster.macros) // self.spec.core.macros
                beats[(domain % 4 * self.spec.cluster.cores + local_core)//2] += 32
            for _ in range(self.spec.macro_spec.context_bytes//32):
                for pair in range(10):
                    size = beats[pair]
                    load_end = self.calendar.transfer(size, load_end,
                        ("mem_read", "stream_in"), size, "LW") if size else load_end+1
            self.load_free = load_end
        else:
            load_start = load_end = 0
        start = max(self.pipeline_free, load_end)
        m = tasks[0]["m"]
        abc = sum(align_up(t["k"], 16) for t in tasks)
        read_psum = sum(4*t["n"] for t in tasks if t["ks"])
        write = sum(4*t["n"] for t in tasks)
        sin = sum(align_up(t["k"], 320) + (align_up(4*t["n"], 320) if t["ks"] else 0) for t in tasks)
        sout = sum(align_up(4*t["n"], 16) for t in tasks)
        adds = sum(t["n"] for t in tasks if t["ks"])
        # Reads contend with reads (including background LW); writes have their
        # own budget under the explicit conflict-free bank-layout assumption.
        demands = {"mem_read": abc+read_psum, "mem_write": write, "stream_in": sin, "stream_out": sout}
        ii = max(self.spec.macro_spec.values["input_ii_cycles"], *(ceil(Fraction(size)/caps[key]) for key, size in demands.items()))
        if adds:
            ii = max(ii, self.spec.native_reference_spec.vxm_service_cycles({"int32_add": adds}))
        # First row follows the full dependency path. Later rows overlap resources.
        capture = max(ceil(Fraction(abc)/caps["mem_read"]),
                      ceil(Fraction(sum(align_up(t["k"],320) for t in tasks))/caps["stream_in"]),
                      max(t["domain"] % 4 + 1 for t in tasks))
        drain = self.spec.execution["fixed_latency_cycles"]["local_pipeline"] + max(ceil(t["k"]/16) for t in tasks)
        stagger = max((ceil(t["n"]/128)-1)*self.spec.values["core_stagger_cycles"] for t in tasks)
        # Serialize first-row read/add/write across K dependencies, while allowing
        # independent output groups to use distinct transposed writer SuperLanes.
        groups = Counter()
        for t in tasks:
            local_write = max(ceil(min(t["n"],self.spec.core.n)*4/self.spec.values["int32_writer_bpc"]),
                ceil(Fraction(4*t["n"])/caps["mem_write"]),
                ceil(Fraction(align_up(4*t["n"],16))/caps["stream_out"]))
            local_read_add = 0
            if t["ks"]:
                v = self.spec.native_reference_spec.values
                local_read_add = max(ceil(Fraction(4*t["n"])/caps["mem_read"]),
                    ceil(Fraction(align_up(4*t["n"],320))/caps["stream_in"]))
                local_read_add += v["vxm_add_latency_cycles"] + self.spec.native_reference_spec.vxm_service_cycles({"int32_add": t["n"]})
            groups[t["ns"]] += local_write + local_read_add
        global_read = max(ceil(Fraction(read_psum)/caps["mem_read"]),
                          ceil(Fraction(sum(align_up(4*t["n"],320) for t in tasks if t["ks"]))/caps["stream_in"]))
        global_add = 0
        if adds:
            v = self.spec.native_reference_spec.values
            global_add = v["vxm_add_latency_cycles"] + self.spec.native_reference_spec.vxm_service_cycles({"int32_add": adds})
        global_write = max(ceil(Fraction(write)/caps["mem_write"]), ceil(Fraction(sout)/caps["stream_out"]))
        output = max(max(groups.values()), global_read+global_add+global_write)
        first = (bool(switches)*self.spec.execution["fixed_latency_cycles"]["context_switch"]
                 + capture + self.spec.macro_spec.values["input_ii_cycles"] + drain + stagger + output)
        duration = first + (m-1)*ii
        # Reserve steady per-row demand throughout fill/drain too. This is a
        # conservative reservation and is not an extra byte/energy action.
        self.calendar.reserve(start, start+duration,
            {key: Fraction(size,ii) for key,size in demands.items()}, "row_pipeline")
        self.pipeline_free = start+duration
        return {"load_start": load_start, "load_end": load_end, "pipeline_start": start,
            "pipeline_end": self.pipeline_free, "first_row_cycles": first, "row_interval_cycles": ii,
            "rows": m, "load_wait_for_context_until": release,
            "pipeline_cycles": duration, "row_payload_bytes": demands,
            "writer_slots": [{"plane": t["domain"]//4, "cluster": t["domain"]%4,
                "core": core, "writer_superlane": 19-(5*(t["domain"]%4)+core)}
                for t in tasks for core in range(ceil(t["n"]/128))]}

    def report(self, waves, useful_ops):
        total = self.pipeline_free
        pipeline = sum(w["pipeline_cycles"] for w in waves)
        clock = self.spec.clock_freq_ghz
        return attach_timing_breakdowns({"status": "analytical_estimate", "model": "single_hem_static_waves_v2", "scope": "single_hem",
            "weight_load_overlap": self.load_overlap, "total_cycles": total, "chip_cycles": None, "time_ns": total/clock if clock is not None else None,
            "throughput_gops": useful_ops*clock/total if clock is not None else None,
            "load_elapsed_cycles_sum": sum(w["load_end"]-w["load_start"] for w in waves),
            "parameters": self.parameters, "clock": deepcopy(self.spec.native_reference_spec.parameters["clock_ghz"]),
            "resources": self.calendar.summary(),
            "assumptions": [
                "Up to eight independent K/N tasks per wave; no fabricated eightfold acceleration for a short wave.",
                "Shared resident MEM allocation follows native TSP policy with CIM K16/N32 padding.",
                "Conflict-free bank layout is assumed: MEM reads and writes overlap on independent budgets; bank placement and capacity per bank are not checked.",
                "LW scans wordline/Core-pair order; inactive slots idle. Selected load overlap: " + self.load_overlap + "; background transfers use residual MEM/input-stream bandwidth.",
                "Compressed row pipeline uses max shared-resource demand; per-row fill/drain reservations conservatively retain that demand.",
                "Distinct transposed writer SuperLanes use the 256 B/cycle INT32 bypass target; not one serial writer per Plane.",
                "K segments accumulate via shared MEM/VXM; no direct inter-domain reduction or FP conversion.",
                "Wave-level drain and context release are conservative; exact Core rolling IW and collector occupancy are not simulated.",
                "Design-source analytical estimate, not calibrated hardware speedup or a guaranteed latency bound."]},
            elapsed={"load_wait": total-pipeline, "row_pipeline": pipeline})
