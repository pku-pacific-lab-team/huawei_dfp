"""Single-core GEMM counts and resource-constrained tile scheduling."""
from ..metrics import attach_contract, attach_timing_breakdowns
# 单核 MNK：冷 L2 的块级复用、核内双缓冲和资源约束流水。
import math
from fractions import Fraction

from ...hardware.ascend import BUFFERS, ROUTES
from .tiling import align_up, buffer_slots, make_plan
from .schedule import Schedule
from .cache import PanelCache


def _timing_inputs(spec):
    """所有服务值从 JSON 读取；假设身份随参数保留，缺项不再隐式补齐。"""
    spec.require_timing()
    names = spec.timing_parameter_names()
    provenance = spec.provenance()
    parameters = {name: {**provenance[name], "selection": "configured"} for name in names}
    proxies = [name for name in names if parameters[name]["status"] == "assumed"]
    rate = Fraction(str(parameters["cube.macs_per_cycle"]["value"]))
    routes = {route: Fraction(str(value)) for route, value in spec.route_services().items()}
    return rate, routes, parameters, proxies


class AscendSimulator:
    def __init__(self, spec):
        self.spec = spec

    def simulate(self, workload, mode="count", tile=None, l1_k=None,
                 *, buffers=None, trace=False):
        if workload.data_format != "INT8" or workload.accumulation_dtype != "INT32" or workload.output_dtype != "INT32":
            raise ValueError("Ascend supports INT8 inputs / INT32 accumulation and output only; use --format INT8")
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        execution = self.spec.execution
        tile = tuple(execution["default_tile_mkn"]) if tile is None else tile
        l1_k = execution["rules"]["default_l1_k"] if l1_k is None else l1_k
        buffers = execution["buffer_slots"] if buffers is None else buffers
        if type(buffers) is not int or buffers not in (1, 2):
            raise ValueError("buffers must be 1 or 2")
        if trace and mode != "timing":
            raise ValueError("trace requires timing mode")
        timed = mode == "timing"
        spec = self.spec
        if timed:
            rate, routes, parameters, proxies = _timing_inputs(spec)
        spec.require_count_capacities()
        plan = make_plan(workload, spec.capacities, spec.chip["hbm_usable_bytes"], tile, l1_k)
        cache = PanelCache(spec.l2_capacity())
        input_tail = None
        allocation = buffer_slots(plan, spec.capacities, buffers)
        released = {name: [None] * entry["slots"] for name, entry in allocation.items()}
        schedule = Schedule(execution["rules"]["route_resources"]) if timed else None
        stages = {r: {"bytes": 0, "transfers": 0, "cycles": 0 if timed else None} for r in ROUTES}
        peak = dict.fromkeys(BUFFERS, 0)
        macs = compute_cycles = output_tiles = compute_blocks = 0
        panels = chunks = 0

        def transfer(route, size, dependencies=(), **coordinates):
            # 每次搬运分别向上取整；先合并 byte 再取整会少计小传输的服务周期。
            stages[route]["bytes"] += size
            stages[route]["transfers"] += 1
            if timed:
                service = math.ceil(size / routes[route]) + execution["fixed_latency_cycles"]["transfer_startup"]
                stages[route]["cycles"] += service
                return schedule.add(route, service, dependencies, bytes=size, **coordinates)

        def load_input(key, size, dependency, **coordinates):
            nonlocal input_tail
            state = cache.access(key, size)
            coords = {**coordinates, "cache_status": state, "cache_key": list(key)}
            # Program-order completion makes logical LRU decisions match the
            # timed cache state, even if custom queue names are configured.
            dependencies = [dependency, input_tail]
            if state == "bypass":
                loaded = transfer("hbm_to_l1", size, dependencies, **coords)
            else:
                if state == "miss":
                    fetch = transfer("hbm_to_l2", size, dependencies, **coords)
                    fill = transfer("l2_fill", size, [fetch], **coords)
                    dependencies = [fill]
                loaded = transfer("l2_to_l1", size, dependencies, **coords)
            input_tail = loaded
            return loaded

        # Output remains in L0C for the complete K reduction; only final C is exported.
        for mi in range(0, workload.M, plan.m):
            m = align_up(min(plan.m, workload.M - mi), 16)
            for ni in range(0, workload.N, plan.n):
                n = align_up(min(plan.n, workload.N - ni), 16)
                output_tiles += 1
                tile_id = output_tiles - 1
                c_slot = tile_id % allocation["L0C"]["slots"]
                first_compute = last_compute = None
                peak["L0C"] = max(peak["L0C"], 4 * m * n)
                for ki in range(0, workload.K, plan.l1_k):
                    # L1 一次取较大的 K panel，再分成若干个送入 L0 的 K block。
                    # m/n/panel 已按硬件粒度补齐，下面累计的是补齐后的工作和搬运量。
                    panel = align_up(min(plan.l1_k, workload.K - ki), 32)
                    peak["L1"] = max(peak["L1"], (m + n) * panel)
                    l1_slot = panels % allocation["L1"]["slots"]
                    coords = {"tile": tile_id, "panel": panels}
                    load_a = load_input(("A", mi, ki, m, panel), m * panel, released["L1"][l1_slot], operand="A", **coords)
                    load_b = load_input(("B", ki, ni, panel, n), panel * n, released["L1"][l1_slot], operand="B", **coords)
                    for kk in range(0, panel, plan.k):
                        k = min(plan.k, panel - kk)
                        peak["L0A"] = max(peak["L0A"], m * k)
                        peak["L0B"] = max(peak["L0B"], k * n)
                        a_slot = compute_blocks % allocation["L0A"]["slots"]
                        b_slot = compute_blocks % allocation["L0B"]["slots"]
                        a = transfer("l1_l0a", m * k, [load_a, released["L0A"][a_slot]], block=compute_blocks, **coords)
                        b = transfer("l1_l0b", k * n, [load_b, released["L0B"][b_slot]], block=compute_blocks, **coords)
                        macs += m * k * n
                        if timed:
                            service = math.ceil(m * k * n / rate) + execution["fixed_latency_cycles"]["compute_startup"]
                            compute_cycles += service
                            last_compute = schedule.add("compute", service,
                                                        [a, b, last_compute, released["L0C"][c_slot]],
                                                        block=compute_blocks, macs=m * k * n, **coords)
                            if first_compute is None:
                                first_compute = last_compute
                            schedule.hold("L0A", a_slot, m * k, a, last_compute)
                            schedule.hold("L0B", b_slot, k * n, b, last_compute)
                        released["L0A"][a_slot] = released["L0B"][b_slot] = last_compute
                        compute_blocks += 1
                    # MTE1 FIFO makes the last B copy the last reader of this panel.
                    released["L1"][l1_slot] = b
                    if timed:
                        schedule.hold("L1", l1_slot, (m + n) * panel, load_a, b)
                    panels += 1
                for offset in range(0, 4 * m * n, plan.ub_chunk_bytes):
                    # 已完成全部 K 累加，才分段经 UB 导出；不会每个 K 块都写回 C。
                    size = min(plan.ub_chunk_bytes, 4 * m * n - offset)
                    peak["UB"] = max(peak["UB"], size)
                    ub_slot = chunks % allocation["UB"]["slots"]
                    export = transfer("l0c_ub", size, [last_compute, released["UB"][ub_slot]], tile=tile_id, chunk=chunks)
                    store = transfer("ub_to_hbm", size, [export], tile=tile_id, chunk=chunks)
                    released["UB"][ub_slot] = store
                    if timed:
                        schedule.hold("UB", ub_slot, size, export, store)
                    chunks += 1
                released["L0C"][c_slot] = export
                if timed:
                    schedule.hold("L0C", c_slot, 4 * m * n, first_compute, export)

        service_cycles = compute_cycles + sum(stage["cycles"] for stage in stages.values()) if timed else None
        scheduled = schedule.run(trace) if timed else None
        cycles = scheduled["total_cycles"] if timed else None
        # GHz 的单位是 cycle/ns，因此 cycles / GHz 直接得到 ns。
        time_ns = cycles / spec.clock_ghz if timed and spec.clock_ghz is not None else None
        useful = 2 * workload.M * workload.K * workload.N
        actions = {r: {"count": stages[r]["bytes"], "unit": "byte", "scope": "core"} for r in ROUTES}
        actions["hbm_read_bytes"] = {"count": actions.pop("hbm_to_l2")["count"] + actions.pop("hbm_to_l1")["count"],
                                     "unit": "byte", "scope": "core"}
        actions["l2_fill_bytes"] = actions.pop("l2_fill")
        actions["l2_read_bytes"] = actions.pop("l2_to_l1")
        actions["hbm_write_bytes"] = actions.pop("ub_to_hbm")
        for action in actions.values():
            action["basis"] = "padded_payload; not physical HBM transactions"
        actions["cube_mac"] = {"count": macs, "unit": "MAC", "scope": "core", "basis": "padded_arithmetic"}
        result = {
            "schema_version": 12, "model": "ascend910_analytical", "mode": mode,
            "workload": workload.to_dict(),
            "boundary": {"scope": "single_core", "simulated_cores": 1,
                         "start": "prepacked_padded_A_B_in_HBM", "end": "packed_padded_INT32_C_in_HBM",
                         "input_preparation_included": False, "output_padding_trim_included": False,
                         "numeric_values_simulated": False},
            "strategy": {"name": "output_stationary", "loop_order": "MNK", "selected": plan.to_dict(),
                         "cross_output_tile_local_reuse": False,
                         "cross_output_tile_l2_reuse": True,
                         "overlap": "resource_constrained",
                         "buffering": {"requested_slots": buffers, "effective": allocation,
                                       "policy": "keep tile; two slots only where usable capacity permits"}},
            "counts": {"useful_ops": useful, "executed_padded_ops": 2 * macs, "cube_mac": macs,
                       "compute_blocks": compute_blocks, "int8_fractals": macs // (16 * 32 * 16),
                       "output_tiles": output_tiles, "logical_output_bytes": 4 * workload.M * workload.N},
            "stages": stages,
            "cache": {"model": "operand_panel_lru", "status": "assumed", **cache.summary(),
                      "accounting": "miss_bytes = HBM reads including oversize bypass; read_bytes = all cached input requests; fill_bytes = allocated misses. Rates are byte/request weighted, not cache-line hit rates."},
            "memory": {
                **{name: {"scope": "core", "capacity_bytes": spec.capacities[name],
                          "peak_bytes": allocation[name]["allocated_bytes"], "peak_payload_bytes": peak[name],
                          "allocation_basis": "fixed slots reserved for the entire task", **allocation[name]}
                   for name in BUFFERS},
                "l2": {"scope": "core_share_of_chip", "chip_capacity_bytes": spec.chip["l2_bytes"],
                       "capacity_bytes": cache.capacity, "peak_bytes": cache.peak,
                       "resident_bytes_at_end": cache.used,
                       "allocation_basis": "floor(chip L2 / architectural cores); all capacity usable; no idle-core borrowing or cross-core sharing",
                       "unit_of_replacement": "complete padded L1 operand panel; operand, origin and shape identify disjoint input regions"},
                "hbm": {"scope": "chip", "physical_bytes": spec.chip["hbm_physical_bytes"],
                        "usable_bytes": spec.chip["hbm_usable_bytes"],
                        "logical_footprint_bytes": plan.hbm_logical_footprint_bytes,
                        "padded_footprint_bytes": plan.hbm_padded_footprint_bytes,
                        "allocation_basis": "prepacked padded A/B/C coexist; no host offload or extra full PSUM"}},
            "timing": {"status": "analytical_estimate" if timed else "not_computed",
                       "policy": "resource_pipeline",
                       "compute_cycles": compute_cycles if timed else None,
                       "total_cycles": cycles, "time_ns": time_ns,
                       "throughput_gops": useful / time_ns if time_ns else None,
                       "missing_parameters": spec.missing_timing()},
            "action_counts": actions,
            "energy": {"status": "not_evaluated", "total_pj": None, "reason": "Issue 2 supplies action inputs; coefficients and energy evaluation are separate work."},
            "hardware_derived": spec.derived(), "parameter_provenance": spec.provenance(),
            "output_path": spec.output_path,
            "limitations": [
                "Single-core coarse analytical estimate; chip peak is specification arithmetic only.",
                "Panel-level LRU and analytical FIFO scheduling; no cache lines, set associativity, bank/NoC/DRAM contention or cross-core scheduling.",
                "Cold, equally partitioned read-only input cache; outputs bypass. Oversize panels bypass without allocation. These policies are assumptions, not original-910 hardware behavior.",
                "HBM/L2 actions count padded payload, not physical transactions. Store-and-forward fetch/fill/read and omitted local DMA port limits require calibration.",
                "Cube throughput includes MAC accumulation; there is no extra fixed Accumulate cycle.",
                "INT32 export path and assumed capacities require target validation.",
                "No host launch, layout conversion, full-model execution, energy or area prediction.",
            ],
        }
        if timed:
            result["timing"].update(model="single_core_l2_pipeline_v1", chip_cycles=None,
                overlapped_service_cycles=service_cycles - cycles,
                schedule=scheduled,
                parameters=parameters, proxy_parameters=proxies,
                derived_services={
                    "hbm": {"scope": "single_core", "status": "assumed",
                            "share_bytes_per_second": float(spec.hbm_service()[0]),
                            "service_bytes_per_cycle": float(spec.hbm_service()[1]),
                            "routes": ["hbm_to_l2", "hbm_to_l1", "ub_to_hbm"],
                            "policy": "even chip-bandwidth share across architectural cores; no idle-core borrowing",
                            "input_parameters": ["chip.hbm_bandwidth_bytes_per_second", "chip.architectural_cores", "clock_freq_ghz"]},
                    "l2": {"scope": "single_core", "status": "assumed",
                           "service_bytes_per_cycle": float(spec.l2_service()),
                           "capacity_bytes": cache.capacity, "routes": ["l2_fill", "l2_to_l1"],
                           "policy": "equal chip shares; L2 fill/read share one service; no idle-core borrowing",
                           "input_parameters": ["chip.l2_bytes", "chip.l2_bytes_per_cycle", "chip.architectural_cores"]}},
                scheduling_assumptions={
                    "status": "assumed", "route_resources": {r: list(v) for r, v in execution["rules"]["route_resources"].items()},
                    "source": "https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0015.html",
                    "conditions": "CANN motivates independent queues. Exact original-910 arbitration and L0C export mapping are uncalibrated. l0c_export is a separate proxy queue; services are non-preemptive and reserve resources for their full duration."},
                assumptions=[
                    "Configured Cube/local/L2 services retain provenance. HBM rate is derived from bandwidth, core count and clock; L2 capacity and byte/cycle rate are equally partitioned. Required unknown inputs fail.",
                    "HBM fetch/output/bypass share one allocated service; L2 fill/read share another. Cold panel misses fetch, fill, then read serially. Input requests complete in program order; compute/output may overlap. Clean input eviction has no writeback; C bypasses L2.",
                    "MTE1 serializes A/B copies. Fixed buffer slots are released after their last reader; L0C holds the entire K reduction.",
                    "Zero configured startup allowances omit uncalibrated fixed latency; each modeled transfer charges its configured allowance. The model is not a calibrated physical upper/lower bound.",
                    "Assumed service values remain assumed in hardware provenance and derived specification peaks; missing required services fail."])
            attach_timing_breakdowns(result["timing"],
                service={"compute": compute_cycles, **{r: s["cycles"] for r, s in stages.items()}})
        effective = {**execution, "default_tile_mkn": list(tile), "buffer_slots": buffers,
                     "rules": {**execution["rules"], "default_l1_k": l1_k}}
        return attach_contract(result, effective)
