"""Single-core, serial stage formulas; no event, cache or resource scheduler."""
# 单个 AI Core 的固定 MNK 路径：GM→L1→L0A/B→Cube/L0C→UB→GM。
# stages 分别记录五条搬运路径，macs 单独记录 Cube 的计算工作。
import math
from fractions import Fraction

from ...hardware.ascend import BUFFERS, ROUTES
from .tiling import align_up, make_plan


def _timing_inputs(spec):
    """优先使用配置服务率；只在任务报告中为缺项声明理想代理。"""
    geometry = {name: spec.parameters[f"cube.{name}"].value for name in "mkn"}
    provenance = spec.provenance()
    parameters, proxies = {}, []

    def select(name, fallback, conditions, basis):
        original = provenance[name]
        if original["value"] is None:
            value = fallback()
            parameters[name] = dict(value=value, unit=original["unit"], scope="core", status="assumed",
                source="; ".join(dict.fromkeys(provenance[key]["source"] for key in basis)),
                source_version="Original Ascend 910; analytical proxy v1, 2026-11-16",
                conditions=conditions, basis_fields=list(basis), selection="assumed_proxy")
            proxies.append(name)
        else:
            parameters[name] = {**original, "selection": "configured"}
        return Fraction(str(parameters[name]["value"]))

    def hbm_share():
        if spec.chip["hbm_bytes_per_cycle"] is None:
            raise ValueError("Ascend timing proxy requires chip.hbm_bytes_per_cycle for missing GM routes")
        return spec.chip["hbm_bytes_per_cycle"] / spec.architectural_cores

    rate = select("cube.macs_per_cycle", lambda: math.prod(geometry.values()) * spec.cube_units,
        "Assume one INT8 16x32x16 fractal per Cube per cycle as ideal service, including accumulation. "
        "Geometry is documented; this cycles contract is not. No separate instruction/fill latency or calibrated utilization.",
        ("cube.m", "cube.k", "cube.n", "cube.units_per_core"))
    sizes = {"l1_l0a": geometry["m"] * geometry["k"], "l1_l0b": geometry["k"] * geometry["n"],
             "l0c_ub": 4 * geometry["m"] * geometry["n"]}
    dimensions = {"l1_l0a": ("cube.m", "cube.k"), "l1_l0b": ("cube.k", "cube.n"),
                  "l0c_ub": ("cube.m", "cube.n")}
    routes = {}
    for route in ROUTES:
        if route in ("gm_l1", "ub_gm"):
            routes[route] = select("transfers." + route, hbm_share,
                "Even chip HBM byte/reference-cycle share across architectural cores; no idle-core borrowing. "
                "Use as integrated GM route service, without another HBM/NoC charge. Fixed DMA latency omitted.",
                ("chip.hbm_bytes_per_cycle", "chip.architectural_cores"))
        else:
            routes[route] = select("transfers." + route, lambda: sizes[route],
                "Assume one route-specific INT8 input or INT32 output fractal per cycle. "
                "This payload proxy is not a published port bandwidth; no contention or fixed DMA latency.",
                dimensions[route])
    return rate, routes, parameters, proxies


class AscendSimulator:
    def __init__(self, spec):
        self.spec = spec

    def simulate(self, workload, mode="count", tile=(64, 128, 64), l1_k=256):
        if workload.data_format != "INT8" or workload.accumulation_dtype != "INT32" or workload.output_dtype != "INT32":
            raise ValueError("Ascend supports INT8 inputs / INT32 accumulation and output only; use --format INT8")
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        timed = mode == "timing"
        spec = self.spec
        if timed:
            rate, routes, parameters, proxies = _timing_inputs(spec)
        plan = make_plan(workload, spec.capacities, tile, l1_k)
        stages = {r: {"bytes": 0, "transfers": 0, "cycles": 0 if timed else None} for r in ROUTES}
        peak = dict.fromkeys(BUFFERS, 0)
        macs = compute_cycles = output_tiles = compute_blocks = 0

        def transfer(route, size):
            # 每次搬运分别向上取整；先合并 byte 再取整会少计小传输的服务周期。
            stages[route]["bytes"] += size
            stages[route]["transfers"] += 1
            if timed:
                stages[route]["cycles"] += math.ceil(size / routes[route])

        # Output remains in L0C for the complete K reduction; only final C is exported.
        for mi in range(0, workload.M, plan.m):
            m = align_up(min(plan.m, workload.M - mi), 16)
            for ni in range(0, workload.N, plan.n):
                n = align_up(min(plan.n, workload.N - ni), 16)
                output_tiles += 1
                peak["L0C"] = max(peak["L0C"], 4 * m * n)
                for ki in range(0, workload.K, plan.l1_k):
                    # L1 一次取较大的 K panel，再分成若干个送入 L0 的 K block。
                    # m/n/panel 已按硬件粒度补齐，下面累计的是补齐后的工作和搬运量。
                    panel = align_up(min(plan.l1_k, workload.K - ki), 32)
                    peak["L1"] = max(peak["L1"], (m + n) * panel)
                    transfer("gm_l1", m * panel)
                    transfer("gm_l1", panel * n)
                    for kk in range(0, panel, plan.k):
                        k = min(plan.k, panel - kk)
                        peak["L0A"] = max(peak["L0A"], m * k)
                        peak["L0B"] = max(peak["L0B"], k * n)
                        transfer("l1_l0a", m * k)
                        transfer("l1_l0b", k * n)
                        macs += m * k * n
                        compute_blocks += 1
                        if timed:
                            compute_cycles += math.ceil(m * k * n / rate)
                for offset in range(0, 4 * m * n, plan.ub_chunk_bytes):
                    # 已完成全部 K 累加，才分段经 UB 导出；不会每个 K 块都写回 C。
                    size = min(plan.ub_chunk_bytes, 4 * m * n - offset)
                    peak["UB"] = max(peak["UB"], size)
                    transfer("l0c_ub", size)
                    transfer("ub_gm", size)

        # 当前时间模型假设计算、搬运阶段串行，不在这里加入重叠或按核数缩短时间。
        cycles = compute_cycles + sum(stage["cycles"] for stage in stages.values()) if timed else None
        # GHz 的单位是 cycle/ns，因此 cycles / GHz 直接得到 ns。
        time_ns = cycles / spec.clock_ghz if timed and spec.clock_ghz is not None else None
        useful = 2 * workload.M * workload.K * workload.N
        actions = {r: {"count": stages[r]["bytes"], "unit": "byte", "scope": "core"} for r in ROUTES}
        actions["cube_mac"] = {"count": macs, "unit": "MAC", "scope": "core"}
        result = {
            "schema_version": 3, "model": "ascend910_analytical", "mode": mode,
            "workload": workload.to_dict(),
            "boundary": {"scope": "single_core", "simulated_cores": 1,
                         "start": "prepacked_padded_A_B_in_GM", "end": "packed_padded_INT32_C_in_GM",
                         "input_preparation_included": False, "output_padding_trim_included": False,
                         "numeric_values_simulated": False},
            "strategy": {"name": "output_stationary", "loop_order": "MNK", "selected": plan.to_dict(),
                         "cross_output_tile_local_reuse": False, "overlap": "none"},
            "counts": {"useful_ops": useful, "executed_padded_ops": 2 * macs, "cube_mac": macs,
                       "compute_blocks": compute_blocks, "int8_fractals": macs // (16 * 32 * 16),
                       "output_tiles": output_tiles, "logical_output_bytes": 4 * workload.M * workload.N},
            "stages": stages,
            "memory": {name: {"scope": "core", "capacity_bytes": spec.capacities[name], "peak_bytes": peak[name]}
                       for name in BUFFERS},
            "timing": {"status": "analytical_estimate" if timed else "not_computed",
                       "policy": "serial_stage_sum", "compute_cycles": compute_cycles if timed else None,
                       "total_cycles": cycles, "time_ns": time_ns,
                       "throughput_gops": useful / time_ns if time_ns else None,
                       "missing_parameters": spec.missing_timing()},
            "action_counts": actions,
            "energy": {"status": "not_evaluated", "total_pj": None, "reason": "Issue 2 supplies action inputs; coefficients and energy evaluation are separate work."},
            "hardware_derived": spec.derived(), "parameter_provenance": spec.provenance(),
            "output_path": spec.output_path,
            "limitations": [
                "Single-core coarse analytical estimate; chip peak is specification arithmetic only.",
                "No core mapping, pipeline overlap, events, cache behavior, bank/NoC/DRAM contention.",
                "GM routes are integrated effective services; requests are not measured HBM transactions.",
                "Cube throughput includes MAC accumulation; there is no extra fixed Accumulate cycle.",
                "INT32 export path and assumed capacities require target validation.",
                "No host launch, layout conversion, full-model execution, energy or area prediction.",
            ],
        }
        if timed:
            result["timing"].update(model="single_core_serial_proxy_v1", chip_cycles=None,
                stages_cycles={"compute": compute_cycles, **{r: s["cycles"] for r, s in stages.items()}},
                parameters=parameters, proxy_parameters=proxies,
                assumptions=[
                    "Configured services take priority; absent rates use only the explicitly listed proxies.",
                    "All stages and blocks are serial; no chip execution, cache reuse, overlap or contention.",
                    "Proxy service omits fixed startup/fill costs and is not a calibrated physical upper/lower bound.",
                    "Hardware provenance, missing parameters and specification peaks are unchanged; proxy rates do not fill hardware unknowns."])
        return result
