"""Coarse native TSP analytical model. No event scheduler or chip speedup."""
# 阅读主线：分块与驻留检查 → plane 权重复用 → MEM/stream 动作 → 可选行服务时间。
# 执行范围固定为一个 hemisphere，规格中的整片资源只用于库存和峰值推导。
import copy
import itertools
import math
from collections import Counter
from .tiling import TSPTilingEngine, align_up

LOOP_ORDERS = {"KNM": "weight reuse across M", "KMN": "K then M then N",
               "MKN": "M then K then N", "NKM": "N then K then M"}


def _timing_inputs(spec, plan, N, need):
    """在本次分配上解析服务参数，不回填原生硬件配置或改变原生对比计数。"""
    v = dict(spec.values)
    parameters, proxies = {}, []
    source = ("https://github.com/pku-pacific-lab-team/LPU-HUAWEI-2012/blob/"
              "700b6e6509b1742878091382df0d0df3e92732e1/docs/public/tsp_architecture_doc_20260805.md")
    # 已知 MEM 端口是代理的基础，不凭空创造缺失的物理供给。
    missing = [key for key in ("mem_read_bpc", "mem_write_bpc", "mem_shared_bpc") if v[key] is None]
    if missing:
        raise ValueError("Missing timing base parameters: " + ", ".join(missing))

    def select(name, fallback, conditions):
        original = spec.parameters[name]
        if v[name] is None:
            v[name] = fallback()
            parameters[name] = dict(value=v[name], unit=original["unit"], scope=original["scope"],
                status="assumed", source=source, source_version="TSP allocated-service proxy v1, 2026-11-16",
                conditions=conditions, selection="assumed_proxy")
            proxies.append(name)
        else:
            parameters[name] = {**copy.deepcopy(original), "selection": "configured"}

    width = spec.stream_width_bytes
    streams = plan["streams_per_direction"] * width
    slices = len(plan["mem_slices"])
    select("install_bpc", lambda: streams,
        "Allocated stream payload ceiling; MEM read/shared limits still apply. No extra installation-port penalty. "
        "Design sections 6.5-6.6 describe the load wave; the mapping to this aggregate service is assumed.")
    install_bw = min(slices * v["mem_read_bpc"], slices * v["mem_shared_bpc"], streams, v["install_bpc"])
    group = width * max(1, math.floor(install_bw / width))
    select("install_first_cycles", lambda: math.ceil(group / install_bw) + spec.n_superlanes - 1,
        "First payload group service plus full SuperLane command-wave propagation (SL-1). "
        "Reserve full chain for tails; do not add another load-wave or SRAM startup charge.")
    select("mxm_row_ii", lambda: 1,
        "Native resident-weight ACC II=1 is documented; mapping it to one allocated hemisphere row is assumed. "
        "MEM/stream/add constraints still limit actual row II, so this is not guaranteed task throughput.")
    select("mxm_first_cycles", lambda: spec.n_superlanes,
        "One local ACC step per SuperLane through the full chain, including tails; no extra internal pipeline. "
        "Do not also charge the command-wave stagger as a separate compute latency.")
    select("row_io_latency_cycles", lambda: 0,
        "Omit additional fixed MEM/SR route latency; explicit payload services remain. "
        "No claim that physical routing or SRAM startup is instantaneous.")
    if "vxm_add_ops_per_cycle" in need:
        select("vxm_add_ops_per_cycle", lambda: min(spec.total_lanes, streams / 4),
            "One dedicated INT32 add per lane per cycle, capped by allocated stream bytes/4. "
            "A single-operation service proxy, not use of all 5120 ALUs or an ISA schedule.")
        select("vxm_add_first_cycles", lambda: max(1, math.ceil(
            min(plan["n"], align_up(N, spec.supercell_dim)) / v["vxm_add_ops_per_cycle"])),
            "Reserve one complete selected output-row add service, even for smaller tail rows; "
            "unknown additional VXM pipeline latency is omitted.")
    for name in sorted(need - parameters.keys()):
        parameters[name] = {**copy.deepcopy(spec.parameters[name]), "selection": "configured"}
    return v, parameters, proxies


def _generate_tile_iter(order, M, K, N, tm, tk, tn):
    # order 只改变遍历先后；返回值始终为 (ms, ks, ns, m, k, n)，后三项是有效尺寸。
    if order not in LOOP_ORDERS:
        raise ValueError(f"Unsupported loop order: {order}")
    groups = {key: [(i, min(size, length-i)) for i in range(0, length, size)]
              for key, length, size in zip("MKN", (M,K,N), (tm,tk,tn))}
    for group in itertools.product(*(groups[key] for key in order)):
        indices = dict(zip(order, group))
        yield (*(indices[key][0] for key in "MKN"), *(indices[key][1] for key in "MKN"))


class TSPSimulator:
    def __init__(self, tsp_spec, *, data_format="INT8"):
        if data_format != "INT8":
            raise ValueError("Reference TSP supports INT8 only; use --formats INT8")
        self.tsp_spec = tsp_spec

    def simulate(self, M, K, N, loop_order="KNM", *, mode="count", scope="single_hem",
                 tile=(320,320,640), mem_slices=None, streams=None, reserved_bytes=0):
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        if scope != "single_hem":
            raise ValueError("unsupported_scope: chip timing/mapping is not modeled; use single_hem")
        if loop_order not in LOOP_ORDERS:
            raise ValueError(f"Unsupported loop order: {loop_order}")
        spec = self.tsp_spec
        v = spec.values
        plan = TSPTilingEngine(spec).derive_tile_sizes(M,K,N, tile=tile, mem_slices=mem_slices,
                                                     streams=streams, reserved_bytes=reserved_bytes)
        need = {"mem_read_bpc", "mem_write_bpc", "mem_shared_bpc", "install_bpc", "install_first_cycles",
                "mxm_first_cycles", "mxm_row_ii", "row_io_latency_cycles"}
        # 只有 K 被分成多段时才需要 VXM 合并部分和，因此按任务决定必需的时间参数。
        if K > plan["k"]:
            need.update(("vxm_add_ops_per_cycle", "vxm_add_first_cycles"))
        missing = sorted(key for key in need if v[key] is None)
        timed = mode == "timing"
        if timed:
            v, parameters, proxies = _timing_inputs(spec, plan, N, need)
        slices, width = len(plan["mem_slices"]), spec.stream_width_bytes
        stream_bw = plan["streams_per_direction"] * width
        if timed:
            read_bw = slices * v["mem_read_bpc"]
            write_bw = slices * v["mem_write_bpc"]
            shared_bw = slices * v["mem_shared_bpc"]
            install_bw = min(read_bw, shared_bw, stream_bw, v["install_bpc"])
        counts = Counter({k: 0 for k in ("mxm_mac", "weight_install_count", "weight_plane_installs",
            "mem_read_activation_bytes", "mem_read_weight_bytes", "weight_install_bytes", "weight_stream_bytes",
            "mem_read_psum_bytes", "mem_write_psum_bytes", "mem_write_final_bytes", "vxm_add_ops",
            "activation_stream_bytes", "result_stream_bytes", "psum_stream_bytes", "write_stream_bytes", "tiles")})
        # plane_tags 跟踪各计算平面当前保留的 B；psum_progress 跟踪各输出块已完成的 K。
        plane_tags = {}
        psum_progress = {}
        # Totals are analytical service times, not independent events or additive busy-time claims.
        install_cycles = row_cycles = 0
        first_result = None
        max_row_ii = 0
        allocated_peak_mac = spec.n_mxm_planes_per_hemisphere * spec.mxm_plane_dim**2
        for ms,ks,ns,m,k,n in _generate_tile_iter(loop_order,M,K,N,plan["m"],plan["k"],plan["n"]):
            kp, np = align_up(k,spec.supercell_dim), align_up(n,spec.supercell_dim)
            # (ms, ns) 标识同一个 C 块；前一段结束位置必须等于本段 ks，避免漏算或重复。
            tag = (ms,ns)
            previous = psum_progress.get(tag, 0)
            if previous != ks:
                raise ValueError("K reduction order is inconsistent")
            later, final = previous > 0, ks + k == K
            psum_progress[tag] = ks+k
            counts["tiles"] += 1
            counts["mxm_mac"] += m*kp*np
            changed_bytes = 0
            for plane in range(math.ceil(np / spec.mxm_plane_dim)):
                cols = min(spec.mxm_plane_dim, np-plane*spec.mxm_plane_dim)
                # 两个 plane 沿 N 拼接。权重 tag 不包含 M，换输入行时可继续使用同一 B。
                weight_tag = (ks, ns+plane*spec.mxm_plane_dim, kp, cols)
                if plane_tags.get(plane) != weight_tag:
                    changed_bytes += kp*cols
                    counts["weight_plane_installs"] += 1
                    plane_tags[plane] = weight_tag
            # 一次块级安装可能更新一个或两个 plane，所以事件数与 plane 安装数分开记。
            counts["weight_install_count"] += int(changed_bytes > 0)
            counts["mem_read_weight_bytes"] += changed_bytes
            counts["weight_install_bytes"] += changed_bytes
            weight_groups = math.ceil(changed_bytes/width)
            counts["weight_stream_bytes"] += weight_groups*width
            counts["mem_read_activation_bytes"] += m*kp
            # 第一段 K 没有旧部分和可读；中间结果与最终 C 使用同一驻留空间。
            counts["mem_read_psum_bytes"] += 4*m*np if later else 0
            counts["mem_write_final_bytes" if final else "mem_write_psum_bytes"] += 4*m*np
            counts["vxm_add_ops"] += m*np if later else 0
            # MEM 记录补齐后的 payload；stream 按整组向量传输，尾组也占完整宽度。
            act_stream = align_up(kp,width)
            out_stream = align_up(4*np,width)
            counts["activation_stream_bytes"] += m*act_stream
            counts["result_stream_bytes"] += m*out_stream
            counts["psum_stream_bytes"] += m*out_stream if later else 0
            counts["write_stream_bytes"] += m*out_stream
            if timed:
                before = install_cycles+row_cycles
                inst = 0
                if changed_bytes:
                    # Several 320-byte streams may install in the same cycle.
                    # Rounding every vector to one cycle would discard that parallelism.
                    # 多条 stream 可在同一周期安装权重，先形成并行组，再计算发出组数。
                    install_group = width * max(1, math.floor(install_bw / width))
                    install_issues = math.ceil(changed_bytes / install_group)
                    ii_install = max(1,math.ceil(install_group/install_bw))
                    first_install = v["install_first_cycles"]
                    if first_install < ii_install:
                        raise ValueError("install_first_cycles cannot be shorter than first-group service")
                    inst = first_install+(install_issues-1)*ii_install
                ca = max(math.ceil(kp/read_bw), math.ceil(kp/shared_bw), math.ceil(act_stream/stream_bw))
                co = math.ceil(out_stream/stream_bw)
                cr = max(math.ceil(4*np/read_bw), math.ceil(4*np/shared_bw), co) if later else 0
                cw = max(math.ceil(4*np/write_bw), math.ceil(4*np/shared_bw), co)
                ii_add = max(1,math.ceil(np/v["vxm_add_ops_per_cycle"])) if later else 0
                add_first = v["vxm_add_first_cycles"] if later else 0
                if later and add_first < ii_add:
                    raise ValueError("vxm_add_first_cycles cannot be shorter than a row add service")
                # Sum actions that share a resource BEFORE applying the pipeline maximum.
                # 共用 MEM 读写端口的需求要先相加，不能把共享端口误当成独立读、写端口。
                mem_read_row = kp+(4*np if later else 0)
                ii_mem = max(math.ceil(mem_read_row/read_bw), math.ceil(4*np/write_bw),
                             math.ceil((mem_read_row+4*np)/shared_bw))
                ii_stream = max(math.ceil((act_stream+out_stream)/stream_bw),
                                math.ceil((out_stream+(out_stream if later else 0))/stream_bw))
                ii_mxm = max(v["mxm_row_ii"], math.ceil(kp*np/allocated_peak_mac))
                if v["mxm_first_cycles"] < math.ceil(kp*np/allocated_peak_mac):
                    raise ValueError("mxm_first_cycles violates the allocated MAC capacity")
                # 不同资源可流水工作，稳态行间隔由最慢资源决定。
                ii = max(ii_mem,ii_stream,ii_mxm,ii_add)
                # 首行经历完整依赖链；后续 m-1 行各增加一个 II。不同 tile 仍串行相加。
                first = v["row_io_latency_cycles"]+ca+v["mxm_first_cycles"]+co+cr+add_first+cw
                if first_result is None:
                    first_result = before+inst+first
                install_cycles += inst
                row_cycles += first+(m-1)*ii
                max_row_ii = max(max_row_ii,ii)
        counts.update(useful_ops=2*M*K*N, executed_padded_ops=2*counts["mxm_mac"],
                      logical_output_bytes=4*M*N, logical_cross_k_adds=(math.ceil(K/plan["k"])-1)*M*N)
        cycles = install_cycles+row_cycles if timed else None
        time_ns = cycles/spec.clock_freq_ghz if timed and spec.clock_freq_ghz is not None else None
        actions = {key: {"count": count, "unit": "MAC" if key=="mxm_mac" else "op" if key=="vxm_add_ops" else "byte",
                         "scope": "single_hem"}
                   for key,count in counts.items()
                   if (key.endswith("_bytes") and key!="logical_output_bytes") or key in {"mxm_mac","vxm_add_ops"}}
        result = {"schema_version": 2, "model": "tsp_analytical_reference", "reference_id": spec.reference_id,
            "profile_id": spec.profile_id, "mode": mode,
            "workload": {"operation":"gemm","M":M,"K":K,"N":N,"data_format":"INT8","accumulation_dtype":"INT32","output_dtype":"INT32"},
            "boundary": {"scope":"single_hem","active_hemispheres":1,"start":"packed_A_B_in_local_MEM",
                         "end":"INT32_C_in_local_MEM","numeric_values_simulated":False,
                         "host_transfer_included":False,"packing_included":False},
            "policy": {"loop_order":loop_order,"selected_tile_mkn":[plan[x] for x in "mkn"],
                       "mem_slices":plan["mem_slices"],"streams_per_direction":plan["streams_per_direction"],
                       "tail_policy":plan["tail_policy"],"inter_tile_overlap":"none",
                       "row_pipeline":"compressed ideal steady stream; shared resource demands added before max",
                       "vxm_budget":"explicit single-task INT32 add service; no ALU replication"},
            "memory":plan["memory"], "counts":dict(counts),"action_counts":actions,
            "timing":{"status":"analytical_estimate" if timed else "not_computed","scope":"single_hem",
                      "total_cycles":cycles,"chip_cycles":None,"time_ns":time_ns,
                      "throughput_gops":counts["useful_ops"]/time_ns if time_ns else None,
                      "install_cycles":install_cycles if timed else None,"row_service_cycles":row_cycles if timed else None,
                      "first_row_write_cycle":first_result,"max_row_interval_cycles":max_row_ii if timed else None,
                      "missing_parameters":missing},
            "energy":{"status":"not_evaluated","total_pj":None,"tops_per_watt":None},
            "area":{"status":"not_evaluated","total_mm2":None,"tops_per_mm2":None},
            "hardware_inventory":spec.area_inventory(),"hardware_derived":spec.derived(),
            "parameter_provenance":copy.deepcopy(spec.parameters),
            "validation":{"hardware_validated":False,"source_note":"See parameter_provenance for source verification; documented specifications do not establish hardware calibration."},
            "limitations":["Single-side resident analytical model; chip inventory is not chip execution.",
                "Uniform packed striping and tail masking are software assumptions, not a verified ISA layout.",
                "MEM actions count padded payload bytes, not physical bank transactions; stream bytes are rounded to vector groups.",
                "No per-bank conflicts, stream time slots, remote MEM, supercell events or inter-tile overlap.",
                "First latencies are end-to-end service inputs; no automatic +19 or additional internal reduction hops.",
                "Padded MAC count is issued logical work; inactive-cell switching and power gating are not modeled."]}
        if timed:
            result["timing"].update(model="single_hem_row_proxy_v1", parameters=parameters,
                proxy_parameters=proxies, stages_cycles={"install": install_cycles, "row_service": row_cycles},
                assumptions=[
                    "Configured values take priority; only absent service/latency fields use the listed allocation-dependent proxies.",
                    "Resident-weight compute II is not total task II: shared MEM, directional stream and VXM services remain constrained.",
                    "Full SuperLane startup is reserved for tails; unknown additional SRAM/SR/VXM fixed delays are omitted.",
                    "Original compressed row pipeline, weight tags and serial tiles are retained; no background load overlap or physical event scheduler.",
                    "This is an uncalibrated estimate, not a guaranteed physical upper/lower bound. Hardware unknowns and peaks remain unchanged."])
            result["policy"]["vxm_budget"] = "configured or explicit single-task INT32 proxy; no ALU replication"
            result["limitations"][4] = "Resolved first latencies include the disclosed proxy allowances; no additional hidden reduction or routing hops."
        return result
