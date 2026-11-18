"""Native TSP row pipeline with one buffered weight-prefetch block."""
from ..metrics import attach_contract, attach_timing_breakdowns
# 阅读主线：分块与驻留检查 → plane 权重复用 → MEM/stream 动作 → 可选行服务时间。
# 执行范围固定为一个 hemisphere，规格中的整片资源只用于库存和峰值推导。
import copy
import itertools
import math
from collections import Counter
from .tiling import TSPTilingEngine, align_up
from .pipeline import schedule_prefetch
from ...hardware.tsp import VXM_ADD_PARAMETERS

LOOP_ORDERS = {"KNM": "weight reuse across M", "KMN": "K then M then N",
               "MKN": "M then K then N", "NKM": "N then K then M"}


def _timing_inputs(spec, need):
    """读取固定配置值；资源分配和任务耗时在执行公式中计算。"""
    missing = sorted(key for key in need if spec.values[key] is None)
    if missing:
        base = {"mem_read_bpc", "mem_write_bpc"}
        label = "Missing timing base parameters: " if set(missing) <= base else "Missing timing parameters: "
        raise ValueError(label + ", ".join(missing))
    parameters = {name: {**copy.deepcopy(spec.parameters[name]), "selection": "configured"}
                  for name in sorted(need)}
    proxies = [name for name, item in parameters.items() if item["status"] == "assumed"]
    return parameters, proxies


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
                 tile=None, mem_slices=None, streams=None, reserved_bytes=0):
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        if scope != "single_hem":
            raise ValueError("unsupported_scope: chip timing/mapping is not modeled; use single_hem")
        if loop_order not in LOOP_ORDERS:
            raise ValueError(f"Unsupported loop order: {loop_order}")
        spec = self.tsp_spec
        tile = tuple(spec.execution["default_tile_mkn"]) if tile is None else tile
        v = spec.values
        plan = TSPTilingEngine(spec).derive_tile_sizes(M,K,N, tile=tile, mem_slices=mem_slices,
                                                     streams=streams, reserved_bytes=reserved_bytes)
        need = {"mem_read_bpc", "mem_write_bpc", "weight_buffer_bytes", "install_bpc", "install_propagation_cycles",
                "mxm_first_cycles", "mxm_row_ii", "row_io_latency_cycles"}
        # 只有 K 被分成多段时才需要 VXM 合并部分和，因此按任务决定必需的时间参数。
        if K > plan["k"]:
            need.update(VXM_ADD_PARAMETERS)
        missing = sorted(key for key in need if v[key] is None)
        timed = mode == "timing"
        if timed:
            parameters, proxies = _timing_inputs(spec, need)
        slices, width = len(plan["mem_slices"]), spec.stream_width_bytes
        stream_bw = plan["streams_per_direction"] * width
        if timed:
            read_bw = slices * v["mem_read_bpc"]
            write_bw = slices * v["mem_write_bpc"]
            derived_services = {
                "stream_bytes_per_cycle": stream_bw,
                "mem_read_bytes_per_cycle": read_bw,
                "mem_write_bytes_per_cycle": write_bw,
                "weight_load_peak_bytes_per_cycle": min(read_bw, stream_bw),
                "install_bytes_per_cycle": v["install_bpc"],
                "weight_buffer_bytes": v["weight_buffer_bytes"],
                "vxm_row_compute_cycles": {"min": None, "max": None},
                "vxm_row_first_cycles": {"min": None, "max": None},
            }
        counts = Counter({k: 0 for k in ("mxm_mac", "weight_install_count", "weight_plane_installs",
            "mem_read_activation_bytes", "mem_read_weight_bytes", "weight_install_bytes", "weight_stream_bytes",
            "mem_read_psum_bytes", "mem_write_psum_bytes", "mem_write_final_bytes", "vxm_add_ops",
            "activation_stream_bytes", "result_stream_bytes", "psum_stream_bytes", "write_stream_bytes", "tiles")})
        # plane_tags 跟踪各计算平面当前保留的 B；psum_progress 跟踪各输出块已完成的 K。
        plane_tags = {}
        psum_progress = {}
        tiles = []
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
            if changed_bytes > v["weight_buffer_bytes"]:
                raise ValueError("unsupported_capacity: pending TSP weights exceed weight_buffer_bytes")
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
                ca = max(math.ceil(kp/read_bw), math.ceil(act_stream/stream_bw))
                co = math.ceil(out_stream/stream_bw)
                cr = max(math.ceil(4*np/read_bw), co) if later else 0
                cw = max(math.ceil(4*np/write_bw), co)
                ii_add = spec.vxm_service_cycles({"int32_add": np}) if later else 0
                # 首行先完成显式传输，再按纯计算速率执行 add；不再用 stream 带宽压低算力。
                # 每个尾块用自己的 np，不沿用最大输出行的计算时间。
                add_first = v["vxm_add_latency_cycles"] + ii_add if later else 0
                if later:
                    for key, value in (("vxm_row_compute_cycles", ii_add), ("vxm_row_first_cycles", add_first)):
                        observed = derived_services[key]
                        observed["min"] = value if observed["min"] is None else min(observed["min"], value)
                        observed["max"] = value if observed["max"] is None else max(observed["max"], value)
                # Sum actions that share a resource BEFORE applying the pipeline maximum.
                # 激活和旧部分和共用读预算；假设 bank 布局无冲突，写服务可独立并发。
                mem_read_row = kp+(4*np if later else 0)
                ii_mem = max(math.ceil(mem_read_row/read_bw), math.ceil(4*np/write_bw))
                ii_stream = max(math.ceil((act_stream+out_stream)/stream_bw),
                                math.ceil((out_stream+(out_stream if later else 0))/stream_bw))
                ii_mxm = max(v["mxm_row_ii"], math.ceil(kp*np/allocated_peak_mac))
                if v["mxm_first_cycles"] < math.ceil(kp*np/allocated_peak_mac):
                    raise ValueError("mxm_first_cycles violates the allocated MAC capacity")
                # 不同资源可流水工作，稳态行间隔由最慢资源决定。
                ii = max(ii_mem,ii_stream,ii_mxm,ii_add)
                # 首行经历完整依赖链；后续行每 II 启动。LW 在行窗口的剩余资源上预取。
                first = v["row_io_latency_cycles"]+ca+v["mxm_first_cycles"]+co+cr+add_first+cw
                tiles.append({"offset_mkn": [ms, ks, ns], "shape_mkn": [m, k, n], "rows": m,
                    "weight_bytes": changed_bytes, "weight_stream_bytes": weight_groups*width,
                    "first_row_cycles": first, "row_interval_cycles": ii,
                    "row_payload_bytes": {"mem_read": mem_read_row, "mem_write": 4*np,
                        "stream_to_mxm": act_stream+out_stream,
                        "stream_to_vxm": out_stream+(out_stream if later else 0)}})
                max_row_ii = max(max_row_ii,ii)
        counts.update(useful_ops=2*M*K*N, executed_padded_ops=2*counts["mxm_mac"],
                      logical_output_bytes=4*M*N, logical_cross_k_adds=(math.ceil(K/plan["k"])-1)*M*N)
        scheduled = schedule_prefetch(tiles,
            {"mem_read": read_bw, "mem_write": write_bw,
             "stream_to_mxm": stream_bw, "stream_to_vxm": stream_bw},
            v["install_bpc"], v["install_propagation_cycles"], v["weight_buffer_bytes"]) if timed else None
        cycles = scheduled["total_cycles"] if timed else None
        time_ns = cycles/spec.clock_freq_ghz if timed and spec.clock_freq_ghz is not None else None
        actions = {key: {"count": count, "unit": "MAC" if key=="mxm_mac" else "op" if key=="vxm_add_ops" else "byte",
                         "scope": "single_hem"}
                   for key,count in counts.items()
                   if (key.endswith("_bytes") and key!="logical_output_bytes") or key in {"mxm_mac","vxm_add_ops"}}
        result = {"schema_version": 10, "model": "tsp_analytical_reference", "reference_id": spec.reference_id,
            "profile_id": spec.profile_id, "mode": mode,
            "workload": {"operation":"gemm","M":M,"K":K,"N":N,"data_format":"INT8","accumulation_dtype":"INT32","output_dtype":"INT32"},
            "boundary": {"scope":"single_hem","active_hemispheres":1,"start":"packed_A_B_in_local_MEM",
                         "end":"INT32_C_in_local_MEM","numeric_values_simulated":False,
                         "host_transfer_included":False,"packing_included":False},
            "policy": {"loop_order":loop_order,"selected_tile_mkn":[plan[x] for x in "mkn"],
                       "mem_slices":plan["mem_slices"],"streams_per_direction":plan["streams_per_direction"],
                       "tail_policy":plan["tail_policy"],"inter_tile_overlap":"weight_prefetch",
                       "weight_load_overlap":"background", "pending_weight_slots":1,
                       "row_pipeline":"compressed ideal steady stream; shared resource demands added before max",
                       "vxm_budget":"explicit single-task INT32 add service; no ALU replication"},
            "memory":plan["memory"], "counts":dict(counts),"action_counts":actions,
            "timing":{"status":"analytical_estimate" if timed else "not_computed","scope":"single_hem",
                      "total_cycles":cycles,"chip_cycles":None,"time_ns":time_ns,
                      "throughput_gops":counts["useful_ops"]/time_ns if time_ns else None,
                      "max_row_interval_cycles":max_row_ii if timed else None,
                      "missing_parameters":missing},
            "energy":{"status":"not_evaluated","total_pj":None,"tops_per_watt":None},
            "area":{"status":"not_evaluated","total_mm2":None,"tops_per_mm2":None},
            "hardware_inventory":spec.area_inventory(),"hardware_derived":spec.derived(),
            "parameter_provenance":copy.deepcopy(spec.parameters),
            "validation":{"hardware_validated":False,"source_note":"See parameter_provenance for source verification; documented specifications do not establish hardware calibration."},
            "limitations":["Single-side resident analytical model; chip inventory is not chip execution.",
                "Uniform packed striping and tail masking are software assumptions, not a verified ISA layout.",
                "MEM actions count padded payload bytes, not physical bank transactions; stream bytes are rounded to vector groups.",
                "MEM read/write services are independent under an assumed conflict-free bank layout; bank allocation/conflicts are not simulated.",
                "No physical stream time slots, remote MEM or supercell events. Only next-weight preparation overlaps tile computation.",
                "One assumed finite weight buffer; LW capture has no additional port limit beyond allocated MEM/stream bandwidth.",
                "IW uses a separate internal port and conservatively waits for prior row-window drain; exact rolling installation is not simulated.",
                "Padded MAC count is issued logical work; inactive-cell switching and power gating are not modeled."]}
        if timed:
            result["schedule"] = scheduled.pop("schedule")
            elapsed_stages = scheduled.pop("elapsed_stages")
            result["timing"].update(scheduled)
            attach_timing_breakdowns(result["timing"], elapsed=elapsed_stages)
            result["timing"].update(model="single_hem_weight_prefetch_v1", parameters=parameters, derived_services=derived_services,
                proxy_parameters=proxies,
                assumptions=[
                    "JSON values are fixed service limits and latencies; required null services fail. Allocation-dependent values are reported separately in derived_services.",
                    "Resident-weight compute II is not total task II: MEM read/write, directional stream and VXM services remain constrained.",
                    "LW reads each changed weight once into a single pending-block buffer. IW pays internal payload service and propagation once, without reading MEM again.",
                    "LW uses residual MEM-read and stream-to-MXM bandwidth; the current row pipeline has priority. Read/write independence does not permit two full-rate reads.",
                    "Row demand/II is reserved across the entire fill/steady/drain window, conservatively reducing prefetch opportunities. Buffer capacity and drained IW are explicit assumptions, not public timing specifications.",
                    "First-row transfer stages and arithmetic are serial; subsequent rows overlap independent resources using max(MEM, stream, MXM, VXM) service intervals. VXM compute is not capped by stream bandwidth again.",
                    "This is an uncalibrated estimate, not a guaranteed physical upper/lower bound. Configured model assumptions are not hardware measurements."])
            result["policy"]["vxm_budget"] = "configured or explicit single-task INT32 proxy; no ALU replication"
        return attach_contract(result, {**spec.execution, "default_tile_mkn": list(tile)})
