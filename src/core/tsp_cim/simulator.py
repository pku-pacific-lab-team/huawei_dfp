"""INT8 hemisphere execution and native TSP comparison."""
from ..cim.simulator import LOOP_ORDERS
from ..cim.macro_model import positive_integer
from .hemisphere import simulate_hemisphere
from ...utils.workload import parse_workloads


class TSPCIMSimulator:
    def __init__(self, tsp_spec, *, data_format="INT8"):
        if data_format != "INT8":
            raise ValueError("TSP-CIM Macro supports INT8 operands and INT32 results only")
        self.tsp_spec = tsp_spec

    def simulate(self, M, K, N, loop_order="KNM", *, mode="count", hemisphere=0,
                 tile_m=None, mem_slices=None, streams=None, reserved_bytes=0, load_overlap=None):
        if mode not in {"count", "timing"}:
            raise ValueError("mode must be count or timing")
        for name, value in (("M", M), ("K", K), ("N", N)):
            positive_integer(value, name)
        return simulate_hemisphere(self.tsp_spec, M, K, N, loop_order, mode=mode,
            hemisphere=hemisphere, tile_m=tile_m, mem_slices=mem_slices,
            streams=streams, reserved_bytes=reserved_bytes, load_overlap=load_overlap)

    def compare_mxm(self, M, K, N, loop_order="KNM", workload=None, *, result=None, **policy):
        # 使用与独立 tsp.py 相同的原生内核，半片对比传递实际模式和分配。
        if workload is None:
            workload = parse_workloads({"gemm": {"M": M, "K": K, "N": N}, "data_format": "INT8"})[0]
        if workload.shape != (M, K, N) or workload.data_format != "INT8":
            raise ValueError("comparison workload must match the INT8 CIM task")
        if result is None:
            result = self.simulate(M, K, N, loop_order, **policy)
        if (tuple(result["workload"][axis] for axis in ("M", "K", "N")) != (M, K, N)
                or result["policy"]["loop_order"] != loop_order):
            raise ValueError("comparison result must match workload and loop order")
        from ...utils.tsp_reference import simulate_reference
        p = result["policy"]
        try:
            reference = simulate_reference(self.tsp_spec.native_reference_spec, workload, mode=result["mode"],
                loop_order=loop_order, tile=(p["selected_tile_mkn"][0],320,640),
                mem_slices=p["mem_slices"], streams=p["streams_per_direction"],
                reserved_bytes=result["memory"]["ranges"]["reserved"]["bytes"])
        except ValueError as exc:
            return {"status": "not_comparable", "native_reference": None, "speedup": None,
                    "energy_ratio": None, "reasons": [str(exc)]}
        return {"status": "aligned_scope_analytical" if result["mode"] == "timing" else "aligned_scope_counts",
            "native_reference": reference, "speedup": None, "energy_ratio": None,
            "analytical_cycle_ratio_native_over_cim": reference["timing"]["total_cycles"]/result["timing"]["total_cycles"]
                if result["mode"] == "timing" else None,
            "load_policy_matched": p["weight_load_overlap"] == reference["policy"]["weight_load_overlap"],
            "native_load_policy": reference["policy"]["weight_load_overlap"],
            "matched": ["INT8/INT32 GEMM", "single hemisphere", "resident MEM boundary", "MEM/stream allocation", "M grouping"],
            "reasons": ["Ratio compares explicit analytical policies, not measured hardware speedup.",
                "Load-policy match is reported separately; use compare_load_policies to isolate background-LW scheduling effects.",
                "Native K16/N16 and CIM K16/N32 padding and weight-context organizations remain architecture-specific."]}

    def compare_load_policies(self, M, K, N, loop_order="KNM", **policy):
        """Same workload/allocation/counts; only the CIM LW overlap policy changes."""
        policy = {k:v for k,v in policy.items() if k not in {"mode", "load_overlap"}}
        runs = {name:self.simulate(M,K,N,loop_order,mode="timing",load_overlap=name,**policy)
                for name in ("serialized", "background")}
        native = self.compare_mxm(M,K,N,loop_order,result=runs["serialized"])
        return {"native_prefetch_cycles": native["native_reference"]["timing"]["total_cycles"]
                    if native["native_reference"] else None,
                "cim_cycles": {name:r["timing"]["total_cycles"] for name,r in runs.items()},
                "counts_equal": runs["serialized"]["counts"] == runs["background"]["counts"],
                "conditions": "Matched hemisphere, allocation and workload. Analytical policy sensitivity, not measured hardware speedup."}

    def sweep_m(self, m_values, K, N, loop_order="KNM", **policy):
        results = []
        for m in m_values:
            result = self.simulate(m, K, N, loop_order, **policy)
            results.append({"M": m, "result": result,
                            "comparison": self.compare_mxm(m, K, N, loop_order, result=result)})
        return results

    def compare_loop_orders(self, M, K, N, **policy):
        return {order: self.simulate(M, K, N, order, **policy) for order in LOOP_ORDERS}
