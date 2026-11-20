"""Adapt native GEMM and Vector/composite results without recounting work."""
from .actions import Action, ActionLedger
from .primitives import Gemm
from .types import OpResult
from ..core.operator_result import NativeResult


def adapt_native_result(operator, architecture, native, *, include_details=False):
    """Wrap native Vector/composite actions; all counts and timings stay native."""
    if not isinstance(native, NativeResult):
        raise ValueError("Expected a native kernel result")
    report = native.report
    if set(native.action_specs) != set(report["action_counts"]):
        raise ValueError("Native action metadata does not cover the action ledger")
    entries = []
    for key, action in report["action_counts"].items():
        spec = native.action_specs[key]
        entries.append(Action(key, architecture, spec.component, spec.name, spec.category,
                              action["count"], action["unit"], operator.outputs[0].dtype,
                              action["scope"], action["basis"], f"#/native_report/action_counts/{key}",
                              native.action_description, attributes=(("native_action", key),)))
    return OpResult.from_native(operator, architecture, ActionLedger(tuple(entries)), report, include_details=include_details)


# Tuple fields: component, canonical name, category, stored/arithmetic dtype.
ASCEND = {
    "cube_mac": ("cube", "mac", "matrix", "INT8xINT8->INT32"),
    "hbm_read_bytes": ("hbm", "read", "offchip_memory", "INT8"),
    "hbm_write_bytes": ("hbm", "write", "offchip_memory", "INT32"),
    "l2_fill_bytes": ("l2", "write", "onchip_memory", "INT8"),
    "l2_read_bytes": ("l2", "read", "onchip_memory", "INT8"),
    "l1_l0a": ("l1_to_l0a", "transfer", "onchip_transfer", "INT8"),
    "l1_l0b": ("l1_to_l0b", "transfer", "onchip_transfer", "INT8"),
    "l0c_ub": ("l0c_to_ub", "transfer", "onchip_transfer", "INT32"),
}
TPU = {
    "matrix_mac": ("mxu", "mac", "matrix", "INT8xINT8->INT32"),
    "hbm_read_bytes": ("hbm", "read", "offchip_memory", "INT8"),
    "hbm_write_bytes": ("hbm", "write", "offchip_memory", "INT32"),
    "cmem_write_bytes": ("cmem", "write", "onchip_memory", "INT8"),
    "cmem_read_bytes": ("cmem", "read", "onchip_memory", "INT8"),
    **{f"vmem_{phase}_{direction}_bytes": ("vmem", direction, "onchip_memory", dtype)
       for phase, dtype, directions in (("input", "INT8", ("read", "write")),
                                         ("psum", "INT32", ("read", "write")),
                                         ("final", "INT32", ("write",)),
                                         ("export", "INT32", ("read",)))
       for direction in directions},
}
MEM = {f"mem_{direction}_{phase}_bytes": ("mem", direction, "onchip_memory", dtype)
       for direction, phase, dtype in (("read", "activation", "INT8"), ("read", "weight", "INT8"),
                                        ("read", "psum", "INT32"), ("write", "psum", "INT32"),
                                        ("write", "final", "INT32"))}
STREAMS = {f"{phase}_stream_bytes": ("stream", "transfer", "interconnect", dtype)
           for phase, dtype in (("weight", "INT8"), ("activation", "INT8"),
                                ("result", "INT32"), ("psum", "INT32"), ("write", "INT32"))}
TSP = {**MEM, **STREAMS,
       "mxm_mac": ("mxm", "mac", "matrix", "INT8xINT8->INT32"),
       "weight_install_bytes": ("weight_buffer_to_mxm", "transfer", "onchip_transfer", "INT8"),
       "vxm_add_ops": ("vxm", "add", "vector", "INT32")}
CIM = {**MEM, **{k: v for k, v in STREAMS.items() if k not in {"weight_stream_bytes", "result_stream_bytes"}},
       "useful_mac": ("macro", "mac", "matrix", "INT8xINT8->INT32"),
       "issued_mac": ("macro", "mac", "matrix", "INT8xINT8->INT32"),
       "cross_macro_k_adds": ("inter_superlane_reduction", "add", "vector", "INT32"),
       "cross_tile_k_adds": ("vxm", "add", "vector", "INT32"),
       **{name: ("macro", name, "cim_internal", dtype) for name, dtype in (
           ("program_bytes", "INT8"), ("context_programs", "INT8"), ("context_switches", "INT8"),
           ("program_wordlines", "INT8"), ("launches", "INT8"), ("bit_serial_cycles", "INT8"),
           ("activation_capture_bytes", "INT8"), ("result_bytes", "INT32"),
           ("weight_word_bit_products", "INT8"), ("local_tree_adds", "unspecified"), ("shift_accumulates", "unspecified"))}}
ALIASES = {
    "logical_activation_broadcast_bytes": ("mem_read_activation_bytes",),
    "logical_inter_superlane_adds": ("cross_macro_k_adds",),
    "logical_inter_k_tile_adds": ("cross_tile_k_adds",),
    "logical_cluster_partial_bytes": ("mem_write_psum_bytes", "mem_write_final_bytes"),
}
PROFILES = {
    "ascend": (12, {"ascend910_analytical"}, ASCEND, "single_core"),
    "tpu": (13, {"tpu_v4_count", "tpu_v4_analytical"}, TPU, "single_tensorcore"),
    "tsp": (10, {"tsp_analytical_reference"}, TSP, "single_hem"),
    "tsp_cim": (10, {"tsp_cim_hemisphere_reference"}, CIM, "single_hem"),
}


def adapt_gemm_report(architecture, report, *, operator=None, include_details=False):
    """Keep every native action traceable and reject unreviewed report versions."""
    if not isinstance(architecture, str) or architecture not in PROFILES:
        raise ValueError("Unsupported architecture")
    version, models, mapping, scope = PROFILES[architecture]
    if report.get("schema_version") != version or report.get("model") not in models:
        raise ValueError("Unsupported native report schema/model; review the action mapping")
    if report["boundary"]["scope"] != scope or report["execution_contract"]["scope"] != scope:
        raise ValueError("Native report execution scope does not match architecture")
    w = report["workload"]
    if (w.get("operation"), w.get("data_format"), w.get("accumulation_dtype"), w.get("output_dtype")) != ("gemm", "INT8", "INT32", "INT32"):
        raise ValueError("GEMM adapter requires INT8 inputs and INT32 accumulation/output")
    op = Gemm(w["M"], w["K"], w["N"])
    if operator is not None:
        if not isinstance(operator, Gemm) or (operator.M, operator.K, operator.N) != (op.M, op.K, op.N):
            raise ValueError("GEMM descriptor does not match native workload")
        op = operator
    if report["counts"]["useful_ops"] != 2 * op.useful_macs or report["counts"]["logical_output_bytes"] != op.spec.outputs[0].nbytes:
        raise ValueError("GEMM logical work does not match tensor shapes")
    aliases = ALIASES if architecture == "tsp_cim" else {}
    if set(report["action_counts"]) != set(mapping) | set(aliases):
        raise ValueError("Unmapped or missing native actions; review the action mapping")
    if report["mode"] not in {"count", "timing"}:
        raise ValueError("Unsupported native run mode")
    for key in ("total_cycles", "time_ns"):
        value = report["timing"][key]
        if report["mode"] == "count" and value is not None:
            raise ValueError("Count mode must not contain timing results")
    entries = []
    for key, native in report["action_counts"].items():
        target = aliases.get(key, (key,))[0]
        component, name, category, dtype = mapping[target]
        role = "summary" if key in aliases else "reference" if architecture == "tsp_cim" and key == "useful_mac" else "action"
        basis = "logical" if architecture == "tpu" else "padded"
        if architecture == "tsp_cim":
            basis = ("logical" if key == "useful_mac" or key.startswith(("cross_", "logical_")) else
                     "padded" if key == "issued_mac" or key.startswith(("mem_", "activation_stream", "psum_stream", "write_stream")) else "modeled_internal")
        # Preserve phase/route identities. A source transfer is never expanded
        # into invented SRAM endpoint actions or physical DRAM transactions.
        entries.append(Action(
            id=key, architecture=architecture, component=component, name=name, category=category,
            count=native["count"], unit=native["unit"], dtype=dtype, scope=native["scope"], basis=basis,
            source=f"#/native_report/action_counts/{key}",
            source_basis=native.get("basis", "Native TSP pads K/N and stream groups; see native policy.tail_policy"),
            role=role, attributes=(("native_action", key),), summarizes=aliases.get(key, ())))
    return OpResult.from_native(op.spec, architecture, ActionLedger(tuple(entries)), report, include_details=include_details)
