"""Describe block boundaries without inventing conversion or transfer costs."""
from .types import OpResult


def io_contract(result):
    arch, op = result.architecture, result.operator
    hbm = arch in {"ascend", "tpu"}
    gemm = op.kind == "gemm"
    location = {"memory": "HBM" if hbm else "local_MEM", "architecture": arch,
                "scope": "chip" if hbm else "single_hem",
                "hemisphere": result.boundary.get("selected_hemisphere"),
                "physical_device": None}
    layout = "native_packed" if gemm and arch != "tpu" else "logical_dense"
    if op.kind in {"residual", "swiglu", "swiglu_ffn", "kv_cache", "decode_attention",
                   "rmsnorm_int8", "qkv_projection", "attention_output_projection", "rope", "embedding", "vocabulary_projection", "mla_preprocess", "mla_cache", "mla_decode_attention"}:
        layout = result.boundary["layout"]
    def tensors(values):
        return [{**tensor.to_dict(), "location": dict(location), "layout": layout} for tensor in values]
    contract = {"schema_version": 1, "inputs": tensors(op.inputs), "outputs": tensors(op.outputs),
            "name_scope": "Within one operator call; graph callers must qualify names by node identity.",
            "intermediate_residency": "No UB/VMEM/register retention across operator calls is promised.",
            "conversion": "No implicit cast, dequantization or requantization; stored dtype is binding.",
            "traffic": "Standalone actions already include each block's boundary accesses; connections do not remove them."}
    if op.kind in {"kv_cache", "decode_attention", "mla_cache", "mla_decode_attention"}:
        contract["state_aliases"] = dict(result.boundary["state_aliases"])
        contract["cache_update"] = result.boundary["cache_update"]
    return contract


def plan_connection(producer, consumer, *, output_index=0, input_index=0):
    """Inspect one proposed edge. This does not execute or sum a graph.

    Even matching descriptions need a placement/allocation decision. Names in
    separate calls are not evidence that the same physical storage is shared.
    """
    if not isinstance(producer, OpResult) or not isinstance(consumer, OpResult):
        raise ValueError("Connection planning requires two OpResult objects")
    for index, tensors in ((output_index, producer.operator.outputs), (input_index, consumer.operator.inputs)):
        if type(index) is not int or not 0 <= index < len(tensors):
            raise ValueError("Invalid connection tensor index")
    source = io_contract(producer)["outputs"][output_index]
    target = io_contract(consumer)["inputs"][input_index]
    if source["shape"] != target["shape"]:
        raise ValueError("Connection shape mismatch; implicit reshape/broadcast is not modeled")
    dtype_change = source["dtype"] != target["dtype"]
    a, b = source["location"], target["location"]
    transfer = a["architecture"] != b["architecture"] or a["memory"] != b["memory"]
    if a["hemisphere"] is not None and b["hemisphere"] is not None and a["hemisphere"] != b["hemisphere"]:
        transfer = True
    layout_change = source["layout"] != target["layout"] or source["layout"] in {"native_packed", "symbolic_padded_views", "symbolic_padded_head_views"}
    needs = []
    if dtype_change:
        needs.append("explicit_conversion_model")
    if layout_change:
        needs.append("layout_validation")
    needs.append("transfer_model" if transfer else "placement_and_allocation_validation")
    return {"schema_version": 1, "status": "requires_boundary_work", "source": source, "target": target,
            "requirements": needs,
            "conversion": {"status": "required_unmodeled" if dtype_change else "not_required",
                           "method": None, "scale_and_rounding_policy": None,
                           "source_payload_bytes": source["logical_bytes"], "target_payload_bytes": target["logical_bytes"],
                           "semantics": "Logical payload sizes only; not additional actions, allocated peak bytes, or a conversion algorithm."},
            "residency": {"status": "transfer_required" if transfer else "unverified_placement",
                          "reuse_assumed": False, "standalone_traffic_removed": False},
            "latency_s": None, "additional_action_counts": None, "energy_pj": None,
            "semantics": "Descriptor-only plan. Unmodeled boundary work is unknown, never zero cost."}
