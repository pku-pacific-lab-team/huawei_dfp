"""Explicit analytical execution policies, separate from physical inventory."""
from copy import deepcopy


PROFILES = {
    "ascend": ("single_core", "HBM", {"resource_pipeline"}, {"serialized"},
               {"compute_startup", "transfer_startup"}),
    "tpu": ("single_tensorcore", "HBM", {"resource_pipeline", "serial"}, {"serialized"},
            {"transfer_startup"}),
    "tsp": ("single_hem", "local_MEM", {"row_pipeline"}, {"background"}, set()),
    "tsp_cim": ("single_hem", "local_MEM", {"static_wave_pipeline"}, {"serialized", "background"},
                {"local_pipeline", "context_switch"}),
}


def read_execution(raw, architecture):
    fields = {"scope", "resident_boundary", "schedule", "default_tile_mkn", "buffer_slots",
              "weight_load_overlap", "fixed_latency_cycles", "rules", "status", "source", "version", "conditions"}
    if architecture == "tpu":
        fields.update(("cmem_mode", "cmem_groups_mn", "cmem_buffer_slots"))
    if architecture == "ascend":
        fields.add("l2")
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError("execution: explicit execution policy fields required")
    if architecture == "tpu":
        if type(raw["cmem_buffer_slots"]) is not int or raw["cmem_buffer_slots"] not in (1, 2):
            raise ValueError("execution.cmem_buffer_slots must be 1 or 2")
        if not isinstance(raw["cmem_mode"], str) or raw["cmem_mode"] not in ("bypass", "tiled"):
            raise ValueError("execution.cmem_mode must be bypass or tiled")
        groups = raw["cmem_groups_mn"]
        if not isinstance(groups, list) or len(groups) != 2 or any(type(n) is not int or n <= 0 for n in groups):
            raise ValueError("execution.cmem_groups_mn must contain two positive integer block counts")
    scope, boundary, schedules, overlaps, latencies = PROFILES[architecture]
    if any(not isinstance(raw[key], str) for key in ("scope", "resident_boundary", "schedule", "weight_load_overlap")):
        raise ValueError("execution: scope, boundary and policies must be strings")
    if (raw["scope"] != scope or raw["resident_boundary"] != boundary or
            raw["schedule"] not in schedules or raw["weight_load_overlap"] not in overlaps):
        raise ValueError("execution: unsupported scope, boundary or scheduling policy")
    tile = raw["default_tile_mkn"]
    if not isinstance(tile, list) or len(tile) != 3 or any(type(x) is not int or x <= 0 for x in tile):
        raise ValueError("execution.default_tile_mkn must contain positive M/K/N integers")
    if architecture in {"tsp", "tsp_cim"} and tile[1:] != [320, 640]:
        raise ValueError("execution: TSP reference default K/N must be 320/640")
    if type(raw["buffer_slots"]) is not int or raw["buffer_slots"] not in (1, 2):
        raise ValueError("execution.buffer_slots must be 1 or 2")
    if architecture in {"tsp", "tsp_cim"} and raw["buffer_slots"] != 1:
        raise ValueError("execution: resident TSP model requires one A/B/C allocation")
    values = raw["fixed_latency_cycles"]
    if not isinstance(values, dict) or set(values) != latencies or any(type(x) is not int or x < 0 for x in values.values()):
        raise ValueError("execution.fixed_latency_cycles: required nonnegative integer cycle allowances")
    if raw["status"] != "assumed":
        raise ValueError("execution policy must retain assumed status")
    for key in ("source", "version", "conditions"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError("execution: missing " + key)
    rules = raw["rules"]
    expected = {
        "ascend": {"default_l1_k", "route_resources"},
        "tpu": {"fill_drain", "output_blocks", "input_sharing"},
        "tsp": {"inter_tile_overlap", "mem_read_write"},
        "tsp_cim": {"mapping", "wave_barrier", "mem_read_write"},
    }[architecture]
    if not isinstance(rules, dict) or set(rules) != expected:
        raise ValueError("execution.rules: unsupported rule fields")
    fixed = {
        "tpu": {"fill_drain": "rows_plus_columns_minus_two", "output_blocks": "serial",
                "input_sharing": "ideal_within_block"},
        "tsp": {"inter_tile_overlap": "weight_prefetch", "mem_read_write": "independent_assumed_conflict_free"},
        "tsp_cim": {"mapping": "kn_round_robin_no_m_replication", "wave_barrier": "drain",
                    "mem_read_write": "independent_assumed_conflict_free"},
    }
    if architecture in fixed and rules != fixed[architecture]:
        raise ValueError("execution.rules: unsupported analytical policy")
    if architecture == "ascend":
        l2 = {"mode": "panel_lru", "initial_state": "cold",
              "capacity_policy": "equal_partition_all_capacity",
              "bandwidth_policy": "equal_partition_no_borrow",
              "output_policy": "bypass", "oversized_read_policy": "bypass_no_allocate",
              "miss_pipeline": "fetch_fill_read_serial", "request_order": "program_order"}
        if raw["l2"] != l2:
            raise ValueError("execution.l2: explicit supported panel-LRU policies required")
        if type(rules["default_l1_k"]) is not int or rules["default_l1_k"] <= 0 or rules["default_l1_k"] % 32:
            raise ValueError("execution.rules.default_l1_k must align to K32")
        routes = rules["route_resources"]
        if not isinstance(routes, dict) or set(routes) != {"hbm_to_l2", "l2_fill", "l2_to_l1", "hbm_to_l1", "l1_l0a", "l1_l0b", "compute", "l0c_ub", "ub_to_hbm"}:
            raise ValueError("execution.rules.route_resources: expected all nine stages including L2")
        if any(not isinstance(v, list) or not v or
               any(not isinstance(r, str) or not r for r in v) or len(set(v)) != len(v) for v in routes.values()):
            raise ValueError("execution.rules.route_resources: invalid resource list")
    return deepcopy(raw)
