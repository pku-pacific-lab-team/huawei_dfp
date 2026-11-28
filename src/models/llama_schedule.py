"""Instantiate local operator dependencies and the inter-chip tensor ledger."""
from math import prod


def model_schedule(config, request, stages, chips, profiles):
    b, h, vocab = request.batch, config["hidden_size"], config["vocab_size"]
    schedule, events = [], []
    groups = [[] for _ in stages]
    for chip in chips:
        groups[chip["stage"]].append(chip)
    first, last = stages[0]["chips"], stages[-1]["chips"]

    def node(name, stage, deps, **fields):
        schedule.append(dict(id=name, stage=stage, depends_on=deps, **fields))
        return name

    def event(name, kind, stage, deps, participants, shape, dtype, **fields):
        payload = prod(shape) * (1 if dtype == "INT8" else 4)
        events.append(dict(id=name, kind=kind, stage=stage, producers=deps, consumers=[],
                           participants=participants, shape=shape, dtype=dtype, logical_tensor_bytes=payload,
                           communication_cycles=None, **fields))

    def transfer(name, stage, source_node, routes, shape, dtype, kind):
        # The same already-replicated tensor is delivered once per remote
        # consumer rank. The chosen endpoints do not prescribe physical links.
        routes = [(a,b) for a,b in routes if a != b]
        if not routes:
            return source_node
        node(name, stage, [source_node], operation=kind)
        participants = list(dict.fromkeys(chip for route in routes for chip in route))
        event(name, kind, stage, [source_node], participants, shape, dtype,
              deliveries=[dict(source=a, destination=b) for a,b in routes])
        return name

    def instance(step, stage, kind, prefix, upstream):
        group = groups[stage]
        ids = stages[stage]["chips"]
        # All ranks share operator order. The leader also owns the gathered
        # logits and the external token-selection boundary.
        profile = profiles[group[0]["steps"][step][kind]]
        for op in profile["nodes"]:
            op_id = op["id"]
            deps = [prefix+"/"+x for x in op["depends_on"]]
            if step and op_id == "kv.append":
                deps.append(f"step.{step-1}/" + prefix.split("/", 1)[1] + "/kv.append")
            if any(profile["tensors"][x]["role"] == "input" for x in op["inputs"]):
                deps.append(upstream)
            name = node(prefix+"/"+op_id, stage, deps, profile_slot=kind, profile_node=op_id)
            if op_id == "select_token":
                schedule[-1].update(owner_chip=ids[0], service="external_token_selection", service_cycles=None)
            elif op.get("collective") == "all_reduce_sum_int32":
                event(name, "tp_all_reduce_sum", stage, deps, ids, [b,h], "INT32",
                      input_bytes_per_rank=op["payload_bytes_per_rank"], int32_adds=op["group_int32_adds"])
            elif op_id in ("hidden.all_gather", "logits.gather"):
                embedding = kind == "embedding"
                axis = "embedding_hidden_columns" if embedding else "logit_vocab_range"
                dtype, width = ("INT8", h) if embedding else ("FP32", vocab)
                shards = [dict(chip=c["chip_id"], columns=c[axis],
                               payload_bytes=b*(c[axis][1]-c[axis][0])*(1 if embedding else 4)) for c in group]
                event(name, "tp_all_gather" if embedding else "tp_gather", stage, deps, ids, [b,width], dtype,
                      input_shards=shards, destinations=ids if embedding else ids[:1])
        return name

    prev = node("request.token_ids", 0, [], operation="input", owner_chip=first[0])
    for step in range(request.decode_steps):
        prefix = f"step.{step}/"
        source = first[0] if step == 0 else last[0]
        prev = transfer(prefix+"token_ids", 0, prev, [(source,dst) for dst in first], [b], "INT32",
                        "token_distribution" if step == 0 else "decode_feedback")
        prev = instance(step, 0, "embedding", prefix+"embedding", prev)
        for s, stage in enumerate(stages):
            if s:
                prev_ids = stages[s-1]["chips"]
                routes = [(prev_ids[i % len(prev_ids)], dst) for i,dst in enumerate(stage["chips"])]
                prev = transfer(prefix+f"pp.{s-1}.{s}", s, prev, routes, [b,h], "INT8", "pp_activation")
            for layer in range(*stage["layers"]):
                prev = instance(step, s, "block", prefix+f"layer.{layer}", prev)
        prev = instance(step, len(stages)-1, "head", prefix+"head", prev)
    by_name = {e["id"]: e for e in events}
    for n in schedule:
        for dep in n["depends_on"]:
            if dep in by_name:
                by_name[dep]["consumers"].append(n["id"])
    if prev in by_name:
        by_name[prev]["consumers"].append("model.logits")
    node("model.logits", len(stages)-1, [prev], operation="output", owner_chip=last[0])
    return schedule, events
