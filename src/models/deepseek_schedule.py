"""Expand a single Decode stream into rank invocations and untimed tensor transfers."""

from math import prod

from .deepseek_placement import assignments, attention_shape, expert_shape, span


class DecodeLedger:
    def __init__(self, chips, planner):
        self.chips, self.planner = chips, planner
        self.nodes, self.events, self.routes = [], [], {}

    def node(self, name, deps, **info):
        self.nodes.append(dict(id=name, depends_on=list(dict.fromkeys(deps)), **info))
        return name

    def event(self, name, kind, deps, group, shape, dtype, **info):
        self.events.append(
            dict(
                id=name,
                kind=kind,
                participants=group,
                shape=list(shape),
                dtype=dtype,
                logical_tensor_bytes=prod(shape) * (1 if dtype == "INT8" else 4),
                producers=list(deps),
                consumers=[],
                communication_cycles=None,
                **info,
            )
        )

    def transfer(self, name, kind, deps, routes, shape, dtype, **info):
        remote = [(s, d) for s, d in routes if s != d]
        participants = list(dict.fromkeys(i for pair in routes for i in pair))
        self.node(name, deps, operation=kind, chips=participants)
        if remote:
            self.event(
                name,
                kind,
                deps,
                participants,
                shape,
                dtype,
                deliveries=[dict(source=s, destination=d) for s, d in remote],
                **info,
            )
        return name

    def run(self, step, name, ids, shape, inputs, *, previous_kv=None):
        graphs = []
        for cid in ids:
            chip = self.chips[cid]
            local = shape(chip)
            profile = self.planner.bind_chip(local, chip, name)
            chip["steps"][step]["runs"][name] = profile
            graphs.append(self.planner.shapes(*local))
        base = graphs[0]
        # Non-leader head/ETP ranks stop after publishing their shard/partial.
        for i, op in enumerate(base.nodes):
            label = op["id"]
            active = [cid for cid, g in zip(ids, graphs) if i < len(g.nodes)]
            deps = [name + "/" + p for p in op["depends_on"]]
            for t in op["inputs"]:
                if base.tensors[t]["role"] == "input":
                    deps.extend(inputs[t])
            if previous_kv and label == "kv.append":
                deps.append(previous_kv)
            n = self.node(name + "/" + label, deps, invocation=name, profile_node=label, chips=active)
            if op.get("collective") == "all_reduce_sum_int32":
                tensor = base.tensors[op["inputs"][0]]
                self.event(
                    n,
                    "tp_all_reduce_sum",
                    deps,
                    ids,
                    tensor["shape"],
                    "INT32",
                    input_bytes_per_rank=tensor["logical_bytes"],
                    int32_adds=tensor["numel"] * (len(ids) - 1),
                )
            elif label in ("hidden.all_gather", "logits.gather"):
                embed = label == "hidden.all_gather"
                key = "embedding_columns" if embed else "vocab_columns"
                tensor = base.tensors[op["outputs"][0]]
                shards = [
                    dict(
                        chip=cid,
                        columns=self.chips[cid][key],
                        payload_bytes=g.tensors[op["inputs"][0]]["logical_bytes"],
                    )
                    for cid, g in zip(ids, graphs)
                ]
                self.event(
                    n,
                    "tp_all_gather" if embed else "tp_gather",
                    deps,
                    ids,
                    tensor["shape"],
                    tensor["dtype"],
                    input_shards=shards,
                    destinations=ids if embed else ids[:1],
                )
            elif label == "select_token":
                self.nodes[-1].update(service="external_token_selection", service_cycles=None)
        return n

    def finish(self, last, owner):
        self.node("model.logits", [last], operation="output", chips=[owner])
        by_id = {row["id"]: row for row in self.events}
        for node in self.nodes:
            for dep in node["depends_on"]:
                if dep in by_id:
                    by_id[dep]["consumers"].append(node["id"])
        return self.nodes, self.events, self.routes


def expert_layer(dag, c, request, stage, step, prefix, front, route):
    b, h = request.batch, c["hidden_size"]
    att = stage["attention_chips"]
    root = att[0]
    selected = dag.run(step, prefix + "/router", [root], lambda _: ("router", b), {"input": [front]})
    info = assignments(stage, route)
    dag.routes[prefix] = info
    results, previous_return = [], {}
    for entry in [*info["active"], dict(expert="shared", tokens=b, group=stage["shared_group"])]:
        expert, count = entry["expert"], entry["tokens"]
        group = stage["expert_groups"][entry["group"]]
        path = prefix + f"/expert.{expert}"
        # Routing indices are symbolic. A message carries the corresponding
        # activation rows, and the ledger gives the slot order at both ends.
        deps = [front] if expert == "shared" else [front, selected]
        if entry["group"] in previous_return:
            deps = [*deps, previous_return[entry["group"]]]
        dispatch = dag.transfer(
            path + "/dispatch",
            "ep_dispatch",
            deps,
            [(root, d) for d in group],
            (count, h),
            "INT8",
            expert=expert,
            slots=entry.get("slots"),
        )
        output = dag.run(
            step, path, group, lambda chip: expert_shape(chip, stage, count), {"tokens": [dispatch]}
        )
        results.append(
            dag.transfer(
                path + "/return",
                "ep_return",
                [output],
                [(group[0], root)],
                (count, h),
                "FP32",
                expert=expert,
                slots=entry.get("slots"),
            )
        )
        previous_return[entry["group"]] = results[-1]
    combined = dag.run(
        step,
        prefix + "/combine",
        [root],
        lambda _: ("combine", b),
        {"routed": results[:-1], "weights": [selected], "shared": results[-1:], "skip": [front]},
    )
    return dag.transfer(
        prefix + "/broadcast", "moe_output_broadcast", [combined], [(root, d) for d in att], (b, h), "INT8"
    )


def decode_schedule(c, request, stages, chips, planner, routes):
    dag = DecodeLedger(chips, planner)
    b, h = request.batch, c["hidden_size"]
    first, last = stages[0]["attention_chips"], stages[-1]["attention_chips"]
    prev = dag.node("request.token_ids", [], operation="input", chips=first[:1])
    for step, length in enumerate(request.effective_lengths(c)):
        prefix = f"step.{step}"
        source = last[0] if step else first[0]
        prev = dag.transfer(
            prefix + "/token_ids",
            "decode_feedback" if step else "token_distribution",
            [prev],
            [(source, d) for d in first],
            (b,),
            "INT32",
        )
        prev = dag.run(
            step,
            prefix + "/embedding",
            first,
            lambda chip: ("embedding", span(chip, "embedding_columns")),
            {"token_ids": [prev]},
        )
        for si, stage in enumerate(stages):
            ids = stage["attention_chips"]
            if si:
                old = stages[si - 1]["attention_chips"]
                prev = dag.transfer(
                    prefix + f"/pp.{si - 1}.{si}",
                    "pp_activation",
                    [prev],
                    [(old[i % len(old)], d) for i, d in enumerate(ids)],
                    (b, h),
                    "INT8",
                )
            for layer in range(*stage["layers"]):
                dense = layer < c["first_k_dense_replace"]
                label = f"/layer.{layer}"
                name = prefix + label
                prev = dag.run(
                    step,
                    name + "/block",
                    ids,
                    lambda chip: attention_shape(chip, stage, length, dense),
                    {"input": [prev]},
                    previous_kv=f"step.{step - 1}" + label + "/block/kv.append" if step else None,
                )
                if not dense:
                    prev = expert_layer(dag, c, request, stage, step, name, prev, routes[step, layer])

        def head(chip):
            leader = chip["chip_id"] == last[0]
            return ("head", span(chip, "vocab_columns"), leader, leader and step + 1 < request.decode_steps)

        prev = dag.run(step, prefix + "/head", last, head, {"input": [prev]})
    return dag.finish(prev, last[0])
