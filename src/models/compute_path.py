"""Replay one Decode stream with exclusive chip compute and zero-time messages."""


def compute_path_cycles(schedule, events, chips, stages, profiles):
    durations = {name: {op["id"]: op["local_compute_cycles"] for op in profile["nodes"]}
                 for name, profile in profiles.items()}
    inventory = {chip["chip_id"]: chip for chip in chips}
    messages = {e["id"]: e for e in events}
    free = dict.fromkeys(inventory, 0)
    finished, latest = {}, {}

    def ready(deps, chip):
        return max((finished[d].get(chip, latest[d]) for d in deps), default=0)

    for node in schedule:
        name, deps = node["id"], node["depends_on"]
        event = messages.get(name, {})
        times = {}
        if "profile_node" in node:
            step = int(name.partition("/")[0].split(".")[1])
            ids = node["chips"] if "invocation" in node else stages[node["stage"]]["chips"]
            for chip in ids:
                bindings = inventory[chip]["steps"][step]
                key = (bindings["runs"][node["invocation"]] if "invocation" in node
                       else bindings[node["profile_slot"]])
                duration = durations[key].get(node["profile_node"])
                if duration is None:
                    # The LLaMA head schedule includes the leader-only token selection.
                    if node.get("service") == "external_token_selection" and chip != node["owner_chip"]:
                        continue
                    raise ValueError(f"Missing timing at {name} on chip {chip}")
                times[chip] = max(free[chip], ready(deps, chip)) + duration
                if duration:
                    free[chip] = times[chip]
            if event.get("kind") in ("tp_all_reduce_sum", "tp_all_gather", "tp_gather"):
                end = max(times.values())
                times = dict.fromkeys(event.get("destinations", ids), end)
        elif "deliveries" in event:
            # PP can send different rank replicas independently. Co-located
            # consumers keep the local result without a recorded delivery.
            for chip in node.get("chips", []):
                times[chip] = ready(deps, chip)
            for route in event["deliveries"]:
                times[route["destination"]] = ready(deps, route["source"])
        else:
            ids = node.get("chips", [node.get("owner_chip")])
            times = {chip: ready(deps, chip) for chip in ids}
        finished[name] = times
        latest[name] = max(times.values())
    return latest["model.logits"]


def tile_metrics(schedule, events, chips, stages, profiles, clock_hz):
    cycles = compute_path_cycles(schedule, events, chips, stages, profiles)
    private, workspace = {}, 0
    resident = max(chip["resident_bytes"] for chip in chips)
    for profile in profiles.values():
        resident = max(resident, profile["capacity"]["peak"]["required_bytes"])
        for node in profile["nodes"]:
            explicit = sum(sum(profile["tensors"][t]["bytes_per_unit"])
                           for t in node["live_tensors"] if t.endswith(".workspace"))
            workspace = max(workspace, explicit + sum(node["scratch_per_unit"]))
            for level, value in node["private_memory"].items():
                level = level.lower()
                private[level] = max(private.get(level, 0), value["required_bytes"])
    return dict(critical_path_cycles=cycles, critical_path_seconds=cycles / clock_hz if clock_hz else None,
                peak_chip_resident_bytes=resident, peak_chip_workspace_bytes=workspace,
                peak_private_bytes_per_unit=private)
