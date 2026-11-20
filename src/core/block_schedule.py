"""Serial dataflow liveness; shared tensor identities are allocated once."""


def storage_schedule(inputs, outputs, steps, sizes, *, reserved_bytes=0):
    """sizes already includes native padding and per-allocation slice striping."""
    if not steps or not inputs or not outputs:
        raise ValueError("Block schedule requires inputs, outputs and steps")
    if type(reserved_bytes) is not int or reserved_bytes < 0:
        raise ValueError("Reserved storage must be a nonnegative integer")
    if len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise ValueError("Duplicate block boundary tensor")
    known, producers, consumers = set(inputs), {}, {}
    ids = set()
    for index, step in enumerate(steps):
        if step["id"] in ids or not set(step["inputs"]) <= known:
            raise ValueError("Block steps must have unique IDs and topologically available inputs")
        ids.add(step["id"])
        if step["output"] in known:
            raise ValueError("Block output alias or duplicate producer")
        producers[step["output"]] = index
        for name in step["inputs"]:
            consumers.setdefault(name, []).append(index)
        known.add(step["output"])
    if not set(outputs) <= set(producers) or set(sizes) != known:
        raise ValueError("Block output or allocation is missing")
    if any(type(size) is not int or size <= 0 for size in sizes.values()):
        raise ValueError("Tensor allocation must be positive integer bytes")
    lifetimes = []
    for name in sorted(known):
        external, final = name in inputs, name in outputs
        if not external and not final and name not in consumers:
            raise ValueError("Unused intermediate is not part of this block")
        lifetimes.append({"tensor": name, "bytes": sizes[name], "first_step": 0 if external else producers[name],
                          "last_step": len(steps) - 1 if external or final else max(consumers[name]),
                          "persistent_input": external})
    timeline = []
    for index, step in enumerate(steps):
        live = [x for x in lifetimes if x["first_step"] <= index <= x["last_step"]]
        timeline.append({"node": step["id"], "live_tensors": [x["tensor"] for x in live],
                         "storage_bytes": sum(x["bytes"] for x in live) + reserved_bytes})
    return {"status": "validated_symbolic_schedule", "lifetimes": lifetimes, "steps": timeline,
            "reserved_bytes": reserved_bytes, "peak_bytes": max(s["storage_bytes"] for s in timeline),
            "policy": "Inputs remain allocated for the whole block; intermediates release after last consumer; output allocation overlaps its inputs."}
