"""Check compact row schedules against their declared storage lifetime rules.

This validates symbolic values/slots, not addresses, banks or machine layouts.
For a repeated group, first and steady-state iterations have the same storage
transitions, so at most two representative iterations are checked.
"""

from .composite_memory import MemoryKind, MemoryPath


def validate_storage_schedule(op, path, groups):
    if not isinstance(path, MemoryPath):
        raise ValueError("Expected an explicit composite MemoryPath")
    local = path.kind is MemoryKind.LOCAL_DMA
    scalar_slots = {"row_sum": 0, "row_scale": 0} if op.operation == "rmsnorm" else {
        "row_max": 0, "row_sum": 1, "row_reciprocal": 1}
    states, exp_elements, output_elements = {}, 0, 0
    max_tile_slots = 2 if op.operation == "rmsnorm" else 1
    peak = 0
    checked = 0

    def fail(message):
        raise ValueError("Invalid composite storage lifetime: " + message)

    for group in groups:
        phase, n, repeats = group["phase"], group["elements"], group["repetitions_per_row"]
        if type(n) is not int or not 1 <= n <= path.tile or type(repeats) is not int or repeats < 1:
            fail("invalid group extent")
        if phase == "normalize" and exp_elements != op.width:
            fail("normalization reads exponentials before the full row was stored")
        group_exp = group_output = 0
        for _ in range(min(repeats, 2)):
            checked += 1
            slots, tree = {}, None
            current_exp = phase == "normalize"
            def slot(name):
                if name in scalar_slots:
                    return ("scalar", scalar_slots[name])
                if not local:
                    return ("scratch", 0)
                if op.operation == "rmsnorm":
                    second = name == "weight" or (phase == "sum_squares" and name in {"tile_scratch", "tree"})
                    return ("tile", int(second))
                return ("tile", 0)

            def read(name, elements):
                if type(elements) is not int or elements < 1:
                    fail("invalid read extent")
                if name in ("tree_survivors", "tree_partial"):
                    if tree is None or elements > tree or (name == "tree_partial" and tree != 1):
                        fail("reduction survivor read before its producer")
                    return
                if name in scalar_slots:
                    if elements != 1 or states.get(slot(name)) != name:
                        fail("row scalar read before initialization or after overwrite: " + name)
                    return
                if not local and name in ("x", "weight"):
                    if name == "weight" and op.operation != "rmsnorm":
                        fail("weight is not an input")
                    if elements > n:
                        fail("read exceeds tile")
                    return
                if not local and name == "output_exp":
                    if not current_exp or elements > n:
                        fail("output workspace read before initialization")
                    return
                if slots.get(slot(name)) != (name, n) or elements > n:
                    fail("tile read before load or after overwrite: " + name)

            def write(name, elements):
                nonlocal current_exp, tree
                if name in scalar_slots:
                    if elements != 1:
                        fail("row scalar exceeds its slot")
                    states[slot(name)] = name
                elif not local and name in ("output_exp", "output"):
                    current_exp = name == "output_exp"
                else:
                    if elements != n:
                        fail("element-wise write extent differs from tile")
                    if slots.get(slot(name), (None,))[0] == "tree":
                        tree = None
                    slots[slot(name)] = (name, n)

            produced_exp = produced_output = 0
            for stage in group["stages"]:
                if "direction" in stage:
                    if not local:
                        fail("resident MEM path must not invent a DMA")
                    if stage["logical_bytes"] != 4 * n:
                        fail("DMA extent differs from tile")
                    value = {"load_x": "x", "reload_x": "x", "load_weight": "weight",
                             "load_exp_workspace": "output_exp", "export_exp_workspace": "output_exp",
                             "export_output": "output"}.get(stage["name"])
                    if value is None:
                        fail("unrecognized DMA value")
                    if stage["direction"] == "load":
                        if value == "output_exp" and exp_elements != op.width:
                            fail("reload of incomplete exponentials")
                        write(value, n)
                    elif stage["direction"] == "export":
                        read(value, n)
                        produced_exp += n if value == "output_exp" else 0
                        produced_output += n if value == "output" else 0
                    else:
                        fail("invalid DMA direction")
                    continue
                for value in stage["reads"]:
                    read(value["buffer"], value["elements"])
                tree_round = stage["name"].startswith("reduce_")
                if tree_round:
                    remaining = n if tree is None else tree
                    pairs = remaining // 2
                    if ([v["elements"] for v in stage["reads"]] != [2 * pairs]
                            or stage["writes"] != [{"buffer": "tile_scratch", "elements": pairs}]):
                        fail("tree level drops or overwrites an odd survivor")
                    tree = remaining - pairs
                    slots[slot("tree")] = ("tree", n)
                else:
                    for value in stage["writes"]:
                        write(value["buffer"], value["elements"])
                        if not local:
                            produced_exp += value["elements"] if value["buffer"] == "output_exp" else 0
                            produced_output += value["elements"] if value["buffer"] == "output" else 0
                used_tile_slots = len(slots)
                if used_tile_slots > (max_tile_slots if local else 1):
                    fail("tile allocation exceeds modeled slots")
                peak = max(peak, 4 * n * used_tile_slots + 4 * len(states))
            if (group_exp and group_exp != produced_exp) or (group_output and group_output != produced_output):
                fail("repeated group has inconsistent lifetime")
            group_exp, group_output = produced_exp, produced_output
        exp_elements += group_exp * repeats
        output_elements += group_output * repeats
    if output_elements != op.width or (op.operation == "softmax" and exp_elements != op.width):
        fail("output workspace or final output is incomplete")
    limit = path.temporary_capacity_bytes
    if peak > limit:
        fail("live temporary bytes exceed the allocation")
    return {"status": "validated_symbolic_schedule", "representative_tile_iterations": checked,
            "peak_live_temporary_bytes": peak, "allocated_temporary_bytes": limit,
            "checks": ["initialized reads", "slot overwrite order", "odd tree survivors", "full-row exp before normalize", "temporary capacity"],
            "limitations": "Symbolic slot/lifetime checks only; no physical address, bank or register proof."}
