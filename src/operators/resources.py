"""Capacity results, preserving each native memory scope and allocation basis."""


def memory_requirements(memory):
    blocks = {"mem": memory} if "resident_bytes" in memory else memory
    result = {}
    for name, block in blocks.items():
        if name == "lifetime_validation":
            # A validation summary is not another physical memory resource.
            continue
        if not isinstance(block, dict):
            continue
        basis = next((key for key in ("allocated_bytes", "resident_bytes", "padded_footprint_bytes",
                                      "logical_footprint_bytes", "peak_bytes") if key in block), None)
        if basis is None:
            continue
        capacity = block.get("capacity_bytes", block.get("usable_bytes"))
        result[name] = {"scope": block.get("scope"), "required_bytes": block[basis],
                        "capacity_bytes": capacity, "requirement_basis": basis}
        if "mode" in block:
            result[name]["mode"] = block["mode"]
    return result
