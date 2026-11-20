"""One pending weight tile, with LW using residual MEM/stream service.

Row windows remain compressed fluid reservations, not physical stream slots.
IW waits for the preceding row window to drain; this is a conservative model
allowance, not a claim that native hardware cannot hide IW.
"""
from fractions import Fraction
from math import ceil


def schedule_prefetch(tiles, capacities, install_bpc, propagation, buffer_bytes):
    caps = {key: Fraction(str(value)) for key, value in capacities.items()}
    peaks = dict.fromkeys(caps, Fraction(0))
    changes = [i for i, tile in enumerate(tiles) if tile["weight_bytes"]]
    pending = None
    change_index = 0
    now = stall = install = rows = load_cycles = overlap = 0
    timeline, segments = [], []
    loads = {}

    def load_window(start, duration, available, background, row_rates=None):
        nonlocal load_cycles, overlap
        if pending is None or not pending["remaining"]:
            return start
        tile = tiles[pending["index"]]
        # Payload and vector-padded stream traffic must make equal progress.
        demand = {"mem_read": tile["weight_bytes"],
                  "stream_to_mxm": tile["weight_stream_bytes"]}
        rate = min(available[key] / size for key, size in demand.items())
        if rate <= 0:
            return start
        cycles = ceil(pending["remaining"] / rate)
        if duration is not None:
            cycles = min(cycles, duration)
        if not cycles:
            return start
        amount = min(pending["remaining"], rate * cycles)
        rates = {key: size * amount / cycles for key, size in demand.items()}
        for key, value in rates.items():
            peaks[key] = max(peaks[key], value + (row_rates or {}).get(key, 0))
        entry = loads.setdefault(pending["index"], {"start": start, "end": start})
        entry["end"] = start + cycles
        segments.append({"tile_index": pending["index"], "start": start, "end": start + cycles,
                         "overlaps_rows": background,
                         "payload_bytes": float(demand["mem_read"] * amount),
                         "stream_bytes": float(demand["stream_to_mxm"] * amount),
                         "rates_bpc": {key: float(value) for key, value in rates.items()}})
        pending["remaining"] -= amount
        load_cycles += cycles
        overlap += cycles if background else 0
        return start + cycles

    for index, tile in enumerate(tiles):
        changed = tile["weight_bytes"]
        arrival = now
        if changed:
            if changed > buffer_bytes:
                raise ValueError("unsupported_capacity: pending TSP weights exceed weight_buffer_bytes")
            if pending is None:
                pending = {"index": index, "remaining": Fraction(1)}
            assert pending["index"] == index
            now = load_window(now, None, caps, False)
            assert not pending["remaining"]
            stall += now - arrival
            # Internal buffer-to-array traffic does not read MEM or streams again.
            iw = propagation + ceil(Fraction(changed) / Fraction(str(install_bpc)))
            install_start = now
            now += iw
            install += iw
            pending = None  # Buffer becomes free only after IW has consumed it.
            change_index += 1
        else:
            install_start = now

        start = now
        duration = tile["first_row_cycles"] + (tile["rows"] - 1) * tile["row_interval_cycles"]
        row_rates = {key: Fraction(size, tile["row_interval_cycles"])
                     for key, size in tile["row_payload_bytes"].items()}
        if any(rate > caps[key] for key, rate in row_rates.items()):
            raise ValueError("TSP row service exceeds resource capacity")
        for key, value in row_rates.items():
            peaks[key] = max(peaks[key], value)
        if pending is None and change_index < len(changes):
            target = changes[change_index]
            if tiles[target]["weight_bytes"] > buffer_bytes:
                raise ValueError("unsupported_capacity: pending TSP weights exceed weight_buffer_bytes")
            pending = {"index": target, "remaining": Fraction(1)}
        # One future weight block can span several M tiles that reuse current B.
        residual = {key: value - row_rates.get(key, 0) for key, value in caps.items()}
        load_window(start, duration, residual, True, row_rates)
        now += duration
        rows += duration
        timeline.append({**tile, "install_start": install_start, "install_end": start,
                         "pipeline_start": start, "pipeline_end": now,
                         "weight_load_stall_cycles": install_start - arrival,
                         "first_row_write_cycle": start + tile["first_row_cycles"],
                         "row_reserved_bpc": {key: float(value) for key, value in row_rates.items()}})
    for index, tile in enumerate(timeline):
        if index in loads:
            tile["weight_load"] = loads[index]
    return {"total_cycles": now, "install_cycles": install, "row_service_cycles": rows,
            "weight_load_cycles": load_cycles, "weight_load_overlap_cycles": overlap,
            "weight_load_stall_cycles": stall,
            "elapsed_stages": {"weight_load_stall": stall, "install": install, "row_service": rows},
            "first_row_write_cycle": timeline[0]["first_row_write_cycle"],
            "resources": {key: {"capacity_bpc": float(value), "peak_reserved_bpc": float(peaks[key])}
                          for key, value in caps.items()},
            "schedule": {"tiles": timeline, "weight_load_segments": segments,
                         "pending_weight_slots": 1, "weight_buffer_bytes": buffer_bytes,
                         "peak_pending_weight_bytes": max(tile["weight_bytes"] for tile in tiles)}}
