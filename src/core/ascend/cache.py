"""Cold, byte-capacity LRU of complete operand panels, not hardware cache lines.

The caller serializes input requests in program order. Inputs are immutable;
outputs bypass this cache, so evictions are clean and need no writeback.
"""
from collections import OrderedDict


class PanelCache:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("L2 per-core capacity must be a positive integer")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.used = self.peak = 0
        self.stats = dict(requests=0, hits=0, misses=0, requested_bytes=0,
                          hit_bytes=0, miss_bytes=0, fill_bytes=0, read_bytes=0,
                          bypass_requests=0, bypass_bytes=0, evictions=0)

    def access(self, key, size):
        if type(size) is not int or size <= 0:
            raise ValueError("L2 panel size must be a positive integer")
        s = self.stats
        s["requests"] += 1
        s["requested_bytes"] += size
        if key in self.entries:
            if self.entries[key] != size:
                raise ValueError("L2 panel identity must include its shape")
            self.entries.move_to_end(key)
            s["hits"] += 1
            s["hit_bytes"] += size
            s["read_bytes"] += size
            return "hit"
        s["misses"] += 1
        s["miss_bytes"] += size
        if size > self.capacity:
            s["bypass_requests"] += 1
            s["bypass_bytes"] += size
            return "bypass"
        while self.used + size > self.capacity:
            _, evicted_size = self.entries.popitem(last=False)
            self.used -= evicted_size
            s["evictions"] += 1
        self.entries[key] = size
        self.used += size
        self.peak = max(self.peak, self.used)
        s["fill_bytes"] += size
        s["read_bytes"] += size
        return "miss"

    def summary(self):
        s = self.stats
        return {**s, "resident_bytes": self.used, "peak_resident_bytes": self.peak,
                "resident_panels": len(self.entries),
                "request_hit_rate": s["hits"] / s["requests"] if s["requests"] else 0,
                "byte_hit_rate": s["hit_bytes"] / s["requested_bytes"] if s["requested_bytes"] else 0}
