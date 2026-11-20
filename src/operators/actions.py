"""Count-only ledger. Grouped counts are not an energy charging policy."""
from dataclasses import asdict, dataclass

from .types import require_text


CATEGORIES = frozenset({"matrix", "vector", "nonlinear", "onchip_memory", "offchip_memory",
                        "onchip_transfer", "interconnect", "cim_internal"})
BASES = frozenset({"logical", "padded", "modeled_internal"})


@dataclass(frozen=True)
class Action:
    id: str
    architecture: str
    component: str
    name: str
    category: str
    count: int | None
    unit: str
    dtype: str
    scope: str
    basis: str
    source: str
    source_basis: str
    role: str = "action"
    status: str = "available"
    attributes: tuple = ()
    summarizes: tuple = ()

    def __post_init__(self):
        for field in ("id", "architecture", "component", "name", "unit", "dtype", "scope", "source", "source_basis"):
            require_text(getattr(self, field), field)
        if self.category not in CATEGORIES or self.basis not in BASES:
            raise ValueError("Unsupported action category or count basis")
        if self.role not in {"action", "reference", "summary"} or self.status not in {"available", "unavailable"}:
            raise ValueError("Unsupported action role or status")
        if self.status == "available":
            if type(self.count) is not int or self.count < 0:
                raise ValueError("Available action count must be a nonnegative integer")
        elif self.count is not None:
            raise ValueError("Unavailable action count must be None, not zero")
        if not isinstance(self.attributes, (tuple, list)):
            raise ValueError("Action attributes must be key/value pairs")
        attrs = []
        for pair in self.attributes:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Action attributes must be key/value pairs")
            require_text(pair[0], "attribute key")
            require_text(pair[1], "attribute value")
            attrs.append(tuple(pair))
        if len({k for k, _ in attrs}) != len(attrs):
            raise ValueError("Duplicate attribute key")
        object.__setattr__(self, "attributes", tuple(sorted(attrs)))
        if not isinstance(self.summarizes, (tuple, list)) or any(not isinstance(s, str) or not s for s in self.summarizes):
            raise ValueError("summarizes must contain action IDs")
        object.__setattr__(self, "summarizes", tuple(self.summarizes))
        if (self.role == "summary") != bool(self.summarizes):
            raise ValueError("Only summaries must identify their underlying actions")

    def grouping_key(self):
        return (self.architecture, self.component, self.name, self.category, self.unit,
                self.dtype, self.scope, self.basis, self.attributes)

    def to_dict(self):
        result = asdict(self)
        result["attributes"] = dict(self.attributes)
        result["summarizes"] = list(self.summarizes)
        return result


@dataclass(frozen=True)
class ActionLedger:
    entries: tuple

    def __post_init__(self):
        if not isinstance(self.entries, (tuple, list)) or any(not isinstance(a, Action) for a in self.entries):
            raise ValueError("Ledger requires Action entries")
        object.__setattr__(self, "entries", tuple(self.entries))
        indexed = {a.id: a for a in self.entries}
        if len(indexed) != len(self.entries):
            raise ValueError("Duplicate action ID; repeated work needs distinct IDs")
        if len({(a.architecture, a.source) for a in self.entries}) != len(self.entries):
            raise ValueError("Duplicate action source; do not recount the same native field")
        for action in self.entries:
            if action.role != "summary":
                continue
            if len(set(action.summarizes)) != len(action.summarizes):
                raise ValueError("Duplicate summary source")
            targets = [indexed.get(key) for key in action.summarizes]
            if any(t is None or t.role != "action" for t in targets):
                raise ValueError("Summary must reference existing non-summary actions")
            if any(t.unit != action.unit or t.architecture != action.architecture for t in targets):
                raise ValueError("Summary units and architecture must match")
            expected = None if any(t.count is None for t in targets) else sum(t.count for t in targets)
            if action.count != expected:
                raise ValueError("Summary count differs from underlying actions")

    def grouped_counts(self):
        """Sum only identical semantic keys, excluding references and summaries.

        A missing contributor propagates as unavailable. Different components,
        dtypes, attributes, scopes and bases are never combined merely by unit.
        """
        groups = {}
        for action in self.entries:
            if action.role != "action":
                continue
            key = action.grouping_key()
            if key not in groups:
                row = action.to_dict()
                for field in ("id", "source", "source_basis", "role", "summarizes"):
                    row.pop(field)
                row.update(count=0, status="available", action_ids=[])
                groups[key] = row
            row = groups[key]
            row["action_ids"].append(action.id)
            row["count"] = None if row["count"] is None or action.count is None else row["count"] + action.count
            row["status"] = "unavailable" if row["count"] is None else "available"
        return list(groups.values())

    def to_dict(self):
        return {"schema_version": 1, "coverage": "partial_native_actions",
                "entries": [a.to_dict() for a in self.entries],
                "grouped_counts": self.grouped_counts(),
                "semantics": "Count views only. References and summaries are excluded from grouped counts. "
                             "Internal primitives and composite actions require an explicit, non-overlapping coefficient policy before energy evaluation. "
                             "Absent actions are not evidence of zero physical activity."}

    def counts_dict(self):
        """One view per action; roles prevent aliases from being charged twice."""
        fields = ("component", "name", "category", "count", "unit", "dtype", "scope", "basis", "role", "status")
        counts = {}
        for action in self.entries:
            row = {key: getattr(action, key) for key in fields}
            attributes = {key: value for key, value in action.attributes if (key, value) != ("native_action", action.id)}
            if attributes:
                row["attributes"] = attributes
            counts[action.id] = row
        return counts
