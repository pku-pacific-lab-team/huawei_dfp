"""Native kernel output, independent of public operator/report adapters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    component: str
    name: str
    category: str


@dataclass(frozen=True)
class NativeResult:
    report: dict
    action_specs: dict[str, ActionSpec]
    action_description: str
