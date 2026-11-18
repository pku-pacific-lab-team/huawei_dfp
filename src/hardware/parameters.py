"""Strict, source-preserving parameters for the native TSP reference."""
# TSP、Macro、CIM-TSP 共用字段校验：值、单位、作用域和来源必须一起提供。
import math


FIELDS = {"value", "unit", "scope", "status", "source", "source_version", "conditions", "verification"}


def read_parameter(raw, name, unit, scope, *, integer=False, zero=False, required=True):
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise ValueError(f"{name}: expected {sorted(FIELDS)}")
    if raw["unit"] != unit or raw["scope"] != scope:
        raise ValueError(f"{name}: expected unit={unit}, scope={scope}")
    if raw["status"] not in {"documented", "measured", "assumed", "unknown"}:
        raise ValueError(f"{name}: invalid status")
    for key in ("source", "source_version", "conditions", "verification"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{name}: missing {key}")
    value = raw["value"]
    # None 只能表示允许缺失且标为 unknown 的参数；零值只有调用方明确许可时才合法。
    if value is None:
        if required or raw["status"] != "unknown":
            raise ValueError(f"{name}: missing value or inconsistent status")
        return None
    # Python 的 bool 属于 int 子类，必须单独拒绝，避免把 true 当成容量或单元数量。
    if raw["status"] == "unknown" or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: invalid value/status")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name}: value must be finite and {'nonnegative' if zero else 'positive'}")
    if integer and type(value) is not int:
        raise ValueError(f"{name}: expected integer")
    return value
