"""Architecture-independent GEMM workload parsing (M, K, N order)."""
# 公共任务只描述 A[M,K]×B[K,N]→C[M,N] 及数据类型，各架构再判断自己能否执行。
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from ..hardware.dataformat import KNOWN_FORMATS


def resolve_config_path(filename, config_dir, category):
    """Accept an absolute path, an existing cwd-relative path, or a config name."""
    path = Path(filename)
    if path.is_absolute() or path.is_file():
        return path.resolve()
    return (Path(config_dir) / category / path).resolve()


def load_workload_config(filename, config_dir):
    path = resolve_config_path(filename, config_dir, "workload")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class Workload:
    M: int
    K: int
    N: int
    data_format: str
    accumulation_dtype: str
    output_dtype: str
    declared_data_format: str
    format_overridden: bool

    def __post_init__(self):
        if any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise ValueError("GEMM M, K, N must be positive integers")
        if self.data_format not in KNOWN_FORMATS:
            raise ValueError(f"Unknown data format: {self.data_format}")

    @property
    def shape(self):
        return self.M, self.K, self.N

    def to_dict(self):
        return {"operation": "gemm", **asdict(self), "useful_ops": 2 * self.M * self.K * self.N}


def parse_workloads(config, formats=None):
    if not isinstance(config, dict) or config.get("operation", "gemm") != "gemm":
        raise ValueError("Only a GEMM workload object is supported")
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported workload schema_version")
    unknown = set(config) - {"schema_version", "operation", "description", "gemm", "data_format",
                             "accumulation_dtype", "output_dtype", "custom_overheads"}
    if unknown:
        raise ValueError(f"Unsupported workload fields: {sorted(unknown)}")
    if config.get("custom_overheads") is not None:
        raise ValueError("custom_overheads is architecture-specific; remove it from the workload")
    dims = config.get("gemm")
    if not isinstance(dims, dict) or set(dims) != {"M", "K", "N"}:
        raise ValueError("gemm must specify exactly M, K, N")
    if any(type(dims[x]) is not int or dims[x] <= 0 for x in ("M", "K", "N")):
        raise ValueError("GEMM M, K, N must be positive integers")
    declared = config.get("data_format")
    # 同时保留文件声明与实际选择的精度；只有调用方传入 formats 才做显式覆盖。
    if not isinstance(declared, str) or declared not in KNOWN_FORMATS:
        raise ValueError(f"Unknown or missing workload data_format: {declared!r}")
    selected = [declared] if formats is None else ([formats] if isinstance(formats, str) else list(formats))
    if not selected:
        raise ValueError("At least one data format is required")
    result = []
    # 去重但保留用户给出的先后顺序，保证多精度任务的输出顺序稳定。
    for fmt in dict.fromkeys(selected):
        if fmt not in KNOWN_FORMATS:
            raise ValueError(f"Unknown data format: {fmt}")
        accumulator = "INT32" if fmt in {"INT8", "INT16"} else "FP32"
        # 公共解析器识别的格式可能多于某个仿真器支持的格式；后者在执行入口再次检查。
        # Legacy models only expose a 4-byte result. Reject richer requests rather than ignore them.
        if config.get("accumulation_dtype", accumulator) != accumulator:
            raise ValueError(f"{fmt} requires accumulation_dtype={accumulator} in this model")
        if config.get("output_dtype", accumulator) != accumulator:
            raise ValueError(f"{fmt} requires output_dtype={accumulator}; conversion is not modeled")
        result.append(Workload(**dims, data_format=fmt, accumulation_dtype=accumulator,
                               output_dtype=accumulator, declared_data_format=declared,
                               format_overridden=formats is not None and fmt != declared))
    return result
