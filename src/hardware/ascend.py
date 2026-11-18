"""Auditable original-910 specifications for a coarse analytical GEMM entry."""
# 规格同时保存数值与来源。容量供分块使用，路径速率供可选 timing 使用。
from dataclasses import asdict, dataclass
import math


BUFFERS = ("L1", "L0A", "L0B", "L0C", "UB")
ROUTES = ("gm_l1", "l1_l0a", "l1_l0b", "l0c_ub", "ub_gm")


def require_fields(raw, keys, name):
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ValueError(f"{name}: expected fields {sorted(keys)}")


@dataclass(frozen=True)
class Parameter:
    value: float | int | None
    unit: str
    scope: str
    status: str
    source: str
    version: str
    conditions: str

    @classmethod
    def read(cls, raw, unit, scope, name, *, integer=False, required=False):
        require_fields(raw, cls.__dataclass_fields__, name)
        p = cls(**raw)
        if p.unit != unit or p.scope != scope or p.status not in {"documented", "measured", "assumed", "unknown"}:
            raise ValueError(f"{name}: invalid unit, scope or provenance status")
        if any(not isinstance(v, str) or not v.strip() for v in (p.source, p.version, p.conditions)):
            raise ValueError(f"{name}: source, version and conditions are required")
        if p.value is None:
            if required or p.status != "unknown":
                raise ValueError(f"{name}: missing required value or inconsistent status")
        elif (p.status == "unknown" or isinstance(p.value, bool) or not isinstance(p.value, (int, float))
              or not math.isfinite(p.value) or p.value <= 0 or (integer and type(p.value) is not int)):
            raise ValueError(f"{name}: expected positive finite {'integer' if integer else 'number'}")
        return p


class AscendSpec:
    def __init__(self, config):
        require_fields(config, ("schema_version", "model", "description", "clock_freq_ghz",
                                "memory", "cube", "vector", "sfu", "transfers", "output_path", "chip"), "hardware")
        if type(config["schema_version"]) is not int or config["schema_version"] != 3 or config["model"] != "ascend910_analytical":
            raise ValueError("Expected schema 3, model ascend910_analytical")
        self.config = config
        self.parameters = {}

        def read(raw, unit, scope, name, **kw):
            p = Parameter.read(raw, unit, scope, name, **kw)
            self.parameters[name] = p
            return p.value

        self.clock_ghz = read(config["clock_freq_ghz"], "GHz", "core", "clock_freq_ghz")
        # usable 约束当前分块，physical 用于硬件库存；两种容量不能相互替代。
        self.capacities, self.physical_capacities = {}, {}
        require_fields(config["memory"], BUFFERS, "memory")
        for name in BUFFERS:
            memory = config["memory"][name]
            require_fields(memory, ("physical_bytes", "usable_bytes"), f"memory.{name}")
            physical = read(memory["physical_bytes"], "byte", "core", f"memory.{name}.physical_bytes", integer=True, required=True)
            usable = read(memory["usable_bytes"], "byte", "core", f"memory.{name}.usable_bytes", integer=True, required=True)
            if usable > physical:
                raise ValueError(f"{name}: usable capacity exceeds physical capacity")
            self.physical_capacities[name], self.capacities[name] = physical, usable
        cube = config["cube"]
        require_fields(cube, ("m", "k", "n", "units_per_core", "macs_per_cycle"), "cube")
        for key, expected in (("m", 16), ("k", 32), ("n", 16)):
            if read(cube[key], "element", "core", f"cube.{key}", integer=True, required=True) != expected:
                raise ValueError("INT8 reference tiling requires a 16x32x16 fractal")
        self.cube_units = read(cube["units_per_core"], "unit", "core", "cube.units_per_core", integer=True, required=True)
        # 16×32×16 只定义计算形状，不能由此认定一个分形恰好在一周期完成。
        self.macs_per_cycle = read(cube["macs_per_cycle"], "MAC/cycle", "core", "cube.macs_per_cycle")
        self.compute_units = {"cube": self.cube_units}
        for name in ("vector", "sfu"):
            block = config[name]
            require_fields(block, ("units_per_core", "fp32_ops_per_cycle"), name)
            self.compute_units[name] = read(block["units_per_core"], "unit", "core", f"{name}.units_per_core", integer=True)
            read(block["fp32_ops_per_cycle"], "op/cycle", "core", f"{name}.fp32_ops_per_cycle")
        require_fields(config["transfers"], ROUTES, "transfers")
        # 单核端到端路径速率独立声明，不把下面 chip 的 HBM/L2 峰值直接填入核内路径。
        self.routes = {route: read(config["transfers"][route], "byte/cycle", "core", f"transfers.{route}") for route in ROUTES}
        self.output_path = config["output_path"]
        require_fields(self.output_path, ("status", "source", "description"), "output_path")
        if self.output_path["status"] not in {"assumed", "documented", "measured"} or not self.output_path["source"]:
            raise ValueError("INT32 export path requires explicit provenance")
        chip = config["chip"]
        chip_fields = {"architectural_cores": ("core", True), "l2_bytes": ("byte", True),
                       "hbm_bytes": ("byte", True), "l2_bytes_per_cycle": ("byte/cycle", False),
                       "hbm_bytes_per_cycle": ("byte/cycle", False), "noc_bytes_per_cycle": ("byte/cycle", False),
                       "interconnect_bytes_per_second": ("byte/s", False)}
        require_fields(chip, chip_fields, "chip")
        self.chip = {key: read(chip[key], unit, "chip", f"chip.{key}", integer=integer,
                              required=key == "architectural_cores") for key, (unit, integer) in chip_fields.items()}
        self.architectural_cores = self.chip["architectural_cores"]

    def missing_timing(self):
        # 周期模型需要 Cube 和五条路径的服务率；时钟缺失只会使 ns 换算不可用。
        names = ["cube.macs_per_cycle", *[f"transfers.{r}" for r in ROUTES]]
        return [name for name in names if self.parameters[name].value is None]

    def require_timing(self):
        missing = self.missing_timing()
        if missing:
            raise ValueError("Timing parameters missing: " + ", ".join(missing))

    def derived(self):
        """Nominal specification arithmetic, never a chip execution prediction."""
        core_ops = 2 * self.macs_per_cycle if self.macs_per_cycle is not None else None
        core_tops = core_ops * self.clock_ghz / 1000 if core_ops is not None and self.clock_ghz is not None else None
        return {
            "mac_ops": 2,
            "int8_fractal_macs": 16 * 32 * 16,
            "core_int8_ops_per_cycle": core_ops,
            "core_int8_peak_tops": core_tops,
            "chip_int8_peak_tops": core_tops * self.architectural_cores if core_tops is not None else None,
            "vector_fp32_peak_gops_per_core": self._gops("vector"),
            "sfu_fp32_peak_gops_per_core": self._gops("sfu"),
            "route_bandwidth_gbps": {r: bw * self.clock_ghz * 8 if bw is not None and self.clock_ghz is not None else None
                                     for r, bw in self.routes.items()},
            "area_inventory": {
                # 核内物理资源乘核数，共享 L2/片外 HBM 各计一份；库存尚不是物理面积。
                "scope": "chip", "core_count": self.architectural_cores,
                "compute_units": {k: v * self.architectural_cores if v is not None else None for k, v in self.compute_units.items()},
                "core_sram_bytes_by_level": {k: v * self.architectural_cores for k, v in self.physical_capacities.items()},
                "shared_l2_bytes": self.chip["l2_bytes"], "external_hbm_bytes": self.chip["hbm_bytes"],
                "area_mm2": None,
            },
            "status": "derived_from_declared_parameters; assumed_inputs_remain_assumed",
        }

    def _gops(self, unit):
        value = self.parameters[f"{unit}.fp32_ops_per_cycle"].value
        return value * self.clock_ghz if value is not None and self.clock_ghz is not None else None

    def provenance(self):
        return {name: asdict(p) for name, p in self.parameters.items()}
