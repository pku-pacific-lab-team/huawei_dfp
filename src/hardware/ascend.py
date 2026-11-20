"""Auditable original-910 specifications for a coarse analytical GEMM entry."""
from .execution import read_execution
from .documentation import without_documentation
# 规格同时保存数值与来源。容量供分块使用，路径速率供可选 timing 使用。
from dataclasses import asdict, dataclass
from fractions import Fraction
import math


BUFFERS = ("L1", "L0A", "L0B", "L0C", "UB")
LOCAL_ROUTES = ("l1_l0a", "l1_l0b", "l0c_ub")
ROUTES = ("hbm_to_l2", "l2_fill", "l2_to_l1", "hbm_to_l1", *LOCAL_ROUTES, "ub_to_hbm")
VECTOR_OPERATION_COSTS = {
    "int32_add": "int32_add_cost",
    "fp32_add": "fp32_basic_op_cost",
    "fp32_mul": "fp32_basic_op_cost",
    "fp32_exp": "fp32_exp_cost",
    "fp32_rsqrt": "fp32_rsqrt_cost",
}
VECTOR_SERVICE_PARAMETERS = ("base_ops_per_cycle", *dict.fromkeys(VECTOR_OPERATION_COSTS.values()))


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
        config = without_documentation(config)
        if type(config.get("schema_version")) is not int or config["schema_version"] != 9 or config.get("model") != "ascend910_analytical":
            raise ValueError("Expected schema 9, model ascend910_analytical; archive cube.units_per_core and vector.units_per_core from schema 8")
        self.execution = read_execution(config.get("execution"), "ascend")
        require_fields(config, ("execution", "schema_version", "model", "description", "clock_freq_ghz",
                                "memory", "cube", "vector", "transfers", "output_path", "chip"), "hardware")
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
        require_fields(cube, ("m", "k", "n", "macs_per_cycle"), "cube")
        for key, expected in (("m", 16), ("k", 32), ("n", 16)):
            if read(cube[key], "element", "core", f"cube.{key}", integer=True, required=True) != expected:
                raise ValueError("INT8 reference tiling requires a 16x32x16 fractal")
        # 16×32×16 只定义计算形状，不能由此认定一个分形恰好在一周期完成。
        self.macs_per_cycle = read(cube["macs_per_cycle"], "MAC/cycle", "core", "cube.macs_per_cycle")
        # 固定服务独立于已归档的单元库存；特殊函数共享 Vector 预算。
        vector = config["vector"]
        require_fields(vector, VECTOR_SERVICE_PARAMETERS, "vector")
        self.vector_parameters = {
            name: read(vector[name], "op/cycle" if name == "base_ops_per_cycle" else "equivalent_op/op",
                       "core", "vector." + name) for name in VECTOR_SERVICE_PARAMETERS}
        require_fields(config["transfers"], LOCAL_ROUTES, "transfers")
        self.local_routes = {route: read(config["transfers"][route], "byte/cycle", "core", f"transfers.{route}") for route in LOCAL_ROUTES}
        self.output_path = config["output_path"]
        require_fields(self.output_path, ("status", "source", "description"), "output_path")
        if self.output_path["status"] not in {"assumed", "documented", "measured"} or not self.output_path["source"]:
            raise ValueError("INT32 export path requires explicit provenance")
        chip = config["chip"]
        chip_fields = {"architectural_cores": ("core", True), "l2_bytes": ("byte", True),
                       "hbm_physical_bytes": ("byte", True), "hbm_usable_bytes": ("byte", True),
                       "l2_bytes_per_cycle": ("byte/cycle", False),
                       "hbm_bandwidth_bytes_per_second": ("byte/s", False)}
        require_fields(chip, chip_fields, "chip")
        self.chip = {key: read(chip[key], unit, "chip", f"chip.{key}", integer=integer,
                              required=key == "architectural_cores") for key, (unit, integer) in chip_fields.items()}
        self.architectural_cores = self.chip["architectural_cores"]
        physical, usable = self.chip["hbm_physical_bytes"], self.chip["hbm_usable_bytes"]
        if usable is not None and (physical is None or usable > physical):
            raise ValueError("HBM usable capacity requires physical capacity and cannot exceed it")

    def require_count_capacities(self):
        missing = ["chip." + k for k in ("hbm_usable_bytes", "l2_bytes") if self.chip[k] is None]
        if missing:
            raise ValueError("Counting parameters missing: " + ", ".join(missing))
        if self.l2_capacity() <= 0:
            raise ValueError("L2 per-core capacity must be positive after equal partition")

    def timing_parameter_names(self):
        return ["cube.macs_per_cycle", *[f"transfers.{r}" for r in LOCAL_ROUTES],
                "chip.hbm_bandwidth_bytes_per_second", "chip.l2_bytes_per_cycle",
                "chip.l2_bytes", "chip.architectural_cores", "clock_freq_ghz"]

    def missing_timing(self):
        # HBM 使用 byte/s，计周期也需要时钟；count 仍可独立运行。
        return [name for name in self.timing_parameter_names() if self.parameters[name].value is None]

    def require_timing(self):
        missing = self.missing_timing()
        if missing:
            raise ValueError("Timing parameters missing: " + ", ".join(missing))

    def hbm_service(self):
        bandwidth = self.chip["hbm_bandwidth_bytes_per_second"]
        share = Fraction(str(bandwidth)) / self.architectural_cores if bandwidth is not None else None
        bpc = share / (Fraction(str(self.clock_ghz)) * 10**9) if share is not None and self.clock_ghz is not None else None
        return share, bpc

    def route_services(self):
        _, bpc = self.hbm_service()
        l2 = self.l2_service()
        return {"hbm_to_l2": bpc, "l2_fill": l2, "l2_to_l1": l2,
                "hbm_to_l1": bpc, **self.local_routes, "ub_to_hbm": bpc}

    def l2_capacity(self):
        capacity = self.chip["l2_bytes"]
        return capacity // self.architectural_cores if capacity is not None else None

    def l2_service(self):
        rate = self.chip["l2_bytes_per_cycle"]
        return Fraction(str(rate)) / self.architectural_cores if rate is not None else None

    def vector_service(self):
        """One core's shared compute budget, independent of inventory counts."""
        v = self.vector_parameters
        base = v["base_ops_per_cycle"]
        return {
            "scope": "single_core", "status": "analytical_compute_service",
            "resource_policy": "one_shared_vector_service",
            "base_ops_per_cycle": base,
            "operation_costs": {op: v[name] for op, name in VECTOR_OPERATION_COSTS.items()},
            "operation_rates_per_cycle": {
                op: None if base is None or v[name] is None else float(Fraction(str(base)) / Fraction(str(v[name])))
                for op, name in VECTOR_OPERATION_COSTS.items()},
            "gemm_operations": [], "nonlinear_included_in_gemm": False,
            "special_function_model": "algorithmic_work_proxy_not_native_instruction_timing",
            "assumed_inputs": ["vector." + name for name in VECTOR_SERVICE_PARAMETERS
                               if self.parameters["vector." + name].status == "assumed"],
            "missing_parameters": ["vector." + name for name in VECTOR_SERVICE_PARAMETERS if v[name] is None],
            "conditions": "Standalone compute occupancy, not full operator latency. Rates share one core budget and cannot be added. "
                          "No transfers, reductions, dependency scheduling, vector tail/issue overhead or special-function startup. "
                          "FP32 denotes operand format, not correctly rounded exp/rsqrt accuracy. "
                          "GEMM accumulates in Cube/L0C and does not invoke this service. "
                          "Unit/core inventory changes do not automatically scale the configured budget.",
        }

    def vector_service_cycles(self, operation_counts):
        """TSP-style weighted work with one final rounding; compute occupancy only."""
        if not isinstance(operation_counts, dict):
            raise ValueError("Vector operation counts must be a dictionary")
        unknown = set(operation_counts) - VECTOR_OPERATION_COSTS.keys()
        if unknown:
            raise ValueError("Unsupported Vector operations: " + ", ".join(sorted(map(str, unknown))))
        if any(type(count) is not int or count < 0 for count in operation_counts.values()):
            raise ValueError("Vector operation counts must be nonnegative integers")
        active = {op: count for op, count in operation_counts.items() if count}
        if not active:
            return 0
        need = ("base_ops_per_cycle", *dict.fromkeys(VECTOR_OPERATION_COSTS[op] for op in active))
        missing = ["vector." + name for name in need if self.vector_parameters[name] is None]
        if missing:
            raise ValueError("Vector service parameters missing: " + ", ".join(missing))
        work = sum(count * Fraction(str(self.vector_parameters[VECTOR_OPERATION_COSTS[op]]))
                   for op, count in active.items())
        return math.ceil(work / Fraction(str(self.vector_parameters["base_ops_per_cycle"])))

    def derived(self):
        """Nominal specification arithmetic, never a chip execution prediction."""
        core_ops = 2 * self.macs_per_cycle if self.macs_per_cycle is not None else None
        core_tops = core_ops * self.clock_ghz / 1000 if core_ops is not None and self.clock_ghz is not None else None
        share, _ = self.hbm_service()
        bandwidths = {r: bw * self.clock_ghz * 8 if bw is not None and self.clock_ghz is not None else None
                      for r, bw in self.local_routes.items()}
        bandwidths.update({r: float(share * 8 / 10**9) if share is not None else None
                           for r in ("hbm_to_l2", "hbm_to_l1", "ub_to_hbm")})
        l2 = self.l2_service()
        bandwidths.update({r: float(l2 * self.clock_ghz * 8) if l2 is not None and self.clock_ghz is not None else None
                           for r in ("l2_fill", "l2_to_l1")})
        return {
            "mac_ops": 2,
            "int8_fractal_macs": 16 * 32 * 16,
            "core_int8_ops_per_cycle": core_ops,
            "core_int8_peak_tops": core_tops,
            "chip_int8_peak_tops": core_tops * self.architectural_cores if core_tops is not None else None,
            "route_bandwidth_gbps": bandwidths,
            "vector_service": self.vector_service(),
            "area_inventory": {
                # 核内物理资源乘核数，共享 L2/片外 HBM 各计一份；库存尚不是物理面积。
                "scope": "chip", "core_count": self.architectural_cores,
                "core_sram_bytes_by_level": {k: v * self.architectural_cores for k, v in self.physical_capacities.items()},
                "shared_l2_bytes": self.chip["l2_bytes"], "external_hbm_bytes": self.chip["hbm_physical_bytes"],
                "area_mm2": None,
            },
            "status": "derived_from_declared_parameters; assumed_inputs_remain_assumed",
        }

    def provenance(self):
        return {name: asdict(p) for name, p in self.parameters.items()}
