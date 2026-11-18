"""The only native reference runner used by standalone and CIM comparison."""
# run_reference 负责从文件装配规格，simulate_reference 负责使用已装配规格运行同一内核。
from ..core.tsp.simulator import TSPSimulator
from .tsp_config import TSPConfigLoader


def run_reference(workload, *, hardware="tsp.json", config_dir=None, mode="count", loop_order="KNM", **policy):
    loader = TSPConfigLoader(config_dir)
    spec = loader.create_tsp_spec(loader.load_hardware_config(hardware))
    return simulate_reference(spec, workload, mode=mode, loop_order=loop_order, **policy)


def simulate_reference(spec, workload, *, mode="count", loop_order="KNM", **policy):
    if workload.accumulation_dtype != "INT32" or workload.output_dtype != "INT32":
        raise ValueError("Native reference requires INT32 accumulation/output")
    result = TSPSimulator(spec, data_format=workload.data_format).simulate(
        *workload.shape, loop_order=loop_order, mode=mode, **policy)
    # 补回声明精度和覆盖标记，避免内核仅由 M/K/N 构造的任务摘要丢失这些信息。
    result["workload"] = workload.to_dict()
    return result


def compare_native(spec, workload, *, loop_order="KNM"):
    # 参考计数有助于核对工作量，但两边执行边界和时序未闭合，不能据此填加速比。
    # Macro migration does not provide a calibrated system timing comparison.
    try:
        reference = simulate_reference(spec, workload, loop_order=loop_order)
    except ValueError as exc:
        return {"status": "not_comparable", "native_reference": None, "speedup": None,
                "energy_ratio": None, "reasons": [str(exc)]}
    return {"status": "not_comparable", "native_reference": reference, "speedup": None,
            "energy_ratio": None, "reasons": [
                "CIM and native reference do not share calibrated execution timing.",
                "Native timing coefficients and energy are not calibrated; reference comparison uses count mode."]}
