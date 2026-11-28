"""Load the four architecture profiles and describe their chip resource shares."""
from ..runtime.backends import BACKENDS, get_backend
from ..utils.ascend_config import AscendConfigLoader
from ..utils.tpu_config import TPUConfigLoader
from ..utils.tsp_config import TSPConfigLoader
from ..utils.tsp_cim_config import TSPCIMConfigLoader


def load_hardware(architecture):
    if architecture not in BACKENDS:
        raise ValueError(f"Unsupported architecture: {architecture}")
    if architecture == "ascend":
        return AscendConfigLoader().load_hardware()
    if architecture == "tpu":
        return TPUConfigLoader().load_hardware()
    loader = TSPCIMConfigLoader() if architecture == "tsp_cim" else TSPConfigLoader()
    return loader.create_tsp_spec(loader.load_hardware_config())


def chip_resources(architecture, spec):
    """Rates in this report describe the allocation already used inside GEMM."""
    get_backend(architecture, spec)
    if architecture == "ascend":
        unit = "ai_core"
        units = spec.architectural_cores
        rates = {"hbm_bytes_per_second": spec.chip["hbm_bandwidth_bytes_per_second"],
                 "l2_bytes_per_cycle": spec.chip["l2_bytes_per_cycle"]}
    elif architecture == "tpu":
        unit = "tensorcore"
        units = spec.values["tensorcores_per_chip"]
        rates = {f"{name}_bytes_per_second": spec.values[f"{name}_bandwidth_bytes_per_second"]
                 for name in ("hbm", "cmem_read", "cmem_write")}
    else:
        unit = "hemisphere"
        native = spec.native_reference_spec if architecture == "tsp_cim" else spec
        units = native.n_hemispheres
        rates = {}
        for direction in ("read", "write"):
            rate = native.values[f"mem_{direction}_bpc"]
            rates[f"mem_{direction}_bytes_per_cycle"] = None if rate is None else rate * native.mem_n_slices
    if type(units) is not int or units < 1:
        raise ValueError("Missing hardware execution unit count")
    return {"unit": unit, "units_per_chip": units, "chip_bandwidth": rates,
            "bandwidth_per_unit": {k: None if v is None else v / units for k, v in rates.items()}}
