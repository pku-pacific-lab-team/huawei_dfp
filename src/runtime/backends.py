"""One backend registry shared by GEMM, Vector and normalization execution."""
from dataclasses import dataclass
from typing import Callable

from ..hardware.ascend import AscendSpec
from ..hardware.tpu import TPUSpec
from ..hardware.tsp import TSPSpec
from ..hardware.tsp_cim import TSPCIMSpec
from ..core.ascend.simulator import AscendSimulator
from ..core.tpu.simulator import TPUSimulator
from ..core.tsp.simulator import TSPSimulator
from ..core.tsp_cim.simulator import TSPCIMSimulator
from ..core.ascend.vector import simulate as ascend_vector
from ..core.tpu.vector import simulate as tpu_vector
from ..core.tsp.vector import simulate as tsp_vector
from ..core.tsp_cim.vector import simulate as cim_vector
from ..core.ascend.composite import make_path as ascend_path
from ..core.tpu.composite import make_path as tpu_path
from ..core.tsp.composite import make_path as tsp_path
from ..core.tsp_cim.composite import make_path as cim_path


@dataclass(frozen=True)
class Backend:
    spec_type: type
    gemm_simulator: type
    gemm_uses_shape: bool
    vector_kernel: Callable
    composite_path: Callable


BACKENDS = {
    "ascend": Backend(AscendSpec, AscendSimulator, False, ascend_vector, ascend_path),
    "tpu": Backend(TPUSpec, TPUSimulator, False, tpu_vector, tpu_path),
    "tsp": Backend(TSPSpec, TSPSimulator, True, tsp_vector, tsp_path),
    "tsp_cim": Backend(TSPCIMSpec, TSPCIMSimulator, True, cim_vector, cim_path),
}


def get_backend(architecture, spec):
    if not isinstance(architecture, str) or architecture not in BACKENDS:
        raise ValueError("Unsupported architecture")
    backend = BACKENDS[architecture]
    if not isinstance(spec, backend.spec_type):
        raise ValueError("Hardware Spec does not match architecture")
    return backend
