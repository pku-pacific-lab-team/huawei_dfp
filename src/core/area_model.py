"""First-order area model for TSP and TSP-CIM inventories."""

from typing import Dict, Optional


class AreaModel:
    """Convert hardware inventory counts into a coarse area breakdown."""

    def __init__(self, area_params: Dict[str, float]):
        self.area_params = area_params

    def _param(self, name: str, default: float = 0.0) -> float:
        return float(self.area_params.get(name, default))

    def evaluate(
        self,
        inventory: Dict[str, float],
        throughput_gops: Optional[float] = None,
    ) -> Dict[str, float]:
        global_sram_mm2 = inventory.get("global_mem_sram_mib", 0.0) / max(
            1e-9,
            self._param("global_sram_density_mib_per_mm2", 1.0),
        )
        local_sram_mm2 = inventory.get("local_sram_mib", 0.0) / max(
            1e-9,
            self._param("local_sram_density_mib_per_mm2", 1.0),
        )

        compute_area_mm2 = (
            inventory.get("mxm_planes", 0.0) * self._param("mxm_plane_area_mm2") +
            inventory.get("cim_macros", 0.0) * self._param("cim_macro_area_mm2") +
            inventory.get("vxm_alus", 0.0) * self._param("vxm_alu_area_mm2") +
            inventory.get("mem_slices", 0.0) * self._param("mem_slice_ctrl_area_mm2") +
            inventory.get("sxm_units", 0.0) * self._param("sxm_unit_area_mm2")
        )

        fabric_area_mm2 = (
            inventory.get("stream_endpoints", 0.0) * self._param("stream_endpoint_area_mm2") +
            inventory.get("fabric_domains", 0.0) * self._param("fabric_domain_area_mm2")
        )

        clock_control_area_mm2 = (
            inventory.get("clock_domains", 0.0) * self._param("clock_domain_area_mm2") +
            inventory.get("control_domains", 0.0) * self._param("control_domain_area_mm2")
        )

        subtotal_mm2 = (
            global_sram_mm2 +
            local_sram_mm2 +
            compute_area_mm2 +
            fabric_area_mm2 +
            clock_control_area_mm2
        )

        overhead_area_mm2 = subtotal_mm2 * self._param(
            "routing_overhead_factor",
            inventory.get("routing_complexity_factor", 0.0),
        )
        total_area_mm2 = subtotal_mm2 + overhead_area_mm2

        throughput_tops = (throughput_gops or 0.0) / 1000.0
        return {
            "global_sram_mm2": global_sram_mm2,
            "local_sram_mm2": local_sram_mm2,
            "compute_area_mm2": compute_area_mm2,
            "fabric_area_mm2": fabric_area_mm2,
            "clock_control_area_mm2": clock_control_area_mm2,
            "overhead_area_mm2": overhead_area_mm2,
            "total_area_mm2": total_area_mm2,
            "tops_per_mm2": (
                throughput_tops / total_area_mm2
                if total_area_mm2 > 0.0 else 0.0
            ),
            "mib_per_mm2": (
                (inventory.get("global_mem_sram_mib", 0.0) + inventory.get("local_sram_mib", 0.0)) /
                total_area_mm2 if total_area_mm2 > 0.0 else 0.0
            ),
        }
