import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from src.core.tsp.simulator import TSPSimulator
from src.core.tsp.tiling import TSPTilingEngine
from src.hardware.tsp import TSPSpec
from src.utils.tsp_config import TSPConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader
from src.utils.tsp_reference import run_reference
from src.utils.workload import parse_workloads

ROOT = Path(__file__).resolve().parents[2]


def config_with_timing():
    cfg = TSPConfigLoader().load_hardware_config()
    values = dict(mem_read_bpc=320,mem_write_bpc=320,install_bpc=320,
                  install_propagation_cycles=2,mxm_first_cycles=5,mxm_row_ii=1,row_io_latency_cycles=2,
                  vxm_base_ops_per_cycle=16,vxm_add_latency_cycles=1)
    for name,value in values.items():
        cfg["parameters"][name].update(value=value,status="assumed",source="synthetic unit-test fixture",
            source_version="fixture-v1",verification="not_hardware_measurement",conditions="Hand-computable test only")
    return cfg


class TSPReferenceTest(unittest.TestCase):
    def setUp(self):
        self.cfg = TSPConfigLoader().load_hardware_config()
        self.spec = TSPSpec(self.cfg)

    def test_geometry_capacity_and_modeled_inventory(self):
        self.assertEqual(self.spec.total_lanes,320)
        self.assertEqual(self.spec.total_mxm_planes_chip,4)
        self.assertEqual(self.spec.total_vxm_alus,5120)
        self.assertEqual(self.spec.mem_total_bytes,220*2**20)
        self.assertEqual(self.spec.mem_slice_capacity_bytes,20*128*1024)
        self.assertEqual(self.spec.total_supercells_chip,1600)
        self.assertAlmostEqual(self.spec.peak_tops(),737.28)
        self.assertNotIn("nominal_int8_tops", self.spec.values)
        self.assertEqual(self.spec.stream_bw_per_direction_bytes_per_cycle,10240)
        for name in ("physical_sram_mib", "local_sram_mib", "spare_superlanes"):
            self.assertNotIn(name, self.spec.area_inventory())
        self.assertIn("dependency_verification",self.spec.derived()["geometric_int8_tops_chip"])

    def test_supercell_description_remains_geometry_only(self):
        mesh = self.spec.describe_supercell_usage(256,512,"INT8")
        self.assertEqual(mesh["active_supercell_rows"],16)
        self.assertEqual(mesh["active_supercell_cols_total"],32)
        self.assertEqual(len(mesh["planes"]),2)
        self.assertNotIn("events",mesh)

    def test_schema_and_value_failures(self):
        variants=[]
        cfg=copy.deepcopy(self.cfg); del cfg["parameters"]["clock_ghz"]; variants.append(cfg)
        for name,field,value in [("mem_read_bpc","value",0),("plane_dim","value",float("nan")),
            ("superlanes","value",True),("usable_mem_bytes","unit","MiB"),
            ("planes_per_hemisphere","scope","chip"),("clock_ghz","source_version",""),
            ("usable_mem_bytes","value",440*2**20)]:
            cfg=copy.deepcopy(self.cfg);cfg["parameters"][name][field]=value;variants.append(cfg)
        cfg=copy.deepcopy(self.cfg);cfg["memory_tile_capacity"]="per_bank";variants.append(cfg)
        for cfg in variants:
            with self.subTest(cfg=cfg),self.assertRaises(ValueError): TSPSpec(cfg)

    def test_count_is_useful_without_cycles_or_costs(self):
        r=TSPSimulator(self.spec).simulate(512,256,512)
        self.assertEqual(r["counts"]["useful_ops"],134217728)
        self.assertEqual(r["counts"]["mxm_mac"],67108864)
        self.assertEqual(r["counts"]["weight_install_count"],1)
        self.assertEqual(r["counts"]["vxm_add_ops"],0)
        self.assertIsNone(r["timing"]["total_cycles"])
        self.assertIsNone(r["energy"]["total_pj"])
        self.assertIsNone(r["area"]["total_mm2"])
        self.assertEqual(r["boundary"]["scope"],"single_hem")
        self.assertEqual(TSPSimulator(self.spec).simulate(1,16,16,mode="timing")["timing"]["total_cycles"],44)

    def test_invalid_shapes_scopes_allocations_and_capacity(self):
        sim=TSPSimulator(self.spec)
        for dims in [(0,16,16),(-1,16,16),(True,16,16),(1,0.5,16),(1,32768,16384)]:
            with self.subTest(dims=dims),self.assertRaises(ValueError): sim.simulate(*dims)
        for kwargs in [dict(scope="chip"),dict(mem_slices=[44]),dict(mem_slices=[0,0]),dict(streams=33),
                       dict(tile=(1,1,16)),dict(reserved_bytes=-1),dict(loop_order="bad")]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError): sim.simulate(1,16,16,**kwargs)
        for fmt in ["BF16","FP16","INT16"]:
            with self.assertRaises(ValueError): TSPSimulator(self.spec,data_format=fmt)

    def test_tail_counts_and_all_loop_orders(self):
        for order in ["KNM","KMN","MKN","NKM"]:
            r=TSPSimulator(self.spec).simulate(2,17,17,loop_order=order,tile=(1,16,16))
            c=r["counts"]
            self.assertEqual(c["useful_ops"],1156)
            self.assertEqual(c["mxm_mac"],2048)
            self.assertEqual(c["vxm_add_ops"],64)
            self.assertEqual(c["logical_cross_k_adds"],34)
            self.assertEqual(c["mem_read_psum_bytes"],256)
            self.assertEqual(c["mem_write_psum_bytes"],256)
            self.assertEqual(c["mem_write_final_bytes"],256)
            ranges=list(r["memory"]["ranges"].values())
            self.assertTrue(all(a["end_per_slice"]<=b["start_per_slice"] for a,b in zip(ranges,ranges[1:])))

    def test_k_boundaries_and_weight_reuse(self):
        for k in [319,320,321]:
            for n in [319,320,321,639,640,641]:
                r=TSPSimulator(self.spec).simulate(1,k,n)
                self.assertEqual(r["counts"]["useful_ops"],2*k*n)
                self.assertEqual(r["counts"]["mxm_mac"],math.ceil(k/16)*16*math.ceil(n/16)*16)
        sim=TSPSimulator(self.spec)
        self.assertEqual(sim.simulate(640,640,640,loop_order="KNM")["counts"]["weight_install_count"],2)
        self.assertEqual(sim.simulate(640,640,640,loop_order="MKN")["counts"]["weight_install_count"],4)

    def test_hand_computed_timing_and_no_hemisphere_division(self):
        cfg=config_with_timing()
        sim=TSPSimulator(TSPSpec(cfg))
        # Cold LW=1, IW=3; first row=10; outward stream II=2.
        self.assertEqual(sim.simulate(1,16,16,mode="timing",mem_slices=[0],streams=1)["timing"]["total_cycles"],14)
        self.assertEqual(sim.simulate(3,16,16,mode="timing",mem_slices=[0],streams=1)["timing"]["total_cycles"],18)
        # Outward stream is full: each K pays LW=1 and IW=3; rows=10+13.
        r=sim.simulate(1,32,16,mode="timing",mem_slices=[0],streams=1,tile=(320,16,640))
        self.assertEqual(r["timing"]["total_cycles"],31)
        self.assertEqual(r["counts"]["vxm_add_ops"],16)
        cfg["parameters"]["hemispheres"]["value"]=1
        cfg["parameters"]["usable_mem_bytes"]["value"]=110*2**20
        other=TSPSimulator(TSPSpec(cfg)).simulate(1,16,16,mode="timing",mem_slices=[0],streams=1)
        self.assertEqual(other["timing"]["total_cycles"],14)
        self.assertIsNone(other["timing"]["chip_cycles"])

    def test_shared_supply_and_only_used_parameters_required(self):
        cfg=config_with_timing()
        cfg["parameters"]["vxm_base_ops_per_cycle"].update(value=None,status="unknown")
        cfg["parameters"]["vxm_add_latency_cycles"].update(value=None,status="unknown")
        sim=TSPSimulator(TSPSpec(cfg))
        slow=sim.simulate(32,320,640,mode="timing",mem_slices=[0])
        fast=sim.simulate(32,320,640,mode="timing")
        self.assertGreater(slow["timing"]["total_cycles"],fast["timing"]["total_cycles"])
        self.assertEqual(slow["timing"]["max_row_interval_cycles"],8)  # max(320/320,2560/320).
        with self.assertRaisesRegex(ValueError,"Missing timing parameters:.*vxm_add"):
            sim.simulate(1,321,16,mode="timing")

    def test_fixed_parameters_and_separate_allocated_services(self):
        sim=TSPSimulator(self.spec)
        full=sim.simulate(3,321,640,mode="timing")
        limited=sim.simulate(3,321,640,mode="timing",streams=1,mem_slices=[0])
        self.assertEqual(full["timing"]["parameters"],limited["timing"]["parameters"])
        p=limited["timing"]["parameters"]
        self.assertEqual(p["install_bpc"]["value"],10240)
        self.assertEqual(p["vxm_base_ops_per_cycle"]["value"],320)
        d=limited["timing"]["derived_services"]
        self.assertEqual(d["install_bytes_per_cycle"],10240)
        self.assertEqual(d["stream_bytes_per_cycle"],320)
        self.assertEqual(d["vxm_row_compute_cycles"],{"min":2,"max":2})
        self.assertEqual(full["counts"],limited["counts"])
        self.assertGreater(limited["timing"]["total_cycles"],full["timing"]["total_cycles"])
        self.assertEqual(self.spec.values["install_bpc"],10240)
        # Full outward stream blocks prefetch: LW=640+32; internal IW=39+20.
        # The two row windows still cost 55+79 cycles.
        self.assertEqual(limited["timing"]["elapsed_cycles"]["by_stage"],{"weight_load_stall":672,"install":59,"row_service":134})
        self.assertEqual(limited["timing"]["total_cycles"],865)

    def test_fixed_install_ceiling_and_propagation_are_independent(self):
        cfg=copy.deepcopy(self.cfg)
        param=cfg["parameters"]["install_bpc"]
        param["value"]=160
        sim=TSPSimulator(TSPSpec(cfg))
        r=sim.simulate(1,320,640,mode="timing")
        self.assertEqual(r["timing"]["parameters"]["install_bpc"]["value"],160)
        self.assertEqual(r["timing"]["install_cycles"],19+640*2)
        self.assertEqual(r["timing"]["derived_services"]["install_bytes_per_cycle"],160)
        cfg["parameters"]["install_propagation_cycles"]["value"]=0
        no_delay=TSPSimulator(TSPSpec(cfg)).simulate(1,320,640,mode="timing")
        self.assertEqual(no_delay["timing"]["total_cycles"],r["timing"]["total_cycles"]-19)
        self.assertEqual(no_delay["counts"],r["counts"])
        param.update(value=None,status="unknown")
        sim=TSPSimulator(TSPSpec(cfg))
        self.assertIsNone(sim.simulate(1,16,16)["timing"]["total_cycles"])
        with self.assertRaisesRegex(ValueError,"Missing timing parameters: install_bpc"):
            sim.simulate(1,16,16,mode="timing")

    def test_legacy_dynamic_fields_and_invalid_fixed_latencies_fail(self):
        for version in (2, 3, 4, 5, 6, 7):
            cfg=copy.deepcopy(self.cfg)
            cfg["schema_version"]=version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError,"schema 9"):
                TSPSpec(cfg)
        for old,new in (("install_first_cycles","install_propagation_cycles"),
                        ("vxm_add_first_cycles","vxm_add_latency_cycles")):
            cfg=copy.deepcopy(self.cfg)
            cfg["parameters"][old]=cfg["parameters"].pop(new)
            with self.assertRaisesRegex(ValueError,"Parameter keys mismatch"):
                TSPSpec(cfg)
        cfg=copy.deepcopy(self.cfg)
        cfg["parameters"]["install_bpc"]["derivation"]={"rule":"allocated_stream_bandwidth"}
        with self.assertRaises(ValueError):
            TSPSpec(cfg)
        for name in ("install_propagation_cycles","vxm_add_latency_cycles"):
            for value in (-1,True,0.5):
                cfg=copy.deepcopy(self.cfg)
                cfg["parameters"][name]["value"]=value
                with self.subTest(name=name,value=value), self.assertRaises(ValueError):
                    TSPSpec(cfg)
            cfg=copy.deepcopy(self.cfg)
            cfg["parameters"][name].update(value=None,status="unknown")
            sim=TSPSimulator(TSPSpec(cfg))
            self.assertIsNone(sim.simulate(1,321,16)["timing"]["total_cycles"])
            with self.assertRaisesRegex(ValueError,name):
                sim.simulate(1,321,16,mode="timing")

    def test_retired_parameters_rejected_and_fp32_costs_not_used_by_gemm(self):
        audit = json.loads((ROOT / "calibration/parameter_source_audit.json").read_text(encoding="utf-8"))
        retired = audit["configs"]["configs/hardware/tsp.json"]["retired_parameters"]
        self.assertEqual(len(retired), 10)
        for path, entry in retired.items():
            name = path.removeprefix("parameters.")
            cfg = copy.deepcopy(self.cfg)
            cfg["parameters"][name] = copy.deepcopy(entry["last_configuration"])
            with self.subTest(parameter=name), self.assertRaisesRegex(ValueError, "Parameter keys mismatch"):
                TSPSpec(cfg)
        self.assertFalse(any(v is None for v in self.spec.values.values()))
        baseline = TSPSimulator(self.spec).simulate(2, 321, 641, mode="timing")
        cfg = copy.deepcopy(self.cfg)
        for name in ("vxm_fp32_basic_op_cost", "vxm_fp32_exp_cost", "vxm_fp32_rsqrt_cost"):
            cfg["parameters"][name].update(value=100, status="assumed")
        other = TSPSimulator(TSPSpec(cfg)).simulate(2, 321, 641, mode="timing")
        for key in ("counts", "action_counts", "memory", "timing"):
            self.assertEqual(baseline[key], other[key])
        self.assertEqual(baseline["schema_version"], 10)
        self.assertEqual(baseline["timing"]["total_cycles"], 221)

    def test_vxm_tail_service_and_fixed_pipeline_latency(self):
        sim=TSPSimulator(self.spec)
        # N640 costs 2 compute cycles; the padded N16 tail costs only 1.
        combined=sim.simulate(2,321,641,mode="timing")
        full=sim.simulate(2,321,640,mode="timing")
        tail=sim.simulate(2,321,1,mode="timing")
        self.assertEqual(combined["timing"]["row_service_cycles"],
                         full["timing"]["row_service_cycles"]+tail["timing"]["row_service_cycles"])
        self.assertLess(combined["timing"]["total_cycles"],
                        full["timing"]["total_cycles"]+tail["timing"]["total_cycles"])
        self.assertEqual(combined["timing"]["derived_services"]["vxm_row_compute_cycles"],{"min":1,"max":2})
        cfg=copy.deepcopy(self.cfg)
        cfg["parameters"]["vxm_add_latency_cycles"]["value"]=5
        delayed=TSPSimulator(TSPSpec(cfg)).simulate(2,321,641,mode="timing")
        self.assertEqual(delayed["counts"],combined["counts"])
        self.assertEqual(delayed["timing"]["max_row_interval_cycles"],combined["timing"]["max_row_interval_cycles"])
        self.assertEqual(delayed["timing"]["total_cycles"]-combined["timing"]["total_cycles"],10)
        self.assertEqual(delayed["timing"]["derived_services"]["vxm_row_first_cycles"],{"min":6,"max":7})
        # Pure arithmetic service can be the bottleneck; pipeline latency is not the II.
        cfg["parameters"]["vxm_base_ops_per_cycle"]["value"]=16
        narrow=TSPSimulator(TSPSpec(cfg)).simulate(2,321,640,mode="timing")
        self.assertEqual(narrow["timing"]["max_row_interval_cycles"],40)

    def test_native_comparison_uses_identical_runner(self):
        from src.core.tsp_cim.simulator import TSPCIMSimulator
        loader=TSPCIMConfigLoader()
        spec=loader.create_tsp_spec(loader.load_hardware_config())
        self.assertEqual(spec.mem_total_bytes,self.spec.mem_total_bytes)
        workload=parse_workloads({"gemm":{"M":32,"K":321,"N":32},"data_format":"INT8"})[0]
        cim=TSPCIMSimulator(spec,data_format="INT8")
        comparison=cim.compare_mxm(*workload.shape,workload=workload)
        self.assertEqual(comparison["native_reference"],run_reference(workload,tile=(32,320,640)))
        self.assertIsNone(comparison["speedup"])
        self.assertIsNone(comparison["energy_ratio"])
        self.assertFalse(hasattr(cim,"_calc_mxm_baseline"))
        sweep=cim.sweep_m([1,2],16,16)
        self.assertTrue(all(x["comparison"]["native_reference"]["reference_id"]==self.spec.reference_id for x in sweep))

    def test_cli_count_and_missing_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"result.json"
            common=[sys.executable,str(ROOT/"tsp.py"),"--workload","tiny_gemm.json","--formats","INT8"]
            r=subprocess.run(common+["--mode","count","--output",str(output)],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["mode"],"count")
            r=subprocess.run(common+["--output",str(output)],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["timing"]["total_cycles"],601)
            before=output.read_bytes()
            cfg=copy.deepcopy(self.cfg)
            cfg["parameters"]["mem_read_bpc"].update(value=None,status="unknown")
            hardware=Path(directory)/"missing.json"
            hardware.write_text(json.dumps(cfg),encoding="utf-8")
            r=subprocess.run(common+["--mode","timing","--hardware",str(hardware),"--output",str(output)],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(r.returncode,2)
            self.assertIn("Missing timing",r.stderr)
            self.assertEqual(output.read_bytes(),before)

    def test_proxy_hand_calculation_and_hardware_unknowns_remain(self):
        sim=TSPSimulator(self.spec)
        count=sim.simulate(16,32,16)
        timed=sim.simulate(16,32,16,mode="timing")
        self.assertEqual(timed["timing"]["elapsed_cycles"]["by_stage"],{"weight_load_stall":1,"install":20,"row_service":38})
        self.assertEqual(timed["timing"]["total_cycles"],59)
        self.assertAlmostEqual(timed["timing"]["time_ns"],59/0.9)
        self.assertEqual(len(timed["timing"]["proxy_parameters"]),6)
        for key in ("counts","action_counts","memory","hardware_inventory","hardware_derived","parameter_provenance"):
            self.assertEqual(timed[key],count[key])
        self.assertEqual(timed["timing"]["missing_parameters"],count["timing"]["missing_parameters"])
        self.assertEqual(timed["parameter_provenance"]["mxm_row_ii"]["value"],1)
        self.assertEqual(timed["timing"]["missing_parameters"],[])
        self.assertEqual(timed["timing"]["parameters"]["install_bpc"]["selection"],"configured")

    def test_proxy_full_plane_loading_and_weight_reuse(self):
        sim=TSPSimulator(self.spec)
        r=sim.simulate(1,320,640,mode="timing")
        self.assertEqual(r["timing"]["install_cycles"],39)  # 20 payload issues + 19 wave hops.
        self.assertEqual(r["timing"]["row_service_cycles"],23)
        r=sim.simulate(640,320,640,mode="timing")
        self.assertEqual(r["counts"]["tiles"],2)
        self.assertEqual(r["counts"]["weight_install_count"],1)
        self.assertEqual(r["timing"]["total_cycles"],20+39+2*(23+319))

    def test_proxy_cross_k_stream_budget_and_configured_precedence(self):
        sim=TSPSimulator(self.spec)
        r=sim.simulate(1,321,16,mode="timing")
        self.assertEqual(r["timing"]["total_cycles"],89)  # Cold LW 1 + IW 40 + rows 48; second LW is hidden.
        slow=sim.simulate(3,32,640,mode="timing",tile=(320,16,640),streams=1,mem_slices=[0])
        self.assertEqual(slow["timing"]["parameters"]["vxm_base_ops_per_cycle"]["value"],320)
        self.assertEqual(slow["timing"]["parameters"]["vxm_add_latency_cycles"]["value"],0)
        self.assertEqual(slow["timing"]["derived_services"]["vxm_row_compute_cycles"],{"min":2,"max":2})
        self.assertGreaterEqual(slow["timing"]["max_row_interval_cycles"],16)
        cfg=copy.deepcopy(self.cfg)
        cfg["parameters"]["mxm_first_cycles"].update(value=5,status="assumed")
        custom=TSPSimulator(TSPSpec(cfg)).simulate(16,32,16,mode="timing")
        self.assertEqual(custom["timing"]["total_cycles"],44)
        self.assertEqual(custom["timing"]["parameters"]["mxm_first_cycles"]["selection"],"configured")

    def test_proxy_unknown_clock_and_missing_base(self):
        cfg=copy.deepcopy(self.cfg)
        cfg["parameters"]["clock_ghz"].update(value=None,status="unknown")
        r=TSPSimulator(TSPSpec(cfg)).simulate(1,1,1,mode="timing")
        self.assertEqual(r["timing"]["total_cycles"],44)
        self.assertIsNone(r["timing"]["time_ns"])
        for name in ("mem_read_bpc","mem_write_bpc"):
            bad=copy.deepcopy(cfg)
            bad["parameters"][name].update(value=None,status="unknown")
            sim=TSPSimulator(TSPSpec(bad))
            self.assertIsNone(sim.simulate(1,1,1)["timing"]["total_cycles"])
            with self.assertRaisesRegex(ValueError,name):sim.simulate(1,1,1,mode="timing")

    def test_clock_unknown_and_coordinate_validation(self):
        cfg=config_with_timing()
        cfg["parameters"]["clock_ghz"].update(value=None,status="unknown")
        spec=TSPSpec(cfg)
        r=TSPSimulator(spec).simulate(1,16,16,mode="timing",mem_slices=[0],streams=1)
        self.assertEqual(r["timing"]["total_cycles"],14)
        self.assertIsNone(r["timing"]["time_ns"])
        self.assertIsNone(r["hardware_derived"]["geometric_int8_tops_chip"]["value"])
        self.assertEqual(spec.validate_sr_coordinate(44),44)
        for index in (-1,45,True):
            with self.assertRaises(ValueError): spec.validate_sr_coordinate(index)
        for shape in [(0,16), (321,16), (16,641)]:
            with self.assertRaises(ValueError): spec.describe_supercell_usage(*shape,"INT8")

    def test_cim_reference_conflicts_cannot_be_overridden(self):
        loader=TSPCIMConfigLoader()
        cfg=loader.load_hardware_config()
        cfg["mem"]={"total_bytes":320*2**20}
        with self.assertRaisesRegex(ValueError,"conflict"):
            loader.create_tsp_spec(cfg)
        with tempfile.TemporaryDirectory() as directory:
            raw=json.loads((ROOT/"configs/hardware/tsp_cim.json").read_text(encoding="utf-8"))
            raw["native_reference"]=str(ROOT/"configs/hardware/tsp.json")
            raw["macro_reference"]=str(ROOT/"configs/hardware/cim_macro.json")
            raw["streams"]={"n_streams_per_direction":64}
            path=Path(directory)/"cim.json"
            path.write_text(json.dumps(raw),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"conflict"):
                loader.load_hardware_config(str(path))

    def test_custom_timing_file_cli_and_inconsistent_service(self):
        cfg=config_with_timing()
        bad=copy.deepcopy(cfg)
        bad["parameters"]["install_propagation_cycles"]["value"]=1
        bad["parameters"]["install_bpc"]["value"]=16
        slow=TSPSimulator(TSPSpec(bad)).simulate(1,16,16,mode="timing")
        self.assertEqual(slow["timing"]["install_cycles"],17)  # Internal IW: 1 fixed + 256/16 cycles.
        bad=copy.deepcopy(cfg)
        bad["parameters"]["vxm_base_ops_per_cycle"]["value"]=10000
        with self.assertRaisesRegex(ValueError,"budget"):
            TSPSpec(bad)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic.json"
            output=Path(directory)/"result.json"
            path.write_text(json.dumps(cfg),encoding="utf-8")
            r=subprocess.run([sys.executable,str(ROOT/"tsp.py"),"--formats","INT8","--workload","tiny_gemm.json",
                              "--mode","timing","--hardware",str(path),"--output",str(output)],
                             cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
            self.assertEqual(r.returncode,0,r.stderr)
            result=json.loads(output.read_text(encoding="utf-8"))
            self.assertGreater(result["timing"]["total_cycles"],0)
            self.assertFalse(result["validation"]["hardware_validated"])

    def test_internal_install_port_is_independent_of_lw_stream_supply(self):
        cfg=config_with_timing()
        cfg["parameters"]["install_bpc"]["value"]=10240
        r=TSPSimulator(TSPSpec(cfg)).simulate(1,320,640,mode="timing")
        # Internal IW: 204800/10240 + 2 propagation; LW has its own service.
        self.assertEqual(r["timing"]["install_cycles"],22)
        restricted=TSPSimulator(TSPSpec(cfg)).simulate(1,320,640,mode="timing",streams=1)
        self.assertEqual(restricted["timing"]["install_cycles"],22)
        self.assertEqual(r["timing"]["weight_load_cycles"],20)
        self.assertEqual(restricted["timing"]["weight_load_cycles"],640)
        cfg["parameters"]["install_bpc"]["value"]=320
        slower=TSPSimulator(TSPSpec(cfg)).simulate(1,320,640,mode="timing")
        self.assertEqual(slower["timing"]["install_cycles"],642)


if __name__ == "__main__":
    unittest.main()
