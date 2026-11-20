from tests.operators.inspection import inspect_result
import copy
import json
import unittest

from src.operators import Gemm, adapt_gemm_report, simulate_gemm
from src.core.ascend.simulator import AscendSimulator
from src.core.tpu.simulator import TPUSimulator
from src.core.tsp.simulator import TSPSimulator
from src.core.tsp_cim.simulator import TSPCIMSimulator
from src.utils.ascend_config import AscendConfigLoader
from src.utils.tpu_config import TPUConfigLoader
from src.utils.tsp_cim_config import TSPCIMConfigLoader


def specs():
    loader = TSPCIMConfigLoader()
    cim = loader.create_tsp_spec(loader.load_hardware_config())
    return {"ascend": AscendConfigLoader().load_hardware(), "tpu": TPUConfigLoader().load_hardware(),
            "tsp": cim.native_reference_spec, "tsp_cim": cim}


def native(arch, spec, op, mode):
    simulators = {"ascend": AscendSimulator, "tpu": TPUSimulator,
                  "tsp": TSPSimulator, "tsp_cim": TSPCIMSimulator}
    sim = simulators[arch](spec)
    work = op.to_workload()
    r = sim.simulate(work, mode=mode) if arch in {"ascend", "tpu"} else sim.simulate(*work.shape, mode=mode)
    r["workload"] = work.to_dict()
    return r


class GemmAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = specs()

    def test_decode_tails_and_cross_k_preserve_native_results(self):
        for shape in ((1, 1, 1), (2, 17, 33), (4, 321, 641)):
            op = Gemm(*shape)
            for arch, spec in self.specs.items():
                for mode in ("count", "timing"):
                    with self.subTest(shape=shape, arch=arch, mode=mode):
                        reference = native(arch, spec, op, mode)
                        result = inspect_result(simulate_gemm(op, arch, spec, mode=mode, include_details=True))
                        self.assertEqual(result["native_report"], reference)
                        self.assertEqual(result["boundary"], reference["boundary"])
                        self.assertEqual(result["resource_requirements"]["memory"], reference["memory"])
                        self.assertEqual(result["work_counts"]["logical"]["operations"], 2 * shape[0] * shape[1] * shape[2])
                        self.assertEqual(result["operator"]["outputs"][0]["logical_bytes"], 4 * shape[0] * shape[2])
                        self.assertEqual(result["latency"]["cycle_count"], reference["timing"]["total_cycles"])
                        ns = reference["timing"]["time_ns"]
                        self.assertEqual(result["latency"]["latency_s"], None if ns is None else ns / 1e9)
                        self.assertEqual(result["energy"], {"status": "not_evaluated", "total_pj": None})
                        json.dumps(result, allow_nan=False)
                        ledger = result["action_ledger"]["entries"]
                        self.assertEqual({a["id"] for a in ledger}, set(reference["action_counts"]))
                        for a in ledger:
                            source = reference["action_counts"][a["id"]]
                            self.assertEqual((a["count"], a["unit"], a["scope"]),
                                             (source["count"], source["unit"], source["scope"]))

    def test_padding_and_cim_aliases_do_not_become_extra_actions(self):
        results = {arch: inspect_result(simulate_gemm(Gemm(1, 17, 33), arch, spec, include_details=True)) for arch, spec in self.specs.items()}
        tpu = {a["id"]: a for a in results["tpu"]["action_ledger"]["entries"]}
        ascend = {a["id"]: a for a in results["ascend"]["action_ledger"]["entries"]}
        self.assertEqual((tpu["matrix_mac"]["count"], tpu["matrix_mac"]["basis"]), (17 * 33, "logical"))
        self.assertEqual((ascend["cube_mac"]["count"], ascend["cube_mac"]["basis"]), (16 * 32 * 48, "padded"))
        cim = results["tsp_cim"]["action_ledger"]
        matrix = [a for a in cim["grouped_counts"] if a["category"] == "matrix"]
        self.assertEqual(len(matrix), 1)
        self.assertEqual(matrix[0]["count"], 32 * 64)
        self.assertEqual(sum(a["role"] == "summary" for a in cim["entries"]), 4)
        for arch in ("tsp", "tsp_cim"):
            self.assertFalse(any(a["category"] == "offchip_memory" for a in results[arch]["action_ledger"]["entries"]))
            self.assertEqual(results[arch]["coverage"]["offchip_memory"], "outside_native_boundary")
        routes = [a for a in results["ascend"]["action_ledger"]["entries"] if a["category"] == "onchip_transfer"]
        self.assertEqual(len(routes), 3)  # no synthetic read + write endpoints

    def test_unknown_schema_actions_or_precision_are_rejected(self):
        r = native("tpu", self.specs["tpu"], Gemm(1, 1, 1), "count")
        variants = []
        bad = copy.deepcopy(r); bad["schema_version"] += 1; variants.append(bad)
        bad = copy.deepcopy(r); bad["action_counts"]["invented"] = bad["action_counts"]["matrix_mac"]; variants.append(bad)
        bad = copy.deepcopy(r); del bad["action_counts"]["matrix_mac"]; variants.append(bad)
        bad = copy.deepcopy(r); bad["workload"]["output_dtype"] = "INT8"; variants.append(bad)
        bad = copy.deepcopy(r); bad["counts"]["useful_ops"] += 1; variants.append(bad)
        for report in variants:
            with self.assertRaises(ValueError):
                adapt_gemm_report("tpu", report, include_details=True)
        with self.assertRaises(ValueError):
            adapt_gemm_report("ascend", r, include_details=True)
        for arch in ("unknown", None):
            with self.assertRaises(ValueError):
                simulate_gemm(Gemm(1, 1, 1), arch, self.specs["tpu"], include_details=True)
        with self.assertRaises(ValueError):
            simulate_gemm(Gemm(1, 1, 1), "tsp", self.specs["tpu"], include_details=True)

    def test_capacity_and_missing_timing_still_fail_in_native_kernel(self):
        for arch in ("tsp", "tsp_cim"):
            with self.subTest(arch=arch), self.assertRaisesRegex(ValueError, "unsupported_capacity"):
                simulate_gemm(Gemm(1, 8192, 28672), arch, self.specs[arch], include_details=True)
        from src.hardware.tpu import TPUSpec
        from tests.tpu.support import hardware_config
        config = hardware_config()
        config["parameters"]["clock_hz"].update(value=None, status="unknown")
        hardware = TPUSpec(config)
        result = inspect_result(simulate_gemm(Gemm(1, 1, 1), "tpu", hardware, include_details=True))
        self.assertEqual(result["latency"]["status"], "not_evaluated")
        with self.assertRaises(ValueError):
            simulate_gemm(Gemm(1, 1, 1), "tpu", hardware, mode="timing", include_details=True)

    def test_adapter_and_export_do_not_mutate_or_alias_native_report(self):
        report = native("tsp_cim", self.specs["tsp_cim"], Gemm(1, 17, 33), "count")
        original = copy.deepcopy(report)
        result = adapt_gemm_report("tsp_cim", report, include_details=True)
        self.assertEqual(report, original)
        report["counts"]["useful_ops"] = -1
        out = inspect_result(result)
        out["native_report"]["counts"]["useful_ops"] = 0
        self.assertEqual(inspect_result(result)["native_report"], original)

    def test_unknown_clock_preserves_cycle_only_native_estimate(self):
        from src.hardware.tsp import TSPSpec
        from src.utils.tsp_config import TSPConfigLoader
        config = TSPConfigLoader().load_hardware_config()
        config["parameters"]["clock_ghz"].update(value=None, status="unknown")
        result = inspect_result(simulate_gemm(Gemm(1, 17, 33), "tsp", TSPSpec(config), mode="timing", include_details=True))
        self.assertEqual(result["latency"]["status"], "cycles_only")
        self.assertGreater(result["latency"]["cycle_count"], 0)
        self.assertIsNone(result["latency"]["latency_s"])

    def test_policy_overrides_stay_in_native_execution(self):
        spec, op = self.specs["tpu"], Gemm(2, 257, 33)
        policy = dict(tile=(128, 128, 128), buffers=1, schedule="serial", cmem_mode="bypass")
        original = TPUSimulator(spec).simulate(op.to_workload(), mode="timing", **policy)
        result = inspect_result(simulate_gemm(op, "tpu", spec, mode="timing", **policy, include_details=True))
        self.assertEqual(result["native_report"], original)
        with self.assertRaises(TypeError):
            simulate_gemm(op, "tpu", spec, invented_policy=True, include_details=True)
