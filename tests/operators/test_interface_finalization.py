from tests.operators.inspection import inspect_result
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.operators import (Gemm, OperatorSpec, RMSNorm, Softmax, TensorSpec, Vector,
                           adapt_gemm_report, plan_connection, simulate_composite, simulate_gemm, simulate_vector)
from src.hardware.vector import VectorExecution, load_vector_execution
from tests.operators.test_gemm import specs
from tests.operators import test_composite_cli


class InterfaceFinalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hardware = specs()

    def test_attributes_are_immutable_normalized_semantics(self):
        x, w = TensorSpec("x", (2, 7), "FP32"), TensorSpec("w", (7,), "FP32")
        a, b = RMSNorm(x, w, axis=-1), RMSNorm(x, w, axis=1)
        self.assertEqual(a, b)
        self.assertEqual(a.spec, b.spec)
        self.assertEqual(hash(a.spec), hash(b.spec))
        self.assertNotEqual(a.spec, RMSNorm(x, w, epsilon=1e-6).spec)
        attrs = {"axis": 1, "epsilon": 1e-5}
        op = OperatorSpec("rmsnorm", (x, w), a.spec.outputs, "FP32", attrs)
        attrs["axis"] = 0
        out = op.to_dict()
        out["attributes"]["epsilon"] = 1
        self.assertEqual(dict(op.attributes), {"axis": 1, "epsilon": 1e-5})
        for attrs in ({"epsilon": float("nan")}, {"axis": [1]}, (("x", 1), ("x", 2)), (("x",),)):
            with self.assertRaises(ValueError):
                replace(op, attributes=attrs)

    def test_gemm_keeps_tensor_identity_without_changing_native_results(self):
        a, b = TensorSpec("activation", (2, 3), "INT8"), TensorSpec("weight", (3, 7), "INT8")
        op = Gemm.from_tensors(a, b, output_name="projection")
        self.assertEqual(op.spec.inputs, (a, b))
        self.assertEqual(op.spec.outputs[0].name, "projection")
        for arch, spec in self.hardware.items():
            original = simulate_gemm(Gemm(2, 3, 7), arch, spec, mode="timing", include_details=True)
            named = simulate_gemm(op, arch, spec, mode="timing", include_details=True)
            self.assertEqual(original.native_report, named.native_report)
            self.assertEqual(original.action_ledger, named.action_ledger)
            self.assertEqual(named.operator, op.spec)
            self.assertEqual([t["name"] for t in inspect_result(named)["io_contract"]["inputs"]], ["activation", "weight"])
            with self.assertRaises(ValueError):
                adapt_gemm_report(arch, original.native_report, operator=Gemm(2, 4, 7), include_details=True)
        self.assertEqual(Gemm.from_tensors(TensorSpec("C", (2, 3), "INT8"), b).output_name, "C_")
        with self.assertRaises(ValueError):
            Gemm.from_tensors(a, b, output_name="activation")
        with self.assertRaises(ValueError):
            Gemm.from_tensors(a, TensorSpec("activation", (3, 7), "INT8"))

    def test_named_vector_composite_and_result_attributes(self):
        x, w = TensorSpec("x", (2, 7), "FP32"), TensorSpec("w", (7,), "FP32")
        for arch, spec in self.hardware.items():
            rms = inspect_result(simulate_composite(RMSNorm(x, w, output_name="norm"), arch, spec, include_details=True))
            soft = inspect_result(simulate_composite(Softmax(x, output_name="prob"), arch, spec, include_details=True))
            vec = inspect_result(simulate_vector(Vector("fp32_exp", (x,), output_name="exp"), arch, spec, include_details=True))
            self.assertEqual(rms["operator"]["attributes"], {"axis": 1, "epsilon": 1e-5})
            self.assertEqual(soft["operator"]["attributes"], {"axis": 1})
            self.assertEqual(vec["operator"]["attributes"], {})
            for report, name in ((rms, "norm"), (soft, "prob"), (vec, "exp")):
                self.assertEqual(report["operator"]["outputs"][0]["name"], name)
                self.assertEqual(report["io_contract"]["outputs"][0]["name"], name)
                self.assertEqual(report["native_report"]["workload"]["attributes"], report["operator"]["attributes"])
        for make in (lambda: RMSNorm(x, w, output_name="w"), lambda: Softmax(x, output_name="x"), lambda: Vector("fp32_exp", (x,), output_name="")):
            with self.assertRaises(ValueError):
                make()

    def test_connection_requires_conversion_without_inventing_cost_or_reuse(self):
        for arch, spec in self.hardware.items():
            producer = simulate_gemm(Gemm(2, 3, 7), arch, spec, include_details=True)
            consumer = simulate_composite(RMSNorm(TensorSpec("x", (2, 7), "FP32"), TensorSpec("w", (7,), "FP32")), arch, spec, include_details=True)
            before = (producer.to_dict(), consumer.to_dict())
            edge = plan_connection(producer, consumer)
            self.assertEqual(edge["conversion"]["status"], "required_unmodeled")
            self.assertEqual(edge["conversion"]["source_payload_bytes"], 56)
            self.assertEqual(edge["conversion"]["target_payload_bytes"], 56)
            self.assertIsNone(edge["latency_s"])
            self.assertIsNone(edge["additional_action_counts"])
            self.assertFalse(edge["residency"]["reuse_assumed"])
            self.assertEqual(before, (producer.to_dict(), consumer.to_dict()))
            reverse = simulate_gemm(Gemm(2, 7, 3), arch, spec, include_details=True)
            edge = plan_connection(consumer, reverse)
            self.assertEqual(edge["conversion"]["target_payload_bytes"], 14)
            self.assertIn("explicit_conversion_model", edge["requirements"])

    def test_connections_do_not_infer_physical_device_or_cross_hemisphere_reuse(self):
        op = Softmax(TensorSpec("x", (2, 7), "FP32"))
        a = simulate_composite(op, "tsp_cim", self.hardware["tsp_cim"], hemisphere=0, include_details=True)
        b = simulate_composite(op, "tsp_cim", self.hardware["tsp_cim"], hemisphere=1, include_details=True)
        self.assertEqual(plan_connection(a, b)["residency"]["status"], "transfer_required")
        same = plan_connection(a, a)
        self.assertEqual(same["conversion"]["status"], "not_required")
        self.assertEqual(same["residency"]["status"], "unverified_placement")
        c = simulate_composite(op, "tpu", self.hardware["tpu"], include_details=True)
        self.assertIn("transfer_model", plan_connection(a, c)["requirements"])
        with self.assertRaises(ValueError):
            plan_connection(a, simulate_gemm(Gemm(2, 8, 3), "tpu", self.hardware["tpu"], include_details=True))
        for index in (True, -1, 3):
            with self.assertRaises(ValueError):
                plan_connection(a, b, input_index=index)

    def test_schema2_null_and_malformed_policy_rejected_for_all_architectures(self):
        for arch in self.hardware:
            cfg = load_vector_execution(arch).config
            for bad in (None, [], False, "", {}):
                with self.subTest(arch=arch, bad=bad), self.assertRaisesRegex(ValueError, "Composite policy fields"):
                    VectorExecution({**cfg, "composite": bad}, arch)
            old = deepcopy(cfg)
            old.pop("composite")
            old["schema_version"] = 1
            self.assertIsNone(VectorExecution(old, arch).composite)

    def test_cli_rejects_null_profile_before_overwriting_and_exports_attributes(self):
        with tempfile.TemporaryDirectory() as folder:
            output, cfgpath, workload = (Path(folder) / name for name in ("result.json", "profile.json", "workload.json"))
            config = load_vector_execution("tpu").config
            config["composite"] = None
            cfgpath.write_text(json.dumps(config), encoding="utf-8")
            output.write_text("preserve", encoding="utf-8")
            runner = test_composite_cli.CompositeCLITest()
            for arch in self.hardware:
                for op in ("fp32_exp", "rmsnorm"):
                    p = runner.run_entry(arch, folder, "--workload", f"operators/{op}.json", "--vector-config", str(cfgpath), "--output", str(output))
                    self.assertEqual(p.returncode, 2, p.stderr)
                    self.assertIn("Composite policy fields", p.stderr)
                    self.assertEqual(output.read_text(encoding="utf-8"), "preserve")
            workload.write_text(json.dumps(dict(schema_version=1,operation="softmax",inputs=[dict(name="x",shape=[2,7],dtype="FP32")],axis=-1,output_name="prob")), encoding="utf-8")
            p = runner.run_entry("tpu", folder, "--workload", str(workload), "--output", str(output))
            self.assertEqual(p.returncode, 0, p.stderr)
            r = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(r["operator"]["attributes"], {"axis": 1})
            self.assertEqual(r["operator"]["outputs"][0]["name"], "prob")
