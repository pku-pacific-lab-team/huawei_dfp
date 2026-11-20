from dataclasses import replace
import unittest

from src.operators import Action, ActionLedger, Gemm, OperatorSpec, TensorSpec


def action(id="read", **changes):
    base = Action(id=id, architecture="test", component="mem", name="read",
                  category="onchip_memory", count=16, unit="byte", dtype="INT8",
                  scope="core", basis="logical", source=f"#/native/{id}", source_basis="test fixture")
    return replace(base, **changes)


class TensorContractTest(unittest.TestCase):
    def test_storage_bytes_and_accumulation_are_separate(self):
        a = TensorSpec("activation", [2, 17], "INT8")
        b = TensorSpec("weight", [17, 33], "INT8")
        op = Gemm.from_tensors(a, b)
        self.assertEqual(a.shape, (2, 17))
        self.assertEqual((a.numel, a.nbytes), (34, 34))
        self.assertEqual(op.useful_macs, 2 * 17 * 33)
        self.assertEqual(op.spec.accumulation_dtype, "INT32")
        self.assertEqual(op.spec.outputs[0].nbytes, 2 * 33 * 4)
        self.assertEqual(TensorSpec("scale", (), "FP32").nbytes, 4)
        self.assertEqual(TensorSpec("temporary", (2, 33), "FP32").nbytes, 264)
        self.assertEqual(op.to_workload().shape, (2, 17, 33))

    def test_invalid_shapes_dtypes_and_mismatched_inputs_fail(self):
        for shape in ((0,), (-1,), (True,), (1.5,), "16", None):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                TensorSpec("x", shape, "INT8")
        for dtype in ("int8", "unknown", None, []):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                TensorSpec("x", (1,), dtype)
        for shape in ((0, 1, 1), (1, True, 1), (1, 1, 0.5)):
            with self.assertRaises(ValueError):
                Gemm(*shape)
        for b in (TensorSpec("b", (4, 5), "INT8"), TensorSpec("b", (3, 5), "FP32"),
                  TensorSpec("b", (3,), "INT8")):
            with self.assertRaises(ValueError):
                Gemm.from_tensors(TensorSpec("a", (2, 3), "INT8"), b)

    def test_descriptors_do_not_alias_caller_lists(self):
        shape = [3, 4]
        a = TensorSpec("a", shape, "INT8")
        inputs = [a]
        op = OperatorSpec("future_op", inputs, [TensorSpec("b", shape, "FP32")])
        shape[0] = 100
        inputs.clear()
        exported = op.to_dict()
        exported["inputs"][0]["shape"][0] = 200
        self.assertEqual(op.inputs[0].shape, (3, 4))
        with self.assertRaises(ValueError):
            OperatorSpec("x", [a, a], [a])


class ActionLedgerTest(unittest.TestCase):
    def test_summary_is_verified_and_excluded_from_grouped_counts(self):
        entries = (action("a"), action("b", count=20),
                   action("sum", count=36, role="summary", summarizes=("a", "b")),
                   action("logical", count=10, role="reference"))
        ledger = ActionLedger(entries)
        self.assertEqual(ledger.grouped_counts()[0]["count"], 36)
        self.assertEqual(ledger.grouped_counts()[0]["action_ids"], ["a", "b"])
        self.assertEqual(len(ledger.to_dict()["entries"]), 4)
        with self.assertRaisesRegex(ValueError, "differs"):
            ActionLedger((*entries[:2], replace(entries[2], count=37)))

    def test_units_components_bases_and_attributes_are_not_blended(self):
        changes = ({"component": "other_mem"}, {"basis": "padded"}, {"dtype": "INT32"},
                   {"scope": "chip"}, {"architecture": "other"}, {"unit": "bit"},
                   {"attributes": (("phase", "psum"),)})
        ledger = ActionLedger((action(), *(action(str(i), **kw) for i, kw in enumerate(changes))))
        self.assertEqual(len(ledger.grouped_counts()), 8)
        self.assertTrue(all(row["count"] == 16 for row in ledger.grouped_counts()))

    def test_unknown_is_preserved_instead_of_counted_as_zero(self):
        ledger = ActionLedger((action(), action("unknown", count=None, status="unavailable")))
        self.assertIsNone(ledger.grouped_counts()[0]["count"])
        self.assertEqual(ledger.grouped_counts()[0]["status"], "unavailable")
        for kwargs in ({"count": -1}, {"count": True}, {"count": 1.2}, {"count": None},
                       {"count": 0, "status": "unavailable"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                action(**kwargs)

    def test_duplicate_sources_and_invalid_alias_graphs_are_rejected(self):
        with self.assertRaises(ValueError):
            ActionLedger((action(), action()))
        with self.assertRaisesRegex(ValueError, "Duplicate action source"):
            ActionLedger((action(), action("different_id", source="#/native/read")))
        for refs in (("missing",), ("read", "read"), ("summary",)):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                ActionLedger((action(), action("summary", role="summary", summarizes=refs)))
        with self.assertRaises(ValueError):
            ActionLedger((action(), action("summary", unit="bit", role="summary", summarizes=("read",))))

    def test_export_is_detached(self):
        ledger = ActionLedger((action(attributes=(("phase", "input"),)),))
        out = ledger.to_dict()
        out["entries"][0]["attributes"]["phase"] = "changed"
        out["grouped_counts"][0]["count"] = 0
        self.assertEqual(ledger.grouped_counts()[0]["count"], 16)
        self.assertEqual(ledger.entries[0].attributes, (("phase", "input"),))
