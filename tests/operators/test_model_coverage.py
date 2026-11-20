"""Coverage claims must retain concrete code/test evidence and explicit gaps."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ModelCoverageTest(unittest.TestCase):
    def test_inventory_claims_have_evidence_and_all_gaps_have_owners(self):
        d = json.loads((ROOT / "validation/issue3/model_coverage.json").read_text(encoding="utf-8"))
        self.assertEqual(d["schema_version"], 1)
        self.assertEqual({m["name"] for m in d["models"]}, {"LLaMA3-70B", "DeepSeek-R1"})
        incomplete = []
        for model in d["models"]:
            self.assertTrue(model["reference_ids"] and set(model["reference_ids"]) <= set(d["references"]))
            self.assertEqual(len({b["id"] for b in model["blocks"]}), len(model["blocks"]))
            for block in model["blocks"]:
                self.assertEqual(block["owner"], "issue3")
                self.assertIn(block["status"], d["status_definitions"])
                self.assertTrue(block["notes"])
                if block["status"] in {"implemented", "partial"}:
                    self.assertTrue(block["implementation"] and block["tests"])
                    for file in block["implementation"]:
                        self.assertTrue((ROOT / file).is_file(), file)
                    for module in block["tests"]:
                        self.assertTrue((ROOT / (module.replace(".", "/") + ".py")).is_file(), module)
                if block["status"] != "implemented":
                    incomplete.append((model["name"], block["id"]))
        self.assertEqual(d["full_target_model_coverage"], not bool(incomplete))

    def test_conventional_cache_is_not_claimed_as_mla(self):
        d = json.loads((ROOT / "validation/issue3/model_coverage.json").read_text(encoding="utf-8"))
        r1 = next(m for m in d["models"] if m["name"] == "DeepSeek-R1")
        cache = next(b for b in r1["blocks"] if b["id"] == "mla_compressed_cache")
        if cache["status"] == "implemented":
            self.assertTrue(any("mla" in module for module in cache["tests"]))
        else:
            self.assertFalse(d["full_target_model_coverage"])


if __name__ == "__main__":
    unittest.main()
