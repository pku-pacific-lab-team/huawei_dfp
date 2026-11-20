import copy
import hashlib
import json
from pathlib import Path
import unittest

from src.hardware.vector import VectorExecution, load_vector_execution


ROOT = Path(__file__).resolve().parents[2]


class VectorConfigTest(unittest.TestCase):
    def test_profile_hash_assumptions_and_new_service_sources(self):
        profile = load_vector_execution("ascend")
        cfg = profile.config
        audit = json.loads((ROOT / "calibration/operator_execution_audit.json").read_text(encoding="utf-8"))
        record = audit["vector_execution_profiles"]["configs/execution/vector.json"]
        digest = hashlib.sha256(json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(record["canonical_json_sha256"], digest)
        self.assertEqual(set(record["parameters"]), set(cfg["parameters"]))
        for name, parameter in cfg["parameters"].items():
            self.assertEqual(parameter["status"], "assumed")
            for evidence in record["parameters"][name]["evidence"]:
                source = audit["sources"][evidence["source_id"]]
                self.assertEqual(source["classification"], "modeling_assumption")
                self.assertIn('id="' + evidence["section"] + '"', (ROOT / source["path"]).read_text(encoding="utf-8"))
        self.assertEqual(load_vector_execution("tsp").policy, load_vector_execution("tsp_cim").policy)

    def test_strict_profile_validation_and_detached_copy(self):
        original = load_vector_execution("ascend").config
        for path, value in ((["schema_version"], True), (["model"], "unknown"),
                            (["policies", "ascend", "schedule"], "overlap"),
                            (["policies", "ascend", "tile_elements"], 0),
                            (["parameters", "ascend_ub_read_bpc", "value"], float("nan")),
                            (["parameters", "ascend_ub_read_bpc", "scope"], "chip")):
            config = copy.deepcopy(original)
            location = config
            for key in path[:-1]:
                location = location[key]
            location[path[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                VectorExecution(config, "ascend")
        config = copy.deepcopy(original)
        parsed = VectorExecution(config, "ascend")
        config["policies"]["ascend"]["tile_elements"] = 1
        self.assertEqual(parsed.policy["tile_elements"], 1024)
