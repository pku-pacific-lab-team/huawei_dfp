import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_issue4
from src.runtime.model_cli import _lock_report_paths, write_model_reports


ROOT = Path(__file__).resolve().parents[2]
REPORT = dict(schema_version=1, model="m", source={}, architecture="tpu", mode="count",
              request={"batch": 2}, stages=[], communication_events=[])


class ModelCliTest(unittest.TestCase):
    def test_custom_execution_profiles_in_timing_and_tile_search(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            shutil.copytree(ROOT / "configs/execution", folder / "execution")
            output = folder / "report.json"
            cmd = [sys.executable, str(ROOT / "tpu.py"), "--model", "llama3_70b",
                   "--config-dir", str(folder), "--hardware", str(ROOT / "configs/hardware/tpu_v4_reference.json"),
                   "--sequence-length", "3", "--decode-steps", "1", "--output", str(output)]

            def run(*options):
                subprocess.run([*cmd, *options], check=True, capture_output=True)
                return json.loads(output.read_bytes())

            def cast_cycles(report):
                profile = report["chips"][0]["steps"][0]["block"]
                return next(n["local_compute_cycles"] for n in report["profiles"][profile]["nodes"]
                            if n["id"] == "attention_norm.to_fp32")

            default = run()
            block = folder / "execution/blocks.json"
            profile = json.loads(block.read_bytes())
            profile["parameters"]["to_fp32_cost"]["value"] = 1_000_000
            block.write_text(json.dumps(profile), encoding="utf-8")
            custom = run()
            self.assertGreater(cast_cycles(custom), cast_cycles(default))
            vector = folder / "execution/vector.json"
            profile = json.loads(vector.read_bytes())
            profile["policies"]["tpu"]["transfer_startup_cycles"] = 100
            vector.write_text(json.dumps(profile), encoding="utf-8")
            timed, tuned = run(), run("--auto-tile")
            self.assertGreater(cast_cycles(timed), cast_cycles(custom))
            cycles = sum(c["steps"][0]["local_compute_cycles"] for c in timed["chips"])
            self.assertEqual(tuned["tile_search"]["baseline"]["critical_path_cycles"], cycles)
            self.assertEqual(tuned["step_totals"], default["step_totals"])

    def test_failed_pair_restores_both_reports(self):
        replace = Path.replace
        with tempfile.TemporaryDirectory() as folder:
            compute, comm = (Path(folder) / name for name in ("compute", "comm"))

            def fail_second(src, dest):
                if dest == compute:
                    raise PermissionError("second replacement failed")
                return replace(src, dest)

            for old_comm in (False, True):
                for old_compute in (False, True):
                    expected = {}
                    for path, exists in ((comm, old_comm), (compute, old_compute)):
                        if exists:
                            expected[path] = b"previous " + path.name.encode()
                            path.write_bytes(expected[path])
                    with patch.object(Path, "replace", fail_second), self.assertRaises(PermissionError):
                        write_model_reports(REPORT, compute, comm)
                    self.assertEqual({p: p.read_bytes() for p in Path(folder).iterdir()}, expected)
                    for path in expected:
                        path.unlink()

    @unittest.skipUnless(os.name == "nt", "Windows read-only replacement semantics")
    def test_read_only_compute_file_preserves_communication(self):
        with tempfile.TemporaryDirectory() as folder:
            compute, comm = (Path(folder) / name for name in ("compute", "comm"))
            compute.write_bytes(b"compute1")
            comm.write_bytes(b"comm1")
            compute.chmod(stat.S_IREAD)
            try:
                with self.assertRaises(PermissionError):
                    write_model_reports(REPORT, compute, comm)
                self.assertEqual((compute.read_bytes(), comm.read_bytes()), (b"compute1", b"comm1"))
            finally:
                compute.chmod(stat.S_IREAD | stat.S_IWRITE)

    @unittest.skipUnless(os.name == "nt", "Windows delete-on-close publisher locks")
    def test_second_publisher_can_retry_after_contention(self):
        with tempfile.TemporaryDirectory() as folder:
            compute, comm = (Path(folder) / name for name in ("compute", "comm"))
            with _lock_report_paths([compute]), self.assertRaises(OSError):
                write_model_reports(REPORT, compute, comm)
            self.assertEqual(list(Path(folder).iterdir()), [])
            write_model_reports(REPORT, compute, comm)
            self.assertEqual(json.loads(comm.read_bytes())["request"], json.loads(compute.read_bytes())["request"])

    def test_issue4_refuses_other_evidence_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(verify_issue4, "ROOT", root), patch.object(verify_issue4, "run_tests") as writer:
                for target in ("validation", "validation/issue2", "validation/issue2/run",
                               "validation/issue3", "validation/issue3/run/retry"):
                    with self.subTest(target=target), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        verify_issue4.main(["--output-dir", str(root / target)])
                    writer.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_issue4_snapshot_identifies_content_without_git(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile = root / "configs/models/llama3_70b.json"
            profile.parent.mkdir(parents=True)
            profile.write_bytes(b"{}\r\n")
            before = verify_issue4.source_snapshot(root)
            self.assertEqual(verify_issue4.git_version(root), {"git_commit": None, "git_dirty": None})
            self.assertEqual(set(before["files"]), {"configs/models/llama3_70b.json"})
            profile.write_bytes(b"{}\n")
            (profile.parent / "acceptance.json").write_text("{}")
            self.assertEqual(verify_issue4.source_snapshot(root), before)
            profile.unlink()
            self.assertNotEqual(verify_issue4.source_snapshot(root)["sha256"], before["sha256"])

    def test_issue4_rejects_source_changes_during_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / "results"
            source = root / "tpu.py"
            source.write_text("# original")

            def run_case(*args):
                source.write_text("# edited")
                return {}, []

            with (patch.object(verify_issue4, "ROOT", root), patch.object(verify_issue4, "discover"),
                  patch.object(verify_issue4, "run_tests", return_value=(None, {"passed": 1}, True)),
                  patch.object(verify_issue4, "case", side_effect=run_case),
                  contextlib.redirect_stdout(io.StringIO())):
                self.assertEqual(verify_issue4.main(["--output-dir", str(out)]), 1)
            report = json.loads((out / "acceptance.json").read_bytes())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["snapshot_unchanged_during_run"])
            self.assertNotEqual(report["source_version"]["sha256"], verify_issue4.source_snapshot(root)["sha256"])
