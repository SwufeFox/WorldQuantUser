import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase, mock

from tools import atom_runner, atom_screen
from tools._runner import CliResult


class AtomRunnerTests(TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.base = Path(work.name)
        self.root = self.base / "research"
        self.source = self.base / "arms.json"
        self.source.write_text(json.dumps([
            {"label": "same", "expr": "rank(close)", "settings": {"decay": 2}},
            {"label": "same", "expr": "rank(close)", "settings": {"decay": 4}},
        ]), encoding="utf-8")

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = atom_runner.main(["--research-dir", str(self.root), *map(str, args)])
        return code, json.loads(stdout.getvalue()) if stdout.getvalue() else {}, stderr.getvalue()

    def test_offline_dry_run_preserves_settings_and_creates_no_state(self):
        code, result, error = self.invoke(self.source, "--dry-run", "--concurrency", 2)
        self.assertEqual(code, 0, error)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["arm_order"], ["1: same", "2: same"])
        self.assertEqual([record["payload"]["settings"]["decay"]
                          for record in result["results"]], [2, 4])
        self.assertFalse(self.root.exists())

    def test_saved_scan_projects_full_is_changes_and_refuses_replay(self):
        output = self.root / "atom-results.jsonl"
        self.root.mkdir()
        legacy = {"label": "legacy", "alpha_id": "a1", "sharpe": 1.6,
                  "checks": {"LOW_SUB_UNIVERSE_SHARPE": {"result": "PASS"}}}
        output.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
        batch_path = None

        def run_cli(root, command, *args, input_text=None):
            nonlocal batch_path
            self.assertEqual((root, command), (self.root, "probe"))
            if args[0] == "-":
                self.assertIn("--no-record", args)
                index = args.index("--batch-path")
                batch_path = Path(args[index + 1])
                batch_path.parent.mkdir()
                spec = json.loads(input_text)
                batch = {"format": "probe-batch", "mechanism": spec["mechanism"],
                         "run_id": "test-run", "recording_policy": {"enabled": False},
                         "arms": spec["arms"], "records": [
                             {"status": "COMPLETE", "alpha_id": "a1",
                              "metrics": {"sharpe": 1.6},
                              "checks": {"LOW_SUB_UNIVERSE_SHARPE": {"result": "FAIL"}},
                              "out_of_sample": {"train": {"sharpe": 0.4}}},
                             {"status": "PENDING"}]}
            else:
                self.assertEqual(args[0], "--recover")
                self.assertIsNone(input_text)
                batch = json.loads(batch_path.read_text(encoding="utf-8"))
                batch["records"][0]["checks"]["LOW_SUB_UNIVERSE_SHARPE"] = {
                    "result": "PASS", "value": 0.523, "limit": 0.517}
                batch["records"][1] = {"status": "COMPLETE", "alpha_id": "a2"}
            batch_path.write_text(json.dumps(batch), encoding="utf-8")
            return CliResult(True, {"path": str(batch_path)})

        with mock.patch.object(atom_runner, "run_cli", side_effect=run_cli) as cli:
            code, result, error = self.invoke(self.source)
            self.assertEqual((code, result["appended"]), (0, 2), error)
            first_screen = io.StringIO()
            with contextlib.redirect_stdout(first_screen):
                self.assertEqual(atom_screen.main(["--research-dir", str(self.root)]), 0)
            self.assertIn("0 pass sub-universe, 1 fail", first_screen.getvalue())
            code, _, error = self.invoke(self.source)
            self.assertEqual(code, 2)
            self.assertIn("already exists", error)
            renamed = self.source.with_name("renamed.json")
            self.source.rename(renamed)
            code, _, error = self.invoke(renamed)
            self.assertEqual(code, 2)
            self.assertIn("already exists", error)
            cli.assert_called_once()

            code, result, error = self.invoke("--recover", batch_path)
            self.assertEqual((code, result["appended"]), (0, 2), error)
            code, result, error = self.invoke("--recover", batch_path)
            self.assertEqual((code, result["appended"]), (0, 0), error)

        original, complete, pending, updated, recovered = map(json.loads, output.read_text().splitlines())
        self.assertEqual(original, legacy)
        self.assertEqual([row["label"] for row in (complete, pending, recovered)], ["same"] * 3)
        self.assertEqual((complete["metrics_scope"], complete["sharpe"]), ("full_is", 1.6))
        self.assertEqual(complete["checks"]["LOW_SUB_UNIVERSE_SHARPE"]["result"], "FAIL")
        self.assertEqual((pending["status"], recovered["alpha_id"]), ("PENDING", "a2"))
        self.assertEqual(pending["arm_index"], recovered["arm_index"])
        self.assertEqual(atom_screen.load(output)["a1"], updated)
        final_screen = io.StringIO()
        with contextlib.redirect_stdout(final_screen):
            self.assertEqual(atom_screen.main(["--research-dir", str(self.root), "--top", "1"]), 0)
        self.assertIn("1 pass sub-universe, 0 fail", final_screen.getvalue())
        self.assertIn("PASS 0.523/0.517", final_screen.getvalue())

    def test_refuses_unrelated_batch_before_calling_cli(self):
        other = self.base / "other.json"
        other.write_text('{"format":"probe-batch","mechanism":"unrelated"}', encoding="utf-8")
        with mock.patch.object(atom_runner, "run_cli") as cli:
            code, _, error = self.invoke("--resume", other)
        self.assertEqual(code, 2)
        self.assertIn("not a batch produced by atom_runner", error)
        cli.assert_not_called()
