#!/usr/bin/env python3
"""Tests for ledger identity, lineage and leg-level exposure.

These pin the two things that were wrong in the ledger rather than the code style:
records without an alpha id lost their settings when keyed, and exposure could only
be guessed because leg names are positional.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import alpha_registry as ar  # noqa: E402
import research_cli as cli  # noqa: E402

COMPOSITE = "a = rank(cap); n = rank(volume); c = rank(close); a + n + c"


def entry(expression=COMPOSITE, alpha_id=None, settings=None, **extra):
    return ar.build_entry(
        {"alpha_id": alpha_id, "expression": expression,
         "settings": settings or {"universe": "TOP500", "decay": 4},
         "metrics": {"fitness": 1.1}, **extra})


class IdentityTests(unittest.TestCase):
    def test_fingerprint_separates_universes(self):
        self.assertNotEqual(
            ar.settings_fingerprint({"universe": "TOP500", "decay": 4}),
            ar.settings_fingerprint({"universe": "TOP3000", "decay": 4}))

    def test_fingerprint_separates_decays(self):
        self.assertNotEqual(
            ar.settings_fingerprint({"universe": "TOP500", "decay": 4}),
            ar.settings_fingerprint({"universe": "TOP500", "decay": 12}))

    def test_fingerprint_ignores_cosmetic_settings(self):
        base = {"universe": "TOP500", "decay": 4}
        noisy = dict(base, language="FASTEXPR", visualization=False)
        self.assertEqual(ar.settings_fingerprint(base),
                         ar.settings_fingerprint(noisy))

    def test_record_without_id_keys_on_expression_and_settings(self):
        # The same formula under two universes used to collapse into one entry.
        a = entry(settings={"universe": "TOP500", "decay": 4})
        b = entry(settings={"universe": "TOP3000", "decay": 4})
        self.assertNotEqual(a["key"], b["key"])

    def test_record_with_id_keys_on_the_id(self):
        a = entry(alpha_id="ABC12345")
        self.assertEqual(a["key"], "ABC12345")


class LegTests(unittest.TestCase):
    def test_legs_are_derived_when_not_stored(self):
        self.assertEqual(ar.legs_of(entry()), ["a", "n", "c"])

    def test_unparseable_expression_has_no_legs(self):
        self.assertEqual(ar.legs_of(entry(expression="rank(cap)")), [])

    def test_shared_legs_match_on_body_not_on_name(self):
        # Both declare a leg called "a" but the bodies differ; they are not the
        # same exposure and must not be reported as sharing.
        one = entry(expression="a = rank(cap); b = rank(close); a + b",
                    alpha_id="ONE")
        two = entry(expression="a = rank(volume); b = rank(close); a + b",
                    alpha_id="TWO")
        rows = ar.shared_legs([one, two], "ONE")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["shared_legs"], 1)

    def test_identical_composites_share_every_leg(self):
        one = entry(alpha_id="ONE")
        two = entry(alpha_id="TWO")
        rows = ar.shared_legs([one, two], "ONE")
        self.assertEqual(rows[0]["shared_legs"], 3)

    def test_leg_index_groups_by_body(self):
        index = ar.leg_index([entry(alpha_id="ONE"), entry(alpha_id="TWO")])
        self.assertEqual(len(index), 3)
        self.assertEqual(sorted(index["rank(cap)"]), ["ONE", "TWO"])


class LineageTests(unittest.TestCase):
    def _ledger(self):
        parent = entry(alpha_id="PARENT")
        child = ar.build_entry(
            {"alpha_id": "CHILD", "expression": COMPOSITE,
             "settings": {"universe": "TOP500", "decay": 4},
             "parent": parent["expression_hash"], "op": "drop_leg:c"})
        return [parent, child]

    def test_chain_walks_to_the_parent(self):
        chain = ar.lineage(self._ledger(), "CHILD")
        self.assertEqual([item["alpha_id"] for item in chain], ["CHILD", "PARENT"])

    def test_children_are_found_from_the_parent(self):
        children = ar.descendants(self._ledger(), "PARENT")
        self.assertEqual([item["alpha_id"] for item in children], ["CHILD"])

    def test_missing_parent_ends_the_walk(self):
        self.assertEqual(len(ar.lineage(self._ledger(), "PARENT")), 1)

    def test_unknown_reference_is_none(self):
        self.assertIsNone(ar.resolve(self._ledger(), "NOPE"))


class RelabelTests(unittest.TestCase):
    def test_family_is_set_and_verdict_preserved(self):
        updated = ar.relabel(entry(alpha_id="X"), family="quality")
        self.assertEqual(updated["family"], "quality")
        self.assertEqual(updated["source"], "relabel")

    def test_unknown_family_is_refused(self):
        with self.assertRaises(ValueError):
            ar.relabel(entry(alpha_id="X"), family="not-a-family")

    def test_unknown_verdict_is_refused(self):
        with self.assertRaises(ValueError):
            ar.relabel(entry(alpha_id="X"), verdict="nonsense")


class CoverageTests(unittest.TestCase):
    def test_counts_entries_missing_family(self):
        ledger = [entry(alpha_id="A"), ar.relabel(entry(alpha_id="B"), family="quality")]
        summary = ar.unlabeled_summary(ledger)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["missing_family"], 1)
        self.assertEqual(summary["family_coverage"], 0.5)


class LedgerCliTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.registry = str(Path(directory.name) / "ledger.jsonl")
        self.login = mock.patch.object(
            cli.ml, "load_session", side_effect=AssertionError("offline command logged in"))
        self.login.start()
        self.addCleanup(self.login.stop)

    def invoke(self, *arguments, expected=0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(list(arguments))
        self.assertEqual(code, expected, stderr.getvalue())
        return json.loads(stdout.getvalue())

    def ledger(self, action, *arguments, expected=0):
        return self.invoke("ledger", action, "--registry", self.registry,
                           *arguments, expected=expected)

    def store(self, alpha_id, *, fitness=1.4, sharpe=1.8, **changes):
        labels = {"verdict": "candidate", "family": "quality",
                  "idea": "Idea 70", "mechanism": "cashflow revision"}
        labels.update({key: changes.pop(key) for key in tuple(changes) if key in labels})
        record = {"alpha_id": alpha_id, "expression": "rank(cashflow_op/cap)",
                  "metrics": {"fitness": fitness, "sharpe": sharpe},
                  "status": "COMPLETE"}
        record.update(changes)
        stored, _written = ar.record_entry(ar.build_entry(record, **labels), self.registry)
        return stored

    def test_schema_discovers_six_ledger_commands_and_removes_old_alpha_commands(self):
        commands = self.invoke("schema")["data"]["commands"]["subcommands"]
        actions = commands["ledger"]["subcommands"]
        self.assertEqual(set(actions), {"record", "list", "label", "impact", "lineage", "coverage"})
        self.assertTrue(set(actions).difference({"list"}).isdisjoint(
            commands["alpha"]["subcommands"]))
        self.assertNotIn("registry", commands["alpha"]["subcommands"])
        self.assertNotIn("find", commands["alpha"]["subcommands"])
        flags = {flag for option in actions["list"]["options"] for flag in option["flags"]}
        self.assertTrue({"--verdict", "--idea", "--min-sharpe", "--min-fitness", "--family",
                         "--field", "--mechanism", "--expression", "--limit", "--format",
                         "--registry"}.issubset(flags))

    def test_unfiltered_list_shows_all_entries_including_book_without_login(self):
        for index in range(52):
            self.store("A%02d" % index, status="ACTIVE" if index == 0 else "COMPLETE")
        data = self.ledger("list")["data"]
        self.assertEqual(data["registry_entries"], 52)
        self.assertEqual(data["matched"], 52)
        self.assertEqual(data["shown"], 52)
        self.assertEqual(len(data["entries"]), 52)
        self.assertEqual(data["summary"]["total"], 52)
        self.assertEqual(next(row for row in data["entries"] if row["alpha_id"] == "A00")["status"],
                         "ACTIVE")

    def test_all_merged_filters_stack_and_repeatable_verdicts_are_a_union(self):
        self.store("MATCH")
        self.store("BOOK", verdict="watch", status="ACTIVE", fitness=1.5)
        self.store("LOW_SHARPE", sharpe=0.5)
        self.store("LOW_FITNESS", fitness=0.5)
        self.store("WRONG_FAMILY", family="event")
        self.store("WRONG_FIELD", expression="rank(close/cap)")
        self.store("WRONG_MECHANISM", mechanism="other")
        self.store("WRONG_IDEA", idea="Idea 71")
        self.store("WRONG_VERDICT", verdict="rejected")
        data = self.ledger(
            "list", "--verdict", "candidate", "--verdict", "watch", "--idea", "IDEA 70",
            "--family", "quality", "--field", "CASHFLOW_OP", "--mechanism", "REVISION",
            "--expression", "/CAP", "--min-sharpe", "1.5", "--min-fitness", "1.0")["data"]
        self.assertEqual([row["alpha_id"] for row in data["entries"]], ["BOOK", "MATCH"])
        self.assertEqual(data["matched"], 2)
        self.assertEqual(data["query"]["family"], "quality")
        self.assertEqual(data["query"]["min_sharpe"], 1.5)
        self.assertEqual(data["verdict_counts"], {"watch": 1, "candidate": 1})
        self.assertTrue(data["already_tested"])

    def test_limit_and_markdown_use_shown_entries_but_counts_describe_all_matches(self):
        self.store("HIGH", fitness=1.0)
        self.store("ZERO", fitness=0.0)
        self.store("NEGATIVE", fitness=-0.5)
        data = self.ledger("list", "--limit", "2", "--format", "markdown")["data"]
        self.assertEqual([row["alpha_id"] for row in data["entries"]], ["HIGH", "ZERO"])
        self.assertEqual(data["shown"], 2)
        self.assertEqual(data["matched"], 3)
        self.assertEqual(data["summary"]["total"], 3)
        self.assertIn("HIGH", data["markdown"])
        self.assertIn("ZERO", data["markdown"])
        self.assertNotIn("NEGATIVE", data["markdown"])

    def test_label_list_roundtrip_appends_history_and_preserves_measurements(self):
        original = self.store("A")
        labelled = self.ledger("label", "A", "--family", "event", "--verdict", "rejected",
                               "--mechanism", "event drift", "--idea", "Idea 99",
                               "--note", "control falsified the mechanism")["data"]
        self.assertTrue(labelled["written"])
        data = self.ledger("list", "--family", "event", "--verdict", "rejected")["data"]
        self.assertEqual(data["registry_entries"], 1)
        self.assertEqual(data["entries"], [labelled["entry"]])
        self.assertEqual(data["entries"][0]["metrics"], original["metrics"])
        self.assertEqual(data["entries"][0]["note"], "control falsified the mechanism")
        history = [json.loads(line) for line in Path(self.registry).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["family"], "quality")
        self.assertEqual(history[1]["family"], "event")

    def test_history_append_exposes_only_latest_metrics(self):
        self.store("A", fitness=0.2, sharpe=0.5)
        self.store("A", fitness=1.6, sharpe=2.0)
        data = self.ledger("list", "--min-fitness", "1.0", "--min-sharpe", "1.5")["data"]
        self.assertEqual(data["registry_entries"], 1)
        self.assertEqual(data["matched"], 1)
        self.assertEqual(data["entries"][0]["metrics"]["fitness"], 1.6)

    def test_record_fetches_current_metrics_and_persists_labels_and_split(self):
        payload = {
            "id": "ONLINE", "regular": {"code": "rank(close)"},
            "settings": {"region": "USA", "testPeriod": "P1Y"},
            "status": "UNSUBMITTED",
            "is": {"sharpe": 1.7, "fitness": 1.2,
                   "checks": [{"name": "LOW_SHARPE", "result": "PASS"}]},
            "train": {"sharpe": 1.9}, "test": {"sharpe": 0.7},
        }
        with mock.patch.object(cli.ml, "load_session", return_value=object()), \
                mock.patch.object(cli.ml, "fetch_alpha_metrics", return_value=payload):
            recorded = self.ledger("record", "ONLINE", "--family", "quality",
                                   "--verdict", "candidate", "--mechanism", "quality drift",
                                   "--idea", "Idea 81", "--note", "current platform read")["data"]
        self.assertTrue(recorded["appended"])
        listed = self.ledger("list", "--field", "close")["data"]["entries"]
        self.assertEqual(listed, [recorded["entry"]])
        self.assertEqual(listed[0]["metrics"]["sharpe"], 1.7)
        self.assertEqual(listed[0]["family"], "quality")
        self.assertEqual(listed[0]["note"], "current platform read")
        self.assertEqual(listed[0]["out_of_sample"]["test"]["sharpe"], 0.7)

    def test_impact_lineage_and_coverage_remain_offline(self):
        parent = self.store("PARENT", expression=COMPOSITE)
        self.store("CHILD", expression=COMPOSITE, parent=parent["expression_hash"], op="reweight")
        self.assertEqual(self.ledger("impact", "PARENT")["data"]["rows"][0]["alpha_id"], "CHILD")
        self.assertEqual([row["alpha_id"] for row in self.ledger("lineage", "CHILD")["data"]["chain"]],
                         ["CHILD", "PARENT"])
        self.assertEqual(self.ledger("coverage")["data"]["summary"]["total"], 2)

    def test_empty_ledger_and_unmatched_filters_are_successful_lists(self):
        data = self.ledger("list")["data"]
        self.assertEqual(data["entries"], [])
        self.assertFalse(data["already_tested"])
        self.store("A")
        data = self.ledger("list", "--field", "never_used")["data"]
        self.assertEqual(data["registry_entries"], 1)
        self.assertEqual(data["matched"], 0)
        self.assertEqual(data["entries"], [])

    def test_negative_limit_is_rejected(self):
        payload = self.ledger("list", "--limit", "-1", expected=2)
        self.assertEqual(payload["error"]["code"], "INVALID_INPUT")


if __name__ == "__main__":
    unittest.main()