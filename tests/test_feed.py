"""Retrieval preserves evidence, query outcomes and limits on local knowledge."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import research_paths
from research_app import feed


class FeedTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = self.directory.name
        patcher = mock.patch.object(research_paths, "research_dir", return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, **overrides):
        values = dict(topic=["receivables"], from_threads=False, topics=False,
                      topics_file=None, limit=5, total=3, max_topics=4,
                      from_year=2005, min_citations=0, seen=None,
                      draft_cards=None, dry_run=False, registry=None)
        values.update(overrides)
        return SimpleNamespace(**values)

    def write_file(self, name, text):
        path = os.path.join(self.root, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def paper(self, title="Receivables", **overrides):
        row = dict(title=title, abstract="Mechanism and identification evidence.",
                   doi="10.1/" + title, url="https://example.org/" + title,
                   year=2018, citations=33, journal="JAR")
        row.update(overrides)
        return row

    def run_feed(self, papers, args=None, action=False):
        with mock.patch("paper_search.search_papers", return_value={"papers": papers}):
            command = feed.cmd_feed_action if action else feed.cmd_feed
            return command(args or self.args(), None)

    def test_questions_accept_documented_tags_and_legacy_bold_headings(self):
        path = self.write_file("open-threads.md", """# Open
- [open] **Supplier network pricing** — evidence; next check
- [OPEN] Plain query without punctuation
- [open] Plain heading：evidence to exclude
- [open] Plain heading two — evidence to exclude
- **H-070（2026-11-11，受控）**：plain and centred are evidence, not the heading
- **递延收入四种读法**：分母决定一切
- Legacy plain heading: evidence
- [closed] **PEAD** — conclusion
- [CLOSED] Old direction: conclusion
- untagged prose is not a question
""")
        self.assertEqual(feed.open_questions(path), [
            "Supplier network pricing", "Plain query without punctuation", "Plain heading",
            "Plain heading two", "H-070（2026-11-11，受控）", "递延收入四种读法",
            "Legacy plain heading"])

    def test_query_excludes_codes_and_uses_only_distinctive_tokens(self):
        self.assertEqual(feed.query_from_question("H-070 2026 递延收入"), "")
        query = feed.query_from_question("H-070 supplier network pricing and information diffusion")
        self.assertEqual(query, "supplier network pricing information diffusion")

    def test_unsearchable_questions_remain_visible_even_without_any_query(self):
        path = self.write_file("open-threads.md", "- [open] **递延收入四种读法** — evidence\n")
        with mock.patch("paper_search.search_papers") as search:
            result = feed.cmd_feed(self.args(topic=None, from_threads=True), None)
        search.assert_not_called()
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["questions_without_query"], [{
            "question": "递延收入四种读法", "source": path,
            "reason": "no usable English query tokens"}])

    def test_topics_file_augments_threads_without_hiding_unsearchable_questions(self):
        self.write_file("open-threads.md", "- [open] **递延收入** — evidence\n")
        topics = self.write_file("topics.txt", "# English queries\nSupplier network\nsupplier   network\n")
        with mock.patch("paper_search.search_papers", return_value={"papers": []}) as search:
            result = feed.cmd_feed(self.args(topic=None, from_threads=True, topics=True,
                                             topics_file=topics), None)
        self.assertEqual(search.call_count, 1)
        self.assertEqual(result["topics"], ["Supplier network"])
        self.assertEqual(len(result["questions_without_query"]), 1)

    def test_failures_empty_and_success_are_reported_and_all_topics_searched(self):
        args = self.args(topic=["first", " FIRST ", "broken", "empty", "last", "deferred"],
                         max_topics=4, total=1)
        with mock.patch("paper_search.search_papers", side_effect=[
                {"papers": [self.paper()]}, RuntimeError("token=SECRET https://private"),
                {"papers": []}, {"papers": [self.paper("Later")]}]) as search:
            result = feed.cmd_feed(args, None)
        self.assertEqual([call.args[0] for call in search.call_args_list],
                         ["first", "broken", "empty", "last"])
        self.assertEqual([row["status"] for row in result["searches"]],
                         ["success", "failure", "empty", "success"])
        self.assertEqual(result["searches"][1]["error"],
                         {"code": "SEARCH_FAILED", "type": "RuntimeError"})
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(result["unsearched_topics"], ["deferred"])

    def test_malformed_response_is_distinct_from_empty_success(self):
        with mock.patch("paper_search.search_papers", side_effect=[
                {"unexpected": []}, {"papers": [None, {}, self.paper()]}]):
            result = feed.cmd_feed(self.args(topic=["bad", "usable"]), None)
        self.assertEqual(result["searches"][0]["status"], "failure")
        self.assertEqual(result["searches"][0]["error"]["code"], "INVALID_RESPONSE")
        self.assertEqual(result["searches"][1]["invalid"], 2)
        self.assertEqual(result["count"], 1)

    def test_round_robin_selection_balances_topics_and_retains_shared_provenance(self):
        first = self.paper("First", citations=1000)
        second = self.paper("Second", citations=900)
        later = self.paper("Later", citations=1)
        with mock.patch("paper_search.search_papers", side_effect=[
                {"papers": [second, first]}, {"papers": [first, later]}]):
            result = feed.cmd_feed(self.args(topic=["alpha", "beta"], total=2), None)
        self.assertEqual([row["title"] for row in result["papers"]], ["First", "Later"])
        self.assertEqual(result["papers"][0]["queries"], ["alpha", "beta"])
        self.assertEqual([row["picked"] for row in result["searches"]], [1, 1])
        self.assertEqual(feed.load_seen(result["seen"]), {feed.paper_key(first), feed.paper_key(later)})
        again = self.run_feed([first, second, later])
        self.assertEqual([row["title"] for row in again["papers"]], ["Second"])
        self.assertEqual(again["seen_before"], 2)

    def test_titles_and_absent_relevance_keywords_never_suppress_papers(self):
        papers = [self.paper("PEAD placebo tests"),
                  self.paper("Post-earnings announcement drift"),
                  self.paper("Supply chain organisation", abstract="Network structure")]
        result = self.run_feed(papers)
        self.assertEqual(result["count"], 3)
        self.assertNotIn("dropped_closed", result)
        self.assertNotIn("queue", result)
        self.assertFalse(os.path.exists(os.path.join(self.root, "inspiration", "queue.jsonl")))

    def test_closure_caution_requires_valid_exact_match_and_retains_attribution(self):
        path = self.write_file("closed-families.json", json.dumps({"closed": [
            {"id": "option-test", "match": {"family": "option"}, "reason": "IV construction failed"},
            {"id": "prose", "reason": "PEAD placebo was inconclusive"},
            {"id": "scoped", "match": {"family": "option", "construction": "specific"}},
            {"id": "invalid", "match": "option"}, None]}))
        result = self.run_feed([self.paper("Declared", family="option"),
                                self.paper("Option implied volatility"),
                                self.paper("Different", family="quality")])
        self.assertEqual(result["count"], 3)
        by_title = {row["title"]: row for row in result["papers"]}
        caution = by_title["Declared"]["closure_cautions"]
        self.assertEqual(len(caution), 1)
        self.assertEqual(caution[0]["source"], path)
        self.assertEqual(caution[0]["id"], "option-test")
        self.assertEqual(caution[0]["reason"], "IV construction failed")
        self.assertEqual(by_title["Option implied volatility"]["closure_cautions"], [])
        self.assertEqual(by_title["Different"]["closure_cautions"], [])

    def test_missing_register_is_visible_and_never_suppresses_papers(self):
        result = self.run_feed([self.paper("Placebo identification")])
        self.assertEqual(result["count"], 1)
        self.assertFalse(result["closed_register"]["available"])
        self.assertEqual(result["papers"][0]["closure_cautions"], [])

    def test_dry_run_does_not_create_research_seen_or_draft_directories(self):
        absent = os.path.join(self.root, "absent")
        args = self.args(dry_run=True, draft_cards=os.path.join(absent, "drafts.jsonl"))
        with mock.patch.object(research_paths, "research_dir", return_value=absent):
            result = self.run_feed([self.paper()], args)
        self.assertEqual(result["count"], 1)
        self.assertFalse(os.path.exists(absent))

    def test_dry_run_leaves_existing_seen_and_drafts_byte_identical(self):
        seen = self.write_file("seen.jsonl", '{"key":"older"}\n')
        draft = self.write_file("draft.jsonl", '{"status":"DRAFT"}\n')
        self.run_feed([self.paper()], self.args(dry_run=True, seen=seen, draft_cards=draft))
        with open(seen, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), '{"key":"older"}\n')
        with open(draft, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), '{"status":"DRAFT"}\n')

    def test_feed_and_draft_retain_source_links_and_bounded_abstract(self):
        draft = os.path.join(self.root, "drafts.jsonl")
        paper = self.paper(abstract="evidence " * 200)
        result = self.run_feed([paper], self.args(draft_cards=draft))
        row = result["papers"][0]
        self.assertEqual(row["doi"], paper["doi"])
        self.assertEqual(row["url"], paper["url"])
        self.assertEqual(row["abstract"], paper["abstract"][:600])
        with open(draft, encoding="utf-8") as handle:
            card = json.loads(handle.read())
        self.assertEqual(card["source"]["url"], paper["url"])
        self.assertEqual(card["abstract"], row["abstract"])
        self.assertIsNone(card["hypothesis"]["falsification"])
        self.assertIsNone(card["novelty"]["closest_existing_family"])

    def test_action_separates_book_exposure_prior_trials_and_unknown_blocks(self):
        registry = self.write_file("alpha-registry.jsonl", "\n".join(json.dumps(row) for row in [
            {"key": "active", "status": "ACTIVE", "expression": "rank(anl4_eps_std)"},
            {"key": "failed", "status": "COMPLETE", "expression": "rank(news_sentiment)",
             "verdict": "falsified", "note": "control won", "batch": "test-1"},
            {"key": "blocked", "status": "ERROR", "expression": "rank(news_sentiment)",
             "verdict": "no_go"}]))
        papers = [self.paper("Analyst forecast dispersion"),
                  self.paper("Other research", abstract="news sentiment and identification"),
                  self.paper("Patent originality", abstract="innovation")]
        # Use the established classifier's interface to isolate the ledger evidence contract.
        def classify(expression):
            return "analyst_dispersion" if "anl4" in expression else "news_sentiment"
        with mock.patch("research_app.mechanisms.block_of", side_effect=classify):
            result = self.run_feed(papers, self.args(registry=registry), action=True)
        self.assertNotIn("action", result)
        active, tested, unknown = result["papers"]
        self.assertEqual(active["book_exposure"]["count"], 1)
        self.assertEqual(tested["book_exposure"]["count"], 0)
        self.assertEqual(tested["prior_trials"]["count"], 1)
        self.assertEqual(tested["prior_trials"]["ledger_entries"], 2)
        self.assertEqual(tested["prior_trials"]["evidence"][0]["verdict"], "falsified")
        self.assertEqual(tested["prior_trials"]["evidence"][0]["note"], "control won")
        self.assertEqual(tested["prior_trials"]["source"], registry)
        self.assertIsNone(unknown["book_exposure"]["count"])
        self.assertIsNone(unknown["prior_trials"]["count"])
        self.assertEqual(tested["block_guess"], "news_sentiment")
        for row, source in zip(result["papers"], papers):
            self.assertEqual(row["doi"], source["doi"])
            self.assertEqual(row["url"], source["url"])
            self.assertEqual(row["abstract"], source["abstract"])

    def test_action_missing_registry_keeps_exposure_and_trials_unknown(self):
        result = self.run_feed([self.paper("Analyst forecast dispersion")], action=True)
        self.assertFalse(result["registry"]["exists"])
        row = result["papers"][0]
        self.assertIsNone(row["book_exposure"]["count"])
        self.assertIsNone(row["prior_trials"]["count"])

    def test_action_empty_registry_reports_only_zero_local_matches(self):
        registry = self.write_file("alpha-registry.jsonl", "")
        result = self.run_feed([self.paper("Analyst forecast dispersion")],
                               self.args(registry=registry), action=True)
        self.assertEqual(result["papers"][0]["book_exposure"]["count"], 0)
        self.assertEqual(result["papers"][0]["prior_trials"]["count"], 0)
        self.assertEqual(result["papers"][0]["prior_trials"]["evidence"], [])

    def test_action_bounds_evidence_but_retains_total_and_registry_source(self):
        registry = self.write_file("alpha-registry.jsonl", "\n".join(json.dumps({
            "key": "trial-%d" % index, "status": "COMPLETE",
            "expression": "rank(anl4_eps_std)", "note": "evidence " * 100,
            "mechanism": "mechanism " * 100,
        }) for index in range(8)))
        with mock.patch("research_app.mechanisms.block_of", return_value="analyst_dispersion"):
            result = self.run_feed([self.paper("Analyst forecast dispersion")],
                                   self.args(registry=registry), action=True)
        prior = result["papers"][0]["prior_trials"]
        self.assertEqual(prior["count"], 8)
        self.assertEqual(prior["ledger_entries"], 8)
        self.assertEqual(prior["evidence_shown"], 5)
        self.assertEqual(len(prior["evidence"]), 5)
        self.assertEqual(prior["source"], registry)
        self.assertEqual(prior["evidence"][0]["key"], "trial-3")
        self.assertLessEqual(len(prior["evidence"][0]["note"]), 240)
        self.assertLessEqual(len(prior["evidence"][0]["mechanism"]), 160)
