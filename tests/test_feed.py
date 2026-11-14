"""Retrieval preserves evidence, query outcomes and limits on local knowledge."""

import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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
                      draft_cards=None, dry_run=False, registry=None, workers=4, timeout=30)
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

    def test_a_bare_field_or_operator_name_is_not_a_query(self):
        # 2026-11-12: these four went out as queries and the digest answered with a
        # biofilm paper, a bone-biology paper, photoemission physics and bread staling.
        for heading in ("销售离散度为什么远强于 EPS 分歧",
                        "「必须组内 rank」是不是构造定律",
                        "delay 0 只差 0.23",
                        "缓冲里六条 stale 的取舍"):
            self.assertEqual(feed.query_from_question(heading), "", heading)
        self.assertEqual(feed.query_from_question("stale news and investor reaction"),
                         "stale news investor reaction")

    def test_a_derived_query_needs_two_content_words(self):
        self.assertEqual(feed.query_from_question("accruals"), "")
        self.assertEqual(feed.query_from_question("accruals quality"), "accruals quality")

    def test_a_thread_can_carry_its_own_english_query(self):
        self.assertEqual(feed.explicit_query(
            "- [open] **销售分歧** — 证据；query: earnings forecast dispersion"), 
            "earnings forecast dispersion")
        self.assertIsNone(feed.explicit_query("- [open] **销售分歧** — 证据"))

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
        outcomes = {"first": {"papers": [self.paper()]},
                    "empty": {"papers": []}, "last": {"papers": [self.paper("Later")]}}
        def search_topic(topic, **_options):
            if topic == "broken":
                raise RuntimeError("token=SECRET https://private")
            return outcomes[topic]
        with mock.patch("paper_search.search_papers", side_effect=search_topic) as search:
            result = feed.cmd_feed(args, None)
        self.assertCountEqual([call.args[0] for call in search.call_args_list],
                              ["first", "broken", "empty", "last"])
        self.assertEqual([row["status"] for row in result["searches"]],
                         ["success", "failure", "empty", "success"])
        self.assertEqual(result["searches"][1]["error"],
                         {"code": "SEARCH_FAILED", "type": "RuntimeError"})
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(result["unsearched_topics"], ["deferred"])

    def test_malformed_response_is_distinct_from_empty_success(self):
        outcomes = {"bad": {"unexpected": []}, "usable": {"papers": [None, {}, self.paper()]}}
        with mock.patch("paper_search.search_papers", side_effect=lambda topic, **_: outcomes[topic]):
            result = feed.cmd_feed(self.args(topic=["bad", "usable"]), None)
        self.assertEqual(result["searches"][0]["status"], "failure")
        self.assertEqual(result["searches"][0]["error"]["code"], "INVALID_RESPONSE")
        self.assertEqual(result["searches"][1]["invalid"], 2)
        self.assertEqual(result["count"], 1)

    def test_round_robin_selection_balances_topics_and_retains_shared_provenance(self):
        first = self.paper("First", citations=1000)
        second = self.paper("Second", citations=900)
        later = self.paper("Later", citations=1)
        outcomes = {"alpha": {"papers": [second, first]}, "beta": {"papers": [first, later]}}
        with mock.patch("paper_search.search_papers", side_effect=lambda topic, **_: outcomes[topic]):
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
        paper = self.paper(abstract="evidence " * 200,
                           landing_page_url="https://publisher.example/article",
                           pdf_url="https://publisher.example/article.pdf")
        result = self.run_feed([paper], self.args(draft_cards=draft))
        row = result["papers"][0]
        self.assertEqual(row["doi"], paper["doi"])
        self.assertEqual(row["url"], paper["url"])
        self.assertEqual(row["landing_page_url"], paper["landing_page_url"])
        self.assertEqual(row["pdf_url"], paper["pdf_url"])
        self.assertEqual(row["abstract"], paper["abstract"][:600])
        with open(draft, encoding="utf-8") as handle:
            card = json.loads(handle.read())
        self.assertEqual(card["source"]["url"], paper["url"])
        self.assertEqual(card["source"]["landing_page_url"], paper["landing_page_url"])
        self.assertEqual(card["source"]["pdf_url"], paper["pdf_url"])
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
        contrasts = [action for action in result["next_actions"]
                     if action["kind"] == "contrast_prior_evidence"]
        self.assertEqual(len(contrasts), 1)
        self.assertEqual(contrasts[0]["paper_index"], 0)
        self.assertEqual(contrasts[0]["source"], registry)
        self.assertEqual(contrasts[0]["evidence_keys"], ["active"])
        self.assertIn("not a joint transaction", result["note"])
        self.assertNotIn("abstract", json.dumps(result["next_actions"]))

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

    def test_explicit_queries_apply_to_plain_and_bold_threads_in_one_read(self):
        path = self.write_file("open-threads.md", """- [open] **销售分歧** — 证据；query: earnings forecast dispersion
- [open] 递延收入；query： deferred revenue quality
- **旧标题**：证据; query: supplier network pricing
- [open] **销售分歧**；query: analyst coverage information
- [closed] 旧问题; query: must never search
- [open] **无英文问题** — 证据
""")
        original_open = open
        with mock.patch("builtins.open", wraps=original_open) as opened, \
                mock.patch("paper_search.search_papers", return_value={"papers": []}) as search:
            result = feed.cmd_feed(self.args(topic=None, from_threads=True), None)
        self.assertEqual(result["topics"], ["earnings forecast dispersion", "deferred revenue quality",
                                             "supplier network pricing", "analyst coverage information"])
        self.assertCountEqual([call.args[0] for call in search.call_args_list], result["topics"])
        reads = [call for call in opened.call_args_list if os.fspath(call.args[0]) == path]
        self.assertEqual(len(reads), 1)
        self.assertEqual([row["question"] for row in result["questions_without_query"]], ["无英文问题"])

    def test_workers_overlap_provider_calls_and_enforce_the_bound(self):
        entered = threading.Event()
        release = threading.Event()
        guard = threading.Lock()
        active = 0
        peak = 0
        topics = ["topic-%d" % index for index in range(6)]

        def search(topic, **options):
            nonlocal active, peak
            self.assertEqual(options["timeout"], 7.5)
            with guard:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    entered.set()
            try:
                if not release.wait(5):
                    raise RuntimeError("test did not release provider calls")
                return {"papers": [self.paper(topic)]}
            finally:
                with guard:
                    active -= 1

        with mock.patch("paper_search.search_papers", side_effect=search) as provider, \
                mock.patch.object(feed.runtime, "note") as progress, \
                ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(feed.cmd_feed, self.args(
                topic=topics, max_topics=6, workers=2, timeout=7.5, total=6, dry_run=True), None)
            try:
                self.assertTrue(entered.wait(5), "two provider calls must overlap")
                with guard:
                    self.assertEqual(active, 2)
            finally:
                release.set()
            result = future.result(timeout=5)
        self.assertEqual(peak, 2)
        self.assertEqual(provider.call_count, 6)
        self.assertEqual(progress.call_count, 6)
        self.assertEqual(result["count"], 6)

    def test_reverse_completion_keeps_reports_picks_and_provenance_in_topic_order(self):
        topics = ["alpha", "beta", "gamma"]
        started = threading.Barrier(4, timeout=5)
        release = {topic: threading.Event() for topic in topics}
        finished = [threading.Event() for _ in topics]
        completion_order = []
        shared = self.paper("Shared", citations=1000)
        outcomes = {"alpha": [shared, self.paper("Alpha")],
                    "beta": [shared, self.paper("Beta")], "gamma": [self.paper("Gamma")]}

        def search(topic, **_options):
            started.wait()
            if not release[topic].wait(5):
                raise RuntimeError("test did not release provider call")
            return {"papers": outcomes[topic]}

        def note(message):
            index = int(message.split()[2].split("/")[0]) - 1
            completion_order.append(index)
            finished[index].set()

        with mock.patch("paper_search.search_papers", side_effect=search), \
                mock.patch.object(feed.runtime, "note", side_effect=note), \
                ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(feed.cmd_feed, self.args(topic=topics, workers=3, dry_run=True), None)
            try:
                started.wait()
                for index in (2, 1, 0):
                    release[topics[index]].set()
                    self.assertTrue(finished[index].wait(5))
                result = future.result(timeout=5)
            finally:
                for event in release.values():
                    event.set()
        self.assertEqual(completion_order, [2, 1, 0])
        self.assertEqual([row["query"] for row in result["searches"]], topics)
        self.assertEqual([row["title"] for row in result["papers"]], ["Shared", "Beta", "Gamma"])
        self.assertEqual(result["papers"][0]["query"], "alpha")
        self.assertEqual(result["papers"][0]["queries"], ["alpha", "beta"])
        self.assertEqual([row["picked"] for row in result["searches"]], [1, 1, 1])

    def test_concurrent_feeds_select_unique_picks_after_network_finishes(self):
        providers = threading.Barrier(2, timeout=5)
        papers = [self.paper("Paper-%d" % index) for index in range(4)]
        draft = os.path.join(self.root, "drafts.jsonl")

        def search(_topic, **_options):
            # Holding the seen lock during provider I/O would deadlock this barrier.
            providers.wait()
            return {"papers": papers}

        with mock.patch("paper_search.search_papers", side_effect=search), \
                ThreadPoolExecutor(max_workers=2) as runner:
            futures = [runner.submit(feed.cmd_feed, self.args(total=2, draft_cards=draft), None)
                       for _ in range(2)]
            results = [future.result(timeout=10) for future in futures]
        picks = [{row["doi"] for row in result["papers"]} for result in results]
        self.assertEqual([len(keys) for keys in picks], [2, 2])
        self.assertFalse(picks[0] & picks[1])
        self.assertEqual(picks[0] | picks[1], {row["doi"] for row in papers})
        self.assertEqual(sorted(result["seen_before"] for result in results), [0, 2])
        self.assertEqual(len(feed.load_seen(results[0]["seen"])), 4)
        with open(draft, encoding="utf-8") as handle:
            cards = [json.loads(line) for line in handle]
        self.assertEqual(len(cards), 4)
        self.assertEqual({card["source"]["doi"] for card in cards}, picks[0] | picks[1])

    def test_invalid_provider_rows_do_not_discard_other_evidence(self):
        rows = [None, [], {}, self.paper("Bad DOI", doi=[]),
                self.paper("Bad rank", citations="unknown"),
                self.paper("Bad abstract", abstract={"text": "invalid"}),
                self.paper("Not finite", citations=float("nan")), self.paper("Usable")]
        # A nonempty, non-string DOI cannot become a stable key.
        rows[3]["doi"] = ["10.1/invalid"]
        result = self.run_feed(rows)
        self.assertEqual(result["searches"][0]["invalid"], 7)
        self.assertEqual([row["title"] for row in result["papers"]], ["Usable"])

    def test_seen_commit_preserves_prior_bytes_and_flushes_before_replace(self):
        seen = os.path.join(self.root, "seen.jsonl")
        previous = b'{"key":"older","note":"keep"}\r\nnull\nnot-json\n{"key":"last"}'
        with open(seen, "wb") as handle:
            handle.write(previous)
        calls = []
        fsync, replace = os.fsync, os.replace

        def flush(descriptor):
            calls.append("fsync")
            return fsync(descriptor)

        def commit(source, destination):
            self.assertTrue(calls)
            self.assertEqual(calls[-1], "fsync")
            calls.append("replace")
            return replace(source, destination)

        with mock.patch.object(feed.os, "fsync", side_effect=flush), \
                mock.patch.object(feed.os, "replace", side_effect=commit):
            result = self.run_feed([self.paper()], self.args(seen=seen))
        with open(seen, "rb") as handle:
            content = handle.read()
        self.assertTrue(content.startswith(previous + b"\n"))
        self.assertEqual(feed.load_seen(seen), {"older", "last", feed.paper_key(self.paper())})
        self.assertEqual(json.loads(content[len(previous) + 1:])["title"], "Receivables")
        self.assertEqual(result["count"], 1)
        self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(self.root)))

    def test_draft_replace_failure_does_not_consume_picks_or_change_prior_files(self):
        seen = self.write_file("seen.jsonl", '{"key":"older"}\n')
        draft = self.write_file("drafts.jsonl", '{"status":"DRAFT","note":"older"}\n')
        replace = os.replace

        def commit(source, destination):
            if destination == os.path.realpath(draft):
                raise PermissionError("private filesystem detail")
            return replace(source, destination)

        with mock.patch.object(feed.os, "replace", side_effect=commit):
            with self.assertRaises(feed.CliError) as failure:
                self.run_feed([self.paper()], self.args(seen=seen, draft_cards=draft))
        self.assertEqual(failure.exception.code, "INVALID_INPUT")
        self.assertNotIn("private", failure.exception.message)
        self.assertEqual(feed.load_seen(seen), {"older"})
        with open(draft, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), '{"status":"DRAFT","note":"older"}\n')
        self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(self.root)))
        retried = self.run_feed([self.paper()], self.args(seen=seen, draft_cards=draft))
        self.assertEqual(retried["count"], 1)

    def test_seen_replace_failure_preserves_state_and_explains_repeatable_drafts(self):
        seen = self.write_file("seen.jsonl", '{"key":"older"}\n')
        draft = os.path.join(self.root, "drafts.jsonl")
        replace = os.replace

        def commit(source, destination):
            if destination == os.path.realpath(seen):
                raise PermissionError("write failed")
            return replace(source, destination)

        with mock.patch.object(feed.os, "replace", side_effect=commit):
            with self.assertRaises(feed.CliError) as failure:
                self.run_feed([self.paper()], self.args(seen=seen, draft_cards=draft))
        self.assertEqual(feed.load_seen(seen), {"older"})
        self.assertIn("repeat draft cards", failure.exception.hint)
        with open(draft, encoding="utf-8") as handle:
            self.assertEqual(json.loads(handle.read())["source"]["doi"], self.paper()["doi"])
        self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(self.root)))
        result = self.run_feed([self.paper()], self.args(seen=seen, draft_cards=draft))
        self.assertEqual(result["count"], 1)
        self.assertIn("not a joint transaction", result["note"])
        with open(draft, encoding="utf-8") as handle:
            self.assertEqual(len(handle.readlines()), 2)

    def test_unreadable_seen_state_fails_closed(self):
        with mock.patch.object(feed, "_read_jsonl", side_effect=PermissionError("private")), \
                mock.patch.object(feed, "_append_jsonl_atomic") as commit:
            with self.assertRaises(feed.CliError):
                self.run_feed([self.paper()])
        commit.assert_not_called()

    def test_next_actions_are_bounded_attributable_and_retries_keep_exact_options(self):
        self.write_file("open-threads.md", "\n".join(
            "- [open] **问题%d** — 证据" % index for index in range(8)))
        failed = ["failed-%d" % index for index in range(8)]
        deferred = ["deferred-%d" % index for index in range(8)]
        papers = [self.paper("Source-%d" % index,
                             landing_page_url="https://publisher.example/%d" % index,
                             pdf_url="https://publisher.example/%d.pdf" % index) for index in range(3)]
        args = self.args(topic=failed + ["usable"] + deferred, from_threads=True,
                         max_topics=9, limit=7, from_year=2011, min_citations=9,
                         workers=2, timeout=7.5, total=3, dry_run=True, action=True,
                         draft_cards=os.path.join(self.root, "drafts.jsonl"))

        def search(topic, **_options):
            if topic in failed:
                raise RuntimeError("SECRET provider detail")
            return {"papers": papers}

        with mock.patch("paper_search.search_papers", side_effect=search), \
                mock.patch.object(feed.runtime, "note") as progress:
            result = feed.cmd_feed(args, None)
        actions = result["next_actions"]
        self.assertEqual(len(actions), 11)
        retries = [row for row in actions if row["kind"] == "retry_search"]
        self.assertEqual([row["query"] for row in retries], failed[:3])
        self.assertEqual(retries[0]["argv"], [
            "--research-dir", self.root, "feed", "--topic", failed[0],
            "--limit", "7", "--total", "3", "--max-topics", "1", "--from-year", "2011",
            "--min-citations", "9", "--workers", "2", "--timeout", "7.5",
            "--seen", result["seen"], "--draft-cards", args.draft_cards, "--dry-run", "--action"])
        self.assertEqual(retries[0]["error"], result["searches"][0]["error"])
        self.assertEqual(len([row for row in actions if row["kind"] == "supply_query"]), 3)
        pending = next(row for row in actions if row["kind"] == "search_deferred")
        self.assertEqual(pending["queries"], deferred[:3])
        self.assertEqual(pending["remaining"], 5)
        reviews = [row for row in actions if row["kind"] == "review_source"]
        self.assertEqual([row["paper_index"] for row in reviews], [0, 1, 2])
        self.assertEqual(reviews[0]["sources"]["pdf_url"], papers[0]["pdf_url"])
        contrast = next(row for row in actions if row["kind"] == "contrast_sources")
        self.assertEqual(contrast["paper_indices"], [0, 1])
        self.assertNotIn("abstract", json.dumps(actions))
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertNotIn("SECRET", str(progress.call_args_list))
        self.assertFalse(os.path.exists(result["seen"]))
        self.assertFalse(os.path.exists(args.draft_cards))


class DataFitTests(unittest.TestCase):
    """Reading keywords do not establish account capabilities or signal validity."""

    def test_keyword_inputs_remain_inferred(self):
        fit = feed.data_fit({"title": "Short interest and the cross-section of stock returns",
                             "abstract": "We study momentum and short interest."})
        self.assertEqual(fit["verdict"], "unknown")
        needs = {item["need"] for item in fit["needs"]}
        self.assertIn("lending and short interest", needs)

    def test_estimate_keywords_do_not_prove_field_availability(self):
        fit = feed.data_fit({"title": "Analyst forecast dispersion and returns",
                             "abstract": "Using consensus estimates we study revision."})
        self.assertEqual(fit["verdict"], "unknown")
        self.assertTrue(all(item["state"] == "inferred" for item in fit["needs"]))

    def test_an_unmatched_paper_stays_unknown_instead_of_guessing(self):
        fit = feed.data_fit({"title": "A history of double-entry bookkeeping",
                             "abstract": "Accounting practice in Renaissance Italy."})
        self.assertEqual(fit["verdict"], "unknown")
        self.assertEqual(fit["needs"], [])
