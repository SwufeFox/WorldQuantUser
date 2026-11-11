"""The inspiration stream: derive, dedupe, and refuse to auto-promote."""

import json
import os
import tempfile
import unittest
from unittest import mock

import research_paths
from research_app import feed


THREADS = """# open
- **H-070（2026-11-11，受控）**：同一对信号分别写成 plain 与 centred 版，两批各三条。
- **递延收入四种读法**：分母决定一切，增速类全废。
- [closed] PEAD（财报意外，事件掩码）—— 安慰剂 0.86，不可区分
"""


class FeedTests(unittest.TestCase):
    def test_questions_skip_closed_lines_and_keep_the_heading(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "open-threads.md")
        open(path, "w", encoding="utf-8").write(THREADS)
        questions = feed.open_questions(path)
        self.assertEqual(len(questions), 2)
        self.assertIn("递延收入四种读法", questions[1])
        self.assertFalse(any("PEAD" in q for q in questions))

    def test_a_query_is_distinctive_words_not_the_whole_sentence(self):
        query = feed.query_from_question("H-070 受控：同一对信号分别写成 plain 与 centred 版")
        self.assertTrue(query)
        self.assertLessEqual(len(query.split()), 8)

    def test_closed_markers_drop_a_paper_by_title(self):
        self.assertTrue(feed.is_closed("Post-Earnings Announcement Drift revisited"))
        self.assertFalse(feed.is_closed("Receivables and deferred revenue quality"))

    def test_the_seen_store_makes_a_second_run_incremental(self):
        tmp = tempfile.mkdtemp()
        seen_path = os.path.join(tmp, "seen.jsonl")
        paper = {"title": "Modelling Receivables and Deferred Revenues to Predict Returns",
                 "abstract": "cross-section of stock returns",
                 "doi": "10.1/x", "year": 2018, "citations": 33, "journal": "JAR"}

        class Args:
            topic = ["receivables"]
            from_threads = False
            limit = 3
            total = 3
            max_topics = 2
            from_year = 2005
            min_citations = 0
            queue = None
            seen = None
            draft_cards = None
            dry_run = False

        args = Args()
        args.seen = seen_path
        args.queue = os.path.join(tmp, "queue.jsonl")

        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp), \
                mock.patch("paper_search.search_papers",
                           return_value={"papers": [paper]}):
            first = feed.cmd_feed(args, None)
            second = feed.cmd_feed(args, None)
        self.assertEqual(first["count"], 1)
        self.assertEqual(second["count"], 0)
        self.assertEqual(second["seen_before"], 1)

    def test_draft_cards_are_unpromotable_on_purpose(self):
        tmp = tempfile.mkdtemp()
        draft = os.path.join(tmp, "drafts.jsonl")
        paper = {"title": "Modelling Receivables and Deferred Revenues to Predict Returns",
                 "doi": "10.1/x", "year": 2018, "citations": 33, "journal": "JAR",
                 "abstract": "cross-section of stock returns"}

        class Args:
            topic = ["receivables"]
            from_threads = False
            limit = 2
            total = 2
            max_topics = 1
            from_year = 2005
            min_citations = 0
            queue = None
            seen = None
            draft_cards = None
            dry_run = False

        args = Args()
        args.seen = os.path.join(tmp, "seen.jsonl")
        args.queue = os.path.join(tmp, "queue.jsonl")
        args.draft_cards = draft

        with mock.patch.object(research_paths, "ensure_dir", return_value=tmp), \
                mock.patch("paper_search.search_papers", return_value={"papers": [paper]}):
            feed.cmd_feed(args, None)
        card = json.loads(open(draft, encoding="utf-8").read().strip())
        self.assertEqual(card["status"], "DRAFT")
        self.assertIsNone(card["hypothesis"]["falsification"])
        self.assertIsNone(card["novelty"]["closest_existing_family"])
