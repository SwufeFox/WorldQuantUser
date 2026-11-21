#!/usr/bin/env python3
"""Tests for offline lint rules that catch what the simulator would reject.

Local lint sees operator names, arity and field ids - it cannot see units. Every rule
here exists because a unit or typing error cost a simulation: the point is to move the
failure forward to the editor.
"""

import json
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import expression_lint
from research_app import app


class GroupProductTests(unittest.TestCase):
    """`bucket` returns a group label, not a number - offline lint can only see the name."""

    def test_arithmetic_on_a_bucket_is_flagged(self):
        # 2026-11-08: three calendar arms multiplied a bucket by 21 and the simulator
        # answered "expected Unit[], found Unit[Group:1]" for each of them.
        report = expression_lint.lint_expression(
            'rank(ts_step(1) - 21*bucket(ts_step(1), range="0,10000,21"))')
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("group_value_misused", codes)

    def test_a_bucket_fed_into_group_rank_is_fine(self):
        report = expression_lint.lint_expression(
            'group_rank(close, bucket(rank(cap), range="0,1,0.5"))')
        codes = [w["code"] for w in report["warnings"]]
        self.assertNotIn("group_value_misused", codes)

    def test_densify_is_treated_the_same_way(self):
        report = expression_lint.lint_expression("rank(2*densify(bucket(rank(cap))))")
        codes = [w["code"] for w in report["warnings"]]
        self.assertIn("group_value_misused", codes)


if __name__ == "__main__":
    unittest.main()

class FormatTests(unittest.TestCase):
    """`fmt`: the book stores multi-line expressions; everything written here did not."""

    SAMPLE = ("leverage = liabilities / assets; gp = (revenue - cogs) / assets; "
              "core = rank( group_rank(gp, subindustry) + 0.5 * group_rank(leverage, subindustry) ); "
              "core + rank(leverage)")

    def test_one_statement_per_line_and_semicolons_kept(self):
        out = expression_lint.pretty_expression(self.SAMPLE)
        self.assertEqual(out.count(";"), self.SAMPLE.count(";"))
        # three statements -> at least three lines, and the closing call is on its own
        self.assertGreaterEqual(len(out.splitlines()), 4)
        self.assertTrue(out.startswith("leverage ="))

    def test_formatting_preserves_the_expression(self):
        import re
        out = expression_lint.pretty_expression(self.SAMPLE)
        strip = lambda text: re.sub(r"\s+", "", text)
        self.assertEqual(strip(out), strip(self.SAMPLE))

    def test_formatting_is_idempotent(self):
        once = expression_lint.pretty_expression(self.SAMPLE)
        self.assertEqual(expression_lint.pretty_expression(once), once)

    def test_a_short_expression_is_left_alone(self):
        self.assertEqual(expression_lint.pretty_expression("rank(close)"), "rank(close)")


class SafeFormatTests(unittest.TestCase):
    def test_unchanged_source_uses_no_parser(self):
        with mock.patch.object(expression_lint.fastexpr_ast, "parse",
                               side_effect=AssertionError("unchanged source was parsed")):
            self.assertEqual(expression_lint.format_fastexpr("rank(close)"), ("rank(close)", None))

    def test_formats_safe_source_and_preserves_syntax_it_cannot_verify(self):
        safe = "x = close; rank(x)"
        unfamiliar = "x = close; rank([x])"
        self.assertEqual(expression_lint.format_fastexpr(safe), ("x = close;\nrank(x)", None))
        self.assertEqual(expression_lint.format_fastexpr(unfamiliar), (unfamiliar, "unsupported_fastexpr"))

    def test_keeps_comments_and_quoted_whitespace(self):
        for comment in ("# follow-up", "// follow-up", "/* follow-up */"):
            with self.subTest(comment=comment):
                source = f"x = close; {comment}\n   rank(x)"
                self.assertEqual(expression_lint.format_fastexpr(source),
                                 (source, "formatting_would_change_comments"))
        quoted = 'group_rank(close, bucket(rank(cap), range="0,  1, 0.5"))'
        self.assertEqual(expression_lint.format_fastexpr(quoted), (quoted, "formatting_would_change_expression"))


def fmt_result(capsys, root, *arguments):
    code = app.main(["--research-dir", str(root), "--quiet", "fmt", *map(str, arguments)])
    return code, json.loads(capsys.readouterr().out)


def test_fmt_previews_safe_suggestions_and_only_writes_explicitly(tmp_path, capsys):
    path = tmp_path / "draft.json"
    unsafe = 'group_rank(close, bucket(rank(cap), range="0,  1, 0.5"))'
    document = {"design": "explore", "arms": [
        {"label": "safe", "expression": "price = close; rank(price)"},
        {"label": "quoted", "expression": unsafe},
        {"label": "python", "expression": "def alpha():\n    pass\n",
         "settings": {"language": "PYTHON", "lookback": 5}},
    ]}
    path.write_text(json.dumps(document), encoding="utf-8")
    original = path.read_bytes()

    code, preview = fmt_result(capsys, tmp_path, path)
    assert code == 0
    data = preview["data"]
    assert data["changed"] == 1
    assert data["written"] is False
    assert data["skipped"] == [
        {"label": "quoted", "reason": "formatting_would_change_expression"},
        {"label": "python", "reason": "python_source"},
    ]
    assert path.read_bytes() == original
    code, written = fmt_result(capsys, tmp_path, path, "--write")
    assert code == 0
    assert written["data"]["written"] is True
    assert json.loads(path.read_bytes()) == data["formatted"]
    assert data["formatted"]["arms"][1:] == document["arms"][1:]
    assert "\n" in data["formatted"]["arms"][0]["expression"]


def test_fmt_rejects_a_batch_or_invalid_spec_without_rewriting(tmp_path, capsys):
    path = tmp_path / "input.json"
    for document in ({"format": "probe-batch", "arms": [{"expression": "x=close;rank(x)"}]},
                     {"design": "explore", "arms": [{"expression": "rank("}]}):
        original = json.dumps(document)
        path.write_text(original, encoding="utf-8")
        code, result = fmt_result(capsys, tmp_path, path, "--write")
        assert code != 0 and result["error"]["code"] == "INVALID_INPUT"
        assert path.read_text(encoding="utf-8") == original


def test_fmt_cannot_save_over_its_input(tmp_path, capsys):
    path = tmp_path / "draft.json"
    original = '{"design": "explore", "arms": [{"expression": "x = close; rank(x)"}]}'
    path.write_text(original, encoding="utf-8")
    code, result = fmt_result(capsys, tmp_path, path, "--save", path)
    assert code != 0 and result["error"]["code"] == "INVALID_INPUT"
    assert path.read_text(encoding="utf-8") == original


def test_fmt_keeps_a_published_campaign_spec_byte_exact(tmp_path, capsys):
    path = tmp_path / "backlog" / "draft.json"
    path.parent.mkdir()
    original = '{"design": "explore", "arms": [{"expression": "x = close; rank(x)"}]}'
    path.write_text(original, encoding="utf-8")
    batch = tmp_path / "backlog-results" / "draft-batch.json"
    batch.parent.mkdir()
    batch.write_text("{}", encoding="utf-8")

    code, preview = fmt_result(capsys, tmp_path, path)
    assert code == 0 and preview["data"]["changed"] == 1
    code, result = fmt_result(capsys, tmp_path, path, "--write")
    assert code != 0 and result["error"]["code"] == "INVALID_INPUT"
    assert path.read_text(encoding="utf-8") == original
    assert batch.read_text(encoding="utf-8") == "{}"


def test_fmt_rejects_and_protects_standalone_python_source(tmp_path, capsys):
    path = tmp_path / "alpha.py"
    path.write_text("print('alpha')\n", encoding="utf-8")
    code, result = fmt_result(capsys, tmp_path, path, "--save", path)
    assert code != 0 and result["error"]["code"] == "INVALID_INPUT"
    assert path.read_text(encoding="utf-8") == "print('alpha')\n"


def test_fmt_can_preview_a_legacy_spec_without_design(tmp_path, capsys):
    path = tmp_path / "legacy.json"
    original = '{"arms": [{"expression": "x = close; rank(x)"}]}'
    path.write_text(original, encoding="utf-8")
    code, preview = fmt_result(capsys, tmp_path, path)
    assert code == 0 and preview["data"]["changed"] == 1
    code, result = fmt_result(capsys, tmp_path, path, "--write")
    assert code != 0 and result["error"]["code"] == "INVALID_INPUT"
    assert path.read_text(encoding="utf-8") == original
