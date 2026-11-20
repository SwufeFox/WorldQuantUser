"""Format BRAIN document blocks as readable text without loading assets."""

import json
from html.parser import HTMLParser


_HEADINGS = {f"h{level}" for level in range(1, 7)}
_PARAGRAPHS = {"p", "div", "section", "blockquote", "table"} | _HEADINGS


class _DocumentHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines = []
        self.line = []
        self.prefix = ""
        self.lists = []
        self.row_cells = None
        self.pre = None
        self.hidden = None

    def _break(self, paragraph=False):
        if self.row_cells is not None:
            self.line.append(" ")
            return
        line = " ".join("".join(self.line).split())
        self.line.clear()
        if line:
            self.lines.append(self.prefix + line)
            self.prefix = ""
        if paragraph and self.lines and self.lines[-1]:
            self.lines.append("")

    def _finish_pre(self):
        code = "".join(self.pre).replace("\r\n", "\n").strip("\n")
        self.pre = None
        if code:
            self.lines.extend((code, ""))

    def handle_starttag(self, tag, _attrs):
        if self.hidden is not None:
            return
        if tag in {"script", "style"}:
            self.hidden = tag
        elif self.pre is not None:
            if tag == "br":
                self.pre.append("\n")
        elif tag == "pre":
            self._break(paragraph=True)
            self.pre = []
        elif tag == "tr":
            self._break()
            self.row_cells = 0
        elif tag in {"td", "th"} and self.row_cells is not None:
            if self.row_cells:
                self.line.append(" | ")
            self.row_cells += 1
        elif tag in {"ul", "ol"}:
            self._break()
            self.lists.append(1 if tag == "ol" else None)
        elif tag == "li":
            self._break()
            number = self.lists[-1] if self.lists else None
            marker = "- " if number is None else f"{number}. "
            self.prefix = "  " * max(0, len(self.lists) - 1) + marker
            if number is not None:
                self.lists[-1] += 1
        elif tag == "br" or tag in _PARAGRAPHS:
            self._break(paragraph=tag in _PARAGRAPHS and not self.prefix)

    def handle_endtag(self, tag):
        if self.hidden is not None:
            if tag == self.hidden:
                self.hidden = None
            return
        if self.pre is not None:
            if tag == "pre":
                self._finish_pre()
        elif tag == "tr":
            self.row_cells = None
            self._break()
        elif tag in {"ul", "ol"}:
            self._break()
            if self.lists:
                self.lists.pop()
        elif tag == "li":
            self._break()
            self.prefix = ""
        elif tag in _PARAGRAPHS:
            self._break(paragraph=not self.lists)

    def handle_data(self, data):
        if self.hidden is None:
            (self.line if self.pre is None else self.pre).append(data)

    def text(self):
        self.close()
        if self.pre is not None:
            self._finish_pre()
        self.row_cells = None
        self._break()
        return "\n".join(self.lines).strip("\n")


def _html_text(html):
    parser = _DocumentHTML()
    parser.feed(html)
    return parser.text()


def _cell_text(cell):
    if isinstance(cell, str):
        return " ".join(_html_text(cell).split())
    return "" if cell is None else json.dumps(cell, ensure_ascii=False)


def _render_block(kind, value):
    if kind == "TEXT" and isinstance(value, str):
        return _html_text(value)
    if kind == "HEADING":
        title = value.get("content") if isinstance(value, dict) else value
        if isinstance(title, str):
            level = str(value.get("level")) if isinstance(value, dict) else "1"
            prefix = "#" * (int(level) if f"h{level}" in _HEADINGS else 1)
            return f"{prefix} {_html_text(title)}" if title else ""
    if kind == "TABLE":
        rows = value.get("data") if isinstance(value, dict) else value
        if isinstance(rows, list) and all(isinstance(row, list) for row in rows):
            return "\n".join(" | ".join(map(_cell_text, row)) for row in rows)
    if kind == "EQUATION" and isinstance(value, str):
        return value.strip()
    if kind == "IMAGE" and isinstance(value, dict):
        parts = [text for key in ("title", "url")
                 if isinstance(text := value.get(key), str) and text]
        if parts:
            return "Image: " + " — ".join(parts)
    if kind == "SIMULATION_EXAMPLE" and isinstance(value, dict):
        return "Simulation example\n" + json.dumps(value, ensure_ascii=False, indent=2)
    return None


def render_blocks(blocks):
    """Return the formatted body and types that require consulting raw blocks."""
    rendered = []
    unsupported = {}
    for block in blocks:
        kind = block.get("type") if isinstance(block, dict) else None
        if not isinstance(kind, str) or not kind:
            unsupported["UNKNOWN"] = None
            continue
        text = _render_block(kind, block.get("value", block.get("content")))
        if text is None:
            unsupported[kind] = None
        elif text:
            rendered.append(text)
    return "\n\n".join(rendered), list(unsupported)
