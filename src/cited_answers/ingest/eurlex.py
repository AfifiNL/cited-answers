"""Deterministic parser for EUR-Lex HTML (the 'eli-subdivision' format of the Official Journal).

Structure relied on (checked against both documents listed in data/SOURCES.md):
  div.eli-subdivision#art_17         an article
    p.oj-ti-art                      "Article 17"
    div.eli-title > p.oj-sti-art     the article title
    div#017.001                      numbered paragraph 1 (absent in one-paragraph articles)
      p.oj-normal / table            text, and points (a), (b) ... as label/text table rows
  div#cpt_III[.sct_1]                chapter / section containers with their titles
  div.eli-container#anx_III          an annex (AI Act only)
    p.oj-ti-grseq-1                  section headings inside an annex

Chunking rules:
  * one chunk per numbered paragraph (or per article when it has none);
  * a paragraph longer than MAX_CHARS made of a lead-in plus points (definitions, lists of
    prohibited practices) is split into one chunk per point, each repeating the lead-in so
    it reads on its own;
  * annexes are split per section and top-level numbered point; dash bullets stay with the
    text they belong to.
Recitals are not included: they explain the provisions but are not citable articles.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

from cited_answers.corpus import Chunk

MAX_CHARS = 1800
_PARA_ID = re.compile(r"^\d{3}\.\d{3}$")
_ART_ID = re.compile(r"^art_(\d+[a-z]?)$")
_ANX_ID = re.compile(r"^anx_([IVXLC]+)$")
_SECTION = re.compile(r"^(Section\s*[A-Z0-9]+)\b")
_WS = re.compile(r"[ \t\r\n     ]+")
_BLOCK = {"p", "table", "div"}


_SPACE_BEFORE_PUNCT = re.compile(r" +([;,.:])(?=\s|$)")


def _clean(text: str) -> str:
    """Collapse whitespace; drop the space a removed footnote marker leaves before ';' etc."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", _WS.sub(" ", text)).strip()


def _classes(tag: Tag) -> list[str]:
    value: Any = tag.get("class") or []
    return list(value) if not isinstance(value, str) else [value]


def _inline(node: Tag | NavigableString) -> str:
    """Inline text: drops footnote markers such as '(57)', renders exponents as '^25'."""
    if isinstance(node, NavigableString):
        return str(node)
    classes = _classes(node)
    if node.name == "a" and node.find("span", class_="oj-note-tag"):
        return ""
    if "oj-note-tag" in classes:
        return ""
    if "oj-super" in classes:
        return "^" + node.get_text()
    return "".join(_inline(c) for c in node.children if isinstance(c, Tag | NavigableString))


def _render(node: Tag) -> list[str]:
    """Render a block into lines. Points become '(a) text' lines, nested points are indented."""
    lines: list[str] = []
    inline: list[str] = []

    def flush() -> None:
        text = _clean("".join(inline))
        if text:
            lines.append(text)
        inline.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            inline.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        if child.name not in _BLOCK:
            inline.append(_inline(child))
            continue
        flush()
        classes = _classes(child)
        if child.name == "p":
            if "oj-ti-art" in classes or "oj-sti-art" in classes:
                continue
            text = _clean(_inline(child))
            if text:
                lines.append(text)
        elif child.name == "table":
            for label, content in _table_rows(child):
                lines.extend(_point_lines(label, content))
        elif "eli-title" not in classes:
            lines.extend(_render(child))
    flush()
    return lines


def _table_rows(table: Tag) -> list[tuple[str, Tag]]:
    """(label, content cell) per row. Rows are [label, text] or, in some annexes,
    [empty, label, text]."""
    body = table.find("tbody", recursive=False) or table
    rows = []
    for tr in body.find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) >= 2:
            rows.append((_clean(tds[-2].get_text()), tds[-1]))
    return rows


def _point_lines(label: str, content: Tag) -> list[str]:
    inner = _render(content)
    if not inner:
        return []
    return [f"{label} {inner[0]}".strip()] + ["  " + line for line in inner[1:]]


def _is_point_label(label: str) -> bool:
    """'(a)', '1.', '(iv)' are citable points; '—' bullets and empty labels are not."""
    return any(ch.isalnum() for ch in label)


def _point_id(label: str) -> str:
    """'(a)' -> 'a', '1.' -> '1', '(12)' -> '12'."""
    return label.strip().strip("().").strip()


def _split_points(node: Tag) -> tuple[list[str], list[tuple[str, list[str]]], list[str]]:
    """Split a paragraph into (lead-in lines, [(point id, lines)], trailing lines)."""
    lead: list[str] = []
    points: list[tuple[str, list[str]]] = []
    tail: list[str] = []
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "table":
            for label, content in _table_rows(child):
                lines = _point_lines(label, content)
                if not lines:
                    continue
                if _is_point_label(label) or not points:
                    points.append((_point_id(label) or "-", lines))
                else:
                    points[-1][1].extend(lines)
        elif child.name in _BLOCK:
            wrapper_lines = _render_one(child)
            (tail if points else lead).extend(wrapper_lines)
    return lead, points, tail


def _render_one(child: Tag) -> list[str]:
    if child.name == "p":
        classes = _classes(child)
        if "oj-ti-art" in classes or "oj-sti-art" in classes:
            return []
        text = _clean(_inline(child))
        return [text] if text else []
    if "eli-title" in _classes(child):
        return []
    return _render(child)


def _container_title(div: Tag) -> str:
    head = div.find("p", class_="oj-ti-section-1", recursive=False)
    title_div = div.find("div", class_="eli-title", recursive=False)
    parts = [_clean(head.get_text()) if head else ""]
    if title_div:
        parts.append(_clean(title_div.get_text(" ")))
    return " ".join(p for p in parts if p)


def _chapter_of(article: Tag) -> str:
    titles: list[str] = []
    for parent in article.parents:
        if isinstance(parent, Tag) and parent.name == "div":
            pid = str(parent.get("id") or "")
            if pid.startswith("cpt_"):
                title = _container_title(parent)
                if title:
                    titles.append(title)
    return " / ".join(reversed(titles))


def parse_articles(soup: BeautifulSoup, law: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for art in soup.find_all("div", class_="eli-subdivision"):
        m = _ART_ID.match(str(art.get("id") or ""))
        if not m:
            continue
        number = m.group(1)
        sti = art.find("p", class_="oj-sti-art")
        base = Chunk(
            id=f"{law}:art{number}",
            law=law,
            kind="article",
            number=number,
            paragraph=None,
            point=None,
            title=_clean(sti.get_text()) if sti else "",
            chapter=_chapter_of(art),
            text="",
            anchor=f"art_{number}",
        )
        paras = [
            (str(int(str(d["id"]).split(".")[1])), d)
            for d in art.find_all("div", recursive=False)
            if _PARA_ID.match(str(d.get("id") or ""))
        ]
        units: list[tuple[str | None, Tag]] = list(paras) if paras else [(None, art)]
        for para, node in units:
            uid = f"{base.id}:p{para}" if para else base.id
            text = "\n".join(_render(node))
            if not text:
                continue
            lead, points, tail = _split_points(node)
            if len(text) <= MAX_CHARS or len(points) < 2:
                chunks.append(replace(base, id=uid, paragraph=para, text=text))
                continue
            lead_text = "\n".join(lead)
            for pid, plines in points:
                body = "\n".join(([lead_text] if lead_text else []) + plines)
                chunks.append(
                    replace(base, id=f"{uid}:pt{pid}", paragraph=para, point=pid, text=body)
                )
            if tail:
                chunks.append(replace(base, id=f"{uid}:tail", paragraph=para, text="\n".join(tail)))
    return chunks


def _section_label(heading: str) -> str:
    """'Section A — Information ...' -> 'Section A'; '4. Control of ...' stays a short title."""
    m = _SECTION.match(heading)
    if m:
        return _clean(m.group(1))
    heading = heading.rstrip(".")
    return heading if len(heading) <= 60 else heading[:57].rstrip() + "..."


def _section_key(label: str, index: int) -> str:
    """Short, stable id part: 'Section A' -> 'sA', '4. Control of ...' -> 's4'."""
    m = _SECTION.match(label) or re.match(r"^(\d+)\.", label)
    if m:
        return "s" + re.sub(r"^Section\s*", "", m.group(1))
    return f"s{index}"


@dataclass
class _Group:
    """A run of annex text that becomes one chunk."""

    section: str | None
    head: list[str]
    point: str | None
    lines: list[str] = field(default_factory=list)


def parse_annexes(soup: BeautifulSoup, law: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for anx in soup.find_all("div", class_="eli-container"):
        m = _ANX_ID.match(str(anx.get("id") or ""))
        if not m:
            continue
        number = m.group(1)
        heads = [_clean(p.get_text()) for p in anx.find_all("p", class_="oj-doc-ti")]
        title = heads[1] if len(heads) > 1 else f"Annex {number}"

        section: str | None = None
        heading_lines: list[str] = []
        last_was_heading = False
        groups = [_Group(None, [], None)]  # intro
        for child in anx.children:
            if not isinstance(child, Tag):
                continue
            classes = _classes(child)
            if child.name == "p" and "oj-doc-ti" in classes:
                continue
            if child.name == "p" and any(c.startswith("oj-ti-grseq") for c in classes):
                heading = _clean(child.get_text())
                if last_was_heading:
                    heading_lines.append(heading)  # "Section 1" + its title on a second line
                else:
                    heading_lines = [heading]
                    section = _section_label(heading)
                last_was_heading = True
                groups.append(_Group(section, list(heading_lines), None))
                continue
            last_was_heading = False
            if child.name == "table":
                for label, content in _table_rows(child):
                    lines = _point_lines(label, content)
                    if not lines:
                        continue
                    if _is_point_label(label):
                        groups.append(_Group(section, list(heading_lines), _point_id(label)))
                    groups[-1].lines.extend(lines)
                continue
            groups[-1].lines.extend(_render_one(child))

        seen: dict[str, int] = {}
        for index, g in enumerate(groups):
            if not g.lines:
                continue
            key = _section_key(g.section, index) + "." if g.section else ""
            key += f"pt{g.point}" if g.point else "intro"
            n = seen.get(key, 0)
            seen[key] = n + 1
            chunks.append(
                Chunk(
                    id=f"{law}:anx{number}:{key}" + (f"-{n + 1}" if n else ""),
                    law=law,
                    kind="annex",
                    number=number,
                    paragraph=g.section,
                    point=g.point,
                    title=title,
                    chapter="",
                    text="\n".join([*g.head, *g.lines]),
                    anchor=f"anx_{number}",
                )
            )
    return chunks


def parse_document(html: str, law: str) -> list[Chunk]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)  # EUR-Lex serves XHTML
        soup = BeautifulSoup(html, "lxml")
    return parse_articles(soup, law) + parse_annexes(soup, law)
