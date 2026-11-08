"""The two legal texts in the corpus and where they come from.

Both are official EUR-Lex HTML renderings of the texts published in the Official Journal
of the European Union. Only the Official Journal version is authentic; see data/SOURCES.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Law:
    key: str
    short: str
    title: str
    celex: str
    eli: str
    oj_reference: str

    @property
    def html_url(self) -> str:
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{self.celex}"

    def anchor_url(self, anchor: str) -> str:
        """Deep link into the EUR-Lex HTML; the anchors (art_17, anx_III) exist in the page."""
        return f"{self.html_url}#{anchor}"


LAWS: dict[str, Law] = {
    "gdpr": Law(
        key="gdpr",
        short="GDPR",
        title="Regulation (EU) 2016/679 (General Data Protection Regulation)",
        celex="32016R0679",
        eli="http://data.europa.eu/eli/reg/2016/679/oj",
        oj_reference="OJ L 119, 4.5.2016, p. 1",
    ),
    "ai_act": Law(
        key="ai_act",
        short="AI Act",
        title="Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        celex="32024R1689",
        eli="http://data.europa.eu/eli/reg/2024/1689/oj",
        oj_reference="OJ L, 2024/1689, 12.7.2024",
    ),
}

LAW_KEYS: tuple[str, ...] = tuple(LAWS)
