"""The answering pipeline as a LangGraph state machine.

    START -> retrieve -> assess_evidence --insufficient--> refuse -> END
                               | sufficient
                               v
                           generate --declined/error--> refuse
                               |
                               v
                            verify --fails, attempts left--> generate (with feedback)
                               |  \\--fails, no attempts left--> refuse
                               v passes
                           finalize -> END

Every edge is decided by deterministic code; the model only fills in `generate`. Each node
appends a trace entry, returned to API callers who ask for `debug`.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from cited_answers.config import Settings
from cited_answers.corpus import Corpus
from cited_answers.glossary import expand_query
from cited_answers.laws import LAW_KEYS
from cited_answers.pipeline.evidence import EvidenceReport, assess_evidence
from cited_answers.pipeline.models import (
    REFUSAL_MESSAGES,
    AnswerClaim,
    AnswerResult,
    Citation,
    Refusal,
)
from cited_answers.pipeline.references import find_references
from cited_answers.pipeline.verify import VerificationReport, verify_draft
from cited_answers.providers.base import AnswerModel, Draft, DraftRequest, ProviderError, Source
from cited_answers.retrieval.base import Hit, Retriever
from cited_answers.text import TermStats, content_terms


class State(TypedDict, total=False):
    question: str
    laws: list[str] | None
    hits: list[Hit]
    sources: list[Source]
    evidence: EvidenceReport
    draft: Draft | None
    verification: VerificationReport | None
    attempts: int
    error: str
    refusal_code: str
    refusal_detail: str
    result: AnswerResult
    trace: Annotated[list[dict[str, Any]], operator.add]


def _timed(name: str, fn: Callable[[State], dict[str, Any]]) -> Any:
    """Wrap a node so it appends {node, ms, ...} to the trace. Returns Any because
    LangGraph's add_node overloads are stricter than a plain callable."""

    def run(state: State) -> dict[str, Any]:
        t0 = time.perf_counter()
        update = fn(state)
        entry = {"node": name, "ms": round((time.perf_counter() - t0) * 1000, 2)}
        entry.update(update.pop("_trace", {}))
        update["trace"] = [entry]
        return update

    return run


class Pipeline:
    def __init__(
        self,
        corpus: Corpus,
        retriever: Retriever,
        model: AnswerModel,
        settings: Settings,
        *,
        embedder_name: str = "",
    ):
        self.corpus = corpus
        self.retriever = retriever
        self.model = model
        self.settings = settings
        self.embedder_name = embedder_name
        self.stats = TermStats.build(c.index_text for c in corpus.chunks)
        self.graph = self._build()

    # ── nodes ────────────────────────────────────────────────────────────────────────────
    def _pinned_hits(self, question: str, laws: Sequence[str]) -> list[Hit]:
        refs = find_references(question)
        keys = refs.article_keys(tuple(laws))
        q_terms = set(content_terms(question))
        pinned: list[Hit] = []
        for key in keys:
            chunks = self.corpus.by_article.get(key, ())
            ranked = sorted(chunks, key=lambda c: -len(q_terms & set(content_terms(c.index_text))))
            pinned.extend(Hit(c, 1.0, {"reference": 1}, pinned=True) for c in ranked[:2])
        return pinned

    def _retrieve(self, state: State) -> dict[str, Any]:
        question = state["question"]
        refs = find_references(question)
        laws = state.get("laws") or list(refs.laws) or list(LAW_KEYS)
        s = self.settings
        query = expand_query(question, refs.laws)
        # Search deeper than top_k: the per-article cap below would otherwise starve the list
        # when one long article dominates the ranking.
        hits = self.retriever.search(query, k=s.top_k * 4, laws=laws)
        merged: list[Hit] = []
        seen: set[str] = set()
        per_article: dict[str, int] = {}
        for hit in [*self._pinned_hits(question, laws), *hits]:
            key = hit.chunk.article_key
            if hit.chunk.id in seen or (
                not hit.pinned and per_article.get(key, 0) >= s.per_article_cap
            ):
                continue
            seen.add(hit.chunk.id)
            per_article[key] = per_article.get(key, 0) + 1
            merged.append(hit)
            if len(merged) >= s.top_k:
                break
        sources = [
            Source(f"S{i}", h.chunk.id, h.chunk.label, h.chunk.title, h.chunk.text)
            for i, h in enumerate(merged[: s.max_sources], start=1)
        ]
        return {
            "laws": laws,
            "hits": merged,
            "sources": sources,
            "attempts": 0,
            "_trace": {
                "laws": laws,
                "query": query,
                "hits": [
                    {
                        "id": h.chunk.id,
                        "score": round(h.score, 5),
                        "ranks": h.ranks,
                        "pinned": h.pinned,
                    }
                    for h in merged
                ],
            },
        }

    def _assess(self, state: State) -> dict[str, Any]:
        report = assess_evidence(
            state["question"],
            state.get("hits", []),
            self.stats,
            min_coverage=self.settings.min_evidence_coverage,
            top_n=self.settings.evidence_top_n,
        )
        update: dict[str, Any] = {
            "evidence": report,
            "_trace": {
                "sufficient": report.sufficient,
                "coverage": report.coverage,
                "missing_terms": report.missing_terms,
                "reason": report.reason,
            },
        }
        if not report.sufficient:
            update["refusal_code"] = "no_evidence"
            update["refusal_detail"] = report.reason
        return update

    def _generate(self, state: State) -> dict[str, Any]:
        attempts = state.get("attempts", 0) + 1
        verification = state.get("verification")
        feedback = verification.feedback() if verification and not verification.ok else ""
        request = DraftRequest(state["question"], state["sources"], feedback)
        try:
            draft = self.model.draft(request)
        except ProviderError as exc:
            return {
                "attempts": attempts,
                "draft": None,
                "refusal_code": "provider_error",
                "refusal_detail": str(exc),
                "_trace": {"attempt": attempts, "error": str(exc)},
            }
        update: dict[str, Any] = {
            "attempts": attempts,
            "draft": draft,
            "_trace": {
                "attempt": attempts,
                "model": self.model.name,
                "refuse": draft.refuse,
                "claims": len(draft.claims),
                "had_feedback": bool(feedback),
            },
        }
        if draft.refuse:
            update["refusal_code"] = "model_declined"
            update["refusal_detail"] = draft.refusal_reason
        return update

    def _verify(self, state: State) -> dict[str, Any]:
        draft = state["draft"]
        assert draft is not None
        report = verify_draft(
            draft,
            {s.label: s for s in state["sources"]},
            min_support=self.settings.min_claim_support,
            min_quote_chars=self.settings.min_quote_chars,
            max_claims=self.settings.max_claims,
        )
        update: dict[str, Any] = {
            "verification": report,
            "_trace": {
                "ok": report.ok,
                "problems": report.problems,
                "support": [c.support for c in report.checks],
            },
        }
        if not report.ok:
            update["refusal_code"] = "unverified"
            update["refusal_detail"] = "; ".join(report.problems)
        return update

    def _finalize(self, state: State) -> dict[str, Any]:
        draft = state["draft"]
        assert draft is not None
        by_label = {s.label: s for s in state["sources"]}
        citations: list[Citation] = []
        ref_of: dict[str, int] = {}
        claims: list[AnswerClaim] = []
        for claim in draft.claims:
            chunk = self.corpus.by_id[by_label[claim.source.strip()].chunk_id]
            if chunk.id not in ref_of:
                ref_of[chunk.id] = len(citations) + 1
                citations.append(
                    Citation(
                        ref=ref_of[chunk.id],
                        chunk_id=chunk.id,
                        article_key=chunk.article_key,
                        label=chunk.label,
                        title=chunk.title,
                        url=chunk.url,
                    )
                )
            claims.append(
                AnswerClaim(
                    text=claim.text.strip(), quote=claim.quote.strip(), ref=ref_of[chunk.id]
                )
            )
        answer = " ".join(f"{c.text} [{c.ref}]" for c in claims)
        result = AnswerResult(
            status="answered",
            question=state["question"],
            answer=answer,
            claims=claims,
            citations=citations,
        )
        return {"result": result, "_trace": {"citations": [c.label for c in citations]}}

    def _refuse(self, state: State) -> dict[str, Any]:
        code = state.get("refusal_code", "no_evidence")
        refusal = Refusal(
            code=code,
            message=REFUSAL_MESSAGES[code],
            detail=state.get("refusal_detail", ""),
        )
        result = AnswerResult(status="refused", question=state["question"], refusal=refusal)
        return {"result": result, "_trace": {"code": code}}

    # ── routing ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _after_assess(state: State) -> str:
        return "generate" if state["evidence"].sufficient else "refuse"

    @staticmethod
    def _after_generate(state: State) -> str:
        draft = state.get("draft")
        return "verify" if draft is not None and not draft.refuse else "refuse"

    def _after_verify(self, state: State) -> str:
        report = state["verification"]
        assert report is not None
        if report.ok:
            return "finalize"
        return "generate" if state["attempts"] < self.settings.max_attempts else "refuse"

    def _build(self) -> Any:
        g = StateGraph(State)
        g.add_node("retrieve", _timed("retrieve", self._retrieve))
        g.add_node("assess_evidence", _timed("assess_evidence", self._assess))
        g.add_node("generate", _timed("generate", self._generate))
        g.add_node("verify", _timed("verify", self._verify))
        g.add_node("finalize", _timed("finalize", self._finalize))
        g.add_node("refuse", _timed("refuse", self._refuse))
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "assess_evidence")
        g.add_conditional_edges(
            "assess_evidence", self._after_assess, {"generate": "generate", "refuse": "refuse"}
        )
        g.add_conditional_edges(
            "generate", self._after_generate, {"verify": "verify", "refuse": "refuse"}
        )
        g.add_conditional_edges(
            "verify",
            self._after_verify,
            {"finalize": "finalize", "generate": "generate", "refuse": "refuse"},
        )
        g.add_edge("finalize", END)
        g.add_edge("refuse", END)
        return g.compile()

    # ── entry point ──────────────────────────────────────────────────────────────────────
    def answer(
        self, question: str, *, laws: Sequence[str] | None = None, debug: bool = False
    ) -> AnswerResult:
        t0 = time.perf_counter()
        question = question.strip()
        if not question:
            result = AnswerResult(
                status="refused",
                question=question,
                refusal=Refusal(code="empty_question", message=REFUSAL_MESSAGES["empty_question"]),
            )
        else:
            final = self.graph.invoke(
                {"question": question, "laws": list(laws) if laws else None, "trace": []}
            )
            result = final["result"]
            result.meta["attempts"] = final.get("attempts", 0)
            if debug:
                result.trace = final.get("trace", [])
        result.meta.update(
            {
                "model": self.model.name,
                "embedder": self.embedder_name,
                "store": self.retriever.name,
                "corpus_version": self.corpus.version,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
        )
        return result
