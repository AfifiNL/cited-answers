"""Citation verifier: every claim must be backed by a verbatim quote from the source it cites.

Checks per claim (all deterministic):
  unknown_source     the claim cites a label that was not among the sources shown
  short_quote        the quote is too short to support anything ("the controller")
  quote_not_found    the quote is not a verbatim span of the cited source
  unsupported_number a number in the claim is not in its quote or citation label
  weak_support       too few of the claim's content words appear in its quote

A draft passes only if every claim passes. On failure the findings go back to the model as
feedback for one more attempt; if that fails too, the pipeline refuses rather than return
a partially supported answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from cited_answers.providers.base import Draft, Source
from cited_answers.text import content_terms, normalize, numbers


@dataclass(frozen=True)
class ClaimCheck:
    index: int
    ok: bool
    support: float
    problems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    checks: list[ClaimCheck]
    problems: list[str]

    def feedback(self) -> str:
        return "Previous answer was rejected by the citation checker:\n" + "\n".join(
            f"- {p}" for p in self.problems
        )


def support_score(claim_text: str, quote: str) -> float:
    claim_terms = set(content_terms(claim_text))
    if not claim_terms:
        return 0.0
    return len(claim_terms & set(content_terms(quote))) / len(claim_terms)


def verify_draft(
    draft: Draft,
    sources: Mapping[str, Source],
    *,
    min_support: float,
    min_quote_chars: int,
    max_claims: int,
) -> VerificationReport:
    problems: list[str] = []
    if not draft.claims:
        return VerificationReport(False, [], ["the answer contains no claims"])
    if len(draft.claims) > max_claims:
        problems.append(f"too many claims ({len(draft.claims)} > {max_claims})")
    checks: list[ClaimCheck] = []
    for i, claim in enumerate(draft.claims, start=1):
        issues: list[str] = []
        src = sources.get(claim.source.strip())
        quote = claim.quote.strip()
        support = support_score(claim.text, quote)
        if src is None:
            issues.append(f"unknown_source: claim {i} cites '{claim.source}', not a given source")
        if len(quote) < min_quote_chars:
            issues.append(f"short_quote: claim {i} quote is shorter than {min_quote_chars} chars")
        elif src is not None and normalize(quote) not in normalize(src.text):
            issues.append(f"quote_not_found: claim {i} quote is not verbatim in {claim.source}")
        allowed_numbers = numbers(quote) | (numbers(src.citation) if src else set())
        stray = sorted(numbers(claim.text) - allowed_numbers)
        if stray:
            issues.append(f"unsupported_number: claim {i} states {', '.join(stray)} not in quote")
        if support < min_support:
            issues.append(
                f"weak_support: claim {i} shares {support:.0%} of its words with its quote "
                f"(minimum {min_support:.0%})"
            )
        checks.append(ClaimCheck(i, not issues, round(support, 4), issues))
        problems.extend(issues)
    return VerificationReport(not problems, checks, problems)
