"""
Verifier (critic) agent: this is where hallucination detection happens.

Strategy (NLI-style claim verification, the same idea RAGAS' faithfulness
metric uses under the hood):
  1. Decompose the draft answer into atomic factual claims.
  2. For each claim, ask the LLM to judge: is it ENTAILED by the retrieved
     context, CONTRADICTED, or NOT_MENTIONED (i.e. unsupported)?
  3. hallucination_score = fraction of claims that are NOT entailed.
  4. If hallucination_score exceeds a threshold, write a structured critique
     describing exactly which claims are unsupported and why, which the
     generator must address on the next loop iteration.
"""
from __future__ import annotations

import json
import re

from core.llm_client import LLMClient
from core.state import ClaimCheck, RAGState

DECOMPOSE_PROMPT = """Break the following ANSWER into a list of atomic, \
independently-checkable factual claims. Ignore hedges, transitions, and \
citation markers like [doc_id] — extract only the substantive claims.

ANSWER:
{answer}

Respond with a JSON array of strings, nothing else. Example:
["Claim one stated plainly.", "Claim two stated plainly."]"""

VERIFY_PROMPT = """You are a strict fact-checker. Determine whether each CLAIM \
is supported by the CONTEXT passages. Use only the context — not outside \
knowledge.

CONTEXT PASSAGES:
{context}

CLAIMS TO CHECK:
{claims_json}

For each claim, respond with a JSON array of objects with this exact shape:
[{{"claim": "...", "verdict": "ENTAILED" | "CONTRADICTED" | "NOT_MENTIONED", \
"evidence_doc_id": "doc_id or null", "confidence": 0.0-1.0}}]

Respond with ONLY the JSON array."""

CRITIQUE_TEMPLATE = """The following claims in the draft answer are NOT \
adequately supported by the retrieved context:

{unsupported_list}

The generator must either: (a) remove these claims, (b) qualify them as \
uncertain, or (c) the retriever should fetch better evidence to support them. \
Hallucination score: {score:.2f} (threshold: {threshold:.2f})."""


def _extract_json(text: str) -> str:
    """LLMs sometimes wrap JSON in prose or code fences; pull out the array."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else text


class VerifierAgent:
    def __init__(self, llm: LLMClient, hallucination_threshold: float = 0.2):
        self.llm = llm
        self.threshold = hallucination_threshold

    def _decompose(self, answer: str) -> list[str]:
        raw = self.llm.complete(
            system="You extract atomic factual claims from text.",
            user=DECOMPOSE_PROMPT.format(answer=answer),
            max_tokens=500,
            temperature=0.0,
        )
        try:
            claims = json.loads(_extract_json(raw))
            return [c for c in claims if isinstance(c, str) and c.strip()]
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat each sentence as a claim
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 8]

    def _verify_claims(self, claims: list[str], context: str) -> list[ClaimCheck]:
        if not claims:
            return []
        raw = self.llm.complete(
            system="You are a strict, literal fact-checker. No outside knowledge.",
            user=VERIFY_PROMPT.format(context=context, claims_json=json.dumps(claims)),
            max_tokens=1200,
            temperature=0.0,
        )
        try:
            results = json.loads(_extract_json(raw))
        except (json.JSONDecodeError, TypeError):
            # Conservative fallback: mark everything unsupported so the loop
            # retries rather than silently shipping unverified claims.
            return [
                ClaimCheck(claim=c, supported=False, confidence=0.0, evidence_doc_id=None)
                for c in claims
            ]

        checks: list[ClaimCheck] = []
        for r in results:
            verdict = r.get("verdict", "NOT_MENTIONED")
            checks.append(
                ClaimCheck(
                    claim=r.get("claim", ""),
                    supported=(verdict == "ENTAILED"),
                    confidence=float(r.get("confidence", 0.0)),
                    evidence_doc_id=r.get("evidence_doc_id"),
                )
            )
        return checks

    def __call__(self, state: RAGState) -> RAGState:
        context = "\n\n".join(f"[{c.doc.doc_id}] {c.doc.text}" for c in state["retrieved_chunks"])

        claims = self._decompose(state["draft_answer"])
        checks = self._verify_claims(claims, context)
        state["claim_checks"] = checks

        if checks:
            unsupported = [c for c in checks if not c["supported"]]
            hallucination_score = len(unsupported) / len(checks)
        else:
            hallucination_score = 0.0
            unsupported = []

        state["hallucination_score"] = hallucination_score
        state["verified"] = hallucination_score <= self.threshold

        if state["verified"]:
            state["final_answer"] = state["draft_answer"]
            state["critique"] = ""
            state["history"].append(
                f"[Verifier] PASS — hallucination score {hallucination_score:.2f} "
                f"<= threshold {self.threshold:.2f}"
            )
        else:
            unsupported_list = "\n".join(f"- \"{c['claim']}\"" for c in unsupported)
            state["critique"] = CRITIQUE_TEMPLATE.format(
                unsupported_list=unsupported_list or "(none listed)",
                score=hallucination_score,
                threshold=self.threshold,
            )
            state["history"].append(
                f"[Verifier] FAIL — hallucination score {hallucination_score:.2f} "
                f"> threshold {self.threshold:.2f} ({len(unsupported)}/{len(checks)} claims unsupported)"
            )

        return state
