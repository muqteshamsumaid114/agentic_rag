"""Shared state schema passed between LangGraph nodes."""
from __future__ import annotations

from typing import List, Optional, TypedDict

from core.retriever import RetrievedChunk


class ClaimCheck(TypedDict):
    claim: str
    supported: bool
    confidence: float
    evidence_doc_id: Optional[str]


class RAGState(TypedDict):
    query: str
    retrieved_chunks: List[RetrievedChunk]
    draft_answer: str
    final_answer: str
    critique: str
    claim_checks: List[ClaimCheck]
    hallucination_score: float          # 0 = fully grounded, 1 = fully hallucinated
    iteration: int
    max_iterations: int
    verified: bool
    history: List[str]                  # human-readable trace of what each agent did
