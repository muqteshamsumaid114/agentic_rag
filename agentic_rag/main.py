"""
End-to-end demo.

Run with:
    export ANTHROPIC_API_KEY=sk-...
    python main.py

This will:
  1. Build the hybrid index (FAISS dense + BM25 sparse) over sample_docs.json
  2. Run the multi-agent self-correcting pipeline on a few questions
  3. Print the full agent trace, final answer, and hallucination score
  4. Run the same questions through the unanswerable-on-purpose case to show
     the loop catching and flagging a hallucination risk
"""
from __future__ import annotations

import json
from pathlib import Path

from core.llm_client import LLMClient
from core.orchestrator import run_pipeline
from core.retriever import (
    BM25SparseIndex,
    CrossEncoderReranker,
    Document,
    FAISSDenseIndex,
    HybridRetriever,
)

DATA_PATH = Path(__file__).parent / "data" / "sample_docs.json"


def build_retriever() -> HybridRetriever:
    with open(DATA_PATH) as f:
        raw_docs = json.load(f)
    documents = [Document(doc_id=d["doc_id"], text=d["text"]) for d in raw_docs]

    dense = FAISSDenseIndex()
    sparse = BM25SparseIndex()
    reranker = CrossEncoderReranker()

    retriever = HybridRetriever(dense, sparse, reranker)
    retriever.index(documents)
    return retriever


def print_report(query: str, state: dict) -> None:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    print("\n--- Agent trace ---")
    for line in state["history"]:
        print(" ", line)

    print("\n--- Retrieved evidence (final iteration) ---")
    for c in state["retrieved_chunks"]:
        print(f"  [{c.doc.doc_id}] rerank={c.rerank_score:.3f}  {c.doc.text[:90]}...")

    print("\n--- Claim-level verification ---")
    for chk in state["claim_checks"]:
        flag = "[OK]" if chk["supported"] else "[FAIL]"
        print(f"  {flag} ({chk['confidence']:.2f}) {chk['claim']}")

    print(f"\nHallucination score: {state['hallucination_score']:.2f}  |  Verified: {state['verified']}")
    print("\n--- FINAL ANSWER ---")
    print(state["final_answer"] or state["draft_answer"])


def main():
    retriever = build_retriever()
    llm = LLMClient(model="llama-3.3-70b-versatile")

    questions = [
        "How does JWST's mirror compare to Hubble's, and why does JWST need to be so cold?",
        # Deliberately out-of-scope question to exercise the self-correction
        # loop: the docs say nothing about JWST's exact resolution in arcseconds.
        "What is JWST's exact angular resolution in arcseconds at 2 microns?",
    ]

    for q in questions:
        state = run_pipeline(q, retriever, llm, max_iterations=3, hallucination_threshold=0.15)
        print_report(q, state)


if __name__ == "__main__":
    main()
