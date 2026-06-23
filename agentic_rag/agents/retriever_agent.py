"""Retriever agent: pulls context via hybrid search + rerank, and can
reformulate the query if the verifier rejects the answer for lack of support."""
from __future__ import annotations

from core.llm_client import LLMClient
from core.retriever import HybridRetriever
from core.state import RAGState

REFORMULATE_PROMPT = """You are a search query specialist. The current retrieved \
context was judged insufficient to answer the user's question reliably.

Original question: {query}

Reviewer critique: {critique}

Write ONE improved search query that would surface better evidence. \
Respond with only the query text, nothing else."""


class RetrieverAgent:
    def __init__(self, retriever: HybridRetriever, llm: LLMClient, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def __call__(self, state: RAGState) -> RAGState:
        query = state["query"]

        # On retry iterations, reformulate the query based on the verifier's critique
        if state["iteration"] > 0 and state.get("critique"):
            reformulated = self.llm.complete(
                system="You rewrite search queries to retrieve better evidence.",
                user=REFORMULATE_PROMPT.format(query=state["query"], critique=state["critique"]),
                max_tokens=64,
                temperature=0.0,
            ).strip()
            query = reformulated or state["query"]
            state["history"].append(f"[Retriever] Reformulated query -> '{query}'")
        else:
            state["history"].append(f"[Retriever] Initial query -> '{query}'")

        chunks = self.retriever.retrieve(query, top_k=self.top_k)
        state["retrieved_chunks"] = chunks
        state["history"].append(
            f"[Retriever] Retrieved {len(chunks)} chunks "
            f"(top rerank score: {chunks[0].rerank_score:.3f})" if chunks else "[Retriever] No chunks found"
        )
        return state
