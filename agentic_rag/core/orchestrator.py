"""
LangGraph orchestration.

Graph shape:

    retriever -> generator -> verifier --(fail, iter < max)--> retriever
                                  |
                                  +--(pass OR iter == max)--> END

The conditional edge out of `verifier` is what implements the self-correction
loop: a failed verification routes back to the retriever (which reformulates
the query) rather than straight back to the generator, since often the root
cause of a hallucination is missing evidence, not a careless generator.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.generator_agent import GeneratorAgent
from agents.retriever_agent import RetrieverAgent
from agents.verifier_agent import VerifierAgent
from core.llm_client import LLMClient
from core.retriever import HybridRetriever
from core.state import RAGState


def _finalize_node(state: RAGState) -> RAGState:
    """
    Runs after a verifier FAIL when retries remain. Conditional-edge router
    functions in LangGraph can only return a routing key — they cannot
    mutate and persist state — so the iteration bump and bookkeeping must
    happen in a real node, not in the router itself.
    """
    state["iteration"] += 1
    state["history"].append(f"[Orchestrator] Looping back to retriever (iteration {state['iteration']})")
    return state


def _exhausted_node(state: RAGState) -> RAGState:
    """Runs when retries are exhausted without passing verification."""
    state["final_answer"] = (
        state["draft_answer"]
        + "\n\n[Note: this answer could not be fully verified against the "
        "retrieved sources after the maximum number of refinement attempts. "
        "Treat unsupported claims with caution.]"
    )
    state["history"].append("[Orchestrator] Max iterations reached — returning best-effort draft")
    return state


def _route_after_verification(state: RAGState) -> str:
    """Pure routing decision — reads state but does not mutate it."""
    if state["verified"]:
        return "end"
    if state["iteration"] + 1 >= state["max_iterations"]:
        return "exhausted"
    return "retry"


def build_graph(retriever: HybridRetriever, llm: LLMClient, hallucination_threshold: float = 0.2):
    retriever_agent = RetrieverAgent(retriever, llm)
    generator_agent = GeneratorAgent(llm)
    verifier_agent = VerifierAgent(llm, hallucination_threshold=hallucination_threshold)

    graph = StateGraph(RAGState)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("generator", generator_agent)
    graph.add_node("verifier", verifier_agent)
    graph.add_node("loop_back", _finalize_node)
    graph.add_node("exhausted", _exhausted_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verification,
        {"retry": "loop_back", "exhausted": "exhausted", "end": END},
    )
    graph.add_edge("loop_back", "retriever")
    graph.add_edge("exhausted", END)

    return graph.compile()


def run_pipeline(
    query: str,
    retriever: HybridRetriever,
    llm: LLMClient,
    max_iterations: int = 3,
    hallucination_threshold: float = 0.2,
) -> RAGState:
    app = build_graph(retriever, llm, hallucination_threshold)
    init_state: RAGState = {
        "query": query,
        "retrieved_chunks": [],
        "draft_answer": "",
        "final_answer": "",
        "critique": "",
        "claim_checks": [],
        "hallucination_score": 1.0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "verified": False,
        "history": [],
    }
    return app.invoke(init_state)
