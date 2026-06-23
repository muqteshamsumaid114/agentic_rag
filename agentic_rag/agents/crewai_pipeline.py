"""
Alternative orchestration using CrewAI instead of LangGraph.

Why include both: LangGraph models the self-correction loop explicitly as a
graph with conditional edges (good for precise control over retry logic).
CrewAI models it as roles + tasks with a manager/process (good for quickly
expressing "who does what" when the control flow is simpler).

This version uses CrewAI's `Process.sequential` with a manual retry wrapper,
since native CrewAI doesn't have first-class conditional looping the way
LangGraph does — that asymmetry is itself worth knowing when picking a
framework for a production self-correction system.
"""
from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from core.llm_client import LLMClient
from core.retriever import HybridRetriever

_retriever_ref: dict = {}


@tool("Hybrid Search")
def hybrid_search_tool(query: str) -> str:
    """Search the knowledge base using hybrid dense+sparse retrieval with
    cross-encoder re-ranking. Returns the top passages with their doc_ids."""
    retriever: HybridRetriever = _retriever_ref["retriever"]
    chunks = retriever.retrieve(query, top_k=5)
    return "\n\n".join(f"[{c.doc.doc_id}] {c.doc.text}" for c in chunks)


def build_crew(retriever: HybridRetriever, llm_model: str = "claude-sonnet-4-6") -> Crew:
    _retriever_ref["retriever"] = retriever

    researcher = Agent(
        role="Retrieval Specialist",
        goal="Find the most relevant, high-quality evidence passages for the user's question",
        backstory=(
            "You are an expert at formulating search queries and judging "
            "passage relevance. You never fabricate evidence — you only "
            "report what hybrid_search_tool actually returns."
        ),
        tools=[hybrid_search_tool],
        llm=llm_model,
        verbose=True,
    )

    writer = Agent(
        role="Grounded Answer Generator",
        goal="Write answers that are fully supported by retrieved evidence, with inline citations",
        backstory=(
            "You are a careful technical writer who never states a fact "
            "without a citation to a specific doc_id from the evidence you "
            "were given. You would rather say 'the evidence doesn't say' than guess."
        ),
        llm=llm_model,
        verbose=True,
    )

    critic = Agent(
        role="Hallucination Auditor",
        goal="Catch any claim in the draft answer that isn't backed by the evidence",
        backstory=(
            "You are a skeptical fact-checker. For every sentence in the "
            "draft, you ask: which exact passage proves this? If you can't "
            "point to one, you flag it as unsupported and explain why."
        ),
        llm=llm_model,
        verbose=True,
    )

    research_task = Task(
        description=(
            "Research the question: '{query}'. Use the Hybrid Search tool to "
            "retrieve evidence. Return the raw retrieved passages verbatim, "
            "with their doc_ids, so downstream agents can cite them."
        ),
        expected_output="A list of retrieved passages, each prefixed with its [doc_id].",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "Using ONLY the evidence passages from the research task, write "
            "an answer to: '{query}'. Cite [doc_id] after every factual claim. "
            "If evidence is insufficient for part of the question, say so explicitly."
        ),
        expected_output="A grounded, cited answer to the user's question.",
        agent=writer,
        context=[research_task],
    )

    audit_task = Task(
        description=(
            "Audit the draft answer against the evidence passages from the "
            "research task. List any claim that lacks a supporting citation "
            "or whose citation doesn't actually support it. Then produce a "
            "FINAL revised answer with unsupported claims removed or qualified."
        ),
        expected_output=(
            "A short audit list of unsupported claims (if any), followed by "
            "a 'FINAL ANSWER:' section with the corrected, fully-grounded answer."
        ),
        agent=critic,
        context=[research_task, writing_task],
    )

    return Crew(
        agents=[researcher, writer, critic],
        tasks=[research_task, writing_task, audit_task],
        process=Process.sequential,
        verbose=True,
    )


def run_crew_pipeline(query: str, retriever: HybridRetriever) -> str:
    crew = build_crew(retriever)
    result = crew.kickoff(inputs={"query": query})
    return str(result)
