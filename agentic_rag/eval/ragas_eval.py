"""
Evaluation harness using RAGAS.

Computes the standard RAG metrics:
  - faithfulness: is the answer grounded in retrieved context? (the external,
    library-computed analogue of our internal VerifierAgent score)
  - answer_relevancy: does the answer actually address the question?
  - context_precision: are retrieved passages relevant (ranked well)?
  - context_recall: did retrieval surface what's needed to answer fully?

We also report our own hallucination_score next to RAGAS faithfulness so you
can sanity-check the in-pipeline verifier against an independent metric —
big divergence between the two is a signal the verifier prompt needs tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from core.llm_client import LLMClient
from core.orchestrator import run_pipeline
from core.retriever import HybridRetriever


@dataclass
class EvalExample:
    question: str
    ground_truth: str  # reference answer, needed for context_recall


def run_eval_suite(
    examples: List[EvalExample],
    retriever: HybridRetriever,
    llm: LLMClient,
    max_iterations: int = 3,
    hallucination_threshold: float = 0.2,
) -> "object":
    """Runs the full agentic pipeline on each example, then scores the
    results with RAGAS. Returns the RAGAS result object (has .to_pandas())."""

    questions, answers, contexts, ground_truths, internal_scores = [], [], [], [], []

    for ex in examples:
        state = run_pipeline(
            ex.question,
            retriever,
            llm,
            max_iterations=max_iterations,
            hallucination_threshold=hallucination_threshold,
        )
        questions.append(ex.question)
        answers.append(state["final_answer"] or state["draft_answer"])
        contexts.append([c.doc.text for c in state["retrieved_chunks"]])
        ground_truths.append(ex.ground_truth)
        internal_scores.append(state["hallucination_score"])

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    df = result.to_pandas()
    df["internal_hallucination_score"] = internal_scores
    df["internal_faithfulness_proxy"] = [1 - s for s in internal_scores]
    df["faithfulness_divergence"] = (df["faithfulness"] - df["internal_faithfulness_proxy"]).abs()

    return df


def summarize(df) -> str:
    cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    lines = ["RAGAS Evaluation Summary", "=" * 40]
    for col in cols:
        if col in df.columns:
            lines.append(f"{col:22s}: {df[col].mean():.3f}")
    lines.append("-" * 40)
    lines.append(f"{'internal hallucination':22s}: {df['internal_hallucination_score'].mean():.3f}")
    lines.append(f"{'mean divergence vs RAGAS':22s}: {df['faithfulness_divergence'].mean():.3f}")
    return "\n".join(lines)
