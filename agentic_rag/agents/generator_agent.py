"""Generator agent: drafts an answer strictly grounded in retrieved context.
On retries, it must explicitly address the verifier's critique."""
from __future__ import annotations

from core.llm_client import LLMClient
from core.state import RAGState

SYSTEM_PROMPT = """You are a careful, evidence-bound answer generator.

Rules:
- Only state facts that are directly supported by the provided CONTEXT.
- Every factual sentence must be traceable to at least one context passage.
- If the context is insufficient to answer fully, say so explicitly rather \
than filling gaps with prior knowledge.
- Cite passages inline using [doc_id] right after the claim they support.
- Be concise and direct."""

USER_TEMPLATE = """QUESTION:
{query}

CONTEXT PASSAGES:
{context}

{critique_block}

Write the answer now, citing [doc_id] after each claim."""

CRITIQUE_BLOCK_TEMPLATE = """PREVIOUS DRAFT WAS REJECTED. Reviewer feedback to address:
{critique}

Revise your answer to fix these specific issues. Do not repeat unsupported claims."""


class GeneratorAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def __call__(self, state: RAGState) -> RAGState:
        chunks = state["retrieved_chunks"]
        context = "\n\n".join(
            f"[{c.doc.doc_id}] {c.doc.text}" for c in chunks
        ) or "(no context retrieved)"

        critique_block = ""
        if state["iteration"] > 0 and state.get("critique"):
            critique_block = CRITIQUE_BLOCK_TEMPLATE.format(critique=state["critique"])

        user_msg = USER_TEMPLATE.format(
            query=state["query"], context=context, critique_block=critique_block
        )

        draft = self.llm.complete(system=SYSTEM_PROMPT, user=user_msg, max_tokens=600, temperature=0.2)

        state["draft_answer"] = draft
        state["history"].append(
            f"[Generator] Produced draft (iteration {state['iteration']}, {len(draft)} chars)"
        )
        return state
