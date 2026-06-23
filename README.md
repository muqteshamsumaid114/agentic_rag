# Multi-Agent Self-Correcting RAG Pipeline

A retrieval-augmented generation system where a **Retriever**, **Generator**, and
**Verifier** agent critique and refine each other's outputs in a loop, with
hybrid (dense + sparse) search, cross-encoder re-ranking, claim-level
hallucination detection, and RAGAS evaluation.

## Why this design

Most RAG demos are a single retrieve → generate pass. The interesting
engineering problem is what happens when the generator says something the
retrieved evidence doesn't support. This project treats that as a first-class
case instead of an edge case:

1. **Hybrid retrieval** (`core/retriever.py`) — dense FAISS search catches
   semantic matches; BM25 sparse search catches exact terms/identifiers dense
   embeddings often miss (model numbers, names, acronyms). Results are fused
   with **Reciprocal Rank Fusion** (not just averaged scores, which don't
   compare across different scoring scales) and then re-ranked with a
   cross-encoder, which is slower but far more precise than either retriever
   alone for the final top-k.

2. **Claim-level hallucination detection** (`agents/verifier_agent.py`) — the
   verifier doesn't just ask "does this look right?". It decomposes the draft
   into atomic claims, checks each one against the retrieved passages for
   entailment, and computes `hallucination_score = unsupported_claims / total_claims`.
   This is the same decompose-and-check idea RAGAS's `faithfulness` metric
   uses, implemented as an in-the-loop gate rather than a post-hoc metric.

3. **Self-correction loop** (`core/orchestrator.py`, LangGraph) — when the
   hallucination score exceeds a threshold, the critique routes back to the
   **retriever**, not directly to the generator. Often the real cause of a
   hallucination is *missing evidence*, not a careless generator — so the
   retriever reformulates the query based on the critique before the
   generator tries again. Bounded by `max_iterations` to avoid infinite loops.

4. **RAGAS evaluation** (`eval/ragas_eval.py`) — runs the full agentic
   pipeline over a test set and scores it with RAGAS's faithfulness, answer
   relevancy, context precision, and context recall, then compares RAGAS's
   faithfulness score against the pipeline's own internal hallucination
   score as a sanity check (large divergence = the verifier prompt needs work).

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │              query                   │
                    └───────────────┬───────────────────────┘
                                    ▼
                    ┌───────────────────────────┐
                    │     RETRIEVER AGENT        │◄────────────┐
                    │  dense (FAISS) + sparse    │              │ reformulated
                    │  (BM25) → RRF fusion →     │              │ query based on
                    │  cross-encoder rerank      │              │ critique
                    └───────────────┬─────────────┘              │
                                    ▼                             │
                    ┌───────────────────────────┐              │
                    │     GENERATOR AGENT        │              │
                    │  drafts answer grounded    │              │
                    │  in retrieved passages,    │              │
                    │  cites [doc_id] inline     │              │
                    └───────────────┬─────────────┘              │
                                    ▼                             │
                    ┌───────────────────────────┐              │
                    │     VERIFIER AGENT          │              │
                    │  decompose → claim-check    │              │
                    │  each vs evidence →          │              │
                    │  hallucination_score         │              │
                    └───────────────┬─────────────┘              │
                                    │                             │
                       score <= threshold?                       │
                          /              \                        │
                       yes                no, retries left ──────┘
                        │                  \
                        ▼                   no, retries exhausted
                  final answer                      │
                                                      ▼
                                          best-effort answer +
                                          "unverified" flag
```

## Project layout

```
agentic_rag/
├── core/
│   ├── retriever.py       # FAISS dense + BM25 sparse + RRF fusion + cross-encoder rerank
│   ├── llm_client.py      # thin Anthropic API wrapper
│   ├── state.py           # shared LangGraph state schema
│   └── orchestrator.py    # LangGraph graph: wires agents + self-correction loop
├── agents/
│   ├── retriever_agent.py # query reformulation + hybrid search
│   ├── generator_agent.py # grounded answer drafting
│   ├── verifier_agent.py  # claim decomposition + entailment checking + scoring
│   └── crewai_pipeline.py # same 3 roles, alternative CrewAI orchestration
├── eval/
│   └── ragas_eval.py      # RAGAS faithfulness/relevancy/precision/recall harness
├── data/
│   └── sample_docs.json   # toy knowledge base (JWST facts) for the demo
├── main.py                # end-to-end demo: index docs, run pipeline, print trace
├── run_eval.py             # run the RAGAS eval suite over a small test set
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

python main.py        # runs the agentic loop on sample questions, prints full trace
python run_eval.py    # runs RAGAS evaluation over a small test set
```

`main.py` includes one question the sample docs *can't* fully answer
(JWST's exact angular resolution in arcseconds), specifically to demonstrate
the verifier catching the gap and the loop reformulating the query rather
not letting the generator guess.

## LangGraph vs CrewAI

Both are included (`core/orchestrator.py` vs `agents/crewai_pipeline.py`) for
comparison:

- **LangGraph** models the self-correction loop as an explicit graph with a
  conditional edge — the loop, retry limit, and termination conditions are
  all visible in the graph definition. This is the version used by `main.py`.
- **CrewAI** models the same three roles via `Process.sequential` tasks. It
  reads more naturally as "who does what," but CrewAI doesn't have
  first-class conditional looping the way LangGraph does, so a real retry
  loop would need a manual wrapper around `crew.kickoff()`. Good to know
  when choosing a framework for a system where retries are core to the design.

## A bug worth knowing about (and how it was caught)

While testing the orchestration graph, an early version put the iteration
counter increment and the `final_answer` assignment inside the **conditional
edge router function** itself. LangGraph conditional-edge functions can only
*return a routing key* — any state mutation performed inside them is
discarded, not persisted. The symptom was a silent infinite loop (recursion
limit hit, `iteration` never advancing) and, separately, an empty
`final_answer` even when verification passed.

The fix: state mutations now happen only inside real graph **nodes**
(`_finalize_node`, `_exhausted_node`, and inside `VerifierAgent` itself for
the success case); the router (`_route_after_verification`) only reads state
and returns a string. This was caught with a mocked-agent unit test that
asserts on the call sequence and final state rather than running the full
LLM pipeline — worth keeping as a regression test if you extend the graph.

## Extending this

- **Swap FAISS for Weaviate**: `core/retriever.py` has a `WeaviateDenseIndex`
  stub with the same interface as `FAISSDenseIndex` — point `HybridRetriever`
  at it instead.
- **Add a 4th agent**: e.g. a "query router" agent in front that classifies
  whether a question even needs retrieval, or a "citation formatter" agent
  after verification passes.
- **Tune the hallucination threshold**: lower = stricter (more retries,
  fewer hallucinations slip through), higher = faster but more permissive.
  Use `run_eval.py`'s divergence column to see how the internal score tracks
  RAGAS's faithfulness as you adjust it.
