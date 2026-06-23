"""
Run the RAGAS evaluation suite against a small test set.

    export ANTHROPIC_API_KEY=sk-...
    export OPENAI_API_KEY=sk-...   # ragas's default judge LLM uses OpenAI;
                                    # see note at bottom of this file to swap it
    python run_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add the agentic_rag directory to sys.path to enable local imports
sys.path.insert(0, str(Path(__file__).parent / "agentic_rag"))

from core.llm_client import LLMClient
from eval.ragas_eval import EvalExample, run_eval_suite, summarize
from main import build_retriever

TEST_SET = [
    EvalExample(
        question="When was JWST launched and from where?",
        ground_truth="JWST was launched on December 25, 2021 from the Guiana Space Centre in French Guiana.",
    ),
    EvalExample(
        question="How big is JWST's primary mirror compared to Hubble's?",
        ground_truth="JWST's primary mirror is 6.5 meters across, much larger than Hubble's 2.4-meter mirror.",
    ),
    EvalExample(
        question="Why does JWST observe in infrared rather than visible light?",
        ground_truth="JWST observes in infrared so it can see through cosmic dust and detect light from some of the earliest, most distant galaxies.",
    ),
    EvalExample(
        question="Has JWST ever been serviced by astronauts like Hubble was?",
        ground_truth="No servicing missions are planned for JWST, unlike Hubble, which was serviced five times because it orbits much closer to Earth.",
    ),
]


def main():
    retriever = build_retriever()
    llm = LLMClient(model="llama-3.3-70b-versatile")

    df = run_eval_suite(TEST_SET, retriever, llm, max_iterations=3, hallucination_threshold=0.15)
    print(summarize(df))
    df.to_csv("eval_results.csv", index=False)
    print("\nFull per-example results written to eval_results.csv")


if __name__ == "__main__":
    main()

# NOTE: RAGAS's default metrics use an LLM-as-judge, which defaults to OpenAI
# under the hood. To judge with Claude instead, wrap LLMClient in ragas's
# LangchainLLMWrapper around a langchain_anthropic.ChatAnthropic instance and
# pass it via `evaluate(..., llm=your_wrapped_claude)`. Left as OpenAI-default
# here for brevity since RAGAS's OpenAI integration is the most battle-tested path.
