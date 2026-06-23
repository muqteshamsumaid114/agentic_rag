import sys
import streamlit as st

# Bootstrapping helper: if run directly via `python app.py`, start Streamlit CLI
if not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", __file__]
    sys.exit(stcli.main())

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

# Page Configuration
st.set_page_config(
    page_title="Agentic RAG Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    }
    .status-ok {
        color: #10B981;
        font-weight: bold;
    }
    .status-fail {
        color: #EF4444;
        font-weight: bold;
    }
    .agent-step {
        background: #1E293B;
        border-left: 4px solid #818CF8;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path(__file__).parent / "data" / "sample_docs.json"

@st.cache_resource
def get_retriever():
    with open(DATA_PATH) as f:
        raw_docs = json.load(f)
    documents = [Document(doc_id=d["doc_id"], text=d["text"]) for d in raw_docs]

    dense = FAISSDenseIndex()
    sparse = BM25SparseIndex()
    reranker = CrossEncoderReranker()

    retriever = HybridRetriever(dense, sparse, reranker)
    retriever.index(documents)
    return retriever

# Header Section
st.markdown('<div class="main-title">🌌 Multi-Agent Self-Correcting RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">retrieval, reasoning, and self-critique pipeline with real-time hallucination checking.</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.image("https://img.icons8.com/color/96/space-exploration.png", width=80)
st.sidebar.header("Pipeline Configuration")
hallucination_threshold = st.sidebar.slider(
    "Hallucination Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.05,
    help="Maximum allowed proportion of unsupported claims. Lower is stricter."
)
max_iterations = st.sidebar.slider(
    "Max Iterations",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    help="Maximum agentic correction loops to resolve hallucinations."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Tech Stack Details")
st.sidebar.markdown("""
- **Embeddings**: SentenceTransformers (BAAI/bge-small-en-v1.5)
- **Dense Vector Store**: FAISS
- **Sparse Index**: BM25
- **Reranker**: Cross-Encoder (ms-marco-MiniLM-L-6-v2)
- **Orchestration**: LangGraph
- **LLM**: Groq (`llama-3.3-70b-versatile`)
""")

# Sample Questions
sample_questions = [
    "How does JWST's mirror compare to Hubble's, and why does JWST need to be so cold?",
    "What is JWST's exact angular resolution in arcseconds at 2 microns?",
    "Who built JWST and when was it launched?"
]

st.subheader("Select or Enter Query")
selected_sample = st.selectbox("Choose a sample question:", ["-- Custom --"] + sample_questions)

default_text = "" if selected_sample == "-- Custom --" else selected_sample
query = st.text_input("Enter your query:", value=default_text)

if st.button("Run Pipeline", type="primary"):
    if not query:
        st.warning("Please enter a query first!")
    else:
        with st.spinner("Initializing retriever and running LangGraph pipeline..."):
            retriever = get_retriever()
            llm = LLMClient(model="llama-3.3-70b-versatile")
            
            # Execute Pipeline
            state = run_pipeline(
                query=query,
                retriever=retriever,
                llm=llm,
                max_iterations=max_iterations,
                hallucination_threshold=hallucination_threshold
            )
            
            st.success("Pipeline finished execution!")
            
            # Results Columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📝 Final Answer")
                answer = state["final_answer"] or state["draft_answer"]
                st.info(answer)
                
                # Claims Checks
                st.subheader("🔍 Claim-level Verification")
                if state["claim_checks"]:
                    for chk in state["claim_checks"]:
                        status = "🟢 [OK]" if chk["supported"] else "🔴 [FAIL]"
                        conf = f"{chk['confidence']:.2%}"
                        st.markdown(f"**{status}** (confidence: {conf}) — *\"{chk['claim']}\"*")
                else:
                    st.write("No claims verified.")
                    
            with col2:
                # Key Metrics Card
                st.subheader("📊 Execution Metrics")
                is_verified = state["verified"]
                v_color = "green" if is_verified else "red"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>Status: <span style="color:{v_color};">{"Verified ✓" if is_verified else "Unverified ✗"}</span></h3>
                        <p><b>Hallucination Score:</b> {state['hallucination_score']:.2f}</p>
                        <p><b>Iterations:</b> {state['iteration'] + 1} / {max_iterations}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Agent Trace Log
                st.subheader("⚡ Agent Trace")
                for trace in state["history"]:
                    st.markdown(f'<div class="agent-step">{trace}</div>', unsafe_allow_html=True)
            
            # Retrieved chunks details
            with st.expander("📚 Retrieved Evidence Chunks (Top-K Reranked)"):
                for idx, c in enumerate(state["retrieved_chunks"]):
                    st.markdown(f"**[{idx+1}] Document ID: `{c.doc.doc_id}`** (Rerank Score: `{c.rerank_score:.3f}`)")
                    st.code(c.doc.text)
                    st.write("---")

