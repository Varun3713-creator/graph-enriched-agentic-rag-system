import sys
import os
import pickle
import time
import streamlit as st

print("STREAMLIT APP STARTING...")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Agentic RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Dark background */
  .stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #0f1629 50%, #0d1117 100%);
    min-height: 100vh;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0d1117 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
  }

  /* Main cards */
  .rag-card {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }

  .rag-card-glow {
    background: rgba(17, 24, 39, 0.9);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 32px rgba(139, 92, 246, 0.15);
  }

  /* Answer card */
  .answer-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(30, 27, 75, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.5);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 40px rgba(139, 92, 246, 0.2), inset 0 1px 0 rgba(255,255,255,0.05);
  }

  /* Step log */
  .step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.03);
    border-left: 3px solid rgba(99, 102, 241, 0.5);
    transition: all 0.2s ease;
    font-size: 0.88rem;
  }

  .step-icon { font-size: 1.1rem; min-width: 24px; }
  .step-label { color: #a78bfa; font-weight: 600; min-width: 160px; }
  .step-detail { color: #94a3b8; font-size: 0.84rem; }

  /* Source chip */
  .source-chip {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a78bfa;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    margin: 3px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Confidence badge */
  .conf-badge-high {
    display: inline-block;
    background: linear-gradient(90deg, #065f46, #047857);
    border: 1px solid #10b981;
    color: #6ee7b7;
    border-radius: 20px;
    padding: 4px 16px;
    font-weight: 600;
    font-size: 0.9rem;
  }
  .conf-badge-mid {
    display: inline-block;
    background: linear-gradient(90deg, #78350f, #92400e);
    border: 1px solid #f59e0b;
    color: #fcd34d;
    border-radius: 20px;
    padding: 4px 16px;
    font-weight: 600;
    font-size: 0.9rem;
  }
  .conf-badge-low {
    display: inline-block;
    background: linear-gradient(90deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    color: #fca5a5;
    border-radius: 20px;
    padding: 4px 16px;
    font-weight: 600;
    font-size: 0.9rem;
  }

  /* Title styling */
  .hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    padding: 0;
    line-height: 1.2;
  }
  .hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 4px;
  }

  /* Input box */
  .stTextArea textarea {
    background: rgba(17, 24, 39, 0.8) !important;
    border: 1.5px solid rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
  }

  .stTextArea textarea:focus {
    border-color: rgba(139, 92, 246, 0.7) !important;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
  }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #6d28d9 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(109, 40, 217, 0.4) !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(109, 40, 217, 0.5) !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: rgba(17, 24, 39, 0.6) !important;
    border-radius: 10px !important;
    color: #a78bfa !important;
    font-weight: 600 !important;
  }

  /* Metrics */
  [data-testid="metric-container"] {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px;
  }

  /* Section headers */
  .section-header {
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    padding-bottom: 6px;
  }

  /* Status pills */
  .pill-online {
    background: rgba(6, 78, 59, 0.5);
    border: 1px solid #10b981;
    color: #6ee7b7;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
  }
  .pill-offline {
    background: rgba(127, 29, 29, 0.5);
    border: 1px solid #ef4444;
    color: #fca5a5;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load system components ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_system():
    """Load vector store and graph from disk (cached)."""
    from core.vector_store import VectorStore
    from rag.graph import ChunkGraph
    from config.settings import FAISS_INDEX_DIR

    store = VectorStore()
    graph_path = os.path.join(FAISS_INDEX_DIR, "graph.pkl")

    if not store.load():
        return None, None

    if os.path.exists(graph_path):
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
    else:
        graph = ChunkGraph()

    return store, graph


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size: 2.5rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #a78bfa;">Agentic RAG</div>
        <div style="font-size: 0.75rem; color: #64748b;">Advanced AI System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

    store, graph = load_system()
    if store is not None and len(store) > 0:
        st.markdown(f'<span class="pill-online">● Index Loaded</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top: 12px; font-size: 0.85rem; color: #94a3b8;">
            <div>📦 Chunks: <b style="color:#e2e8f0">{len(store)}</b></div>
            <div>🔗 Graph edges: <b style="color:#e2e8f0">{graph.graph.number_of_edges() if graph else 0}</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill-offline">● No Index Found</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="margin-top: 12px; padding: 12px; background: rgba(127,29,29,0.2); border-radius: 8px; 
                    border: 1px solid rgba(239,68,68,0.3); font-size: 0.8rem; color: #fca5a5;">
            Run indexing first:<br><br>
            <code style="background: rgba(0,0,0,0.4); padding: 4px 8px; border-radius: 4px; font-size: 0.75rem;">
            python index_documents.py
            </code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=5)
    show_scores = st.toggle("Show Ranking Scores", value=False)
    show_graph_info = st.toggle("Show Graph Info", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Sample Queries</div>', unsafe_allow_html=True)

    sample_queries = [
        "How do I fix error E05?",
        "What is the Wi-Fi setup procedure?",
        "What does error E12 mean?",
        "What is the current device status?",
        "How do I replace the ink cartridge?",
        "What are the print speed specifications?",
        "How often should I clean the printhead?",
    ]
    for q in sample_queries:
        if st.button(q, key=f"sq_{q[:20]}"):
            st.session_state["preset_query"] = q

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.7rem; color: #374151; text-align: center; border-top: 1px solid rgba(99,102,241,0.1); padding-top: 12px;">
        🧠 Advanced Agentic RAG<br>FAISS + NetworkX + Azure OpenAI
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("""
    <div style="padding: 8px 0 24px;">
        <p class="hero-title">🧠 Advanced Agentic RAG System</p>
        <p class="hero-sub">Intelligent document understanding · Graph-enhanced retrieval · Multi-agent reasoning</p>
    </div>
    """, unsafe_allow_html=True)

# ── Architecture diagram (collapsible) ───────────────────────────────────────
with st.expander("📐 System Architecture", expanded=False):
    st.markdown("""
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px;">
            <div class="section-header">Pipeline Flow</div>
            <div style="font-size: 0.83rem; color: #94a3b8; line-height: 2;">
                🧠 <b style="color:#a78bfa">Planner Agent</b> — Decides strategy<br>
                🔧 <b style="color:#60a5fa">Tool Agent</b> — Executes API tools<br>
                🔍 <b style="color:#34d399">Retriever</b> — FAISS + graph expansion<br>
                📊 <b style="color:#fbbf24">Ranker</b> — Multi-factor scoring<br>
                🧠 <b style="color:#a78bfa">LLM</b> — Generates answer<br>
                🔁 <b style="color:#f87171">Reflection</b> — Validates & scores
            </div>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <div class="section-header">Ranking Formula</div>
            <div style="font-size: 0.82rem; color: #94a3b8; background: rgba(0,0,0,0.3); padding: 12px; border-radius: 8px; font-family: 'JetBrains Mono', monospace;">
                Final Score =<br>
                &nbsp;&nbsp;0.5 × similarity<br>
                + 0.2 × section_relevance<br>
                + 0.2 × graph_connectivity<br>
                + 0.1 × positional_relevance
            </div>
        </div>
        <div style="flex: 1; min-width: 200px;">
            <div class="section-header">Graph Link Types</div>
            <div style="font-size: 0.83rem; color: #94a3b8; line-height: 2;">
                🔵 <b style="color:#60a5fa">NEXT</b> — Sequential chunks<br>
                🟣 <b style="color:#a78bfa">PARENT</b> — Section hierarchy<br>
                🟢 <b style="color:#34d399">SIMILAR</b> — Semantic similarity<br>
                🟡 <b style="color:#fbbf24">ENTITY_MATCH</b> — Shared keywords
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Query input ────────────────────────────────────────────────────────────────
preset = st.session_state.pop("preset_query", "")
query = st.text_area(
    "💬 Ask your question",
    value=preset,
    height=100,
    placeholder="e.g. 'How do I resolve error E05?' or 'What is the current device status?'",
    label_visibility="visible",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    run_clicked = st.button("🚀  Run Agentic Pipeline", use_container_width=True)
with col_clear:
    if st.button("🗑️  Clear", use_container_width=True):
        st.session_state.pop("last_result", None)
        st.rerun()

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_clicked:
    print(f"DEBUG: Run clicked. Query: '{query}'")
    if not query.strip():
        st.warning("Please enter a query.")
    elif store is None or len(store) == 0:
        print("DEBUG: Store is empty/None")
        st.error("⚠️ No index found. Please run `python index_documents.py` first.")
    else:
        print("DEBUG: Starting orchestrator...")
        from agents.orchestrator import Orchestrator

        orchestrator = Orchestrator(store, graph)

        # Progress placeholder
        progress_ph = st.empty()
        with progress_ph.container():
            st.markdown("""
            <div class="rag-card" style="text-align:center; padding: 32px;">
                <div style="font-size: 2rem; animation: pulse 1.5s infinite;">🧠</div>
                <div style="color: #a78bfa; font-weight: 600; margin-top: 12px;">Agentic pipeline running...</div>
                <div style="color: #64748b; font-size: 0.85rem; margin-top: 6px;">Planner → Tools → Retrieval → LLM → Reflection</div>
            </div>
            """, unsafe_allow_html=True)
            
            log_container = st.container()
            log_container.markdown('<div class="section-header">Live Reasoning Log</div>', unsafe_allow_html=True)
            log_step_ph = log_container.empty()

        def update_log(step):
            with log_step_ph.container():
                # Show ALL steps so far
                steps_html = ""
                for s in orchestrator.steps:
                    steps_html += f"""
                    <div class="step-row">
                        <span class="step-icon">{s.icon}</span>
                        <span class="step-label">{s.label}</span>
                        <span class="step-detail">{s.detail}</span>
                    </div>
                    """
                st.markdown(steps_html, unsafe_allow_html=True)

        with st.spinner(""):
            result = orchestrator.run(query, step_callback=update_log)

        progress_ph.empty()
        st.session_state["last_result"] = result
        st.session_state["last_query"] = query

# ── Display results ────────────────────────────────────────────────────────────
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    last_query = st.session_state.get("last_query", "")

    # ── Reasoning Steps ────────────────────────────────────────────────────────
    with st.expander("🧠 Full Agent Reasoning Log", expanded=False):
        st.markdown('<div class="section-header">Step-by-Step Execution</div>', unsafe_allow_html=True)
        for step in result.steps:
            st.markdown(f"""
            <div class="step-row">
                <span class="step-icon">{step.icon}</span>
                <span class="step-label">{step.label}</span>
                <span class="step-detail">{step.detail}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Answer ─────────────────────────────────────────────────────────────────
    conf = result.confidence
    if conf >= 0.8:
        badge_class = "conf-badge-high"
        conf_label = f"✅ {conf:.0%} Confidence"
    elif conf >= 0.6:
        badge_class = "conf-badge-mid"
        conf_label = f"⚠️ {conf:.0%} Confidence"
    else:
        badge_class = "conf-badge-low"
        conf_label = f"❌ {conf:.0%} Confidence"

    st.markdown(f"""
    <div class="answer-card">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div class="section-header" style="margin: 0; border: none; padding: 0;">💬 Generated Answer</div>
            <span class="{badge_class}">{conf_label}</span>
        </div>
        <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.75; white-space: pre-wrap;">{result.answer}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Reflection feedback ────────────────────────────────────────────────────
    refl = result.reflection
    refl_color = "#6ee7b7" if refl.is_grounded else "#fca5a5"
    st.markdown(f"""
    <div class="rag-card" style="padding: 16px 24px;">
        <div style="display: flex; gap: 24px; flex-wrap: wrap; align-items: center;">
            <div>
                <span style="color: #64748b; font-size: 0.8rem;">Grounded</span><br>
                <span style="color: {refl_color}; font-weight: 600;">{'✅ Yes' if refl.is_grounded else '❌ No'}</span>
            </div>
            <div>
                <span style="color: #64748b; font-size: 0.8rem;">Complete</span><br>
                <span style="color: {refl_color}; font-weight: 600;">{'✅ Yes' if refl.is_complete else '❌ No'}</span>
            </div>
            <div style="flex: 1;">
                <span style="color: #64748b; font-size: 0.8rem;">Reflection Feedback</span><br>
                <span style="color: #94a3b8; font-size: 0.88rem;">{refl.feedback or 'No feedback.'}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sources ────────────────────────────────────────────────────────────────
    if result.sources:
        with st.expander(f"📚 Source Chunks ({len(result.sources)} retrieved)", expanded=True):
            retrieval_results_map = {}
            if "last_result" in st.session_state:
                pass  # scores already baked in

            for i, chunk in enumerate(result.sources[:6]):
                st.markdown(f"""
                <div class="rag-card" style="padding: 16px 20px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div>
                            <span class="source-chip">#{i+1}</span>
                            <span class="source-chip">📂 {chunk.source}</span>
                            <span class="source-chip">📑 {chunk.section or 'General'}</span>
                            <span class="source-chip">🏷️ {chunk.chunk_type.value}</span>
                        </div>
                        <span style="color: #64748b; font-size: 0.75rem;">pos: {chunk.position}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6; background: rgba(0,0,0,0.2); 
                                padding: 10px 14px; border-radius: 8px; font-family: 'JetBrains Mono', monospace;
                                white-space: pre-wrap;">{chunk.text[:400]}{'...' if len(chunk.text) > 400 else ''}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Welcome state ──────────────────────────────────────────────────────────────
elif not run_clicked:
    st.markdown("""
    <div style="display: flex; gap: 16px; margin-top: 20px; flex-wrap: wrap;">
        <div class="rag-card" style="flex: 1; min-width: 200px; text-align: center; padding: 28px;">
            <div style="font-size: 2rem; margin-bottom: 12px;">📄</div>
            <div style="color: #a78bfa; font-weight: 600; margin-bottom: 6px;">Document Intelligence</div>
            <div style="color: #64748b; font-size: 0.83rem;">LLM-assisted structure detection with adaptive chunking for PDFs, HTML & text</div>
        </div>
        <div class="rag-card" style="flex: 1; min-width: 200px; text-align: center; padding: 28px;">
            <div style="font-size: 2rem; margin-bottom: 12px;">🔗</div>
            <div style="color: #60a5fa; font-weight: 600; margin-bottom: 6px;">Graph-Enhanced Retrieval</div>
            <div style="color: #64748b; font-size: 0.83rem;">NEXT · PARENT · SIMILAR · ENTITY_MATCH links expand context beyond top-k</div>
        </div>
        <div class="rag-card" style="flex: 1; min-width: 200px; text-align: center; padding: 28px;">
            <div style="font-size: 2rem; margin-bottom: 12px;">🤖</div>
            <div style="color: #34d399; font-weight: 600; margin-bottom: 6px;">Multi-Agent Reasoning</div>
            <div style="color: #64748b; font-size: 0.83rem;">Planner · Tool · Retriever · LLM · Reflection agents work in concert</div>
        </div>
        <div class="rag-card" style="flex: 1; min-width: 200px; text-align: center; padding: 28px;">
            <div style="font-size: 2rem; margin-bottom: 12px;">📊</div>
            <div style="color: #fbbf24; font-weight: 600; margin-bottom: 6px;">Confidence Scoring</div>
            <div style="color: #64748b; font-size: 0.83rem;">Every answer is validated and scored for grounding and completeness</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
