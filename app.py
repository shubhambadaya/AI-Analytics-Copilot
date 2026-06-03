import os
import io
import json
import streamlit as st

# Bridge Streamlit Community Cloud secrets into environment variables BEFORE any
# project module (which reads keys via os.getenv at import time) is imported.
# Locally this is a no-op when no secrets file exists.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

import pandas as pd
import plotly.graph_objects as go

from src.utils.logger import get_logger
from src.utils.config import config
from src.utils.query_log import query_log_store
from src.context.manager import context_manager
from src.context.builder import context_builder
from src.metadata.extractor import profile_dataframe
from src.metadata.dictionary import parse_and_validate_dictionary, merge_metadata_and_dictionary
from src.llm.client import llm_client
from src.llm.analysis_planner import generate_strategic_analysis_plan
from src.llm.planner import generate_analysis_plan
from src.analysis.engine import execute_analysis
from src.visuals.generator import build_plotly_chart
from src.llm.interpreter import generate_insights_and_recommendations

logger = get_logger(__name__)

# Premium, harmonized CSS styles injected directly into the page
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Main font declarations — Apple system font stack (SF Pro) */
html, body, div, span, applet, object, iframe, h1, h2, h3, h4, h5, h6, p, blockquote, pre, a, abbr, acronym, address, big, cite, code, del, dfn, em, img, ins, kbd, q, s, samp, small, strike, strong, sub, sup, tt, var, b, u, i, center, dl, dt, dd, ol, ul, li, fieldset, form, label, legend, table, caption, tbody, tfoot, thead, tr, th, td, article, aside, canvas, details, embed, figure, figcaption, footer, header, hgroup, menu, nav, output, ruby, section, summary, time, mark, audio, video, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', Roboto, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

p, span, div, li, td, th {
    line-height: 1.6;
    color: #1F2937;
}

h1, h2, h3, h4, h5, h6 {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', 'Segoe UI', Roboto, Arial, sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    color: #0F172A;
    line-height: 1.2;
}

/* AI answer output — Apple-inspired reading typography for the Copilot's
   response text (answer, insights, recommendations rendered as markdown). */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 1.05rem;
    line-height: 1.72;
    letter-spacing: -0.011em;
    color: #1D1D1F;            /* Apple near-black text */
    font-weight: 400;
}
[data-testid="stMarkdownContainer"] li {
    margin-bottom: 6px;
}
[data-testid="stMarkdownContainer"] strong {
    font-weight: 600;
    color: #1D1D1F;
}

/* Custom premium card design */
.premium-card {
    background: rgba(255, 255, 255, 1);
    border: 1px solid rgba(229, 231, 235, 1);
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.025);
    margin-bottom: 24px;
    transition: all 0.2s ease-in-out;
}

.premium-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    transform: translateY(-2px);
}

/* Stat badge styling */
.stat-badge {
    background: #EEF2F6;
    border-radius: 8px;
    padding: 8px 12px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #4A5568;
    display: inline-block;
    margin-right: 8px;
    margin-bottom: 8px;
    border: 1px solid #E2E8F0;
}

/* Chat bubble aesthetics */
.chat-bubble-user {
    background-color: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 12px 12px 0px 12px;
    padding: 16px;
    margin: 8px 0;
    text-align: right;
    align-self: flex-end;
}

.chat-bubble-assistant {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px 12px 12px 0px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}

.glow-title {
    background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

/* ---------- Landing / onboarding ---------- */
.hero-sub {
    font-size: 1.15rem;
    color: #475569;
    margin-top: -6px;
    margin-bottom: 28px;
    max-width: 720px;
    line-height: 1.6;
}

.onboard-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid #E5E7EB;
    border-radius: 16px;
    padding: 40px;
    box-shadow: 0 10px 30px -12px rgba(15, 23, 42, 0.12);
    margin-bottom: 24px;
}

.onboard-eyebrow {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #4F46E5;
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 16px;
}

.step-grid {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 4px 0;
}

.step-card {
    flex: 1;
    min-width: 200px;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 18px 20px;
}

.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, #4F46E5, #06B6D4);
    color: #fff;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 10px;
}

.step-card h4 { margin: 0 0 4px 0; font-size: 1rem; }
.step-card p { margin: 0; color: #64748B; font-size: 0.9rem; }

.example-chip {
    display: inline-block;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 999px;
    padding: 8px 16px;
    margin: 6px 8px 0 0;
    color: #334155;
    font-size: 0.92rem;
}

.trust-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 24px; }
.trust-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    color: #065F46;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.85rem;
    font-weight: 500;
}
</style>
"""

def require_password():
    """
    Lightweight access gate for shared deployments. Active only when APP_PASSWORD
    is configured (Streamlit secret or env var); a no-op locally when it isn't, so
    local development isn't blocked. Halts rendering until the correct password.
    """
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        return  # no gate configured (local / dev)
    if st.session_state.get("_authenticated"):
        return

    st.markdown("<div class='glow-title'>AI Analytics Copilot</div>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>🔒 Private preview — please enter the access password to continue.</p>", unsafe_allow_html=True)
    pw = st.text_input("Access password", type="password")
    if pw:
        if pw == expected:
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    st.stop()


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = {}
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = None
    if "draft_dictionary" not in st.session_state:
        st.session_state.draft_dictionary = None
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = None
    if "pending_clarification" not in st.session_state:
        st.session_state.pending_clarification = {}

def main():
    st.set_page_config(
        page_title="AI Analytics Copilot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    require_password()

    # ------------------ SIDEBAR CONFIGURATION ------------------
    st.sidebar.markdown("<h2 style='margin-top:0;'>⚙️ Setup</h2>", unsafe_allow_html=True)

    # Detect LLM APIs configured
    providers = llm_client.get_available_providers()

    st.sidebar.subheader("AI engine")
    if not providers:
        st.sidebar.error("⚠️ No AI model is connected yet. Add an API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY) to the `.env` file to get started.")
    else:
        cols = st.sidebar.columns(len(providers))
        for idx, provider in enumerate(providers):
            cols[idx].markdown(f"<span class='stat-badge' style='background-color:#D1FAE5; border-color:#34D399; color:#065F46; padding: 4px 8px; font-size:0.75rem; text-align:center;'>● {provider.upper()}</span>", unsafe_allow_html=True)

        selected_provider = st.sidebar.selectbox(
            "AI model",
            options=providers,
            index=0 if st.session_state.selected_provider is None else providers.index(st.session_state.selected_provider),
            help="The AI model that powers your answers. Any option works — pick one if unsure."
        )
        st.session_state.selected_provider = selected_provider

    st.sidebar.markdown("---")

    # Multiple Data Ingestion
    st.sidebar.subheader("1. Upload your data")
    uploaded_files = st.sidebar.file_uploader(
        "Add CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Drag and drop one or more spreadsheets (CSV format) to analyze."
    )

    # Ingest Data Dictionary
    st.sidebar.subheader("2. Add business context (optional)")
    uploaded_dict = st.sidebar.file_uploader(
        "Add a data dictionary",
        type=["json", "yaml", "yml"],
        help="Optional: a file describing what your columns mean and how key metrics (like ARPU) are calculated, so answers match your business definitions."
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            fname = uploaded_file.name
            if fname not in st.session_state.datasets:
                with st.spinner(f"Analyzing and profiling {fname}..."):
                    bytes_data = uploaded_file.getvalue()
                    saved_path = context_manager.save_uploaded_file(fname, bytes_data)
                    df = context_manager.load_dataset(saved_path)
                    
                    # Profile dataset
                    metadata = profile_dataframe(df)
                    
                    # Initialize dataset dict state
                    st.session_state.datasets[fname] = {
                        "path": saved_path,
                        "df": df,
                        "metadata": metadata,
                        "dictionary_path": None
                    }
                    st.session_state.history[fname] = []
                    
                    # Auto-set active
                    if st.session_state.active_dataset is None:
                        st.session_state.active_dataset = fname
                    
                    st.toast(f"Ingested and profiled: {fname}", icon="📊")

    # Select Active Dataset dropdown
    if st.session_state.datasets:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Your data")
        active_dataset = st.sidebar.selectbox(
            "Dataset to analyze",
            options=list(st.session_state.datasets.keys()),
            index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset) if st.session_state.active_dataset in st.session_state.datasets else 0,
            help="Choose which uploaded file your questions apply to."
        )
        st.session_state.active_dataset = active_dataset
        
        # Merge dictionary
        if uploaded_dict:
            active_dict_path = st.session_state.datasets[active_dataset]["dictionary_path"]
            if active_dict_path is None or uploaded_dict.name != os.path.basename(active_dict_path):
                with st.spinner("Validating and parsing data dictionary..."):
                    try:
                        bytes_dict = uploaded_dict.getvalue()
                        dict_path = context_manager.save_uploaded_file(uploaded_dict.name, bytes_dict)
                        dictionary_obj = parse_and_validate_dictionary(dict_path)
                        enriched_meta = merge_metadata_and_dictionary(
                            st.session_state.datasets[active_dataset]["metadata"], 
                            dictionary_obj
                        )
                        st.session_state.datasets[active_dataset]["dictionary_path"] = dict_path
                        st.session_state.datasets[active_dataset]["metadata"] = enriched_meta
                        st.toast("Business Data Dictionary validated and merged!", icon="📝")
                    except Exception as e:
                        st.sidebar.error(f"❌ Dictionary Validation Error:\n{str(e)}")

        # Auto-Generate Dictionary Button
        if st.session_state.datasets[active_dataset]["dictionary_path"] is None:
            if st.sidebar.button("✨ Auto-describe my data", use_container_width=True, help="Let the AI suggest what each column means and which metrics matter — you can review and edit before applying."):
                with st.spinner("LLM is inferring business semantics and KPIs..."):
                    from src.llm.dictionary_generator import auto_generate_dictionary
                    try:
                        draft_json = auto_generate_dictionary(
                            st.session_state.datasets[active_dataset]["metadata"],
                            st.session_state.selected_provider
                        )
                        st.session_state.draft_dictionary = draft_json
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Generation failed: {e}")

    # Clear uploads
    if st.sidebar.button("🧹 Clear all data", use_container_width=True, help="Remove all uploaded files and start over."):
        context_manager.delete_all_data()
        st.session_state.datasets = {}
        st.session_state.history = {}
        st.session_state.active_dataset = None
        st.rerun()

    # Global question log — export every question asked in the app. Persists across
    # sessions/redeploys when DATABASE_URL is configured; local JSON otherwise.
    _log_records = query_log_store.get_all()
    if _log_records:
        _log_df = pd.DataFrame(_log_records)
        if "timestamp" in _log_df.columns:
            _log_df.insert(0, "asked_at", pd.to_datetime(_log_df["timestamp"], unit="s"))
            _log_df = _log_df.drop(columns=["timestamp"])
        st.sidebar.download_button(
            label=f"📜 Download question log ({len(_log_records)})",
            data=_log_df.to_csv(index=False).encode("utf-8"),
            file_name="question_log.csv",
            mime="text/csv",
            use_container_width=True,
            help="Export every question asked in the app as a CSV.",
        )

    # ------------------ MAIN SCREEN RENDER ------------------
    st.markdown("<div class='glow-title'>AI Analytics Copilot</div>", unsafe_allow_html=True)
    st.markdown("<p class='hero-sub'>Ask questions about your data in plain English and get clear answers, charts, and recommendations in seconds — no formulas or code required.</p>", unsafe_allow_html=True)

    if st.session_state.draft_dictionary:
        st.markdown("### ✨ Review Auto-Generated Dictionary")
        st.info("The AI has inferred the following business rules and KPIs from the raw dataset. You can edit this JSON directly before applying.")
        edited_json = st.text_area("Draft Dictionary JSON", value=st.session_state.draft_dictionary, height=400)
        
        col1, col2 = st.columns(2)
        if col1.button("✅ Save & Apply Dictionary", use_container_width=True):
            try:
                active_dataset = st.session_state.active_dataset
                dict_path = context_manager.save_uploaded_file(f"auto_dict.json", edited_json.encode('utf-8'))
                dictionary_obj = parse_and_validate_dictionary(dict_path)
                enriched_meta = merge_metadata_and_dictionary(
                    st.session_state.datasets[active_dataset]["metadata"], 
                    dictionary_obj
                )
                st.session_state.datasets[active_dataset]["dictionary_path"] = dict_path
                st.session_state.datasets[active_dataset]["metadata"] = enriched_meta
                st.session_state.draft_dictionary = None # Clear draft
                st.toast("Dictionary successfully generated and applied!", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to parse or validate dictionary edits: {e}")
                
        if col2.button("🗑️ Discard Draft", use_container_width=True):
            st.session_state.draft_dictionary = None
            st.rerun()
            
        st.stop() # Suspend dashboard rendering until dictionary draft is handled

    if not st.session_state.datasets:
        st.markdown(
            """
            <div class="onboard-card">
                <span class="onboard-eyebrow">Getting started</span>
                <h2 style="margin: 0 0 8px 0;">👋 Let's explore your data</h2>
                <p style="color: #64748B; max-width: 680px; margin: 0 0 24px 0; font-size: 1.05rem;">
                    Upload a spreadsheet and ask questions in everyday language — like having your own
                    data analyst on call. You'll get the answer, a chart, and what to do next.
                </p>
                <div class="step-grid">
                    <div class="step-card">
                        <div class="step-num">1</div>
                        <h4>Upload your data</h4>
                        <p>Use the panel on the left to add one or more CSV files.</p>
                    </div>
                    <div class="step-card">
                        <div class="step-num">2</div>
                        <h4>Ask a question</h4>
                        <p>Type what you want to know — no special syntax needed.</p>
                    </div>
                    <div class="step-card">
                        <div class="step-num">3</div>
                        <h4>Get instant insights</h4>
                        <p>Read the answer, explore the chart, and act on the advice.</p>
                    </div>
                </div>
                <div style="margin-top: 28px;">
                    <p style="font-weight: 600; color: #0F172A; margin: 0 0 6px 0;">Try asking…</p>
                    <span class="example-chip">Which customer segments spend the most?</span>
                    <span class="example-chip">What drives higher revenue?</span>
                    <span class="example-chip">Who should we target for an upgrade?</span>
                    <span class="example-chip">Show average spend by age group</span>
                </div>
                <div class="trust-row">
                    <span class="trust-pill">✅ Every number is calculated from your data, not guessed</span>
                    <span class="trust-pill">⚡ Answers in seconds — no formulas or code</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.info("👈 Start by uploading a CSV file in the sidebar on the left.")
        return

    # Active Dataset details
    active_dataset = st.session_state.active_dataset
    dataset_info = st.session_state.datasets[active_dataset]
    df = dataset_info["df"]
    metadata = dataset_info["metadata"]
    dict_path = dataset_info["dictionary_path"]
    
    badge_str = ""
    for name in st.session_state.datasets.keys():
        is_active = (name == active_dataset)
        style = "background-color:#F8FAFC; border-color:#CBD5E1; color:#334155; font-weight: 500;" if is_active else "background-color:#F1F5F9; border-color:#E2E8F0; color:#64748B; opacity: 0.7;"
        active_dot = "<span style='color:#10B981;'>•</span> " if is_active else ""
        badge_str += f"<span class='stat-badge' style='{style}'>{active_dot}{name}</span>"
        
    st.markdown(f"<div style='margin-bottom:20px;'>{badge_str}</div>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div style='margin-bottom:24px; padding: 12px 16px; background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; display: inline-flex; gap: 16px; align-items: center;'>
            <span style='color: #475569; font-weight: 500; font-size: 0.9rem;'>Dataset: <span style='color:#0F172A;'>{active_dataset}</span></span>
            <span style='color: #CBD5E1;'>|</span>
            <span style='color: #475569; font-size: 0.9rem;'>Rows: <span style='color:#0F172A;'>{metadata['dimensions']['rows']:,}</span></span>
            <span style='color: #CBD5E1;'>|</span>
            <span style='color: #475569; font-size: 0.9rem;'>Cols: <span style='color:#0F172A;'>{metadata['dimensions']['columns']}</span></span>
            <span style='color: #CBD5E1;'>|</span>
            <span style='color: #475569; font-size: 0.9rem;'>Size: <span style='color:#0F172A;'>{metadata['dimensions']['memory_bytes'] / 1024 / 1024:.2f} MB</span></span>
            {"<span style='color: #CBD5E1;'>|</span><span style='color: #10B981; font-weight: 500; font-size: 0.9rem;'>• Dictionary Active</span>" if dict_path else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Core tabs
    tab_chat, tab_explore, tab_json, tab_dictionary, tab_architecture, tab_kb = st.tabs([
        "💬 Ask",
        "📋 Data Overview",
        "🧩 Technical Details",
        "📖 Data Dictionary",
        "⚙️ How It Works",
        "🧠 Learned Rules"
    ])

    # ==================== TAB 1: COGNITIVE ANALYTICS CHAT ====================
    with tab_chat:
        st.markdown(f"### 💬 Ask anything about `{active_dataset}`")
        st.caption("For example:  “Which customer segments spend the most?”  ·  “What drives higher revenue?”  ·  “Who should we target for an upgrade?”")
        
        # Display history
        active_history = st.session_state.history.get(active_dataset, [])
        for msg_idx, message in enumerate(active_history):
            if message["role"] == "user":
                st.markdown(f"<div class='chat-bubble-user'>🧑‍💻 <b>You:</b><br>{message['content']}</div>", unsafe_allow_html=True)
            elif message["role"] == "assistant":
                # Agent type badge
                agent_type = message.get("agent_type", "")
                if agent_type == "simple":
                    badge = "<span class='stat-badge' style='background-color:#ECFDF5; border-color:#10B981; color:#065F46;'>⚡ Fast Agent (Flash)</span>"
                elif agent_type == "complex":
                    badge = "<span class='stat-badge' style='background-color:#EEF2FF; border-color:#6366F1; color:#3730A3;'>🧠 Pro Agent (Deep Reasoning)</span>"
                elif agent_type == "predictive":
                    badge = "<span class='stat-badge' style='background-color:#FCE7F3; border-color:#DB2777; color:#9D174D;'>🔮 Predictive ML Agent (Scikit-Learn)</span>"
                elif agent_type == "learn":
                    badge = "<span class='stat-badge' style='background-color:#FEF3C7; border-color:#F59E0B; color:#92400E;'>📝 Knowledge Capture</span>"
                elif agent_type == "clarify":
                    badge = "<span class='stat-badge' style='background-color:#FEF9C3; border-color:#EAB308; color:#854D0E;'>🔎 Needs your input</span>"
                else:
                    badge = ""
                st.markdown(f"<div class='chat-bubble-assistant'>🤖 <b>Copilot:</b> {badge}</div>", unsafe_allow_html=True)

                # Single Query Render
                with st.container():
                    st.markdown(message.get("direct_answer", ""))

                    # Analyst plan: how the Copilot approached the question (complex path)
                    if message.get("approach") or message.get("planned_steps") or message.get("assumptions"):
                        with st.expander("🧭 How I approached this", expanded=False):
                            if message.get("approach"):
                                st.markdown(message["approach"])
                            if message.get("planned_steps"):
                                st.markdown("**Plan:** " + " → ".join(message["planned_steps"]))
                            if message.get("assumptions"):
                                st.markdown("**Assumptions:**")
                                for a in message["assumptions"]:
                                    st.markdown(f"- {a}")

                    if message.get("rendered_charts"):
                        for i, rc in enumerate(message["rendered_charts"]):
                            chart_df = pd.read_json(io.StringIO(rc["chart_df"]))
                            fig = build_plotly_chart(chart_df, rc["chart_spec"])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{active_dataset}_{msg_idx}_{i}")
                            
                    if message.get("has_table"):
                        table_df = pd.read_json(io.StringIO(message["table_df"]))
                        with st.expander("📊 Aggregated Result Data Table"):
                            st.dataframe(table_df, use_container_width=True)
                            
                    if message.get("insights"):
                        st.markdown("##### 💡 Key Business Insights:")
                        for ins in message["insights"]:
                            st.markdown(f"- {ins}")
                            
                    if message.get("recommendations"):
                        st.markdown("##### 🚀 Strategic Recommendations:")
                        for rec in message["recommendations"]:
                            st.markdown(f"- {rec}")
                            
                    if message.get("confidence_score"):
                        conf = message["confidence_score"]
                        color = "#059669" if conf >= 0.75 else "#D97706" if conf >= 0.5 else "#DC2626"
                        label = "High Confidence" if conf >= 0.75 else "Medium Confidence" if conf >= 0.5 else "Low Confidence"
                        st.markdown(f"<div style='margin-top: 12px;'><span class='stat-badge' style='color:{color}; border-color:{color};'>🎯 {label} (Score: {conf:.2f})</span></div>", unsafe_allow_html=True)
                        
                    if message.get("statistical_backing"):
                        with st.expander("🔬 Statistical Evidence"):
                            for evidence in message["statistical_backing"]:
                                st.markdown(f"- {evidence}")
                st.markdown("<hr style='margin:16px 0; border:0; border-top:1px solid #E5E7EB;'>", unsafe_allow_html=True)

        # Input Form
        query_form = st.form(key="business_query_form")
        user_query = query_form.text_input("Your question", placeholder=f"e.g. Which segments drive the most revenue in {active_dataset}?", key="query_input")
        submit_query = query_form.form_submit_button("Get answer", use_container_width=True)

        if submit_query and user_query:
            # If the Copilot previously asked for clarification, treat this reply as
            # the answer and fold it back into the original question.
            pending = st.session_state.pending_clarification.pop(active_dataset, None)
            if pending:
                effective_query = f"{pending['original_query']}\n\n[Stakeholder clarification: {user_query}]"
            else:
                effective_query = user_query

            st.session_state.history[active_dataset].append({"role": "user", "content": user_query})
            # Persist the asked question to the global (cross-session) query log.
            try:
                query_log_store.log_question(
                    user_query, dataset=active_dataset, provider=st.session_state.selected_provider or ""
                )
            except Exception as _e:
                logger.warning(f"Failed to log question: {_e}")
            st.markdown(f"<div class='chat-bubble-user'>🧑‍💻 <b>You:</b><br>{user_query}</div>", unsafe_allow_html=True)

            # Synthesize consolidated multi-table LLM context profile
            context_profile = context_builder.build_llm_context(st.session_state.datasets)

            from src.llm.pipeline import run_headless_pipeline
            all_datasets = {name: info["df"] for name, info in st.session_state.datasets.items()}

            # Iterative ReAct Execution Mode
            status_container = st.empty()
            with status_container.container():
                st.markdown("#### 🧠 Agentic Reasoning Engine Started...")

            final_result = None
            pipeline_gen = run_headless_pipeline(
                query=effective_query,
                context_profile=context_profile,
                provider=st.session_state.selected_provider,
                df=df,
                all_datasets=all_datasets,
                metadata=metadata,
                history=st.session_state.history[active_dataset][:-1]
            )
            
            for state in pipeline_gen:
                if state["status"] == "complete":
                    final_result = state["result"]
                    break
                elif state["status"] == "error":
                    final_result = {"success": False, "error": state.get("content")}
                    break
                elif state["status"] == "clarify":
                    # The Copilot needs input before assuming — present questions and pause.
                    qs = state.get("questions", [])
                    assumptions = state.get("assumptions", [])
                    approach = state.get("approach", "")
                    parts = []
                    if approach:
                        parts.append(approach)
                    parts.append(
                        "**Before I dig in, a quick check so I don't assume:**\n"
                        + "\n".join(f"- {q}" for q in qs)
                    )
                    if assumptions:
                        parts.append(
                            "_Otherwise I'll proceed with these assumptions — just reply to confirm or correct:_\n"
                            + "\n".join(f"- {a}" for a in assumptions)
                        )
                    clar_msg = "\n\n".join(parts)
                    final_result = {
                        "success": True, "role": "assistant", "agent_type": "clarify",
                        "query": user_query, "content": clar_msg, "direct_answer": clar_msg,
                        "has_chart": False, "has_table": False,
                    }
                    st.session_state.pending_clarification[active_dataset] = {"original_query": effective_query}
                    break
                else:
                    # Intermediate Status Update
                    with status_container.container():
                        if state["status"] == "thought":
                            decision = state.get('decision', '')
                            if "SIMPLE" in decision:
                                st.success(f"⚡ **Routed to Fast Agent (Flash)**: {state.get('reasoning')}")
                            elif "COMPLEX" in decision:
                                st.info(f"🧠 **Routed to Pro Agent (Deep Reasoning)**: {state.get('reasoning')}")
                            elif "ANALYST PLAN" in decision:
                                st.info(f"🧭 **Here's my plan**: {state.get('reasoning')}")
                            else:
                                st.info(f"🤔 **Agent Decision**: {decision} — {state.get('reasoning')}")
                        elif state["status"] == "code":
                            with st.expander(f"⚙️ Executing Code (Loop {state.get('step')})"):
                                st.code(state.get("code"), language="python")
                        else:
                            st.markdown(f"*{state.get('content')}*")
                            
            if final_result and final_result.get("success"):
                st.session_state.history[active_dataset].append(final_result)
                status_container.empty()
                st.rerun() # Force re-render of chat history to show the new messages properly
            elif final_result:
                status_container.empty()
                st.error(f"❌ Analysis failed: {final_result.get('error')}")
                # Don't rerun, just show the error in place so the user can read it.
            else:
                status_container.empty()
                st.error("❌ Analysis failed: Pipeline yielded no final result.")
                
            st.markdown("<hr style='margin:24px 0; border:0; border-top:1px solid #E5E7EB;'>", unsafe_allow_html=True)

    # ==================== TAB 2: DETAILED PROFILE EXPLORER ====================
    with tab_explore:
        st.markdown(f"### Dataset Schema Profile: `{active_dataset}`")
        st.caption("Automatic statistical summaries, column type detections, and cardinalities calculated safely locally.")
        
        with st.expander("🔍 Interactive Data Raw Preview (First 50 Rows)", expanded=True):
            st.dataframe(df.head(50), use_container_width=True)
            
        st.markdown("#### Inferred Columns Profiling")
        
        cols_profile = metadata.get("columns", {})
        for col_name, col_meta in cols_profile.items():
            cat = col_meta.get("category", "unknown")
            dtype = col_meta.get("data_type", "unknown")
            null_pct = col_meta.get("null_percentage", 0.0) * 100
            cardinality = col_meta.get("distinct_values", 0)
            desc = col_meta.get("description", "No description provided.")
            samples = ", ".join(col_meta.get("samples", []))
            
            badge_color = "#E0F2FE"
            badge_text_color = "#0369A1"
            if cat == "numeric":
                badge_color = "#D1FAE5"
                badge_text_color = "#065F46"
            elif cat == "datetime":
                badge_color = "#F3E8FF"
                badge_text_color = "#6B21A8"
            elif cat == "id":
                badge_color = "#FEF3C7"
                badge_text_color = "#92400E"
            elif cat == "boolean":
                badge_color = "#ECEFEE"
                badge_text_color = "#1F2937"
                
            data_quality = col_meta.get("data_quality_score", 1.0)
            dq_color = "#10B981" if data_quality > 0.8 else "#F59E0B" if data_quality > 0.5 else "#EF4444"
            
            stats_str = ""
            if cat == "numeric" and col_meta.get("statistics"):
                s = col_meta["statistics"]
                basic = f"Min: `{s.get('min')}` | Max: `{s.get('max')}` | Avg: `{s.get('mean'):.2f}` | Median: `{s.get('median'):.2f}`"
                advanced = f"Shape: `{s.get('distribution_shape', 'unknown')}` | Skew: `{s.get('skewness')}` | Outliers: `{s.get('outlier_count')}`"
                stats_str = f"| {basic} <br> | {advanced}"
            elif cat == "datetime" and col_meta.get("statistics"):
                s = col_meta["statistics"]
                stats_str = f"| Date Range: `{s.get('min')}` **to** `{s.get('max')}`"
            elif cat == "categorical" and col_meta.get("statistics"):
                s = col_meta["statistics"]
                freqs = [f"'{x['value']}': {x['count']} ({x['percentage']*100:.1f}%)" for x in s.get("top_frequencies", [])]
                stats_str = "| Top Frequencies: " + ", ".join(freqs)
                
            st.markdown(
                f"""
                <div class="premium-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <h4 style="margin:0; font-size:1.2rem;">{col_name}</h4>
                        <div>
                            <span style="background:{badge_color}; color:{badge_text_color}; font-weight:600; font-size:0.75rem; padding:4px 8px; border-radius:6px; margin-right:8px;">{cat.upper()}</span>
                            <span style="background:#F3F4F6; color:#374151; font-weight:500; font-size:0.75rem; padding:4px 8px; border-radius:6px;">{dtype}</span>
                        </div>
                    </div>
                    <p style="margin:0 0 8px 0; color:#4B5563; font-size:0.95rem;"><b>Description:</b> {desc}</p>
                    <div style="font-size:0.85rem; color:#6B7280; display:flex; flex-wrap:wrap; gap:16px; align-items:center;">
                        <span>🛡️ Data Quality: <b style="color:{dq_color};">{(data_quality*100):.0f}/100</b></span>
                        <span>❌ Null Rate: <b>{null_pct:.1f}%</b></span>
                        <span>🔑 Inferred Cardinality: <b>{cardinality}</b></span>
                        <span>🧪 Sample Values: <i>{samples}</i></span>
                    </div>
                    {f"<div style='font-size:0.85rem; color:#374151; margin-top:8px; border-top: 1px solid #F3F4F6; padding-top:8px;'>📊 Stats Summary: {stats_str}</div>" if stats_str else ""}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ==================== TAB 3: STRUCTURED JSON METADATA ====================
    with tab_json:
        st.markdown("### Structured Context Profile (JSON) for all active datasets")
        st.caption("Consolidated contextual intelligence profile assembled by the ContextBuilder.")
        
        active_context = context_builder.build_llm_context(st.session_state.datasets)
        pretty_json = json.dumps(active_context, indent=2)
        
        st.download_button(
            label="📥 Download Structured Context Profile",
            data=pretty_json,
            file_name="consolidated_context_profile.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.code(pretty_json, language="json")

    # ==================== TAB 4: DATA DICTIONARY ====================
    with tab_dictionary:
        st.markdown(f"### Enriched Data Dictionary for `{active_dataset}`")
        st.caption("Validates and merges YAML/JSON schemas to build the contextual intelligence layer.")
        
        if dict_path:
            st.success(f"✅ Contextual Business Dictionary Active from: `{os.path.basename(dict_path)}`")
            
            with st.expander("✏️ Edit Raw Dictionary JSON", expanded=False):
                st.info("You can make live updates to the data dictionary logic here.")
                try:
                    with open(dict_path, "r", encoding="utf-8") as f:
                        current_dict_json = f.read()
                    
                    edited_live_json = st.text_area("Dictionary Configuration", value=current_dict_json, height=300)
                    if st.button("💾 Save Changes to Dictionary", use_container_width=True):
                        try:
                            with open(dict_path, "w", encoding="utf-8") as f:
                                f.write(edited_live_json)
                            # Re-parse and merge
                            dictionary_obj = parse_and_validate_dictionary(dict_path)
                            enriched_meta = merge_metadata_and_dictionary(
                                st.session_state.datasets[active_dataset]["metadata"], 
                                dictionary_obj
                            )
                            st.session_state.datasets[active_dataset]["metadata"] = enriched_meta
                            st.toast("Dictionary updated successfully!", icon="✅")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to validate JSON edits: {e}")
                except Exception as e:
                    st.error(f"Could not load dictionary file: {e}")
                    
            st.markdown("#### Logical Grain Granularity")
            st.markdown(
                f"""
                <div class="premium-card" style="background-color: #F8FAFC; border-left: 5px solid #4F46E5;">
                    <p style="margin: 0; font-size: 1.1rem; font-weight: 500; color: #1E293B;">
                        {metadata.get('grain', 'No grain description defined.')}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("#### Business Key Performance Indicators (KPIs)")
            kpis = metadata.get("kpis", [])
            if kpis:
                kpi_records = []
                for kpi in kpis:
                    kpi_records.append({
                        "KPI Name": kpi.get("name"),
                        "Formula / Logic": f"`{kpi.get('formula')}`",
                        "Strategic Description": kpi.get("description")
                    })
                st.write(pd.DataFrame(kpi_records).to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No KPIs defined in this business dictionary.")
                
            st.markdown("<br>#### Column-Level Business Semantics", unsafe_allow_html=True)
            dict_records = []
            for col_name, col_meta in metadata.get("columns", {}).items():
                is_key = col_meta.get("is_join_key", False)
                key_badge = "<span style='background:#FEF3C7; color:#92400E; font-weight:600; padding:2px 6px; border-radius:4px;'>Join Key</span>" if is_key else "<span style='color:#9CA3AF;'>No</span>"
                
                rules = col_meta.get("business_rules", [])
                rules_str = "".join(f"<li style='margin-bottom:2px;'>{r}</li>" for r in rules) if rules else "<span style='color:#9CA3AF;'>None</span>"
                
                dict_records.append({
                    "Column": f"<b>{col_name}</b>",
                    "Semantic Meaning": col_meta.get("description", "No description provided."),
                    "Relationship": key_badge,
                    "Business Rules / Constraints": f"<ul style='margin:0; padding-left:14px;'>{rules_str}</ul>"
                })
            
            st.write(pd.DataFrame(dict_records).to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("ℹ️ No Contextual Business Dictionary has been loaded for this dataset. Automatically inferring structural column summaries.")
            st.markdown(
                """
                **Sample YAML Business Dictionary configuration:**
                ```yaml
                grain: "One row per transaction"
                
                columns:
                  rc_gap:
                    description: "Number of days between recharge and credit expiry"
                    is_join_key: false
                    business_rules:
                      - "Value must be positive"
                      - "Null values should be treated as 0"
                  customer_id:
                    description: "Unique integer identifying a registered subscriber"
                    is_join_key: true
                    
                kpis:
                  - name: "Average Days to Expiry"
                    formula: "mean(rc_gap)"
                    description: "Average days elapsed until credit expiration across recharge logs"
                ```
                """
            )

    # ==================== TAB 5: SYSTEM ARCHITECTURE ====================
    with tab_architecture:
        st.markdown("### ⚙️ How It Works")
        st.caption("How the Copilot turns your question into a trustworthy answer.")

        st.markdown(
            """
            <div class="step-grid">
                <div class="step-card"><div class="step-num">1</div><h4>You ask</h4>
                    <p>Type a business question in plain English.</p></div>
                <div class="step-card"><div class="step-num">2</div><h4>It checks what it needs</h4>
                    <p>If a key term (like “ARPU”) isn’t defined in your data or dictionary, it asks you first instead of guessing.</p></div>
                <div class="step-card"><div class="step-num">3</div><h4>It shares a plan</h4>
                    <p>For deeper questions it lays out its approach and assumptions before starting.</p></div>
            </div>
            <div class="step-grid" style="margin-top:16px;">
                <div class="step-card"><div class="step-num">4</div><h4>It calculates exactly</h4>
                    <p>It writes and runs real code on your data, so every number is computed — never made up.</p></div>
                <div class="step-card"><div class="step-num">5</div><h4>It digs deeper if needed</h4>
                    <p>Open-ended goals are explored step by step; “who should we…” questions train a prediction model.</p></div>
                <div class="step-card"><div class="step-num">6</div><h4>You get the answer</h4>
                    <p>A clear answer, charts, key insights, and recommendations — all backed by the data.</p></div>
            </div>
            <div class="trust-row">
                <span class="trust-pill">✅ Every number is computed from your data, not guessed</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔧 Technical view: the execution pipeline", expanded=False):
            mermaid_code = """graph TD
    classDef llm fill:#E0F2FE,stroke:#0284C7,color:#0369A1;
    classDef fast fill:#ECFDF5,stroke:#10B981,color:#065F46;
    classDef pred fill:#FCE7F3,stroke:#DB2777,color:#9D174D;
    classDef engine fill:#FEF3C7,stroke:#D97706,color:#92400E;
    classDef ui fill:#F3E8FF,stroke:#7E22CE,color:#6B21A8;

    User["You ask a question"]:::ui
    Ctx["Context Builder<br>data profile, KPIs, learned rules"]:::engine
    Router{"Router classifies the question"}:::llm
    User --> Ctx --> Router

    Router -->|teaching a rule| Learn["Save to Knowledge Base"]:::engine
    Router -->|simple| Fast["Fast Agent writes code<br>(reuses cache if seen before)"]:::fast
    Router -->|deep / why-how| Scope["Scope and Clarify<br>ask only if a term is undefined"]:::llm
    Router -->|predict / who| ML["ML Agent defines target and features"]:::pred

    Scope -->|needs input| Ask["Pause and ask you"]:::ui
    Scope -->|clear| Plan["Strategic plan and assumptions shared"]:::llm
    Plan --> Loop["Investigation loop:<br>step, run, reflect"]:::llm

    Fast --> Sandbox
    Loop --> Sandbox
    ML --> MLE["Train model<br>Random Forest"]:::engine

    Sandbox["Sandboxed math<br>exact, never guessed"]:::engine --> SV["Stats and charts"]:::llm
    SV --> Insight["Insight Agent<br>plain-English findings"]:::llm
    MLE --> Insight
    Insight --> Rec["Recommendations<br>grounded in the numbers"]:::llm
    Rec --> UI["Answer, charts and advice"]:::ui
    Learn --> UI"""
            try:
                import base64
                encoded = base64.urlsafe_b64encode(mermaid_code.encode("utf-8")).decode("utf-8")
                st.image(f"https://mermaid.ink/img/{encoded}", use_container_width=True)
                st.caption("Rendered via mermaid.ink — needs an internet connection.")
            except Exception as e:
                st.warning(f"Could not render the diagram: {e}")

    # ==================== TAB 6: SEMANTIC KB ====================
    with tab_kb:
        st.markdown("### 🧠 Learned Rules")
        st.caption("Teach the Copilot your business definitions once — it remembers them and applies them to every future answer.")

        st.markdown(
            "**What this is.** The Copilot keeps a memory of your business rules and definitions. "
            "Once you teach it something, it uses that meaning in every answer from now on — even in future sessions.\n\n"
            "**How to teach it.** Just type the rule as a normal message in the **💬 Ask** tab — no special format needed. For example:"
        )
        st.markdown(
            """
            <div style="margin: 4px 0 8px 0;">
                <span class="example-chip">Active users have more than 5GB of data usage</span>
                <span class="example-chip">ARPU means total recharge divided by subscriber count</span>
                <span class="example-chip">Treat customers on Plan 349+ as high-value</span>
                <span class="example-chip">Always show distributions as histograms</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "The Copilot recognises these as definitions (not questions), confirms it has saved them, and lists them below.\n\n"
            "**It will also ask you.** If you ask about a term it can't find in your data, KPIs, or saved rules "
            "(e.g. *“how do we improve ARPU?”* when no revenue measure is defined), it pauses and asks what you mean — "
            "and your answer becomes part of how it works.\n\n"
            "**Tip:** for exact formulas, defining them in your uploaded **data dictionary** "
            "(the *“2. Add business context”* step) is the most precise option."
        )

        st.markdown("---")
        st.markdown("#### 📂 Currently Active Rules")
        from src.llm.semantic_memory import semantic_store
        rules = semantic_store.get_all_rules()
        if rules:
            for i, rule in enumerate(rules):
                rule_col, del_col = st.columns([0.92, 0.08])
                with rule_col:
                    st.info(f"**{i+1}.** {rule}")
                with del_col:
                    if st.button("🗑️", key=f"del_rule_{i}", help="Delete this rule"):
                        semantic_store.delete_rule(i)
                        st.toast("Rule deleted.", icon="🗑️")
                        st.rerun()

            st.markdown("")
            if st.button("🧹 Clear all rules", use_container_width=True, help="Remove every learned rule."):
                n = semantic_store.clear_all_rules()
                st.toast(f"Cleared {n} rule(s).", icon="🧹")
                st.rerun()
        else:
            st.info("No business rules have been learned yet. Try teaching the Copilot something!")

if __name__ == "__main__":
    main()
