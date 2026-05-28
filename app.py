import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils.logger import get_logger
from src.utils.config import config
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

/* Main font declarations */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* Custom premium card design */
.premium-card {
    background: rgba(255, 255, 255, 1);
    border: 1px solid rgba(229, 231, 235, 1);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.025);
    margin-bottom: 24px;
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
</style>
"""

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = {}
    if "datasets" not in st.session_state:
        st.session_state.datasets = {}
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = None
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = None

def main():
    st.set_page_config(
        page_title="AI Analytics Copilot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    
    # ------------------ SIDEBAR CONFIGURATION ------------------
    st.sidebar.markdown("<h2 style='margin-top:0;'>⚙️ Settings & Ingestion</h2>", unsafe_allow_html=True)
    
    # Detect LLM APIs configured
    providers = llm_client.get_available_providers()
    
    st.sidebar.subheader("LLM Provider Configuration")
    if not providers:
        st.sidebar.error("⚠️ No API keys found! Please set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY inside your `.env` file to activate the Copilot.")
    else:
        cols = st.sidebar.columns(len(providers))
        for idx, provider in enumerate(providers):
            cols[idx].markdown(f"<span class='stat-badge' style='background-color:#D1FAE5; border-color:#34D399; color:#065F46; padding: 4px 8px; font-size:0.75rem; text-align:center;'>● {provider.upper()}</span>", unsafe_allow_html=True)
            
        selected_provider = st.sidebar.selectbox(
            "Active LLM Reasoner",
            options=providers,
            index=0 if st.session_state.selected_provider is None else providers.index(st.session_state.selected_provider)
        )
        st.session_state.selected_provider = selected_provider

    st.sidebar.markdown("---")
    
    # Multiple Data Ingestion
    st.sidebar.subheader("1. Ingest Datasets (CSV)")
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload multiple CSV datasets to explore and analyze."
    )
    
    # Ingest Data Dictionary
    st.sidebar.subheader("2. Ingest Business Dictionary")
    uploaded_dict = st.sidebar.file_uploader(
        "Upload dictionary (JSON/YAML)",
        type=["json", "yaml", "yml"],
        help="Provide column definitions, grains, KPIs, join keys, and business rules."
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
        st.sidebar.subheader("Active Workspace")
        active_dataset = st.sidebar.selectbox(
            "Active Dataset",
            options=list(st.session_state.datasets.keys()),
            index=list(st.session_state.datasets.keys()).index(st.session_state.active_dataset) if st.session_state.active_dataset in st.session_state.datasets else 0
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

    # Clear uploads
    if st.sidebar.button("🧹 Clear Context uploads", use_container_width=True):
        context_manager.delete_all_data()
        st.session_state.datasets = {}
        st.session_state.history = {}
        st.session_state.active_dataset = None
        st.rerun()

    # ------------------ MAIN SCREEN RENDER ------------------
    st.markdown("<div class='glow-title'>Enterprise AI Analytics Copilot</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.15rem; color:#4B5563; margin-top:-10px; margin-bottom:24px;'>Decoupling deterministic calculations from cognitive reasoning to deliver 100% accurate enterprise business intelligence.</p>", unsafe_allow_html=True)

    if not st.session_state.datasets:
        st.markdown(
            """
            <div class="premium-card" style="text-align: center; padding: 48px; border: 2px dashed #CBD5E1; background-color: #F8FAFC;">
                <div style="font-size: 4rem; margin-bottom: 16px;">📊</div>
                <h3>Welcome to your Analytics Copilot</h3>
                <p style="color: #64748B; max-width: 600px; margin: 0 auto 24px auto;">
                    To get started, upload one or more CSV datasets in the sidebar. Once ingested, the Copilot will automatically profile column types and launch the analytics interface.
                </p>
                <div style="display: flex; justify-content: center; gap: 16px;">
                    <span class="stat-badge">⚡ Safe Code Execution</span>
                    <span class="stat-badge">📈 Premium Plotly Visuals</span>
                    <span class="stat-badge">🔒 100% Local Math Checks</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
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
        style = "background-color:#E0F2FE; border-color:#7DD3FC; color:#0369A1; border: 1.5px solid #0284C7;" if is_active else "opacity: 0.6;"
        badge_str += f"<span class='stat-badge' style='{style}'>{'📂 Active: ' if is_active else '📄 '} {name}</span>"
        
    st.markdown(f"<div style='margin-bottom:20px;'>{badge_str}</div>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div style='margin-bottom:20px;'>
            <span class='stat-badge'>📊 Selected: {active_dataset}</span>
            <span class='stat-badge'>📋 Rows: {metadata['dimensions']['rows']:,}</span>
            <span class='stat-badge'>📐 Columns: {metadata['dimensions']['columns']}</span>
            <span class='stat-badge'>💾 Size: {metadata['dimensions']['memory_bytes'] / 1024 / 1024:.2f} MB</span>
            {"<span class='stat-badge' style='background-color:#D1FAE5; border-color:#34D399; color:#065F46;'>📝 Dictionary Active</span>" if dict_path else ""}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Core tabs
    tab_chat, tab_explore, tab_json, tab_dictionary = st.tabs([
        "💬 Analytics Copilot Chat", 
        "🔍 Profile Explorer", 
        "⚙️ Structured JSON Metadata",
        "📖 Data Dictionary"
    ])

    # ==================== TAB 1: COGNITIVE ANALYTICS CHAT ====================
    with tab_chat:
        st.markdown(f"### Ask Business Questions on: `{active_dataset}`")
        st.caption("Ask questions like: 'Why are users downgrading?', 'Show total recharge amounts by gender', or 'What is our tenure trend?'")
        
        # Display history
        active_history = st.session_state.history.get(active_dataset, [])
        for message in active_history:
            if message["role"] == "user":
                st.markdown(f"<div class='chat-bubble-user'>🧑‍💻 <b>You:</b><br>{message['content']}</div>", unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown("<div class='chat-bubble-assistant'>🤖 <b>Copilot:</b></div>", unsafe_allow_html=True)
                with st.container():
                    st.markdown(message["direct_answer"])
                    
                    if message.get("has_chart") and message.get("chart_spec"):
                        chart_df = pd.read_json(message["chart_df"])
                        fig = build_plotly_chart(chart_df, message["chart_spec"])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                    if message.get("has_table"):
                        table_df = pd.read_json(message["table_df"])
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
                            
                    if "confidence_score" in message:
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
        user_query = query_form.text_input("Ask a business question:", placeholder=f"Ask a question about {active_dataset}...", key="query_input")
        submit_query = query_form.form_submit_button("Run Analysis", use_container_width=True)

        if submit_query and user_query:
            st.session_state.history[active_dataset].append({"role": "user", "content": user_query})
            st.markdown(f"<div class='chat-bubble-user'>🧑‍💻 <b>You:</b><br>{user_query}</div>", unsafe_allow_html=True)
            
            # Synthesize consolidated multi-table LLM context profile
            context_profile = context_builder.build_llm_context(st.session_state.datasets)
            
            # Stage 1: High-Level Strategic Blueprint
            with st.spinner("Formulating Strategic Analysis Blueprint..."):
                try:
                    strategic_blueprint = generate_strategic_analysis_plan(
                        query=user_query,
                        context_profile=context_profile,
                        preferred_provider=st.session_state.selected_provider
                    )
                except Exception as e:
                    st.error(f"❌ Failed to construct strategic plan: {str(e)}")
                    return
            
            # Render Strategic blueprint immediately to provide visibility
            with st.expander("🧠 Cognitive Strategic Analysis Blueprint", expanded=True):
                st.markdown(f"**Logical Reasoning:**\n{strategic_blueprint.thought_process}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Analyses Selected to Run:**")
                    for step in strategic_blueprint.analysis_plan:
                        st.markdown(f"- `{step.replace('_', ' ').capitalize()}`")
                        
                    st.markdown("**Suggested Chart Layouts:**")
                    for vis in strategic_blueprint.suggested_visualizations:
                        st.markdown(f"- {vis}")
                with col2:
                    st.markdown("**Essential KPI Metrics:**")
                    for metric in strategic_blueprint.required_metrics:
                        st.markdown(f"- `{metric}`")
                        
                    st.markdown("**Segmentation Dimensions:**")
                    for dim in strategic_blueprint.required_dimensions:
                        st.markdown(f"- `{dim}`")
            
            # Stage 2: Low-Level Pandas Execution Planning
            with st.spinner("Formulating secure Pandas code plan..."):
                try:
                    plan = generate_analysis_plan(
                        query=user_query,
                        context_profile=context_profile,
                        strategic_blueprint=strategic_blueprint.model_dump(),
                        history=st.session_state.history[active_dataset][:-1],
                        preferred_provider=st.session_state.selected_provider
                    )
                except Exception as e:
                    st.error(f"❌ Failed to construct code plan: {str(e)}")
                    return
                
            with st.expander("💭 Generated Pandas Code & Sandboxing Details", expanded=False):
                st.markdown(f"**Thought Process:**\n{plan.thought_process}")
                st.code(plan.pandas_code, language="python")
                if plan.visual_spec.is_visual_requested:
                    st.markdown("**Requested Chart Specification:**")
                    st.json(plan.visual_spec.model_dump())
                    
            # Safe local execution with self-correction
            with st.spinner("Executing Pandas calculations securely..."):
                all_datasets = {name: info["df"] for name, info in st.session_state.datasets.items()}
                exec_result = execute_analysis(df, plan.pandas_code, all_datasets=all_datasets)
                
                # Auto-correction loop
                if not exec_result["success"]:
                    st.warning("⚠️ Initial analysis execution failed. Activating self-correction engine...")
                    
                    correction_prompt = f"""
Your previous Pandas code generated for the user question failed with a runtime exception.

User Question: "{user_query}"
Previous Code:
```python
{plan.pandas_code}
```
Runtime Error Exception:
{exec_result["error"]}

Please write the fixed Pandas script. Make sure you define `analyze(df)` function correctly.
"""
                    try:
                        plan = generate_analysis_plan(
                            query=correction_prompt,
                            context_profile=context_profile,
                            strategic_blueprint=strategic_blueprint.model_dump(),
                            history=st.session_state.history[active_dataset][:-1],
                            preferred_provider=st.session_state.selected_provider
                        )
                        exec_result = execute_analysis(df, plan.pandas_code, all_datasets=all_datasets)
                    except Exception as e:
                        logger.error(f"Self-correction planning failed: {str(e)}")
            
            if not exec_result["success"]:
                st.error("❌ Calculations failed validation or execution.")
                st.markdown(f"**Debug Error Logs:**\n```\n{exec_result['stderr'] or exec_result['error']}\n```")
                return
                
            result_df = exec_result["result"]
            # Extract statistical results from sandbox if generated
            stat_results = exec_result.get("sandbox_globals", {}).get("stat_results", None)
            
            if exec_result["stdout"].strip():
                with st.expander("🖥️ Execution standard output prints", expanded=False):
                    st.text(exec_result["stdout"])
                    
            if stat_results:
                with st.expander("🔬 Statistical Engine Output", expanded=True):
                    st.json(stat_results)
                    
            # Plotly Visuals rendering
            fig = None
            if plan.visual_spec.is_visual_requested:
                with st.spinner("Generating visuals..."):
                    fig = build_plotly_chart(result_df, plan.visual_spec, stat_results=stat_results)
                    
            # Insights Generation
            with st.spinner("Summarizing data and extracting insights..."):
                try:
                    interpretation = generate_insights_and_recommendations(
                        query=user_query,
                        result_data=result_df,
                        original_metadata=metadata,
                        history=st.session_state.history[active_dataset][:-1],
                        preferred_provider=st.session_state.selected_provider
                    )
                except Exception as e:
                    st.error(f"❌ Synthesis failed: {str(e)}")
                    return
                    
            # Render assistant chat response
            st.markdown("<div class='chat-bubble-assistant'>🤖 <b>Copilot:</b></div>", unsafe_allow_html=True)
            st.markdown(interpretation.direct_answer)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
            with st.expander("📊 Aggregated Result Data Table"):
                st.dataframe(result_df, use_container_width=True)
                
            st.markdown("##### 💡 Key Business Insights:")
            for ins in interpretation.insights:
                st.markdown(f"- {ins}")
                
            st.markdown("##### 🚀 Strategic Recommendations:")
            for rec in interpretation.recommendations:
                st.markdown(f"- {rec}")
                
            conf = interpretation.confidence_score
            color = "#059669" if conf >= 0.75 else "#D97706" if conf >= 0.5 else "#DC2626"
            label = "High Confidence" if conf >= 0.75 else "Medium Confidence" if conf >= 0.5 else "Low Confidence"
            st.markdown(f"<div style='margin-top: 12px;'><span class='stat-badge' style='color:{color}; border-color:{color};'>🎯 {label} (Score: {conf:.2f})</span></div>", unsafe_allow_html=True)
            
            if interpretation.statistical_backing:
                with st.expander("🔬 Statistical Evidence"):
                    for evidence in interpretation.statistical_backing:
                        st.markdown(f"- {evidence}")
                
            # Store history
            msg_data = {
                "role": "assistant",
                "content": interpretation.direct_answer,
                "direct_answer": interpretation.direct_answer,
                "has_chart": fig is not None,
                "chart_spec": plan.visual_spec if fig else None,
                "chart_df": result_df.to_json(orient="records") if fig else None,
                "has_table": result_df is not None,
                "table_df": result_df.to_json(orient="records") if result_df is not None else None,
                "insights": interpretation.insights,
                "recommendations": interpretation.recommendations,
                "confidence_score": interpretation.confidence_score,
                "statistical_backing": interpretation.statistical_backing
            }
            st.session_state.history[active_dataset].append(msg_data)
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

if __name__ == "__main__":
    main()
