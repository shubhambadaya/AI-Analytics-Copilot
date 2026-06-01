from typing import Dict, Any, List
import json
from types import SimpleNamespace
import pandas as pd
import src.analysis.stats_engine as stats_engine
from src.llm.analysis_planner import generate_strategic_analysis_plan
from src.llm.agent import evaluate_agent_state
from src.llm.clarifier import run_clarification_check
from src.llm.planner import generate_analysis_plan
from src.analysis.engine import execute_analysis
from src.visuals.generator import build_plotly_chart
from src.llm.interpreter import generate_insights_and_recommendations
from src.llm.golden_queries import golden_store
from src.llm.complexity_router import classify_query_complexity
from src.llm.client import MODEL_FAST, MODEL_PRO
from src.utils.logger import get_logger

# Import the new 5-Agent DAG modules
from src.llm.schema_agent import run_schema_agent
from src.llm.stats_agent import run_stats_agent
from src.llm.viz_agent import run_viz_agent
from src.llm.insight_agent import run_insight_agent
from src.llm.recommender_agent import run_recommender_agent
from src.llm.ml_agent import run_ml_agent

logger = get_logger(__name__)

def run_headless_pipeline(
    query: str, 
    context_profile: Dict[str, Any], 
    provider: str, 
    df: pd.DataFrame, 
    all_datasets: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    history: List[Dict[str, Any]]
):
    """
    Executes the cognitive analytics pipeline with dual-agent routing.
    SIMPLE queries use Gemini Flash for fast single-shot execution.
    COMPLEX queries use Gemini Pro with the full iterative ReAct loop.
    Yields intermediate states to the UI, and finally yields the completed package.
    """
    logger.info(f"Pipeline started for query: '{query}'")
    
    # ==================== STEP 0: COMPLEXITY CLASSIFICATION ====================
    yield {"status": "running", "step": "Routing", "content": "Classifying query complexity..."}
    classification = classify_query_complexity(query, provider=provider)
    complexity = classification.complexity
    
    yield {
        "status": "thought", 
        "step": 0, 
        "decision": f"ROUTE → {complexity}", 
        "reasoning": classification.reasoning
    }
    
    # ==================== LEARN_RULE (detected by the router — no extra LLM call) ====================
    # The router classifies rule-teaching statements directly, so we avoid a second
    # (previously Pro-model) call on every query just to check for LEARN_RULE.
    if complexity == "LEARN_RULE":
        from src.llm.semantic_memory import semantic_store
        rule = classification.learned_rule or classification.reasoning
        semantic_store.save_rule(rule)
        yield {
            "status": "complete",
            "result": {
                "success": True,
                "query": query,
                "role": "assistant",
                "agent_type": "learn",
                "content": "🧠 **Rule Learned & Saved!** I have committed this to my long-term Semantic Knowledge Base.",
                "direct_answer": f"Learned Rule: {rule}",
                "has_chart": False,
                "has_table": False
            }
        }
        return

    if complexity == "SIMPLE":
        yield from _run_simple_path(query, context_profile, provider, df, all_datasets, metadata, history)
    elif complexity == "PREDICTIVE":
        yield from _run_predictive_path(query, context_profile, provider, df, all_datasets, metadata, history)
    else:
        yield from _run_complex_path(query, context_profile, provider, df, all_datasets, metadata, history)


def _run_simple_path(
    query: str, context_profile: Dict[str, Any], provider: str,
    df: pd.DataFrame, all_datasets: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any], history: List[Dict[str, Any]]
):
    """
    SIMPLE Path: Single-shot code generation + execution using Gemini Flash.
    No strategic blueprint, no ReAct loop. Target latency: 3-5 seconds.
    """
    logger.info("Executing SIMPLE path (Gemini Flash)")
    yield {"status": "running", "step": "Simple Agent", "content": "⚡ Fast Agent: Generating analysis in single pass..."}

    plan = None
    exec_result = None

    # 0. Cache short-circuit: reuse the validated code of a near-identical past
    # analysis, skipping the code-generation LLM call entirely. The cached code is
    # still run through the validator/engine, so a stale match (different dataset
    # or columns) fails safely and falls through to normal generation below.
    cached = golden_store.get_cached_code(query)
    if cached:
        yield {"status": "running", "step": "Cache",
               "content": "⚡ Cache hit: reusing a validated past analysis — skipping code generation..."}
        yield {"status": "code", "step": 1, "code": cached["pandas_code"]}
        exec_result = execute_analysis(df, cached["pandas_code"], all_datasets=all_datasets)
        if exec_result["success"]:
            plan = SimpleNamespace(pandas_code=cached["pandas_code"], visual_spec=None)
        else:
            logger.info("Cached code failed to execute; falling back to code generation.")
            exec_result = None

    # 1. Single-shot code generation (using Flash model) — only if no usable cache hit
    if plan is None:
        try:
            plan = generate_analysis_plan(
                query=query,
                context_profile=context_profile,
                history=history,
                preferred_provider=provider,
                model_override=MODEL_FAST
            )
        except Exception as e:
            yield {"status": "error", "content": f"Code generation failed: {e}"}
            return

        yield {"status": "code", "step": 1, "code": plan.pandas_code}

        # 2. Execute
        exec_result = execute_analysis(df, plan.pandas_code, all_datasets=all_datasets)

        # 3. Self-correction (one retry with Flash)
        if not exec_result["success"]:
            yield {"status": "running", "step": "Self-Correction", "content": "Retrying with error feedback..."}
            correction_prompt = f"Previous code failed: {exec_result['error']}\nRewrite the script for: {query}"
            try:
                plan = generate_analysis_plan(
                    query=correction_prompt,
                    context_profile=context_profile,
                    history=history,
                    preferred_provider=provider,
                    model_override=MODEL_FAST
                )
                exec_result = execute_analysis(df, plan.pandas_code, all_datasets=all_datasets)
            except Exception:
                pass

    if not exec_result or not exec_result["success"]:
        err = exec_result["error"] if exec_result else "unknown error"
        yield {"status": "error", "content": f"Analysis failed: {err}"}
        return

    result_df = exec_result["result"]
    stat_results = exec_result.get("sandbox_globals", {}).get("stat_results", None)
    
    # 4. Single-shot interpretation (using Flash)
    yield {"status": "running", "step": "Interpretation", "content": "⚡ Fast Agent: Summarizing results..."}
    memory_buffer = [{
        "step": 1,
        "focus": query,
        "code": plan.pandas_code,
        "status": "success",
        "stat_results": stat_results,
        "raw_df": result_df
    }]
    
    try:
        interpretation = generate_insights_and_recommendations(
            query=query,
            result_data=memory_buffer,
            original_metadata=metadata,
            history=history,
            preferred_provider=provider,
            model_override=MODEL_FAST
        )
    except Exception as e:
        yield {"status": "error", "content": f"Interpretation failed: {e}"}
        return
    
    # 5. Build charts
    rendered_charts = _build_charts(plan, interpretation, result_df, stat_results)
    
    # Golden Store
    if interpretation.confidence_score >= 0.75:
        golden_store.save_golden_query(
            user_query=query,
            pandas_code=plan.pandas_code,
            confidence_score=interpretation.confidence_score
        )
    
    yield {"status": "complete", "result": _build_final_package(
        query, interpretation, memory_buffer, rendered_charts, result_df, agent_type="simple"
    )}


# Upper bound on ReAct investigation iterations, to cap latency and cost.
MAX_INVESTIGATION_STEPS = 5


def _safe_preview(result_df: pd.DataFrame, n: int = 5) -> Any:
    """Small, JSON-native preview of a result frame (pandas handles dates/numpy via to_json)."""
    if result_df is None or result_df.empty:
        return []
    try:
        return json.loads(result_df.head(n).to_json(orient="records"))
    except Exception:
        return []


def _build_recommender_evidence(primary_df, stat_results, successful_steps, max_rows: int = 25) -> str:
    """
    Assemble a compact, numeric evidence payload for the Recommendation Agent so it
    can cite concrete figures (segments, counts, stats) rather than vague advice.
    """
    parts = []
    if primary_df is not None and not primary_df.empty:
        try:
            table = primary_df.head(max_rows).to_markdown(index=False)
        except Exception:
            table = json.dumps(_safe_preview(primary_df, max_rows))
        parts.append(f"Primary result table:\n{table}")

    if stat_results:
        parts.append(f"Statistical tests:\n{json.dumps(stat_results)}")

    # Highlight earlier investigation steps so cross-step findings are available too.
    if len(successful_steps) > 1:
        lines = [
            f"- Step {m['step']} ({m['focus']}): {json.dumps(m.get('data_preview'))[:400]}"
            for m in successful_steps[:-1]
        ]
        parts.append("Earlier investigation findings:\n" + "\n".join(lines))

    return "\n\n".join(parts)


def _build_predictive_evidence(result_df, max_rows: int = 15) -> str:
    """
    Summarize ML scoring output — predicted-class distribution, confidence, key
    drivers, and a sample of the highest-confidence rows — so the Recommendation
    Agent can target the right cohort with concrete numbers.
    """
    if result_df is None or result_df.empty:
        return ""

    parts = []
    pred_cols = [c for c in result_df.columns if c.startswith("predicted_")]
    if pred_cols:
        dist = result_df[pred_cols[0]].value_counts().to_dict()
        parts.append(f"Predicted class distribution ({pred_cols[0]}): {dist}")

    if "confidence_score" in result_df.columns:
        conf = result_df["confidence_score"]
        n_high = int((conf >= 0.7).sum())
        parts.append(
            f"Total scored users: {len(result_df)}; high-confidence (>=0.7): {n_high}; "
            f"mean confidence: {float(conf.mean()):.3f}"
        )

    if "key_factors" in result_df.columns and len(result_df):
        parts.append(f"Top predictive drivers: {result_df['key_factors'].iloc[0]}")

    try:
        sample = result_df.sort_values("confidence_score", ascending=False) \
            if "confidence_score" in result_df.columns else result_df
        parts.append("Highest-confidence scored rows:\n" + sample.head(max_rows).to_markdown(index=False))
    except Exception:
        pass

    return "\n\n".join(parts)


def _run_complex_path(
    query: str, context_profile: Dict[str, Any], provider: str,
    df: pd.DataFrame, all_datasets: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any], history: List[Dict[str, Any]]
):
    """
    COMPLEX Path: Strategic, iterative ReAct investigation.

    A strategic blueprint decomposes the goal into analytical steps, then a bounded
    ReAct loop runs them one at a time — generating code, executing it, and letting
    the Evaluator decide whether to CONTINUE (with a new focus) or COMPLETE. Each
    step's findings accumulate in a memory buffer that drives multi-step synthesis.

    Simple multi-step queries terminate after one iteration (the Evaluator returns
    COMPLETE), so this degrades gracefully; open-ended goals fan out across steps.
    """
    logger.info("Executing COMPLEX path (Strategic ReAct Investigation)")

    # ---------- 0. Scope the question (analyst step) ----------
    # Decide whether a key term is genuinely undefined (ask only if critical), and
    # capture the analyst's assumptions + approach to share with the stakeholder.
    yield {"status": "running", "step": "Scoping",
           "content": "🔎 Reviewing your question and scoping the analysis..."}
    clar = None
    try:
        clar = run_clarification_check(query, context_profile, history, provider, MODEL_FAST)
    except Exception as e:
        logger.warning(f"Clarification check failed, proceeding without it: {e}")

    if clar and clar.needs_clarification and clar.clarifying_questions:
        # Pause and ask the stakeholder rather than guessing an undefined concept.
        yield {
            "status": "clarify",
            "questions": clar.clarifying_questions,
            "assumptions": clar.assumptions,
            "approach": clar.approach,
        }
        return

    # ---------- 1. Strategic blueprint ----------
    yield {"status": "running", "step": "Strategy",
           "content": "🧭 Strategic Planner: Decomposing the goal into an analysis blueprint..."}
    try:
        blueprint = generate_strategic_analysis_plan(query, context_profile, preferred_provider=provider)
    except Exception as e:
        yield {"status": "error", "content": f"Strategic planning failed: {e}"}
        return

    # Share the plan with the stakeholder in analyst voice (then auto-run).
    analyst_approach = clar.approach if clar else ""
    analyst_assumptions = clar.assumptions if clar else []
    planned_steps = [s for s in blueprint.analysis_plan if s]
    plan_lines = []
    if analyst_approach:
        plan_lines.append(analyst_approach)
    if planned_steps:
        plan_lines.append("Plan: " + " → ".join(planned_steps) + ".")
    if analyst_assumptions:
        plan_lines.append("Assumptions: " + "; ".join(analyst_assumptions) + ".")
    yield {"status": "thought", "step": 0, "decision": "ANALYST PLAN",
           "reasoning": " ".join(plan_lines) or blueprint.thought_process}

    goal_context = (
        f"Overall Goal: {query}\n"
        f"Strategic steps: {blueprint.analysis_plan}\n"
        f"Required metrics: {blueprint.required_metrics}\n"
        f"Required dimensions: {blueprint.required_dimensions}"
    )
    # The blueprint's planned steps drive the investigation so it genuinely fans
    # out as planned. Once they're exhausted, the Evaluator governs early-stop and
    # any adaptive follow-up steps (up to the cap).
    pending_steps: List[str] = [s for s in blueprint.analysis_plan if s] or [query]
    next_focus = f"{goal_context}\n\nExecute this planned analytical step: {pending_steps.pop(0)}"

    # ---------- 2. ReAct investigation loop ----------
    memory_buffer: List[Dict[str, Any]] = []
    last_result_df = None

    for step_idx in range(1, MAX_INVESTIGATION_STEPS + 1):
        focus_label = next_focus.splitlines()[-1][:160]
        yield {"status": "running", "step": f"Investigate {step_idx}",
               "content": f"🔬 Step {step_idx}: {focus_label}"}

        # 2a. Generate code for the current focus
        try:
            schema_plan = run_schema_agent(next_focus, context_profile, history, provider, MODEL_PRO)
        except Exception as e:
            yield {"status": "error", "content": f"Schema Agent failed at step {step_idx}: {e}"}
            return

        yield {"status": "thought", "step": step_idx, "decision": f"REASON (step {step_idx})",
               "reasoning": schema_plan.thought_process}
        yield {"status": "code", "step": step_idx, "code": schema_plan.pandas_code}

        # 2b. Execute, with one self-correction retry
        exec_result = execute_analysis(df, schema_plan.pandas_code, all_datasets=all_datasets)
        if not exec_result["success"]:
            correction = (f"Original focus: {next_focus}\n\nYour previous code failed with:\n"
                          f"{exec_result['error']}\nRewrite it.")
            schema_plan = run_schema_agent(correction, context_profile, history, provider, MODEL_PRO)
            exec_result = execute_analysis(df, schema_plan.pandas_code, all_datasets=all_datasets)

        if exec_result["success"]:
            result_df = exec_result["result"]
            last_result_df = result_df
            memory_buffer.append({
                "step": step_idx, "focus": focus_label, "code": schema_plan.pandas_code,
                "status": "success", "raw_df": result_df, "data_preview": _safe_preview(result_df)
            })
        else:
            # Record the failure; the Evaluator can still decide to try another angle.
            memory_buffer.append({
                "step": step_idx, "focus": focus_label, "code": schema_plan.pandas_code,
                "status": "failed", "error": exec_result["error"], "raw_df": None
            })

        # 2c. Reflect — continue the investigation or conclude?
        if step_idx >= MAX_INVESTIGATION_STEPS:
            yield {"status": "thought", "step": step_idx, "decision": "STEP CAP REACHED",
                   "reasoning": "Reached the maximum number of investigation steps; synthesizing findings."}
            break

        # While planned blueprint steps remain, keep following the plan (genuine
        # fan-out). The Evaluator only governs once the plan is exhausted.
        if pending_steps:
            planned_step = pending_steps.pop(0)
            yield {"status": "thought", "step": step_idx, "decision": "NEXT PLANNED STEP",
                   "reasoning": f"Continuing the strategic plan: {planned_step}"}
            next_focus = f"Overall Goal: {query}\n\nExecute this planned analytical step: {planned_step}"
            continue

        action = evaluate_agent_state(query, memory_buffer, provider)
        if action.decision == "COMPLETE":
            yield {"status": "thought", "step": step_idx, "decision": "COMPLETE",
                   "reasoning": action.reasoning}
            break

        yield {"status": "thought", "step": step_idx, "decision": "CONTINUE",
               "reasoning": action.reasoning}
        next_step = action.next_step_focus or "Explore another relevant dimension of the goal."
        next_focus = f"Overall Goal: {query}\n\nNext analytical step: {next_step}"

    # ---------- 3. Synthesis over the full investigation ----------
    successful = [m for m in memory_buffer if m["status"] == "success" and m.get("raw_df") is not None]
    if not successful:
        yield {"status": "error", "content": "Investigation produced no successful analysis steps."}
        return

    primary_df = last_result_df if last_result_df is not None else successful[-1]["raw_df"]

    # 3a. Statistical validation of the primary (most recent) result
    stat_results = {}
    try:
        stats_plan = run_stats_agent(query, primary_df, provider, MODEL_PRO)
        if stats_plan.requested_tests:
            yield {"status": "running", "step": "Stats",
                   "content": f"🔬 Running {len(stats_plan.requested_tests)} deterministic statistical test(s)..."}
            stat_results = stats_engine.run_requested_tests(
                primary_df, [t.model_dump() for t in stats_plan.requested_tests]
            )
    except Exception as e:
        logger.warning(f"Stats step failed: {e}")

    # 3b. Visualization of the primary result
    yield {"status": "running", "step": "Viz", "content": "📊 Visualization Agent: Designing charts..."}
    rendered_charts = []
    try:
        viz_plan = run_viz_agent(query, primary_df, provider, MODEL_PRO)
        for spec in viz_plan.chart_specs:
            fig = build_plotly_chart(primary_df, spec, stat_results)
            if fig:
                rendered_charts.append({
                    "spec": spec, "df_json": primary_df.to_json(orient="records"), "fig": fig
                })
    except Exception as e:
        logger.warning(f"Viz step failed: {e}")

    # 3c. Insight synthesis across EVERY step of the investigation.
    # Pack the per-step journey alongside the stats so the Insight Agent reasons
    # over the whole investigation, not just the final table.
    combined_evidence = {
        "statistical_tests": stat_results,
        "investigation_steps": [
            {"step": m["step"], "focus": m["focus"], "data_preview": m.get("data_preview")}
            for m in successful
        ],
    }
    yield {"status": "running", "step": "Synthesis",
           "content": "🧠 Insight Agent: Synthesizing findings across the full investigation..."}
    try:
        insight_plan = run_insight_agent(query, primary_df, combined_evidence, history, provider, MODEL_PRO)
    except Exception as e:
        yield {"status": "error", "content": f"Insight Agent failed: {e}"}
        return

    # 3d. Strategic recommendations — grounded in the actual numbers, not just the
    # insight text, so the agent can quantify each recommendation.
    recommender_evidence = _build_recommender_evidence(primary_df, stat_results, successful)
    yield {"status": "running", "step": "Strategy",
           "content": "🚀 Recommendation Agent: Formulating business strategy..."}
    try:
        rec_plan = run_recommender_agent(
            query, insight_plan.direct_answer, insight_plan.insights,
            evidence=recommender_evidence, provider=provider, model_override=MODEL_PRO
        )
        recommendations = rec_plan.recommendations
    except Exception as e:
        logger.warning(f"Recommender step failed: {e}")
        recommendations = []

    # Golden Store: cache the final successful analysis if high-confidence
    if insight_plan.confidence_score >= 0.75:
        golden_store.save_golden_query(
            user_query=query,
            pandas_code=successful[-1]["code"],
            confidence_score=insight_plan.confidence_score
        )

    class MockInterpretation:
        pass
    mock_interp = MockInterpretation()
    mock_interp.direct_answer = insight_plan.direct_answer
    mock_interp.insights = insight_plan.insights
    mock_interp.recommendations = recommendations
    mock_interp.confidence_score = insight_plan.confidence_score
    mock_interp.statistical_backing = insight_plan.statistical_backing

    package = _build_final_package(
        query, mock_interp, memory_buffer, rendered_charts, primary_df, agent_type="complex"
    )
    # Persist the analyst plan so the stakeholder sees how it was approached.
    package["approach"] = analyst_approach
    package["assumptions"] = analyst_assumptions
    package["planned_steps"] = planned_steps
    yield {"status": "complete", "result": package}

def _run_predictive_path(
    query: str, context_profile: Dict[str, Any], provider: str,
    df: pd.DataFrame, all_datasets: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any], history: List[Dict[str, Any]]
):
    """
    PREDICTIVE Path: Specialized DAG for Machine Learning scoring.
    """
    logger.info("Executing PREDICTIVE path (ML Agent)")
    memory_buffer = []
    
    # 1. ML Agent (Code Generation)
    yield {"status": "running", "step": "ML Agent", "content": "🤖 ML Agent: Selecting features and defining predictive target..."}
    try:
        ml_plan = run_ml_agent(query, context_profile, history, provider, MODEL_PRO)
    except Exception as e:
        yield {"status": "error", "content": f"ML Agent failed: {e}"}
        return
        
    yield {"status": "thought", "step": 1, "decision": "PREDICT", "reasoning": ml_plan.thought_process}
    yield {"status": "code", "step": 1, "code": ml_plan.pandas_code}
    
    # Execute Code
    exec_result = execute_analysis(df, ml_plan.pandas_code, all_datasets=all_datasets)
    
    if not exec_result["success"]:
        yield {"status": "running", "step": "Self-Correction", "content": "ML Agent correcting code..."}
        correction_query = f"Original Query: {query}\n\nYour previous code failed with:\n{exec_result['error']}\nRewrite it."
        ml_plan = run_ml_agent(correction_query, context_profile, history, provider, MODEL_PRO)
        exec_result = execute_analysis(df, ml_plan.pandas_code, all_datasets=all_datasets)
        
    if not exec_result["success"]:
        yield {"status": "error", "content": f"Execution failed after retry: {exec_result['error']}"}
        return
        
    result_df = exec_result["result"]
    memory_buffer.append({
        "step": 1, "focus": "Predictive Modeling", "code": ml_plan.pandas_code,
        "status": "success", "raw_df": result_df
    })
    
    # 2. Insight Synthesis
    yield {"status": "running", "step": "Synthesis", "content": "4️⃣ Insight Agent: Synthesizing predictive results..."}
    try:
        insight_plan = run_insight_agent(query, result_df, None, history, provider, MODEL_PRO)
    except Exception as e:
        yield {"status": "error", "content": f"Insight Agent failed: {e}"}
        return
        
    # 3. Data-aware recommendations grounded in the scored cohort (replaces the
    # previously hardcoded recommendation strings).
    predictive_evidence = _build_predictive_evidence(result_df)
    yield {"status": "running", "step": "Strategy",
           "content": "🚀 Recommendation Agent: Formulating an activation strategy for the scored users..."}
    try:
        rec_plan = run_recommender_agent(
            query, insight_plan.direct_answer, insight_plan.insights,
            evidence=predictive_evidence, provider=provider, model_override=MODEL_PRO
        )
        recommendations = rec_plan.recommendations
    except Exception as e:
        logger.warning(f"Recommender step failed: {e}")
        recommendations = [
            "Operationalize this model by deploying the scored user list to the CRM.",
            "Design targeted campaigns for users in the highest-probability decile.",
        ]

    statistical_backing = ["Predictions powered by Random Forest feature importances."]
    if "key_factors" in result_df.columns and len(result_df):
        statistical_backing.append(str(result_df["key_factors"].iloc[0]))

    class MockInterpretation:
        pass
    mock_interp = MockInterpretation()
    mock_interp.direct_answer = insight_plan.direct_answer
    mock_interp.insights = insight_plan.insights
    mock_interp.recommendations = recommendations
    mock_interp.confidence_score = insight_plan.confidence_score
    mock_interp.statistical_backing = statistical_backing
    
    yield {"status": "complete", "result": _build_final_package(
        query, mock_interp, memory_buffer, [], result_df, agent_type="predictive"
    )}




def _build_charts(plan, interpretation, result_df, stat_results):
    """Build charts from the plan's visual spec and the interpreter's requested charts."""
    rendered_charts = []
    
    # Chart from the planner's visual spec
    if plan.visual_spec and plan.visual_spec.is_visual_requested:
        fig = build_plotly_chart(result_df, plan.visual_spec, stat_results=stat_results)
        if fig:
            rendered_charts.append({
                "spec": plan.visual_spec,
                "df_json": result_df.to_json(orient="records"),
                "fig": fig
            })
    
    # Additional charts from the interpreter
    if hasattr(interpretation, "requested_charts") and interpretation.requested_charts:
        for chart_spec in interpretation.requested_charts:
            fig = build_plotly_chart(result_df, chart_spec, stat_results=None)
            if fig:
                rendered_charts.append({
                    "spec": chart_spec,
                    "df_json": result_df.to_json(orient="records"),
                    "fig": fig
                })
    
    return rendered_charts


def _build_final_package(query, interpretation, memory_buffer, rendered_charts, result_df, agent_type="complex"):
    """Constructs the final output package for the UI."""
    return {
        "success": True,
        "query": query,
        "role": "assistant",
        "agent_type": agent_type,
        "content": interpretation.direct_answer,
        "direct_answer": interpretation.direct_answer,
        "insights": interpretation.insights,
        "recommendations": interpretation.recommendations,
        "confidence_score": interpretation.confidence_score,
        "statistical_backing": interpretation.statistical_backing,
        
        "memory_buffer": [
            {"step": m["step"], "focus": m["focus"], "status": m["status"], "code": m["code"]} 
            for m in memory_buffer
        ],
        
        "rendered_charts": [
            {
                "chart_spec": rc["spec"],
                "chart_df": rc["df_json"]
            }
            for rc in rendered_charts
        ],
        "has_chart": len(rendered_charts) > 0,
        "has_table": len(memory_buffer) > 0,
        "table_df": result_df.to_json(orient="records") if result_df is not None and not result_df.empty else None
    }
