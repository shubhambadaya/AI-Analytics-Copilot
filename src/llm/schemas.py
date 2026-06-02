from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal


class ComplexityClassification(BaseModel):
    """Classifies a user query to route it to the right agent (or capture a taught rule)."""
    complexity: Literal["SIMPLE", "COMPLEX", "PREDICTIVE", "LEARN_RULE"] = Field(
        ...,
        description="SIMPLE: single-step queries. COMPLEX: multi-step queries. PREDICTIVE: machine learning and user scoring. LEARN_RULE: the user is teaching a business rule/definition/preference to remember rather than asking a question."
    )
    reasoning: str = Field(
        ...,
        description="1 sentence explaining the classification."
    )
    learned_rule: Optional[str] = Field(
        None,
        description="If complexity is LEARN_RULE, the exact concise rule/definition/preference to memorize; otherwise null."
    )
    wants_recommendations: bool = Field(
        False,
        description="True ONLY if the user explicitly asks for recommendations, actions, advice, strategy, or how to improve/fix/increase/reduce/grow something. False for purely descriptive or analytical questions (what / how many / show / list / compare / which / why)."
    )

    @field_validator("wants_recommendations", mode="before")
    @classmethod
    def _coerce_recs_flag(cls, v):
        return bool(v) if v is not None else False

class ClarificationCheck(BaseModel):
    """Scopes a complex query before analysis: decides if a clarifying question is genuinely needed, and states the analyst's plan and assumptions."""
    needs_clarification: bool = Field(
        ...,
        description="True ONLY if the question hinges on a metric/term/segment that is NOT a column, NOT a defined KPI, and NOT a learned rule — such that guessing its meaning would materially change the answer. Otherwise False."
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="1-2 concise, specific questions for the stakeholder. Only populated when needs_clarification is True."
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Key assumptions being made to answer the question (e.g. 'Using `Price` as the revenue measure'). Always provide these."
    )
    approach: str = Field(
        ...,
        description="One or two sentences, in plain analyst language, describing how you will approach the question."
    )

    @field_validator("clarifying_questions", "assumptions", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v):
        # The Gemini path is unconstrained JSON, so optional list fields can come
        # back as null instead of []; normalize so validation doesn't fail.
        return v or []


class CritiqueResult(BaseModel):
    """Reviews a synthesized answer against the evidence before it is finalized."""
    passes: bool = Field(..., description="True if the answer fully addresses the question and every claim is supported by the provided data/statistics.")
    issues: List[str] = Field(default_factory=list, description="Specific problems found: unsupported claims, parts of the question left unanswered, or contradictions. Empty if it passes.")
    caveats: List[str] = Field(default_factory=list, description="Honest limitations a stakeholder should know (e.g. small sample, proxy metric, correlation is not causation, missing segments).")
    refined_answer: str = Field(..., description="The corrected/tightened direct answer — keep ONLY what the evidence supports; qualify or remove overreaching claims; do not invent new numbers. If the original was sound, return it unchanged.")
    confidence: float = Field(..., description="Grounded confidence 0.0-1.0 in the refined answer given the evidence (high only when claims rest on significant, sufficient data).")

    @field_validator("issues", "caveats", mode="before")
    @classmethod
    def _coerce_none_to_list(cls, v):
        return v or []


class VisualSpec(BaseModel):
    """
    Structured specification for rendering Plotly charts.
    Helps separate analytical computations from visualization definitions.
    """
    is_visual_requested: bool = Field(
        ..., 
        description="True if the user's query requests or would benefit from a visual representation (chart/graph)."
    )
    chart_type: Optional[Literal["bar", "line", "scatter", "pie", "histogram", "box", "heatmap"]] = Field(
        None, 
        description="The type of chart to generate."
    )
    x_column: Optional[str] = Field(
        None, 
        description="The name of the column to plot on the X-axis."
    )
    y_column: Optional[str] = Field(
        None, 
        description="The name of the column to plot on the Y-axis."
    )
    group_column: Optional[str] = Field(
        None, 
        description="Optional column to group or color by (for multiple series)."
    )
    title: Optional[str] = Field(
        None, 
        description="The title of the chart."
    )
    x_label: Optional[str] = Field(
        None, 
        description="Custom label for the X-axis."
    )
    y_label: Optional[str] = Field(
        None, 
        description="Custom label for the Y-axis."
    )
    barmode: Optional[Literal["group", "stack"]] = Field(
        "group", 
        description="Bar mode layout if chart_type is bar."
    )
    orientation: Optional[Literal["v", "h"]] = Field(
        "v", 
        description="Orientation of bars if chart_type is bar ('v' for vertical, 'h' for horizontal)."
    )
    nbins: Optional[int] = Field(
        None,
        description="Number of bins for histogram charts. Use 15-30 for good granularity."
    )
    color_scale: Optional[str] = Field(
        None,
        description="Color scale for heatmaps. Options: 'RdBu', 'Viridis', 'Plasma', 'YlOrRd', 'Blues'."
    )
    show_values: Optional[bool] = Field(
        False,
        description="If True, annotate bars/points with their numeric values."
    )
    reference_line: Optional[float] = Field(
        None,
        description="Draw a horizontal reference line at this value (e.g. the average)."
    )
    sort_by: Optional[Literal["value_asc", "value_desc", "label"]] = Field(
        None,
        description="Sort the chart data. 'value_desc' shows highest first."
    )
    top_n: Optional[int] = Field(
        None,
        description="Only show the top N items in the chart. Use with sort_by='value_desc'."
    )

class StatisticalResult(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    effect_size_label: Optional[str]
    is_significant: bool
    interpretation: str


class AnalysisPlan(BaseModel):
    """
    Structured plan generated by the LLM containing the logical steps, 
    the secure Pandas execution script, and visual rendering details.
    """
    thought_process: str = Field(
        ..., 
        description="Concise logical reasoning (Max 2 sentences). DO NOT ramble."
    )
    pandas_code: str = Field(
        ..., 
        description=(
            "Valid python script doing the pandas data transformations. "
            "It must load 'df' (already pre-loaded in the global namespace), "
            "perform aggregations/filters, and assign the final aggregated dataframe to a variable named 'result_df'. "
            "Keep the result_df small and aggregated. DO NOT import forbidden packages."
        )
    )
    visual_spec: VisualSpec = Field(
        ..., 
        description="Visual chart specification corresponding to the aggregated results."
    )
    statistical_tests: Optional[List[str]] = None
    confidence_target: Optional[float] = None

class InsightInterpretation(BaseModel):
    """
    Structured response generated by the LLM summarizing the findings,
    extracting business insights, and making domain-level recommendations.
    """
    direct_answer: str = Field(
        ..., 
        description="A direct, concise, and mathematically accurate answer to the user's business question."
    )
    insights: List[str] = Field(
        ..., 
        description="Bullet-point list of high-value business insights, trends, outliers, or patterns found in the data."
    )
    recommendations: List[str] = Field(
        ..., 
        description="Bullet-point list of actionable strategic recommendations based on the findings."
    )
    requested_charts: Optional[List[VisualSpec]] = Field(
        None,
        description="If the insights or answer describe specific data distributions or trends, define them here so the UI can render them alongside your answer."
    )
    confidence_score: float = Field(
        ...,
        description="Confidence score between 0.0 and 1.0."
    )
    statistical_backing: Optional[List[str]] = None

# ==============================================================================
# 5-AGENT DAG SCHEMAS (Decoupled Specialization)
# ==============================================================================

class SchemaAgentPlan(BaseModel):
    """Output for the specialized Schema/Pandas Agent."""
    thought_process: str = Field(..., description="Logical reasoning for data transformations (Max 2 sentences).")
    pandas_code: str = Field(
        ..., 
        description="Valid Python script to transform df into result_df. Keep result_df small and aggregated."
    )

class StatTestRequest(BaseModel):
    """A single statistical test paired with the exact columns it should run against."""
    test: Literal[
        "analyze_distribution", "detect_outliers", "compare_groups",
        "compare_proportions", "compute_correlations", "analyze_trend",
        "detect_changepoints", "value_counts"
    ] = Field(..., description="The deterministic stats_engine function to run.")
    metric_column: Optional[str] = Field(
        None,
        description="Numeric column to analyze. Required for analyze_distribution, detect_outliers, analyze_trend, detect_changepoints, and used as the metric for compare_groups. For value_counts it is the target column to count (categorical or numeric). Must be an exact column name from the data."
    )
    group_column: Optional[str] = Field(
        None,
        description="Categorical column whose values define the groups to compare. Required for compare_groups. Must be an exact column name from the data."
    )
    columns: Optional[List[str]] = Field(
        None,
        description="Two or more numeric columns to correlate. Used only by compute_correlations; if omitted, all numeric columns are used."
    )
    time_column: Optional[str] = Field(
        None,
        description="Optional datetime or ordinal column to use as the time axis for analyze_trend."
    )

class StatsAgentPlan(BaseModel):
    """Output for the specialized Statistical Agent."""
    thought_process: str = Field(..., description="Reasoning for which statistical tests to run based on the data preview.")
    requested_tests: List[StatTestRequest] = Field(
        default_factory=list,
        description="List of statistical tests to execute deterministically, each naming the exact target columns it should run against."
    )

class VizAgentPlan(BaseModel):
    """Output for the specialized Visualization Agent."""
    thought_process: str = Field(..., description="Reasoning for the optimal chart selection based on data dimensions.")
    chart_specs: List[VisualSpec] = Field(default_factory=list, description="List of VisualSpecs to render.")

class InsightAgentPlan(BaseModel):
    """Output for the specialized Insight Agent."""
    direct_answer: str = Field(..., description="Clear, mathematically accurate answer to the user's question.")
    insights: List[str] = Field(..., description="Bullet points of deep insights, trends, and anomalies.")
    confidence_score: float = Field(..., description="Confidence in the findings.")
    statistical_backing: Optional[List[str]] = None

class RecommenderAgentPlan(BaseModel):
    """Output for the specialized Recommendation Agent."""
    recommendations: List[str] = Field(..., description="Actionable, forward-looking business recommendations based on the insights.")

class MLAgentPlan(BaseModel):
    """Output for the specialized ML Predictive Agent."""
    thought_process: str = Field(..., description="Reasoning for ML model selection (Max 2 sentences).")
    pandas_code: str = Field(..., description="Valid Python code using ml_engine.train_and_predict to generate scored results.")
