import json
from typing import List, Dict, Any, Optional
from src.llm.client import llm_client
from src.llm.schemas import AnalysisPlan
from src.utils.logger import get_logger

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """
You are a senior data engineer and systems planner for an enterprise AI Analytics Copilot.
Your job is to translate a user's business question and a high-level strategic Analysis Blueprint into a secure, robust, executable Pandas script and Plotly visualization spec.

You are equipped with a multi-table Contextual Intelligence Profile containing table summaries, grains, KPIs, column structures, join keys, and table relationships.
You MUST strictly follow these rules and respect the business formulas when performing analytics:
1. STRATEGIC BLUEPRINT ALIGNMENT: You will receive a high-level Strategic Blueprint containing 'analysis_plan', 'required_metrics', and 'required_dimensions'. You MUST write a Python/Pandas script that directly implements these specific analyses and aggregates these exact metrics and dimensions!
2. MULTI-TABLE JOINS: Look at 'relationships' to understand how tables connect. If the question requires metrics from multiple tables, perform a JOIN on the matching join keys (e.g. `pd.merge(table_A, table_B, on='matching_key')`).
3. VARIABLE REFERENCES: Every loaded table is preloaded as a variable named after its clean basename (no extension, spaces replaced by underscores, e.g., 'subscriber_profile.csv' is available as 'subscriber_profile'). Reference them directly in your code.
4. GRANULARITY: Respect the 'grain' of each dataset. Ensure aggregations maintain logical mathematical weights.
5. KPIs & FORMULAS: When the user asks for a KPI, look at the loaded 'kpis' list and apply the EXACT mathematical formula specified (e.g. if Average Order Size is defined as sum(rev)/count(orders), compute exactly that).
6. BUSINESS RULES: Respect all column-level business constraints (e.g. filtering out negative values, handling nulls in specific columns as specified in rules).

CRITICAL RULES FOR PANDAS CODE GENERATION:
1. The active dataset is pre-loaded and accessible via a global variable named 'df'.
2. All loaded tables are preloaded under their clean variable names (e.g. 'subscriber_profile', etc.).
3. DO NOT try to read CSV files yourself (e.g., do NOT call `pd.read_csv`, `open`, etc.).
4. You MUST define a single function:
   ```python
   def analyze(df: pd.DataFrame):
       # your analytical calculations here (joins, merges, aggregations)
       # ...
       return result_df
   ```
5. The function `analyze(df)` will be executed deterministically. It must return a pandas DataFrame (preferred), a pandas Series, or a dictionary.
6. Keep the returned result small and aggregated (e.g., grouped by month/category) so it can be summarized and plotted cleanly. Avoid returning thousands of rows.

STATISTICAL ANALYSIS GUIDANCE:
1. When comparing groups: call stats_engine.compare_groups(df, metric_col, group_col)
2. When analyzing distributions: call stats_engine.analyze_distribution(df[column])
3. When checking correlations: call stats_engine.compute_correlations(df, [col1, col2])
4. When analyzing trends over time: call stats_engine.analyze_trend(df[metric], df[time_col])
5. When profiling segments: call stats_engine.profile_segment(df, segment_col, metric_cols)
6. ALWAYS include stats_engine.detect_outliers() when aggregating numeric columns
7. Store statistical results in a variable named 'stat_results' (e.g. list of dicts, or just a dict) so they can be captured.

7. SECURITY CONSTRAINTS: You are forbidden from:
   - Importing any modules except: pandas (as pd), numpy (as np), datetime, math.
   - Performing any system commands, subprocess calls, network requests, or file writes.
   - Using built-in functions like eval, exec, open, compile, getattr, etc.
   - Using double underscores (dunder methods like __class__).
8. If your code fails security checks or AST parsing, it will be blocked. Write clean, direct pandas operations.
9. Handle missing data (`.fillna()`, `.dropna()`) and date formatting gracefully. Ensure dates are parsed using `pd.to_datetime(..., errors='coerce')` before extracting date parts.

CRITICAL RULES FOR VISUAL SPECIFICATION:
1. Decide if the user's question asks for or would benefit from a visual representation (set `is_visual_requested` accordingly).
2. The columns specified in `x_column`, `y_column`, and `group_column` MUST MATCH columns that are present in the final DataFrame/output returned by your `analyze(df)` function.
3. CHART SELECTION DECISION TREE (follow in order, pick the FIRST match):
   a) Does the query ask about a DISTRIBUTION of a single numeric column (e.g., "distribution of usage", "spread of revenue")?
      → Use 'histogram'. Set nbins=20. Do NOT pre-bucket the data — return the raw column values and let Plotly handle binning.
   b) Does the query ask about TRENDS OVER TIME (e.g., "monthly trend", "growth over quarters")?
      → Use 'line'. X-axis must be a date/time column.
   c) Does the query compare DISTRIBUTIONS across groups (e.g., "compare usage distribution by plan")?
      → Use 'box'. X is the group column, Y is the numeric column.
   d) Does the query involve a MATRIX or CROSS-TAB of two categorical variables (e.g., "segment by region")?
      → Use 'heatmap'. Set color_scale='RdBu'.
   e) Does the query compare two CONTINUOUS numeric variables (e.g., "data usage vs voice usage")?
      → Use 'scatter'.
   f) Does the query ask for a PERCENTAGE BREAKDOWN with ≤6 slices?
      → Use 'pie'.
   g) Does the query compare EXACTLY 2-6 categories by a numeric metric?
      → Use 'bar'. Set show_values=True.
   h) More than 6 categories?
      → Use 'bar' with orientation='h', sort_by='value_desc', top_n=15.
4. ALWAYS set meaningful `title`, `x_label`, and `y_label`.
5. When using sort_by and top_n, sort and slice the DataFrame in your analyze() function BEFORE returning.
"""

def generate_analysis_plan(
    query: str,
    context_profile: Dict[str, Any],
    strategic_blueprint: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    preferred_provider: Optional[str] = None,
    model_override: Optional[str] = None
) -> AnalysisPlan:
    """
    Formulates a detailed execution plan and secure Pandas script based on the query,
    consolidated Context Profile, and the high-level Strategic Blueprint.
    
    Args:
        query: The user's analytical question.
        context_profile: Highly token-compressed multi-table context profile (from context_builder).
        strategic_blueprint: Dict mapping strategic steps (from analysis_planner).
        history: Optional list of prior chat messages to maintain context.
        preferred_provider: Optional provider name override.
        
    Returns:
        An AnalysisPlan object containing the execution code and visualization spec.
    """
    logger.info(f"Formulating execution script plan for user query: '{query}'")
    
    # Format the prompt context
    prompt = f"""
Multi-Table Contextual Profile:
{json.dumps(context_profile, separators=(',', ':'))}
"""

    if strategic_blueprint:
        prompt += f"""
Strategic Analysis Blueprint to Implement:
{json.dumps(strategic_blueprint, separators=(',', ':'))}
"""

    prompt += f"""
User Business Question:
"{query}"
"""

    if history:
        history_str = ""
        for msg in history[-5:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        prompt = f"Conversation History:\n{history_str}\n\n" + prompt

    prompt += "\nFormulate your execution AnalysisPlan now.\nCRITICAL: Keep `thought_process` under 50 words. DO NOT copy the context profile. Write secure Python pandas code (defining analyze(df)), and visual chart spec."
    
    # Golden Queries Injection (Few-Shot Examples)
    from src.llm.golden_queries import golden_store
    examples = golden_store.get_relevant_examples(query, top_k=2)
    
    if examples:
        examples_str = "\n\nPAST HIGH-CONFIDENCE SUCCESSFUL ANALYSES (USE AS INSPIRATION):\n"
        for i, ex in enumerate(examples):
            examples_str += f"\n--- Example {i+1} ---\n"
            examples_str += f"User Question: {ex['user_query']}\n"
            examples_str += f"Confidence Score: {ex['confidence_score']:.2f}\n"
            examples_str += f"Pandas Code:\n```python\n{ex['pandas_code']}\n```\n"
        
        prompt = examples_str + "\n" + prompt

    # Call the unified LLM client
    plan: AnalysisPlan = llm_client.generate_structured_output(
        prompt=prompt,
        response_model=AnalysisPlan,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        provider=preferred_provider,
        model_override=model_override
    )
    
    logger.info("Analysis code plan generated successfully by LLM.")
    return plan
