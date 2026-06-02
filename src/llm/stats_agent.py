import json
import pandas as pd
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import StatsAgentPlan
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are the Statistical Agent for an AI Analytics Pipeline.
Your ONLY job is to look at a preview of an aggregated dataset and decide which deterministic statistical tests to run, and name the EXACT columns each test should use.

Available tests and the columns they need:
- `analyze_distribution`: normality, mean, median, skew of one column. Set `metric_column` (numeric).
- `detect_outliers`: anomalies via IQR. Set `metric_column` (numeric).
- `compare_groups`: t-test (2 groups) or ANOVA (3+ groups). Set `metric_column` (numeric) and `group_column` (categorical).
- `compare_proportions`: two-proportion z-test. (Requires explicit success/total counts — only request if the data clearly contains them.)
- `compute_correlations`: Pearson/Spearman matrix. Optionally set `columns` to 2+ numeric columns; omit to use all numeric columns.
- `analyze_trend`: linear regression over time. Set `metric_column` (numeric) and optionally `time_column`.
- `detect_changepoints`: regime shifts in a time series. Set `metric_column` (numeric).
- `value_counts`: frequency breakdown (counts + percentages) of a column; ideal for categorical univariate analysis ("target column + count"). Set `metric_column` to the target column (categorical or numeric).

CRITICAL RULES:
1. ONLY select tests that make sense for the data preview provided.
2. If the data is just a count or simple lookup, return an empty list `[]`.
3. Column names you provide MUST exactly match the column names shown in the data. Use numeric columns where a numeric metric is required and categorical columns for groupings.
4. Do NOT hallucinate stats or column names.
"""

def run_stats_agent(
    query: str, 
    result_df: pd.DataFrame,
    provider: str = None,
    model_override: str = None
) -> StatsAgentPlan:
    logger.info("Statistical Agent evaluating data...")
    
    if result_df is None or result_df.empty:
        return StatsAgentPlan(thought_process="No data to analyze.", requested_tests=[])
        
    data_preview = result_df.head(25).to_dict(orient="records")
    dtypes = {k: str(v) for k, v in result_df.dtypes.items()}

    prompt = (f"User Question: {query}\n\nData Columns/Types: {dtypes}\n\n"
              f"Data Preview (up to 25 of {len(result_df)} rows):\n{json.dumps(data_preview, default=str)}")
        
    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=StatsAgentPlan,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
