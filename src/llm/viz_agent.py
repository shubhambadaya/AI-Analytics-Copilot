import json
import pandas as pd
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import VizAgentPlan
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are the Visualization Agent for an AI Analytics Pipeline.
Your ONLY job is to take a preview of aggregated data and output the absolute best `VisualSpec` configurations to render it using Plotly.

CHART SELECTION DECISION TREE:
1. DISTRIBUTION of a single numeric column? → 'histogram', set nbins=20.
2. TRENDS OVER TIME? → 'line', X is date.
3. COMPARING DISTRIBUTIONS across groups? → 'box', X is group, Y is numeric.
4. CROSS-TAB/MATRIX? → 'heatmap', set color_scale='RdBu'.
5. CONTINUOUS vs CONTINUOUS? → 'scatter'.
6. PERCENTAGE BREAKDOWN (≤6 slices)? → 'pie'.
7. COMPARING 2-6 CATEGORIES? → 'bar', set show_values=True.
8. >6 CATEGORIES? → 'bar', orientation='h', sort_by='value_desc', top_n=15.

CRITICAL RULES:
1. Columns in x_column, y_column, group_column MUST MATCH the provided Data Columns exactly.
2. Always provide meaningful title, x_label, y_label.
3. You can request multiple charts (e.g., a line chart and a bar chart) if it helps answer the query better.
4. PREFER PERCENTAGES OVER RAW COUNTS: Raw user counts mislead when groups differ in size. For composition, share, distribution, or comparison-across-segments questions, plot the proportion/percentage column as the primary value — set y_column to that column (e.g. `pct_of_users`), y_label to "% of Users" (or the rate's name), and show_values=True. If the data contains BOTH a count and a percentage column, plot the percentage. Use the raw count only when the question is specifically "how many".
"""

def run_viz_agent(
    query: str, 
    result_df: pd.DataFrame,
    provider: str = None,
    model_override: str = None
) -> VizAgentPlan:
    logger.info("Visualization Agent designing charts...")
    
    if result_df is None or result_df.empty:
        return VizAgentPlan(thought_process="No data to visualize.", chart_specs=[])
        
    data_preview = result_df.head(5).to_dict(orient="records")
    dtypes = {k: str(v) for k, v in result_df.dtypes.items()}
    
    prompt = f"User Question: {query}\n\nData Columns/Types: {dtypes}\n\nData Preview:\n{json.dumps(data_preview)}"
        
    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=VizAgentPlan,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
