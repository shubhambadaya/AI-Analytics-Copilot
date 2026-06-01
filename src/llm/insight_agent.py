import json
import pandas as pd
from typing import Dict, Any, List
from src.llm.client import llm_client
from src.llm.schemas import InsightAgentPlan
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are the Insight Synthesis Agent for an AI Analytics Pipeline.
Your ONLY job is to write a direct answer and bulleted business insights based on the provided dataset and statistical results.

CRITICAL RULES:
1. ACCURACY: Every number must be mathematically correct and backed by the provided data.
2. DIRECTNESS: Start with a clear direct answer. No fluff.
3. SYNTHESIS: Explain what the trends and statistics MEAN. Do not just restate the table in prose.
4. DO NOT write recommendations. That is another agent's job.
"""

def run_insight_agent(
    query: str, 
    result_df: pd.DataFrame,
    stat_results: Dict[str, Any],
    history: List[Dict[str, Any]] = None,
    provider: str = None,
    model_override: str = None
) -> InsightAgentPlan:
    logger.info("Insight Agent synthesizing findings...")
    
    data_markdown = result_df.to_markdown() if result_df is not None and not result_df.empty else "No matching data found."
    
    prompt = f"User Question: {query}\n\nResult Data:\n{data_markdown}\n\nStatistical Evidence:\n{json.dumps(stat_results)}\n"
    
    if history:
        hist_str = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in history[-3:]])
        prompt = f"Conversation History:\n{hist_str}\n\n" + prompt
        
    return llm_client.generate_structured_output(
        prompt=prompt,
        response_model=InsightAgentPlan,
        system_prompt=SYSTEM_PROMPT,
        provider=provider,
        model_override=model_override
    )
