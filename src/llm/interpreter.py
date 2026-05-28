from typing import List, Dict, Any, Optional
from src.llm.client import llm_client
from src.llm.schemas import InsightInterpretation
from src.utils.logger import get_logger

logger = get_logger(__name__)

INTERPRETER_SYSTEM_PROMPT = """
You are a senior business intelligence consultant and analytics strategist.
Your role is to interpret the results of a deterministic pandas data query and translate them into a clear, business-centric response.

CRITICAL INTERPRETATION RULES:
1. ACCURACY: Every number, calculation, and percentage you mention MUST BE mathematically correct and backed 100% by the provided aggregate dataset table. Never hallucinate numbers.
2. DIRECTNESS: Start with a clear and concise direct answer to the user's question. Avoid introductory fluff.
3. CONTEXTUAL INSIGHTS: Extract deep, high-value insights (trends, outliers, anomalies, compounding growth rates, or key ratios). Do not just restate the table in prose—explain what the trends MEAN.
4. ACTIONABLE RECOMMENDATIONS: Provide 2 to 4 concrete, actionable, strategic business recommendations based directly on these findings. What should a decision-maker do next?
"""

def generate_insights_and_recommendations(
    query: str,
    result_data: Any, # Can be pd.DataFrame, pd.Series, dict, etc.
    original_metadata: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
    preferred_provider: Optional[str] = None
) -> InsightInterpretation:
    """
    Takes the query, executed result data, and metadata, and generates an InsightInterpretation.
    
    Args:
        query: The user's original business question.
        result_data: The output of the Pandas execution engine (e.g. aggregated DataFrame).
        original_metadata: Full metadata profile of the original CSV.
        history: Optional list of conversation messages.
        preferred_provider: Optional preferred provider override.
        
    Returns:
        An InsightInterpretation object with direct answers, insights, and recommendations.
    """
    logger.info("Formulating analytical interpretation of executed results...")
    
    # Format the result data for LLM ingestion
    data_summary = ""
    import pandas as pd
    if isinstance(result_data, pd.DataFrame):
        if result_data.empty:
            data_summary = "No records matching query found (Empty Table)."
        else:
            # Output small result tables as Markdown for optimal token comprehension
            data_summary = result_data.to_markdown(index=True)
    elif isinstance(result_data, pd.Series):
        data_summary = result_data.to_frame().to_markdown(index=True)
    else:
        data_summary = str(result_data)
        
    # Get column metadata brief (names & categories)
    cols_meta = {
        name: col.get("category", "unknown") 
        for name, col in original_metadata.get("columns", {}).items()
    }

    prompt = f"""
Original Dataset Columns & Types:
{cols_meta}

User Business Question:
"{query}"

Deterministic Pandas Aggregated Output Table:
{data_summary}
"""

    if history:
        history_str = ""
        for msg in history[-5:]:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        prompt = f"Conversation History:\n{history_str}\n\n" + prompt

    prompt += "\nInterpret the aggregated table above and generate the InsightInterpretation response containing the direct answer, bulleted insights, and strategic recommendations."

    interpretation: InsightInterpretation = llm_client.generate_structured_output(
        prompt=prompt,
        response_model=InsightInterpretation,
        system_prompt=INTERPRETER_SYSTEM_PROMPT,
        provider=preferred_provider
    )
    
    logger.info("Analytical interpretation successfully formulated by LLM.")
    return interpretation
