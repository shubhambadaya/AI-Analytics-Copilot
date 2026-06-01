import json
from typing import List, Optional, Any
from pydantic import BaseModel, Field
from src.llm.client import llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RouteDecision(BaseModel):
    is_dashboard: bool = Field(..., description="True if the user is asking a broad question that requires multiple charts to answer properly.")
    sub_queries: List[str] = Field(default_factory=list, description="If is_dashboard is True, provide 2-4 highly specific analytical sub-queries. If False, leave empty.")

SYSTEM_PROMPT = """
You are an intelligent Query Router for an Enterprise AI Analytics Copilot.
Your job is to determine if a user's question is a "Single Query" or a "Dashboard Request".

RULES:
1. Single Query (is_dashboard = False): The user is asking a specific, focused question (e.g., "What is the average churn rate?", "Show me the distribution of age", "Compare male vs female revenue").
2. Dashboard Request (is_dashboard = True): The user is asking a broad, exploratory, or profiling question that is best answered with multiple distinct charts (e.g., "Profile my churned users", "Give me a dashboard of sales performance", "Analyze our customer base").

If it is a Dashboard Request, you must break it down into 2-4 highly specific, independent sub-queries. 
Each sub-query must be standalone and executable as a single chart (e.g., ["What is the age distribution?", "What is the gender breakdown?", "How does tenure affect churn?"]).
Do NOT ask for generic summaries. Ask for specific metrics and dimensions.
"""

def route_query(user_query: str, context_profile: dict, provider: Optional[str] = None) -> RouteDecision:
    """
    Evaluates the user query against the context profile to determine if it should fan-out into a multi-chart dashboard.
    """
    logger.info(f"Routing query: '{user_query}'")
    
    # Send a lightweight version of the context just so it knows what columns exist for generating sub-queries
    lightweight_context = {
        "columns": list(context_profile.get("columns", {}).keys()),
        "kpis": [k.get("name") for k in context_profile.get("kpis", [])]
    }
    
    prompt = f"Available Context:\n{json.dumps(lightweight_context)}\n\nUser Question:\n{user_query}\n\nDetermine the routing decision."
    
    try:
        decision: RouteDecision = llm_client.generate_structured_output(
            prompt=prompt,
            response_model=RouteDecision,
            system_prompt=SYSTEM_PROMPT,
            provider=provider
        )
        
        if decision.is_dashboard:
            logger.info(f"Query routed as DASHBOARD with {len(decision.sub_queries)} sub-queries.")
        else:
            logger.info("Query routed as SINGLE.")
            
        return decision
        
    except Exception as e:
        logger.error(f"Routing failed, defaulting to single query. Error: {e}")
        # Failsafe: return a single query route
        return RouteDecision(is_dashboard=False, sub_queries=[])
