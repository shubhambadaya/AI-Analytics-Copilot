import json
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
from src.llm.client import llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AgentAction(BaseModel):
    decision: Literal["CONTINUE", "COMPLETE", "LEARN_RULE"] = Field(
        ...,
        description="Choose 'CONTINUE' if more code execution is needed. Choose 'COMPLETE' if the memory has the answer. Choose 'LEARN_RULE' if the user is explicitly telling you a business rule, definition, or preference to remember for the future."
    )
    reasoning: str = Field(
        ...,
        description="Logical explanation for the decision. If CONTINUE, explain what is missing. If COMPLETE, explain how the current data answers the question."
    )
    next_step_focus: Optional[str] = Field(
        None,
        description="If CONTINUE, provide a specific, actionable instruction for the Code Generator on what Pandas analysis to run next."
    )

EVALUATOR_SYSTEM_PROMPT = """
You are the central ReAct Agent Evaluator for an Enterprise AI Analytics Copilot.
Your job is to read the user's business question and evaluate the current 'Agent Memory' (which contains the results of previously executed Pandas scripts).

You must decide your next action:
1. If the memory is insufficient or empty to answer a data query, output 'CONTINUE' and instruct the next coding step.
2. If the memory contains the final answer to the data query, output 'COMPLETE'.
3. If the user is NOT asking a data query, but is instead explicitly teaching you a business rule, definition, or visual preference (e.g., "Active users are >5GB", "Always use histograms for distribution"), output 'LEARN_RULE'. 
   - If 'LEARN_RULE', you MUST put the exact, concise rule to be memorized inside the `reasoning` field.
"""

EVALUATOR_PROMPT_TEMPLATE = """
User Query: "{query}"

Agent Memory Buffer (Previous Code Executions):
{memory_json}

Evaluate the memory buffer. Do we have all the data required to answer the query?
"""

def evaluate_agent_state(
    query: str,
    memory_buffer: List[Dict[str, Any]],
    preferred_provider: Optional[str] = None
) -> AgentAction:
    """
    Evaluates the current state of the ReAct loop and decides whether to continue or complete.
    """
    logger.info(f"Agent Evaluator assessing state (Memory Size: {len(memory_buffer)} steps)...")
    
    # Compress memory buffer for context
    compressed_memory = []
    for step in memory_buffer:
        # We only send the summaries/stats to the LLM, not massive raw dataframes
        step_summary = {
            "step_index": step["step"],
            "goal": step["focus"],
            "code_executed": step["code"],
            "status": step["status"],
            "statistical_results": step.get("stat_results"),
            "data_summary": step.get("data_preview") # A small json preview of the dataframe
        }
        if "error" in step:
            step_summary["error"] = step["error"]
            
        compressed_memory.append(step_summary)
        
    prompt = EVALUATOR_PROMPT_TEMPLATE.format(
        query=query,
        memory_json=json.dumps(compressed_memory, indent=2)
    )
    
    try:
        action: AgentAction = llm_client.generate_structured_output(
            prompt=prompt,
            response_model=AgentAction,
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            provider=preferred_provider
        )
        logger.info(f"Agent Decision: {action.decision} | Reasoning: {action.reasoning}")
        return action
    except Exception as e:
        logger.error(f"Failed to evaluate agent state: {e}")
        # Default to COMPLETE to avoid infinite loops if API fails
        return AgentAction(decision="COMPLETE", reasoning="Failed to evaluate state, defaulting to complete.")
