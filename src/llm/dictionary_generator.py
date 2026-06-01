import json
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from src.llm.client import llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Define a List-based schema to avoid Gemini's additionalProperties dictionary errors
class KPIDefinition(BaseModel):
    name: str = Field(..., description="Strategic name of the KPI.")
    formula: str = Field(..., description="Math description or pandas code calculation formula.")
    description: str = Field(..., description="Business explanation of what this KPI measures.")

class DraftColumnDefinition(BaseModel):
    name: str = Field(..., description="Exact name of the column.")
    description: str = Field(..., description="The semantic business description of the column.")
    is_join_key: bool = Field(False, description="True if this is a primary/foreign key.")
    business_rules: List[str] = Field(default_factory=list, description="Logical constraints (e.g. 'Must be > 0').")

class DraftDictionary(BaseModel):
    grain: str = Field(..., description="Granularity level of the dataset.")
    columns: List[DraftColumnDefinition] = Field(default_factory=list, description="List of all semantic column definitions.")
    kpis: List[KPIDefinition] = Field(default_factory=list, description="Calculated business KPIs.")

SYSTEM_PROMPT = """
You are a Senior Data Architect and Business Intelligence Strategist.
Your task is to reverse-engineer a Business Data Dictionary from a raw database table profile.

You will be given a JSON summary containing column names, data types, and sample values.
You must infer the business semantics of this dataset and output a structured DraftDictionary containing:
1. `grain`: The row-level granularity of the dataset.
2. `columns`: A list of dictionaries, one for each column, describing its meaning and rules.
3. `kpis`: 2-4 strategic Key Performance Indicators that can be calculated strictly from the available columns. Provide the specific math formula using exact column names.

Do NOT invent columns that do not exist in the provided profile.
Focus on standard enterprise logic (e.g., if there's an 'amount' and 'status', formulate a 'Gross Revenue' KPI).
"""

def auto_generate_dictionary(
    raw_metadata: Dict[str, Any],
    provider: Optional[str] = None
) -> str:
    """
    Passes the raw dataset profile to the LLM to auto-infer a draft Business Dictionary.
    Returns the generated dictionary as a formatted JSON string for user editing.
    """
    logger.info("Auto-generating draft business dictionary via LLM...")
    
    # Compress metadata to save tokens
    compressed_profile = {
        "dataset_name": "Uploaded Dataset",
        "total_rows": raw_metadata.get("rows"),
        "columns": {}
    }
    
    for col_name, col_data in raw_metadata.get("columns", {}).items():
        compressed_profile["columns"][col_name] = {
            "type": col_data.get("inferred_type", col_data.get("type")),
            "samples": col_data.get("samples", [])[:3]
        }
        
    prompt = f"Raw Dataset Profile:\n{json.dumps(compressed_profile, indent=2)}\n\nGenerate the comprehensive BusinessDataDictionary."
    
    try:
        draft_dict: DraftDictionary = llm_client.generate_structured_output(
            prompt=prompt,
            response_model=DraftDictionary,
            system_prompt=SYSTEM_PROMPT,
            provider=provider
        )
        logger.info("Draft dictionary generated successfully.")
        
        # Return as pretty JSON string so user can edit it in a Text Area
        return draft_dict.model_dump_json(indent=4)
        
    except Exception as e:
        logger.error(f"Failed to generate dictionary: {e}")
        raise RuntimeError(f"LLM failed to generate draft dictionary: {e}")
