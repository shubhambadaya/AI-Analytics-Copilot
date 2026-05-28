import json
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Dict, List, Optional, Any, Union
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ==================== VALIDATION SCHEMA LAYER ====================

class ColumnDefinition(BaseModel):
    """Business metadata for a single column."""
    description: str = Field(..., description="The semantic business description of the column.")
    is_join_key: bool = Field(False, description="Indicates if this column is a primary/foreign key used for joining.")
    business_rules: List[str] = Field(default_factory=list, description="Regulatory, business, or integrity constraints.")

class KPIDefinition(BaseModel):
    """Definition of a strategic Key Performance Indicator (KPI)."""
    name: str = Field(..., description="Strategic name of the KPI.")
    formula: str = Field(..., description="Deterministic math description or code calculation formula.")
    description: str = Field(..., description="Business explanation of what this KPI measures.")

class BusinessDataDictionary(BaseModel):
    """The complete, validated Business Data Dictionary object."""
    grain: Optional[str] = Field(None, description="Granularity level of the dataset (e.g. 'One row per transaction').")
    columns: Dict[str, ColumnDefinition] = Field(default_factory=dict, description="Semantic configurations of column metrics.")
    kpis: List[KPIDefinition] = Field(default_factory=list, description="Calculated business KPIs based on this dataset.")

    @model_validator(mode="before")
    @classmethod
    def normalize_columns_and_fields(cls, data: Any) -> Any:
        """
        Normalizes list-based columns format (common in enterprise dictionaries)
        to the expected dictionary format, and maps alternative description tags.
        """
        if not isinstance(data, dict):
            return data
            
        # Extract primary key list for automatic join key inference
        primary_keys = set(data.get("primary_key", []))
        
        raw_cols = data.get("columns")
        if isinstance(raw_cols, list):
            logger.info("List-based 'columns' format detected in business dictionary. Normalizing to dictionary map...")
            normalized_cols = {}
            for item in raw_cols:
                if not isinstance(item, dict):
                    continue
                col_name = item.get("name")
                if not col_name:
                    continue
                    
                # Map alternate description fields if "description" is missing
                desc = item.get("description") or item.get("business_meaning") or item.get("meaning") or ""
                
                # Check join key status
                is_key = item.get("is_join_key", False)
                if not is_key and col_name in primary_keys:
                    is_key = True
                    
                # Extract business rules (support strings or lists)
                rules = item.get("business_rules", [])
                if isinstance(rules, str):
                    rules = [rules]
                elif not isinstance(rules, list):
                    rules = []
                    
                normalized_cols[col_name] = {
                    "description": desc,
                    "is_join_key": is_key,
                    "business_rules": rules
                }
            data["columns"] = normalized_cols
            
        return data

# ==================== PARSER MODULE ====================

def parse_and_validate_dictionary(file_path: str) -> BusinessDataDictionary:
    """
    Parses a JSON or YAML business dictionary and validates it against the Pydantic schema.
    Supports both standard maps and enterprise list formats.
    
    Args:
        file_path: Absolute path to the JSON or YAML file.
        
    Returns:
        A validated BusinessDataDictionary Pydantic instance.
        
    Raises:
        ValueError: If file parsing fails or validation schema fails.
    """
    logger.info(f"Parsing business dictionary from: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                raw_data = yaml.safe_load(f)
            elif file_path.endswith(".json"):
                raw_data = json.load(f)
            else:
                raise ValueError("Unsupported file format. Please upload JSON or YAML (.yaml/.yml) files.")
                
        # Validate against our Pydantic schema (with custom pre-validator normalization)
        dictionary = BusinessDataDictionary.model_validate(raw_data)
        logger.info("Business dictionary parsed and validated successfully.")
        return dictionary
        
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        logger.error(f"Syntax error parsing file '{file_path}': {str(e)}")
        raise ValueError(f"Syntax Error: Failed to parse data dictionary file content. {str(e)}")
    except ValidationError as e:
        logger.error(f"Schema validation failure in dictionary '{file_path}': {str(e)}")
        error_details = []
        for err in e.errors():
            loc = " -> ".join(str(x) for x in err["loc"])
            error_details.append(f"Field '{loc}': {err['msg']}")
        raise ValueError(f"Validation Failure: {'; '.join(error_details)}")
    except Exception as e:
        logger.error(f"Unexpected error loading dictionary: {str(e)}")
        raise ValueError(f"Loader Error: {str(e)}")

# ==================== SCHEMA MERGER LAYER ====================

def merge_metadata_and_dictionary(
    raw_metadata: Dict[str, Any], 
    dictionary: BusinessDataDictionary
) -> Dict[str, Any]:
    """
    Synthesizes the automatic profiling metadata with the rich business dictionary context.
    
    Args:
        raw_metadata: Automatically generated profile (from extractor.py).
        dictionary: Validated BusinessDataDictionary Pydantic model.
        
    Returns:
        A combined, self-contained Contextual Intelligence Profile.
    """
    logger.info("Merging raw metadata profile with business dictionary context...")
    
    # Deep copy raw metadata to prevent mutations
    merged = json.loads(json.dumps(raw_metadata))
    
    # 1. Attach high-level grains and KPIs
    merged["grain"] = dictionary.grain or "No explicit grain definition provided."
    merged["kpis"] = [kpi.model_dump() for kpi in dictionary.kpis]
    
    # 2. Enrich individual column elements
    columns_profile = merged.get("columns", {})
    dict_columns = dictionary.columns
    
    # Case-insensitive mapping preparation for dictionary keys
    clean_dict_cols = {str(k).strip().lower(): v for k, v in dict_columns.items()}
    
    for col_name, col_meta in columns_profile.items():
        clean_key = str(col_name).strip().lower()
        
        if clean_key in clean_dict_cols:
            col_def: ColumnDefinition = clean_dict_cols[clean_key]
            
            # Inject business semantics
            col_meta["description"] = col_def.description
            col_meta["is_join_key"] = col_def.is_join_key
            col_meta["business_rules"] = col_def.business_rules
        else:
            col_meta["description"] = "No description provided in dictionary."
            col_meta["is_join_key"] = False
            col_meta["business_rules"] = []
            
    logger.info("Contextual Intelligence Profile merged successfully.")
    return merged
