import json
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ContextBuilder:
    """
    Context Builder Layer (Contextual Intelligence Gateway).
    Synthesizes multiple table profiles, data dictionaries, KPI definitions,
    and inferred join relationships into a single, token-compressed context profile
    specifically optimized for LLM prompting.
    """
    def __init__(self):
        logger.info("ContextBuilder module initialized.")

    def infer_relationships(self, datasets: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Dynamically infers relationships between multiple datasets
        by identifying shared columns flagged as primary/foreign join keys.
        
        Args:
            datasets: Active session datasets map.
            
        Returns:
            A list of detected join relationship dictionaries.
        """
        logger.info("Scanning datasets to infer relational join paths...")
        relationships = []
        join_key_map = {} # col_name_lower -> list of (table_name, original_col_name)
        
        for table_name, info in datasets.items():
            meta = info.get("metadata", {})
            columns = meta.get("columns", {})
            for col_name, col_meta in columns.items():
                if col_meta.get("is_join_key", False):
                    key_lower = col_name.strip().lower()
                    if key_lower not in join_key_map:
                        join_key_map[key_lower] = []
                    join_key_map[key_lower].append((table_name, col_name))
                    
        # Match join keys present in more than one table
        for key_lower, occurrences in join_key_map.items():
            if len(occurrences) > 1:
                # Generate pairing links (A -> B)
                for i in range(len(occurrences)):
                    for j in range(i + 1, len(occurrences)):
                        tbl_a, col_a = occurrences[i]
                        tbl_b, col_b = occurrences[j]
                        relationships.append({
                            "from_table": tbl_a,
                            "from_column": col_a,
                            "to_table": tbl_b,
                            "to_column": col_b,
                            "matching_key": col_a
                        })
                        logger.info(f"Inferred Join Relationship: [{tbl_a}.{col_a}] <===> [{tbl_b}.{col_b}]")
                        
        return relationships

    def build_llm_context(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesizes active datasets into a highly token-efficient,
        non-redundant structured context dictionary.
        
        Removes heavy descriptive statistics and truncates wide sample lists
        to keep the prompt payload light while maintaining complete semantic awareness.
        """
        logger.info("Constructing token-compressed context object...")
        
        context_profile = {
            "tables": {},
            "relationships": self.infer_relationships(datasets)
        }
        
        for table_name, info in datasets.items():
            meta = info.get("metadata", {})
            dimensions = meta.get("dimensions", {})
            columns = meta.get("columns", {})
            kpis = meta.get("kpis", [])
            grain = meta.get("grain", "No grain granularity specified.")
            
            table_summary = {
                "dimensions": {
                    "rows": dimensions.get("rows", 0),
                    "columns": dimensions.get("columns", 0)
                },
                "grain": grain,
                "kpis": kpis,
                "columns": {}
            }
            
            # Compress column profiles: preserve types, semantic descriptions, join key status, 
            # and business rules, but strip redundant stats and sample details
            for col_name, col_meta in columns.items():
                cat = col_meta.get("category", "unknown")
                dtype = col_meta.get("data_type", "unknown")
                null_pct = round(col_meta.get("null_percentage", 0.0) * 100, 1)
                desc = col_meta.get("description", "")
                is_key = col_meta.get("is_join_key", False)
                rules = col_meta.get("business_rules", [])
                
                # Fetch minimal samples to prevent token bloat
                samples = col_meta.get("samples", [])[:2] # limit to top 2 samples
                
                col_summary = {
                    "category": cat,
                    "type": dtype,
                    "null_rate": f"{null_pct}%",
                    "description": desc,
                    "samples": samples
                }
                
                if is_key:
                    col_summary["is_join_key"] = True
                if rules:
                    col_summary["business_rules"] = rules
                    
                # Include simple numeric boundary values inside prompt for calculation range guides
                stats = col_meta.get("statistics", {})
                if cat == "numeric" and stats:
                    col_summary["range"] = {
                        "min": stats.get("min"),
                        "max": stats.get("max")
                    }
                elif cat == "datetime" and stats:
                    col_summary["range"] = {
                        "start": stats.get("min"),
                        "end": stats.get("max")
                    }
                    
                table_summary["columns"][col_name] = col_summary
                
            context_profile["tables"][table_name] = table_summary
            
        logger.info("Context object construction complete.")
        return context_profile

    def format_context_to_dense_json(self, context_obj: Dict[str, Any]) -> str:
        """
        Formats the context object into a dense JSON string (no indent, no unnecessary spaces)
        maximizing token utilization inside LLM prompt templates.
        """
        return json.dumps(context_obj, separators=(',', ':'))
        
    def format_context_to_yaml(self, context_obj: Dict[str, Any]) -> str:
        """
        Formats the context object into a highly readable, structured YAML block.
        YAML is often read more cleanly by certain LLMs than compressed JSON.
        """
        try:
            import yaml
            return yaml.dump(context_obj, default_flow_style=False, sort_keys=False)
        except ImportError:
            return json.dumps(context_obj, indent=2)

# Instantiated builder instance
context_builder = ContextBuilder()
