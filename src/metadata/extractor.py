import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

def is_id_column(series: pd.Series, col_name: str, total_rows: int) -> bool:
    """
    Deterministically detects if a column serves as an ID or primary key.
    Checks name matches (e.g. 'id', '_id', 'key') and cardinality ratios.
    """
    name_lower = str(col_name).lower()
    
    # 1. Direct name matches
    name_match = (
        name_lower == "id" or 
        name_lower.endswith("_id") or 
        name_lower.startswith("id_") or 
        name_lower.endswith("key") or
        name_lower.endswith("code")
    )
    
    if total_rows == 0:
        return name_match
        
    # Heuristics: unique ratio high-cardinality should only apply for larger datasets
    # to avoid treating tiny unique string series (like unique names in 5 rows) as IDs.
    if total_rows <= 15:
        return name_match
        
    # 2. Cardinality calculation (dropna to avoid treating nulls as unique ids)
    non_null_series = series.dropna()
    if len(non_null_series) == 0:
        return False
        
    unique_count = non_null_series.nunique()
    unique_ratio = unique_count / total_rows
    
    # 3. Exclude floating-point fields since they represent continuous data
    is_float = pd.api.types.is_float_dtype(series)
    
    return (name_match or unique_ratio > 0.98) and not is_float and unique_count > 1

def profile_dataframe(df: pd.DataFrame, max_sample_rows: int = 100000) -> Dict[str, Any]:
    """
    Profiles a pandas DataFrame to extract rich, deterministic metadata.
    Optimized for large datasets: uses fast statistical checks and samples 
    extremely large files for cardinality/type inference to maintain speed.
    
    Args:
        df: The pandas DataFrame to profile.
        max_sample_rows: Maximum rows to use for expensive cardinality/sample calculations.
        
    Returns:
        A dictionary containing the profile metadata.
    """
    logger.info("Profiling DataFrame starting...")
    
    total_rows = len(df)
    total_cols = len(df.columns)
    memory_usage = df.memory_usage(deep=True).sum()
    
    # Scalability: If dataset is huge, sample it for cardinality & type inferences
    use_sample = total_rows > max_sample_rows
    sample_df = df.sample(n=max_sample_rows, random_state=42) if use_sample else df
    if use_sample:
        logger.info(f"Large dataset detected ({total_rows} rows). Sampling {max_sample_rows} rows for type/cardinality inferences.")
        
    columns_profile = {}
    
    for col in df.columns:
        series = df[col]
        sample_series = sample_df[col]
        
        # Calculate null statistics (exact, quick on full series)
        null_count = int(series.isnull().sum())
        null_pct = float(null_count / total_rows) if total_rows > 0 else 0.0
        
        # Cardinality on sample (or full series if not sampled)
        try:
            unique_count = int(sample_series.nunique())
            if use_sample:
                # Extrapolate cardinality fraction for metadata display
                unique_count = int((unique_count / max_sample_rows) * total_rows)
        except Exception:
            unique_count = 0
            
        # Detect broad category
        dtype = str(series.dtype)
        
        # Heuristics-based type categorization
        if is_id_column(sample_series, col, len(sample_df)):
            category = "id"
        elif pd.api.types.is_bool_dtype(series):
            category = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            category = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            category = "datetime"
        else:
            # Let's inspect objects/strings
            # Try to see if it represents dates
            try:
                non_null_samples = sample_series.dropna().head(10).astype(str)
                parsed_dates = pd.to_datetime(non_null_samples, errors='coerce')
                if parsed_dates.notnull().all() and len(non_null_samples) > 0:
                    category = "datetime"
                else:
                    # Check if boolean represented as string (e.g. 'true'/'false', 'y'/'n')
                    unique_vals = set(sample_series.dropna().head(20).astype(str).str.lower())
                    if unique_vals.issubset({"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0"}):
                        category = "boolean"
                    else:
                        category = "categorical"
            except Exception:
                category = "categorical"
        
        # Calculate descriptive statistics (using full series where computationally cheap)
        stats = {}
        data_quality_score = 1.0  # Base score 0.0-1.0
        
        # Adjust quality score based on nulls
        data_quality_score -= null_pct * 0.5
        
        if category == "numeric":
            clean_series = series.dropna()
            
            # Basic stats
            min_val = float(clean_series.min()) if len(clean_series) > 0 else None
            max_val = float(clean_series.max()) if len(clean_series) > 0 else None
            mean_val = float(clean_series.mean()) if len(clean_series) > 0 else None
            median_val = float(clean_series.median()) if len(clean_series) > 0 else None
            std_val = float(clean_series.std()) if len(clean_series) > 0 else None
            
            # Advanced stats
            skew_val = None
            kurt_val = None
            shape = "unknown"
            outliers = 0
            
            if len(clean_series) >= 3:
                skew_val = float(clean_series.skew())
                kurt_val = float(clean_series.kurtosis())
                
                # Shape heuristics
                if abs(skew_val) < 0.5:
                    shape = "symmetric"
                elif skew_val > 0:
                    shape = "right-skewed" if skew_val < 1.5 else "heavily right-skewed"
                else:
                    shape = "left-skewed" if skew_val > -1.5 else "heavily left-skewed"
                
                # Outlier detection (IQR)
                q1 = clean_series.quantile(0.25)
                q3 = clean_series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = int(((clean_series < lower) | (clean_series > upper)).sum())
                    
                    # Penalize data quality if >5% are outliers
                    outlier_pct = outliers / len(clean_series)
                    if outlier_pct > 0.05:
                        data_quality_score -= min(0.3, outlier_pct * 2)
            
            stats = {
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "median": median_val,
                "std": std_val,
                "skewness": skew_val,
                "kurtosis": kurt_val,
                "distribution_shape": shape,
                "outlier_count": outliers
            }
            
        elif category == "datetime":
            try:
                dt_series = pd.to_datetime(series, errors='coerce')
                valid_dt = dt_series.dropna()
                if len(valid_dt) > 0:
                    stats = {
                        "min": str(valid_dt.min()),
                        "max": str(valid_dt.max()),
                        "duration_days": int((valid_dt.max() - valid_dt.min()).days)
                    }
                else:
                    data_quality_score -= 0.5 # High parsing failure rate
            except Exception:
                pass
        elif category == "categorical":
            # Find the most frequent categories (value counts)
            try:
                value_counts = series.value_counts().head(3)
                top_categories = []
                for val, count in value_counts.items():
                    top_categories.append({
                        "value": str(val),
                        "count": int(count),
                        "percentage": float(count / total_rows)
                    })
                stats = {"top_frequencies": top_categories}
            except Exception:
                pass
        
        # Fetch non-null samples (top 5 unique values or first 5 non-null values)
        sample_values = []
        try:
            unique_non_null = sample_series.dropna().unique()
            sample_values = [str(val) for val in unique_non_null[:5]]
        except Exception:
            sample_values = [str(val) for val in sample_series.dropna().head(5)]
            
        columns_profile[col] = {
            "name": col,
            "data_type": dtype,
            "category": category,
            "null_count": null_count,
            "null_percentage": null_pct,
            "distinct_values": unique_count,
            "data_quality_score": round(max(0.0, data_quality_score), 2),
            "statistics": stats,
            "samples": sample_values
        }
        
    profile = {
        "dimensions": {
            "rows": total_rows,
            "columns": total_cols,
            "memory_bytes": int(memory_usage)
        },
        "columns": columns_profile
    }
    
    logger.info(f"Profiling DataFrame finished. Total columns profiled: {total_cols}")
    return profile

def extract_metadata_from_csv(file_path: str) -> Dict[str, Any]:
    """
    Reads a CSV and profiles it.
    
    Args:
        file_path: Absolute path to the CSV file.
        
    Returns:
        A dictionary containing the profile metadata.
    """
    logger.info(f"Reading CSV file from path: {file_path}")
    df = pd.read_csv(file_path)
    return profile_dataframe(df)
