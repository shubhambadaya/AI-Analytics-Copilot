"""
Enterprise Statistical Analysis Engine
=======================================
Deterministic, pre-built statistical functions that the LLM planner can invoke
by name instead of generating raw scipy code. Organized into 6 analysis modules.

All functions return plain dicts/DataFrames so results can be easily serialized,
displayed in Streamlit, and fed back to the LLM interpreter for narration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_distribution(series: pd.Series) -> Dict[str, Any]:
    """
    Comprehensive distribution analysis of a numeric series.
    
    Returns:
        Dict with keys: mean, median, std, skewness, kurtosis,
        normality_test (Shapiro-Wilk), shape_classification, outlier_count.
    """
    from scipy import stats as sp_stats
    
    clean = series.dropna()
    if len(clean) < 8:
        return {
            "error": "Insufficient data for distribution analysis (need ≥8 non-null values)",
            "n": len(clean)
        }
    
    mean_val = float(clean.mean())
    median_val = float(clean.median())
    std_val = float(clean.std())
    skew_val = float(clean.skew())
    kurt_val = float(clean.kurtosis())  # excess kurtosis (Fisher)
    
    # Shapiro-Wilk normality test (use subsample for large datasets)
    sample = clean.sample(min(5000, len(clean)), random_state=42) if len(clean) > 5000 else clean
    try:
        shapiro_stat, shapiro_p = sp_stats.shapiro(sample)
    except Exception:
        shapiro_stat, shapiro_p = float("nan"), float("nan")
    
    is_normal = shapiro_p > 0.05 if not np.isnan(shapiro_p) else None
    
    # Shape classification
    if abs(skew_val) < 0.5:
        shape = "symmetric"
    elif skew_val > 0:
        shape = "right-skewed" if skew_val < 1.5 else "heavily right-skewed"
    else:
        shape = "left-skewed" if skew_val > -1.5 else "heavily left-skewed"
    
    if is_normal:
        shape += " (approximately normal)"
    
    # IQR-based outlier count
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((clean < lower_bound) | (clean > upper_bound)).sum())
    
    return {
        "n": int(len(clean)),
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "std": round(std_val, 4),
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "normality_test": {
            "test": "Shapiro-Wilk",
            "statistic": round(float(shapiro_stat), 6) if not np.isnan(shapiro_stat) else None,
            "p_value": round(float(shapiro_p), 6) if not np.isnan(shapiro_p) else None,
            "is_normal": is_normal
        },
        "shape_classification": shape,
        "outlier_count": outlier_count,
        "outlier_bounds": {
            "lower": round(float(lower_bound), 4),
            "upper": round(float(upper_bound), 4)
        }
    }


def detect_outliers(
    series: pd.Series, 
    method: str = "iqr"
) -> pd.DataFrame:
    """
    Detect outliers in a numeric series using IQR or Z-score method.
    
    Args:
        series: Numeric pandas Series.
        method: 'iqr' (default) or 'zscore'.
        
    Returns:
        DataFrame with columns: value, is_outlier, score, method.
    """
    clean = series.dropna()
    result = pd.DataFrame({"value": clean})
    
    if method == "zscore":
        from scipy import stats as sp_stats
        z_scores = np.abs(sp_stats.zscore(clean, nan_policy="omit"))
        result["score"] = z_scores
        result["is_outlier"] = z_scores > 3.0
    else:  # IQR
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        result["score"] = clean.apply(
            lambda x: max(0, (lower - x) / iqr if x < lower else (x - upper) / iqr if x > upper else 0)
        )
        result["is_outlier"] = (clean < lower) | (clean > upper)
    
    result["method"] = method
    logger.info(f"Outlier detection ({method}): {result['is_outlier'].sum()} outliers found in {len(clean)} values")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: Comparison Testing
# ═══════════════════════════════════════════════════════════════════════════════

def compare_groups(
    df: pd.DataFrame, 
    metric_col: str, 
    group_col: str
) -> Dict[str, Any]:
    """
    Automatically selects and runs the appropriate statistical test to compare
    groups on a metric. Handles 2-group and multi-group scenarios.
    
    Returns:
        Dict with: test_name, statistic, p_value, effect_size, effect_size_label,
        is_significant, group_stats, interpretation.
    """
    from scipy import stats as sp_stats
    
    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        logger.info(f"Metric column '{metric_col}' is categorical. Falling back to Chi-Square / Cramer's V.")
        return compute_cramers_v(df, group_col, metric_col)
        
    groups_data = {}
    for name, grp in df.groupby(group_col):
        vals = grp[metric_col].dropna()
        if len(vals) >= 2:
            groups_data[str(name)] = vals
    
    n_groups = len(groups_data)
    if n_groups < 2:
        return {"error": f"Need at least 2 groups with ≥2 values each, found {n_groups}"}
    
    # Compute per-group descriptive stats
    group_stats = {}
    for name, vals in groups_data.items():
        ci_low, ci_high = _confidence_interval(vals)
        group_stats[name] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 4),
            "median": round(float(vals.median()), 4),
            "std": round(float(vals.std()), 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)]
        }
    
    group_values = list(groups_data.values())
    
    # Check normality for test selection
    all_normal = True
    for vals in group_values:
        if len(vals) >= 8:
            _, p = sp_stats.shapiro(vals.sample(min(5000, len(vals)), random_state=42) if len(vals) > 5000 else vals)
            if p < 0.05:
                all_normal = False
                break
        else:
            all_normal = False
    
    if n_groups == 2:
        g1, g2 = group_values[0], group_values[1]
        if all_normal:
            stat, p = sp_stats.ttest_ind(g1, g2, equal_var=False)  # Welch's t-test
            test_name = "Welch's t-test"
        else:
            stat, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U"
        
        # Cohen's d effect size
        pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        effect_size = float(abs(g1.mean() - g2.mean()) / pooled_std) if pooled_std > 0 else 0.0
        effect_label = _cohens_d_label(effect_size)
    else:
        if all_normal:
            stat, p = sp_stats.f_oneway(*group_values)
            test_name = "One-way ANOVA"
        else:
            stat, p = sp_stats.kruskal(*group_values)
            test_name = "Kruskal-Wallis H"
        
        # Eta-squared effect size for multi-group
        all_vals = pd.concat(group_values)
        grand_mean = all_vals.mean()
        ss_between = sum(len(v) * (v.mean() - grand_mean)**2 for v in group_values)
        ss_total = sum((all_vals - grand_mean)**2)
        effect_size = float(ss_between / ss_total) if ss_total > 0 else 0.0
        effect_label = _eta_squared_label(effect_size)
    
    is_sig = bool(p < 0.05)
    
    interpretation = (
        f"{test_name} shows a {'statistically significant' if is_sig else 'non-significant'} "
        f"difference between groups (p={p:.4f}). "
        f"Effect size: {effect_size:.3f} ({effect_label})."
    )
    
    logger.info(f"Group comparison: {test_name}, p={p:.4f}, effect_size={effect_size:.3f} ({effect_label})")
    
    return {
        "test_name": test_name,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "effect_size": round(effect_size, 4),
        "effect_size_label": effect_label,
        "is_significant": is_sig,
        "n_groups": n_groups,
        "group_stats": group_stats,
        "interpretation": interpretation
    }


def compare_proportions(
    group_a_success: int, group_a_total: int,
    group_b_success: int, group_b_total: int
) -> Dict[str, Any]:
    """
    Two-proportion z-test for comparing rates/percentages between two groups.
    Example: comparing conversion rates, churn rates, etc.
    """
    from scipy import stats as sp_stats
    
    p1 = group_a_success / group_a_total if group_a_total > 0 else 0
    p2 = group_b_success / group_b_total if group_b_total > 0 else 0
    
    # Pooled proportion
    p_pool = (group_a_success + group_b_success) / (group_a_total + group_b_total)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/group_a_total + 1/group_b_total))
    
    if se == 0:
        return {"error": "Cannot compute z-test: zero standard error"}
    
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
    
    return {
        "test_name": "Two-proportion z-test",
        "proportion_a": round(p1, 4),
        "proportion_b": round(p2, 4),
        "difference": round(p1 - p2, 4),
        "z_statistic": round(float(z_stat), 4),
        "p_value": round(float(p_value), 6),
        "is_significant": bool(p_value < 0.05),
        "interpretation": (
            f"Group A rate: {p1:.2%}, Group B rate: {p2:.2%}. "
            f"Difference {'is' if p_value < 0.05 else 'is not'} statistically significant (p={p_value:.4f})."
        )
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: Correlation Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def compute_correlations(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    method: str = "auto"
) -> Dict[str, Any]:
    """
    Compute pairwise correlations with significance testing.
    
    Args:
        df: DataFrame
        columns: Columns to include (default: all numeric)
        method: 'pearson', 'spearman', 'auto' (auto-selects based on normality)
    
    Returns:
        Dict with correlation_matrix, p_value_matrix, significant_pairs.
    """
    from scipy import stats as sp_stats
    
    if columns:
        numeric_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis"}
    
    subset = df[numeric_cols].dropna()
    
    if method == "auto":
        method = "pearson"  # default
        for col in numeric_cols[:3]:  # quick normality check on first few
            if len(subset[col]) >= 8:
                _, p = sp_stats.shapiro(subset[col].sample(min(5000, len(subset)), random_state=42))
                if p < 0.05:
                    method = "spearman"
                    break
    
    n = len(numeric_cols)
    corr_matrix = {}
    p_matrix = {}
    significant_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            col_a, col_b = numeric_cols[i], numeric_cols[j]
            a, b = subset[col_a], subset[col_b]
            
            if method == "spearman":
                corr, p_val = sp_stats.spearmanr(a, b)
            else:
                corr, p_val = sp_stats.pearsonr(a, b)
            
            pair_key = f"{col_a} × {col_b}"
            corr_matrix[pair_key] = round(float(corr), 4)
            p_matrix[pair_key] = round(float(p_val), 6)
            
            if p_val < 0.05:
                strength = _correlation_strength(abs(corr))
                significant_pairs.append({
                    "pair": pair_key,
                    "correlation": round(float(corr), 4),
                    "p_value": round(float(p_val), 6),
                    "strength": strength,
                    "direction": "positive" if corr > 0 else "negative"
                })
    
    # Sort significant pairs by absolute correlation
    significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    logger.info(f"Correlation analysis ({method}): {len(significant_pairs)} significant pairs found")
    
    return {
        "method": method,
        "n_observations": len(subset),
        "correlation_matrix": corr_matrix,
        "p_value_matrix": p_matrix,
        "significant_pairs": significant_pairs,
        "n_significant": len(significant_pairs)
    }


def compute_cramers_v(
    df: pd.DataFrame, 
    col_a: str, 
    col_b: str
) -> Dict[str, Any]:
    """
    Cramér's V association measure for two categorical variables,
    with chi-square test of independence.
    """
    from scipy import stats as sp_stats
    
    contingency = pd.crosstab(df[col_a], df[col_b])
    chi2, p_val, dof, expected = sp_stats.chi2_contingency(contingency)
    
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0
    
    strength = _correlation_strength(cramers_v)
    
    return {
        "test_name": "Chi-Square + Cramér's V",
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_val), 6),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "association_strength": strength,
        "is_significant": bool(p_val < 0.05),
        "interpretation": (
            f"Chi-square test: χ²={chi2:.2f}, p={p_val:.4f}. "
            f"Cramér's V={cramers_v:.3f} ({strength} association). "
            f"The association {'is' if p_val < 0.05 else 'is not'} statistically significant."
        )
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: Trend Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_trend(
    values: pd.Series, 
    time_index: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Linear trend analysis with regression statistics.
    
    Args:
        values: Numeric series of measurements.
        time_index: Optional time/order series (uses integer index if omitted).
    
    Returns:
        Dict with slope, r_squared, p_value, direction, trend_strength.
    """
    from scipy import stats as sp_stats
    
    clean_vals = values.dropna()
    if len(clean_vals) < 3:
        return {"error": "Need at least 3 data points for trend analysis"}
    
    if time_index is not None:
        # Convert dates to ordinal if datetime
        if pd.api.types.is_datetime64_any_dtype(time_index):
            x = time_index.dropna().map(pd.Timestamp.toordinal).values
        else:
            x = pd.to_numeric(time_index, errors="coerce").dropna().values
        # Align lengths
        min_len = min(len(x), len(clean_vals))
        x = x[:min_len]
        y = clean_vals.values[:min_len]
    else:
        x = np.arange(len(clean_vals))
        y = clean_vals.values
    
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x.astype(float), y.astype(float))
    
    r_squared = r_value ** 2
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    
    if r_squared > 0.7:
        strength = "strong"
    elif r_squared > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    
    # Percent change from fitted start to end
    fitted_start = intercept + slope * x[0]
    fitted_end = intercept + slope * x[-1]
    pct_change = ((fitted_end - fitted_start) / abs(fitted_start) * 100) if fitted_start != 0 else 0
    
    logger.info(f"Trend analysis: {direction} ({strength}), R²={r_squared:.3f}, p={p_value:.4f}")
    
    return {
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_squared), 4),
        "p_value": round(float(p_value), 6),
        "std_error": round(float(std_err), 6),
        "direction": direction,
        "trend_strength": strength,
        "is_significant": bool(p_value < 0.05),
        "percent_change": round(float(pct_change), 2),
        "n_points": len(y),
        "interpretation": (
            f"{strength.capitalize()} {direction} trend detected (R²={r_squared:.3f}, p={p_value:.4f}). "
            f"Overall change: {pct_change:+.1f}%."
        )
    }


def detect_changepoints(
    series: pd.Series, 
    threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Simple CUSUM-based changepoint detection for identifying regime shifts.
    
    Args:
        series: Numeric time series.
        threshold: Number of standard deviations to trigger a changepoint.
    
    Returns:
        Dict with changepoints (indices), n_changepoints, segments.
    """
    clean = series.dropna().reset_index(drop=True)
    if len(clean) < 10:
        return {"error": "Need at least 10 data points for changepoint detection"}
    
    mean_val = clean.mean()
    std_val = clean.std()
    
    if std_val == 0:
        return {"changepoints": [], "n_changepoints": 0, "segments": []}
    
    cusum_pos = np.zeros(len(clean))
    cusum_neg = np.zeros(len(clean))
    changepoints = []
    
    for i in range(1, len(clean)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (clean.iloc[i] - mean_val) / std_val - 0.5)
        cusum_neg[i] = min(0, cusum_neg[i-1] + (clean.iloc[i] - mean_val) / std_val + 0.5)
        
        if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
            changepoints.append(int(i))
            cusum_pos[i] = 0
            cusum_neg[i] = 0
    
    # Build segment summaries
    boundaries = [0] + changepoints + [len(clean)]
    segments = []
    for i in range(len(boundaries) - 1):
        seg = clean.iloc[boundaries[i]:boundaries[i+1]]
        segments.append({
            "start_index": boundaries[i],
            "end_index": boundaries[i+1] - 1,
            "mean": round(float(seg.mean()), 4),
            "std": round(float(seg.std()), 4),
            "n": len(seg)
        })
    
    return {
        "changepoints": changepoints,
        "n_changepoints": len(changepoints),
        "segments": segments,
        "threshold_used": threshold
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: Cohort / Segment Profiling
# ═══════════════════════════════════════════════════════════════════════════════

def profile_segment(
    df: pd.DataFrame, 
    segment_col: str, 
    metric_cols: List[str]
) -> pd.DataFrame:
    """
    Generate per-segment descriptive statistics with 95% confidence intervals.
    
    Returns:
        DataFrame with segment, metric, n, mean, std, ci_lower, ci_upper.
    """
    rows = []
    for seg_name, seg_df in df.groupby(segment_col):
        for metric in metric_cols:
            if metric not in seg_df.columns or not pd.api.types.is_numeric_dtype(seg_df[metric]):
                continue
            vals = seg_df[metric].dropna()
            if len(vals) < 2:
                continue
            
            ci_low, ci_high = _confidence_interval(vals)
            rows.append({
                "segment": str(seg_name),
                "metric": metric,
                "n": int(len(vals)),
                "mean": round(float(vals.mean()), 4),
                "median": round(float(vals.median()), 4),
                "std": round(float(vals.std()), 4),
                "ci_lower": round(ci_low, 4),
                "ci_upper": round(ci_high, 4)
            })
    
    result = pd.DataFrame(rows)
    logger.info(f"Segment profiling: {len(result)} segment-metric combinations computed")
    return result


def compute_confidence_interval(
    series: pd.Series, 
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Compute confidence interval for the mean of a numeric series.
    Uses t-distribution for small samples, z for large.
    """
    from scipy import stats as sp_stats
    
    clean = series.dropna()
    n = len(clean)
    if n < 2:
        return {"error": "Need at least 2 values to compute confidence interval"}
    
    mean_val = float(clean.mean())
    se = float(clean.std() / np.sqrt(n))
    
    alpha = 1 - confidence
    if n < 30:
        t_crit = sp_stats.t.ppf(1 - alpha/2, df=n-1)
        margin = t_crit * se
        method = "t-distribution"
    else:
        z_crit = sp_stats.norm.ppf(1 - alpha/2)
        margin = z_crit * se
        method = "z-distribution"
    
    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(mean_val - margin, 4),
        "ci_upper": round(mean_val + margin, 4),
        "margin_of_error": round(float(margin), 4),
        "confidence_level": confidence,
        "method": method,
        "n": n,
        "standard_error": round(se, 4)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6: Confidence Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def compute_analysis_confidence(
    sample_size: int,
    p_values: Optional[List[float]] = None,
    effect_sizes: Optional[List[float]] = None,
    data_completeness: float = 1.0
) -> Dict[str, Any]:
    """
    Compute a composite 0–1 confidence score for an entire analysis.
    
    Factors:
        - Sample size adequacy (0-1)
        - Statistical significance of results (0-1)
        - Effect size magnitude (0-1)
        - Data completeness / quality (0-1)
    
    Returns:
        Dict with overall score, component scores, confidence_label.
    """
    # 1. Sample size score (sigmoid: 0.5 at n=30, 0.9 at n=200)
    sample_score = 1.0 / (1.0 + np.exp(-0.03 * (sample_size - 50)))
    
    # 2. Significance score (avg p-value proximity to threshold)
    if p_values and len(p_values) > 0:
        valid_ps = [p for p in p_values if p is not None and not np.isnan(p)]
        if valid_ps:
            sig_score = np.mean([1.0 if p < 0.01 else 0.7 if p < 0.05 else 0.3 if p < 0.1 else 0.1 for p in valid_ps])
        else:
            sig_score = 0.5  # No test results — neutral
    else:
        sig_score = 0.5  # Descriptive only — neutral
    
    # 3. Effect size score
    if effect_sizes and len(effect_sizes) > 0:
        valid_es = [e for e in effect_sizes if e is not None and not np.isnan(e)]
        if valid_es:
            effect_score = np.mean([min(1.0, e / 0.8) for e in valid_es])  # 0.8+ = perfect
        else:
            effect_score = 0.5
    else:
        effect_score = 0.5
    
    # 4. Data completeness
    completeness_score = min(1.0, max(0.0, data_completeness))
    
    # Weighted composite
    weights = {"sample": 0.3, "significance": 0.25, "effect": 0.2, "completeness": 0.25}
    overall = (
        weights["sample"] * sample_score +
        weights["significance"] * sig_score +
        weights["effect"] * effect_score +
        weights["completeness"] * completeness_score
    )
    
    if overall >= 0.75:
        label = "high"
    elif overall >= 0.5:
        label = "medium"
    else:
        label = "low"
    
    return {
        "overall_score": round(float(overall), 3),
        "confidence_label": label,
        "components": {
            "sample_size_score": round(float(sample_score), 3),
            "significance_score": round(float(sig_score), 3),
            "effect_size_score": round(float(effect_score), 3),
            "completeness_score": round(float(completeness_score), 3)
        },
        "sample_size": sample_size
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _confidence_interval(series: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """Quick CI computation returning (lower, upper)."""
    from scipy import stats as sp_stats
    n = len(series)
    mean = series.mean()
    se = series.std() / np.sqrt(n)
    if n < 30:
        t_crit = sp_stats.t.ppf(1 - (1 - confidence)/2, df=n-1)
        margin = t_crit * se
    else:
        z_crit = sp_stats.norm.ppf(1 - (1 - confidence)/2)
        margin = z_crit * se
    return float(mean - margin), float(mean + margin)


def _cohens_d_label(d: float) -> str:
    """Classify Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _eta_squared_label(eta2: float) -> str:
    """Classify eta-squared effect size."""
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"


def _correlation_strength(r: float) -> str:
    """Classify correlation coefficient strength."""
    r = abs(r)
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "weak"
    elif r < 0.5:
        return "moderate"
    elif r < 0.7:
        return "strong"
    else:
        return "very strong"
