import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List, Union
from src.llm.schemas import VisualSpec
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Premium, executive-grade corporate color palette (Sleek slate, royal blue, emerald, amber, soft grey)
EXECUTIVE_PALETTE = ["#1E3A8A", "#10B981", "#F59E0B", "#EF4444", "#6366F1", "#06B6D4", "#EC4899", "#8B5CF6"]

# Standardized layout guidelines for modern executive dashboards
EXECUTIVE_LAYOUT_CONFIG = {
    "font_family": "Outfit, Inter, system-ui, -apple-system, sans-serif",
    "title_font_size": 22,
    "title_font_color": "#0F172A",  # Deep Slate-900
    "axis_label_color": "#475569",  # Slate-600
    "gridline_color": "rgba(226, 232, 240, 0.8)",  # Slate-200
    "bg_color": "rgba(255, 255, 255, 1)",  # Pure White
    "plot_bg_color": "rgba(248, 250, 252, 0.6)",  # Slate-50 tint
}

# ==================== AUTOMATIC CHART MAPPING LOGIC ====================

def map_analysis_to_chart_type(analysis_type: str) -> str:
    """
    Deterministically maps high-level analysis types to their ideal Plotly chart types.
    
    Args:
        analysis_type: String identifier of analysis (e.g. 'trend_analysis').
        
    Returns:
        Ideal chart type string ('line', 'bar', 'heatmap', 'histogram', etc.).
    """
    clean_type = str(analysis_type).strip().lower()
    
    if "trend" in clean_type or "time_series" in clean_type or "growth" in clean_type:
        return "line"
    elif "cohort" in clean_type or "comparison" in clean_type or "segment" in clean_type or "ranking" in clean_type:
        return "bar"
    elif "correlation" in clean_type or "relationship" in clean_type or "matrix" in clean_type or "affinity" in clean_type:
        return "heatmap"
    elif "distribution" in clean_type or "spread" in clean_type or "density" in clean_type:
        return "histogram"
    elif "outlier" in clean_type or "spread_distribution" in clean_type:
        return "box"
    elif "share" in clean_type or "composition" in clean_type or "breakdown" in clean_type:
        return "pie"
    else:
        # Defaults to bar chart for categorical and logical comparisons
        return "bar"

# ==================== REUSABLE EXECUTIVE CHART BUILDERS ====================

def apply_executive_styling(fig: go.Figure, title: str) -> go.Figure:
    """
    Applies strict corporate executive-style layouts to any Plotly figure.
    Removes visual clutter, rotates labels if needed, and aligns typography.
    """
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'y': 0.95,
            'x': 0.05,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': dict(
                family=EXECUTIVE_LAYOUT_CONFIG["font_family"],
                size=EXECUTIVE_LAYOUT_CONFIG["title_font_size"],
                color=EXECUTIVE_LAYOUT_CONFIG["title_font_color"]
            )
        },
        font=dict(
            family=EXECUTIVE_LAYOUT_CONFIG["font_family"],
            size=12,
            color=EXECUTIVE_LAYOUT_CONFIG["axis_label_color"]
        ),
        margin=dict(l=60, r=40, t=90, b=60),
        hovermode="x unified",
        plot_bgcolor=EXECUTIVE_LAYOUT_CONFIG["plot_bg_color"],
        paper_bgcolor=EXECUTIVE_LAYOUT_CONFIG["bg_color"],
        
        # Gridline and ticks alignments
        xaxis=dict(
            showgrid=True,
            gridcolor=EXECUTIVE_LAYOUT_CONFIG["gridline_color"],
            zeroline=False,
            tickangle=-15,  # Prevents overlapping text labels
            tickfont=dict(size=11),
            linecolor="rgba(148, 163, 184, 0.5)",
            linewidth=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=EXECUTIVE_LAYOUT_CONFIG["gridline_color"],
            zeroline=False,
            tickfont=dict(size=11),
            linecolor="rgba(148, 163, 184, 0.5)",
            linewidth=1
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(226, 232, 240, 1)",
            borderwidth=1,
            font=dict(size=11)
        )
    )
    return fig

def plot_line_chart(
    df: pd.DataFrame, x: str, y: str, group: Optional[str] = None, title: str = ""
) -> go.Figure:
    """Builds a premium, executive-grade trend line chart."""
    fig = px.line(
        df, x=x, y=y, color=group,
        color_discrete_sequence=EXECUTIVE_PALETTE,
        markers=True
    )
    fig.update_traces(line=dict(width=3))
    return apply_executive_styling(fig, title)

def plot_bar_chart(
    df: pd.DataFrame, x: str, y: str, group: Optional[str] = None, 
    barmode: str = "group", orientation: str = "v", title: str = ""
) -> go.Figure:
    """Builds a premium, executive-grade comparative bar chart."""
    # Force categorical x-axis: cast to string so Plotly doesn't treat
    # alphanumeric codes (e.g. offer tags, plan IDs) as continuous numbers.
    plot_df = df.copy()
    cat_col = x if orientation == "v" else y
    if cat_col and cat_col in plot_df.columns:
        plot_df[cat_col] = plot_df[cat_col].astype(str)
    
    fig = px.bar(
        plot_df, x=x, y=y, color=group,
        barmode=barmode,
        orientation=orientation,
        color_discrete_sequence=EXECUTIVE_PALETTE
    )
    # Explicitly set the category axis type
    if orientation == "v":
        fig.update_xaxes(type="category")
    else:
        fig.update_yaxes(type="category")
    return apply_executive_styling(fig, title)

def plot_heatmap_chart(
    df: pd.DataFrame, x: str, y: str, z_val: str, title: str = ""
) -> go.Figure:
    """Builds a highly readable correlation matrix or behavioral heatmap."""
    try:
        # Dynamic pivot representation
        pivot_df = df.pivot(index=x, columns=y, values=z_val)
        fig = px.imshow(pivot_df, color_continuous_scale="RdBu_r", aspect="auto", text_auto=".1f")
    except Exception:
        # Fallback to direct numerical correlations if pivot fails
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto")
        
    fig.update_layout(coloraxis_colorbar=dict(thickness=15, title="Value"))
    return apply_executive_styling(fig, title)

def plot_histogram_chart(
    df: pd.DataFrame, x: str, y: Optional[str] = None, group: Optional[str] = None, title: str = ""
) -> go.Figure:
    """Builds an executive-grade distribution histogram."""
    fig = px.histogram(
        df, x=x, y=y, color=group,
        color_discrete_sequence=EXECUTIVE_PALETTE,
        opacity=0.85
    )
    fig.update_layout(bargap=0.05)
    return apply_executive_styling(fig, title)

def plot_pie_chart(
    df: pd.DataFrame, names: str, values: str, title: str = ""
) -> go.Figure:
    """Builds a clean, executive donut-style chart."""
    plot_df = df.copy()
    if names and names in plot_df.columns:
        plot_df[names] = plot_df[names].astype(str)
    fig = px.pie(
        plot_df, names=names, values=values,
        color_discrete_sequence=EXECUTIVE_PALETTE,
        hole=0.4
    )
    fig.update_traces(textinfo='percent+label', textposition='outside')
    return apply_executive_styling(fig, title)

def apply_statistical_overlays(fig: go.Figure, stat_results: Dict[str, Any]) -> go.Figure:
    """
    Overlays statistical significance markers, p-values, or trendlines
    onto an existing Plotly figure based on the deterministic engine's outputs.
    """
    if not stat_results:
        return fig
        
    annotations = []
    
    # 1. Overlay Correlation & Trend Analysis
    if "correlation_results" in stat_results:
        corr_data = stat_results["correlation_results"]
        if isinstance(corr_data, list):
            for corr in corr_data:
                # Add text annotation
                text = f"<b>{corr.get('method', 'Correlation').title()}</b>: {corr.get('coefficient', 0):.2f}<br><b>p-value</b>: {corr.get('p_value', 1.0):.4f}"
                annotations.append(
                    dict(
                        x=0.02, y=0.98, xref='paper', yref='paper',
                        text=text, showarrow=False,
                        align='left', bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#10B981", borderwidth=1, borderpad=4,
                        font=dict(size=11, color="#065F46")
                    )
                )
    
    if "trend_results" in stat_results:
        trend = stat_results["trend_results"]
        if "slope" in trend and "intercept" in trend:
            # Note: drawing a true trendline requires the x-domain.
            # For simplicity, we just add the regression equation as a badge.
            eq = f"<b>Trend (OLS)</b>: y = {trend['slope']:.3f}x + {trend['intercept']:.3f}<br><b>R²</b>: {trend.get('r_squared', 0):.3f}"
            annotations.append(
                dict(
                    x=0.02, y=0.88, xref='paper', yref='paper',
                    text=eq, showarrow=False,
                    align='left', bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#3B82F6", borderwidth=1, borderpad=4,
                    font=dict(size=11, color="#1E40AF")
                )
            )
            
    # 2. Overlay Comparison Testing (ANOVA / T-Test)
    if "comparison_results" in stat_results:
        comp = stat_results["comparison_results"]
        test_name = comp.get("test_name", "Test")
        p_val = comp.get("p_value", 1.0)
        is_sig = comp.get("is_significant", False)
        
        color = "#10B981" if is_sig else "#94A3B8"
        text = f"<b>{test_name}</b><br>p-value: {p_val:.4g}"
        annotations.append(
            dict(
                x=0.98, y=0.98, xref='paper', yref='paper',
                text=text, showarrow=False,
                align='right', bgcolor="rgba(255,255,255,0.9)",
                bordercolor=color, borderwidth=1, borderpad=4,
                font=dict(size=11, color="#1F2937")
            )
        )
        
    if annotations:
        fig.update_layout(annotations=annotations)
        
    return fig

def build_plotly_chart(df: pd.DataFrame, spec: VisualSpec, stat_results: Optional[Dict[str, Any]] = None) -> Optional[go.Figure]:
    """
    Main visualization entry-point. Deterministically maps and routes 
    result datasets to their ideal Plotly visualization based on VisualSpec.
    
    Args:
        df: Aggregate result dataset returned by analysis calculations.
        spec: VisualSpec configuration.
        stat_results: Optional dictionary containing deterministic statistical test results.
        
    Returns:
        Plotly Figure object styled for executive layouts.
    """
    if not spec.is_visual_requested or spec.chart_type is None:
        logger.info("Visual spec indicates no chart was requested.")
        return None
        
    if df is None or df.empty:
        logger.warning("Empty dataframe passed. Cannot render chart.")
        return None
        
    # Reset index if column headers reside in DataFrame Index ( groupby outputs )
    if spec.x_column and spec.x_column not in df.columns and spec.x_column == df.index.name:
        df = df.reset_index()
    elif spec.y_column and spec.y_column not in df.columns and spec.y_column == df.index.name:
        df = df.reset_index()
        
    # Standardize column targets
    x = spec.x_column
    y = spec.y_column
    group = spec.group_column
    
    # 1. Fallback mapping validations
    if x and x not in df.columns:
        for c in df.columns:
            if str(c).lower() == x.lower():
                x = c
                break
    if y and y not in df.columns:
        for c in df.columns:
            if str(c).lower() == y.lower():
                y = c
                break
    if group and group not in df.columns:
        for c in df.columns:
            if str(c).lower() == group.lower():
                group = c
                break
        else:
            group = None
            
    # 2. Select logical column targets if completely missing
    if (x is None or x not in df.columns) and len(df.columns) > 0:
        x = df.columns[0]
    if (y is None or y not in df.columns) and len(df.columns) > 1:
        y = df.columns[1]
    elif (y is None or y not in df.columns) and len(df.columns) > 0:
        y = df.columns[0]
        
    chart_type = spec.chart_type.lower()
    title = spec.title or f"{chart_type.capitalize()} Analysis representation"
    
    logger.info(f"Rendering visualization engine type '{chart_type}' on active columns (x={x}, y={y})...")
    
    try:
        fig = None
        if chart_type == "line":
            fig = plot_line_chart(df, x, y, group, title)
        elif chart_type == "bar":
            barmode = spec.barmode or "group"
            orientation = spec.orientation or "v"
            fig = plot_bar_chart(df, x, y, group, barmode, orientation, title)
        elif chart_type == "heatmap":
            z_val = y
            y_col = group if group else (df.columns[2] if len(df.columns) >= 3 else "")
            if y_col and y_col in df.columns:
                fig = plot_heatmap_chart(df, x, y_col, z_val, title)
            else:
                fig = plot_heatmap_chart(df, x, y, z_val, title)
        elif chart_type == "histogram":
            fig = plot_histogram_chart(df, x, y, group, title)
        elif chart_type == "pie":
            fig = plot_pie_chart(df, x, y, title)
        else:
            fig = px.scatter(df, x=x, y=y, color=group, color_discrete_sequence=EXECUTIVE_PALETTE)
            fig = apply_executive_styling(fig, title)
            
        if fig and stat_results:
            fig = apply_statistical_overlays(fig, stat_results)
            
        return fig
            
    except Exception as e:
        logger.error(f"Failed to render automatic Plotly figure: {str(e)}")
        return None
