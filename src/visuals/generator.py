import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List, Union
from src.llm.schemas import VisualSpec
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXECUTIVE_PALETTE = [
    "#6366F1",  # Indigo
    "#06B6D4",  # Cyan  
    "#10B981",  # Emerald
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#EC4899",  # Pink
    "#8B5CF6",  # Violet
    "#14B8A6",  # Teal
    "#F97316",  # Orange
    "#3B82F6",  # Blue
]

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
    df: pd.DataFrame, x: str, y: Optional[str] = None, group: Optional[str] = None, title: str = "", nbins: Optional[int] = None
) -> go.Figure:
    """Builds an executive-grade distribution histogram."""
    fig = px.histogram(
        df, x=x, y=y, color=group,
        color_discrete_sequence=EXECUTIVE_PALETTE,
        opacity=0.85,
        nbins=nbins or 20
    )
    fig.update_layout(bargap=0.05)
    return apply_executive_styling(fig, title)

def plot_box_chart(
    df: pd.DataFrame, x: str, y: Optional[str] = None, group: Optional[str] = None, title: str = ""
) -> go.Figure:
    """Builds a box plot for distribution comparisons across groups."""
    fig = px.box(
        df, x=x, y=y,
        color=group if group and group in df.columns else None,
        color_discrete_sequence=EXECUTIVE_PALETTE,
        title=title or "Distribution Comparison"
    )
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

    # stat_results is keyed by test name (e.g. "compute_correlations", possibly
    # suffixed like "analyze_distribution_2" when a test runs on several columns),
    # so match each result by its structure rather than by an exact key. We render
    # at most one badge of each kind to avoid clutter.
    corr_done = trend_done = comp_done = False
    for res in stat_results.values():
        if not isinstance(res, dict) or "error" in res or "skipped" in res:
            continue

        # 1. Correlation: strongest significant pair from compute_correlations
        if not corr_done and res.get("significant_pairs"):
            top = res["significant_pairs"][0]
            text = (f"<b>{res.get('method', 'Correlation').title()}</b>: "
                    f"{top.get('correlation', 0):.2f}<br><b>p-value</b>: {top.get('p_value', 1.0):.4f}")
            annotations.append(
                dict(
                    x=0.02, y=0.98, xref='paper', yref='paper',
                    text=text, showarrow=False,
                    align='left', bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#10B981", borderwidth=1, borderpad=4,
                    font=dict(size=11, color="#065F46")
                )
            )
            corr_done = True

        # 2. Trend: regression equation badge from analyze_trend
        if not trend_done and "slope" in res and "intercept" in res:
            eq = (f"<b>Trend (OLS)</b>: y = {res['slope']:.3f}x + {res['intercept']:.3f}"
                  f"<br><b>R²</b>: {res.get('r_squared', 0):.3f}")
            annotations.append(
                dict(
                    x=0.02, y=0.88, xref='paper', yref='paper',
                    text=eq, showarrow=False,
                    align='left', bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#3B82F6", borderwidth=1, borderpad=4,
                    font=dict(size=11, color="#1E40AF")
                )
            )
            trend_done = True

        # 3. Comparison testing (t-test / ANOVA / Chi-Square) from compare_groups
        if not comp_done and "test_name" in res and "p_value" in res:
            p_val = res.get("p_value", 1.0)
            color = "#10B981" if res.get("is_significant", False) else "#94A3B8"
            text = f"<b>{res['test_name']}</b><br>p-value: {p_val:.4g}"
            annotations.append(
                dict(
                    x=0.98, y=0.98, xref='paper', yref='paper',
                    text=text, showarrow=False,
                    align='right', bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=color, borderwidth=1, borderpad=4,
                    font=dict(size=11, color="#1F2937")
                )
            )
            comp_done = True

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
            nbins_val = getattr(spec, 'nbins', None) if hasattr(spec, 'nbins') else (spec.get('nbins') if isinstance(spec, dict) else None)
            fig = plot_histogram_chart(df, x, y, group, title, nbins=nbins_val)
        elif chart_type == "box":
            fig = plot_box_chart(df, x, y, group, title)
        elif chart_type == "pie":
            fig = plot_pie_chart(df, x, y, title)
        else:
            fig = px.scatter(df, x=x, y=y, color=group, color_discrete_sequence=EXECUTIVE_PALETTE)
            fig = apply_executive_styling(fig, title)
            
        # Apply enhanced spec features
        if fig:
            # x_label / y_label support
            x_label = getattr(spec, 'x_label', None) if hasattr(spec, 'x_label') else (spec.get('x_label') if isinstance(spec, dict) else None)
            y_label = getattr(spec, 'y_label', None) if hasattr(spec, 'y_label') else (spec.get('y_label') if isinstance(spec, dict) else None)
            if x_label:
                fig.update_xaxes(title_text=x_label)
            if y_label:
                fig.update_yaxes(title_text=y_label)

            # Percentage formatting: when the value axis represents a percentage
            # (agent labels it "% ..." or the column is named pct_*/percent*), add a
            # "%" suffix to the axis ticks and the value annotations.
            y_name = str(y).lower() if y else ""
            is_pct = (bool(y_label) and "%" in str(y_label)) or "pct" in y_name or "percent" in y_name

            show_values = getattr(spec, 'show_values', False) if hasattr(spec, 'show_values') else (spec.get('show_values', False) if isinstance(spec, dict) else False)

            if chart_type == "bar":
                orientation = (getattr(spec, 'orientation', None) if hasattr(spec, 'orientation') else None) or "v"
                val_axis = "x" if orientation == "h" else "y"
                if is_pct:
                    (fig.update_xaxes if val_axis == "x" else fig.update_yaxes)(ticksuffix="%")
                if show_values:
                    fig.update_traces(
                        textposition='outside',
                        texttemplate='%{' + val_axis + ':.1f}' + ('%' if is_pct else '')
                    )
            elif chart_type == "line" and is_pct:
                fig.update_yaxes(ticksuffix="%")
                
            # reference_line support
            ref_line = getattr(spec, 'reference_line', None) if hasattr(spec, 'reference_line') else (spec.get('reference_line') if isinstance(spec, dict) else None)
            if ref_line is not None:
                fig.add_hline(y=ref_line, line_dash='dash', line_color='#EF4444', annotation_text='Reference', annotation_position='top left')
            
        if fig and stat_results:
            fig = apply_statistical_overlays(fig, stat_results)
            
        return fig
            
    except Exception as e:
        logger.error(f"Failed to render automatic Plotly figure: {str(e)}")
        return None
