"""
GA TANF Dashboard (modular)
- Single CSV source: data/tanf_with_census.csv
- GeoJSON: data/ga_counties.geojson
- Uses FIPS from CSV for GEOID mapping; county names from GeoJSON for labels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go


# ======================
# Config / Paths
# ======================

GEOJSON_PATH = Path("data/ga_counties.geojson")
TANF_CSV     = Path("data/tanf_with_census.csv")
COI_CSV      = Path("data/child_opportunity_index_georgia_filtered.csv")
CLUSTER_CSV  = Path("data/tanf_clustered.csv")


# ======================
# Utilities
# ======================

def clean_county_name(s: pd.Series) -> pd.Series:
    """Normalize county names for consistent matching/labeling."""
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace(r"\bCounty\b", "", regex=True)  # drop literal "County"
         .str.strip()
         .str.title()
    )


def coerce_num(x):
    """Safely coerce to float; return NaN on failure."""
    try:
        return float(x)
    except Exception:
        return np.nan


# ======================
# Data Loaders
# ======================

@lru_cache(maxsize=4)
def load_geojson(path: Path) -> Tuple[dict, Dict[str, str], Dict[str, str]]:
    """
    Load GA counties GeoJSON and build NAME<->GEOID maps.
    Returns:
        geojson, name_to_geoid, geoid_to_name
    """
    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    name_to_geoid, geoid_to_name = {}, {}
    for feat in geojson["features"]:
        props = feat.get("properties", {})
        nm = str(props.get("NAME", "")).strip()
        geoid = str(props.get("GEOID", "")).strip()
        if nm and geoid:
            nm_clean = clean_county_name(pd.Series([nm])).iloc[0]
            name_to_geoid[nm_clean] = geoid
            geoid_to_name[geoid] = nm_clean

    return geojson, name_to_geoid, geoid_to_name


def load_dataset(csv_path: Path,
                 name_to_geoid: Dict[str, str]) -> pd.DataFrame:
    """
    Load the single TANF-with-census CSV and prepare fields/metrics.
    Expects columns:
        state, county_name, year, Recipients, Black Rec., Black Children, Black Families,
        fips, state_fips, county_fips, total_population, black_population,
        black_families_poverty, black_children_poverty, location_name, state_name
    """
    df = pd.read_csv(csv_path, dtype=str)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    required_cols = {
        "state", "county_name", "year",
        "Recipients", "Black Rec.", "Black Children", "Black Families",
        "fips", "state_fips", "county_fips",
        "total_population", "black_population",
        "black_families_poverty", "black_children_poverty",
        "location_name", "state_name",
    }
    missing = required_cols.difference(set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Normalize
    df["county_name"] = clean_county_name(df["county_name"])
    df["year"] = df["year"].astype(str)
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    for c in [
        "Recipients", "Black Rec.", "Black Children", "Black Families",
        "total_population", "black_population",
        "black_families_poverty", "black_children_poverty",
    ]:
        df[c] = df[c].map(coerce_num)

    # Mapping key for the choropleth
    df["GEOID"] = df["fips"]
    # Fallback by name if any fips missing
    miss = df["GEOID"].isna() | (df["GEOID"] == "")
    df.loc[miss, "GEOID"] = df.loc[miss, "county_name"].map(name_to_geoid)

    # One row per county-year for mapping/trends (no aggregation across years)
    df = (
        df.sort_values(["county_name", "year"])
          .drop_duplicates(subset=["county_name", "year"], keep="first")
          .reset_index(drop=True)
    )

    # Derived metrics
    df["rate_pct"] = (df["Black Rec."] / df["Recipients"]) * 100
    df["black_over_blackpop_pct"] = (df["Black Rec."] / df["black_population"]) * 100
    # New: percent of black children in poverty who received TANF
    #       and percent of black families in poverty who received TANF
    df["children_poverty_pct"] = (df["Black Children"] / df["black_children_poverty"]) * 100
    df["families_poverty_pct"] = (df["Black Families"] / df["black_families_poverty"]) * 100
    for col in ["rate_pct", "black_over_blackpop_pct"]:
        df.loc[~np.isfinite(df[col]), col] = np.nan
    for col in ["children_poverty_pct", "families_poverty_pct"]:
        df.loc[~np.isfinite(df[col]), col] = np.nan

    # For sorting in line charts
    df["Year_num"] = pd.to_numeric(df["year"], errors="coerce")

    return df


def load_cluster_dataset(csv_path: Path,
                        name_to_geoid: Dict[str, str]) -> pd.DataFrame:
    """
    Load the TANF clustering CSV.
    Expects columns: county_name, 2017, 2018, 2019, 2020, 2021, cluster
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)
    
    # Clean county names to match the main dataset
    df["county_name"] = clean_county_name(df["county_name"])
    
    # Create GEOID mapping - try name-based mapping first
    df["GEOID"] = df["county_name"].map(name_to_geoid)
    
    # Remove rows where we couldn't map the county
    df = df[df["GEOID"].notna()].copy()
    
    return df


def load_coi_dataset(csv_path: Path,
                     name_to_geoid: Dict[str, str]) -> pd.DataFrame:
    """
    Load the Child Opportunity Index CSV and prepare fields/metrics.
    Expects columns:
        state_fips, state_usps, state_name, county_fips, county_name, year, 
        z_COI_stt (main metric), plus other COI-related metrics
    """
    df = pd.read_csv(csv_path, dtype={'county_fips': str, 'state_fips': str})
    df.rename(columns={c: c.strip() for c in df.columns}, inplace=True)

    # Data is already filtered for Georgia
    
    # Normalize county names
    df["county_name"] = clean_county_name(df["county_name"])
    df["year"] = df["year"].astype(str)
    
    # Create FIPS code - county_fips already contains the full 5-digit FIPS code
    df["fips"] = df["county_fips"].astype(str)
    
    # Mapping key for the choropleth
    df["GEOID"] = df["fips"]
    # Fallback by name if any fips missing
    miss = df["GEOID"].isna() | (df["GEOID"] == "")
    df.loc[miss, "GEOID"] = df.loc[miss, "county_name"].map(name_to_geoid)

    # One row per county-year (should already be this way)
    df = (
        df.sort_values(["county_name", "year"])
          .drop_duplicates(subset=["county_name", "year"], keep="first")
          .reset_index(drop=True)
    )

    # For sorting in line charts
    df["Year_num"] = pd.to_numeric(df["year"], errors="coerce")

    return df


@lru_cache(maxsize=1)
def metrics_config() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
        METRICS (label->column), METRIC_FMT (column->'%' or 'count')
    """
    metrics = {
        "Black Recipients as % of Total Recipients": "rate_pct",
        "TANF Black Participation Rate (% of Black Population)": "black_over_blackpop_pct",
        "Percentage Black Children in Poverty who received TANF": "children_poverty_pct",
        "Percentage Black Families in Poverty who received TANF": "families_poverty_pct",
        "Black TANF Recipients Count": "Black Rec.",
        "Total TANF Recipients": "Recipients",
        "Total Black Population": "black_population",
    }
    metric_fmt = {
        "rate_pct": "%",
        "black_over_blackpop_pct": "%",
        "children_poverty_pct": "%",
        "families_poverty_pct": "%",
    }
    return metrics, metric_fmt


@lru_cache(maxsize=1)
def coi_metrics_config() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
        COI_METRICS (label->column), COI_METRIC_FMT (column->format type)
    """
    metrics = {
        "Overall Child Opportunity Index (COI) Z-Score": "z_COI_stt",
        "COI Z-Score - Education Domain": "z_ED_stt",
        "COI Z-Score - Health & Environment Domain": "z_HE_stt",
        "COI Z-Score - Social & Economic Domain": "z_SE_stt",
        "Population": "pop",
    }
    metric_fmt = {
        "z_COI_stt": "z-score",
        "z_ED_stt": "z-score",
        "z_HE_stt": "z-score",
        "z_SE_stt": "z-score",
        "pop": "count",
    }
    return metrics, metric_fmt


@lru_cache(maxsize=1)
def get_metric_descriptions() -> Dict[str, str]:
    """
    Returns descriptions for TANF metrics explaining what each metric means.
    """
    return {
        "rate_pct": "Percentage of all TANF recipients in a county who are Black. This shows the racial composition of TANF beneficiaries.",
        "black_over_blackpop_pct": "Percentage of the Black population in a county that receives TANF benefits. This measures TANF participation rates within the Black community.",
        "children_poverty_pct": "Percentage of Black children living in poverty who receive TANF benefits. This shows how well TANF reaches Black children in need.",
        "families_poverty_pct": "Percentage of Black families living in poverty who receive TANF benefits. This measures TANF coverage among Black families facing economic hardship.",
        "Black Rec.": "Total number of Black individuals receiving TANF benefits in the county. This is the raw count of Black TANF recipients.",
        "Recipients": "Total number of individuals receiving TANF benefits in the county, regardless of race. This shows overall TANF caseload size.",
        "black_population": "Total Black population in the county according to census data. This provides demographic context for understanding TANF participation rates."
    }


@lru_cache(maxsize=1)
def get_coi_metric_descriptions() -> Dict[str, str]:
    """
    Returns descriptions for Child Opportunity Index metrics explaining what each metric means.
    """
    return {
        "z_COI_stt": "Overall measure of child opportunity in the county relative to the state average. Positive scores indicate better opportunities, negative scores indicate fewer opportunities.",
        "z_ED_stt": "Education domain score measuring access to quality schools, school readiness programs, and educational resources relative to the state average.",
        "z_HE_stt": "Health & Environment score measuring access to healthcare, environmental quality, and health-promoting resources relative to the state average.",
        "z_SE_stt": "Social & Economic score measuring family economic security, neighborhood safety, and social capital relative to the state average.",
        "pop": "Total population of the county. This provides demographic context for understanding the Child Opportunity Index scores and their impact."
    }


def get_county_rankings(
    df: pd.DataFrame,
    geoid_to_name: Dict[str, str],
    year: str,
    metric: str,
    metric_fmt: Dict[str, str],
    n_counties: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Get top and bottom performing counties for a given metric and year.
    
    Args:
        df: DataFrame with county data
        geoid_to_name: Mapping from GEOID to county name
        year: Year to filter data
        metric: Metric column name
        metric_fmt: Formatting information for metrics
        n_counties: Number of top/bottom counties to return
    
    Returns:
        Tuple of (top_counties, bottom_counties) lists with dicts containing 'county' and 'value'
    """
    # Filter data for the specified year
    dfy = df[df["year"] == str(year)].copy()
    
    # Remove rows with missing values for the metric
    dfy = dfy[dfy[metric].notna()].copy()
    
    if dfy.empty:
        return [], []
    
    # Add county names
    dfy["County"] = dfy["GEOID"].map(geoid_to_name).fillna("Unknown")
    
    # Sort by metric value
    dfy_sorted = dfy.sort_values(metric, ascending=False)
    
    # Get top and bottom counties
    top_counties = []
    bottom_counties = []
    
    # Format values based on metric type
    kind = metric_fmt.get(metric, "count")
    
    def format_value(val):
        if pd.isna(val):
            return "N/A"
        if kind == "%":
            return f"{val:.1f}%"
        elif kind == "z-score":
            return f"{val:.2f}"
        else:
            return f"{int(val):,}"
    
    # Top performing counties
    for _, row in dfy_sorted.head(n_counties).iterrows():
        top_counties.append({
            'county': row['County'],
            'value': format_value(row[metric])
        })
    
    # Bottom performing counties  
    for _, row in dfy_sorted.tail(n_counties).iterrows():
        bottom_counties.append({
            'county': row['County'],
            'value': format_value(row[metric])
        })
    
    # Reverse bottom counties so worst is first
    bottom_counties.reverse()
    
    return top_counties, bottom_counties


def create_ranking_display(counties: List[Dict], title: str, icon: str, is_top: bool = True) -> html.Div:
    """
    Create a formatted display for county rankings.
    
    Args:
        counties: List of dicts with 'county' and 'value' keys
        title: Title for the ranking section
        icon: Emoji icon to display
        is_top: Whether these are top performers (affects styling)
    
    Returns:
        HTML div with formatted county rankings
    """
    color_class = "top-performer" if is_top else "bottom-performer"
    
    if not counties:
        return html.Div([
            html.H4([icon, " ", title]),
            html.P("No data available", style={"color": "#6b7280", "fontSize": "14px", "margin": "0"})
        ], className=color_class)
    
    county_items = []
    for i, county_info in enumerate(counties, 1):
        county_items.append(
            html.Div([
                html.Span(f"{i}.", className="county-rank"),
                html.Span(county_info['county'], className="county-name"),
                html.Span(county_info['value'], className="county-value")
            ], className="county-ranking-item")
        )
    
    return html.Div([
        html.H4([icon, " ", title]),
        html.Div(county_items)
    ], className=color_class)


def calculate_state_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate state-wide averages for each year by aggregating county data.
    Uses appropriate aggregation methods for each metric type.
    
    Args:
        df: DataFrame with county-level data
    
    Returns:
        DataFrame with state averages by year
    """
    # Group by year and calculate state totals
    state_data = []
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year].copy()
        
        # Remove rows with missing data for core calculations
        year_data = year_data.dropna(subset=['Recipients', 'Black Rec.', 'black_population'])
        
        if len(year_data) == 0:
            continue
            
        # Sum totals across all counties
        total_recipients = year_data['Recipients'].sum()
        total_black_recipients = year_data['Black Rec.'].sum()
        total_black_population = year_data['black_population'].sum()
        
        # Sum additional metrics if available
        total_black_children = year_data['Black Children'].sum() if 'Black Children' in year_data.columns else np.nan
        total_black_families = year_data['Black Families'].sum() if 'Black Families' in year_data.columns else np.nan
        total_black_children_poverty = year_data['black_children_poverty'].sum() if 'black_children_poverty' in year_data.columns else np.nan
        total_black_families_poverty = year_data['black_families_poverty'].sum() if 'black_families_poverty' in year_data.columns else np.nan
        
        # Calculate state-level percentages using proper aggregation
        state_rate_pct = (total_black_recipients / total_recipients * 100) if total_recipients > 0 else np.nan
        state_black_over_blackpop_pct = (total_black_recipients / total_black_population * 100) if total_black_population > 0 else np.nan
        state_children_poverty_pct = (total_black_children / total_black_children_poverty * 100) if total_black_children_poverty > 0 else np.nan
        state_families_poverty_pct = (total_black_families / total_black_families_poverty * 100) if total_black_families_poverty > 0 else np.nan
        
        state_data.append({
            'year': year,
            'Recipients': total_recipients,
            'Black Rec.': total_black_recipients,
            'black_population': total_black_population,
            'Black Children': total_black_children,
            'Black Families': total_black_families,
            'black_children_poverty': total_black_children_poverty,
            'black_families_poverty': total_black_families_poverty,
            'rate_pct': state_rate_pct,
            'black_over_blackpop_pct': state_black_over_blackpop_pct,
            'children_poverty_pct': state_children_poverty_pct,
            'families_poverty_pct': state_families_poverty_pct,
            'Year_num': pd.to_numeric(year, errors='coerce')
        })
    
    return pd.DataFrame(state_data)


# ======================
# Figure Builders
# ======================

def make_map_figure(
    df: pd.DataFrame,
    geojson: dict,
    geoid_to_name: Dict[str, str],
    year: str,
    metric: str,
    metrics_label_map: Dict[str, str],
    metric_fmt: Dict[str, str],
    cluster_df: Optional[pd.DataFrame] = None,
    show_clusters: bool = False,
) -> Tuple[go.Figure, str]:
    """Build the choropleth map and the 'missing data' note."""
    
    # Check if we should show clusters instead of metric values
    if show_clusters and cluster_df is not None and metric == "families_poverty_pct":
        return make_cluster_map_figure(geojson, geoid_to_name, cluster_df)
    
    dfy = df[df["year"] == str(year)].copy()

    # Ensure all counties appear, even if missing for that year
    base = pd.DataFrame({"GEOID": list(geoid_to_name.keys())})
    keep_cols = [
        "GEOID", "county_name", "Recipients", "Black Rec.", "black_population",
        "rate_pct", "black_over_blackpop_pct", "children_poverty_pct", "families_poverty_pct"
    ]
    dfy = base.merge(dfy[keep_cols], on="GEOID", how="left")
    dfy["County"] = dfy["GEOID"].map(geoid_to_name)

    def fmt_count(v):
        return f"{int(v):,}" if pd.notna(v) else "N/A"

    def fmt_pct(v):
        return f"{v:.1f}%" if pd.notna(v) else "N/A"
    
    dfy["hover"] = (
        "<b style='color:#000; font-size:14px'>" + dfy["County"].fillna("Unknown") + " County</b><br><br>" +
        "<b style='color:#1f2937'>Demographics:</b><br>" +
        "<span style='color:#111'>Black Population: <b>" + dfy["black_population"].map(fmt_count) + "</b></span><br>" +
        "<span style='color:#111'>Total Recipients: <b>" + dfy["Recipients"].map(fmt_count) + "</b></span><br>" +
        "<span style='color:#111'>Black Recipients: <b>" + dfy["Black Rec."].map(fmt_count) + "</b></span><br><br>" +
        "<b style='color:#1f2937'>Metrics:</b><br>" +
        "<span style='color:#111'>% Black of Recipients: <b style='color:#1f2937'>" + dfy["rate_pct"].map(fmt_pct) + "</b></span><br>" +
        "<span style='color:#111'>% Black Recipients of Black Pop: <b style='color:#1f2937'>" + dfy["black_over_blackpop_pct"].map(fmt_pct) + "</b></span><br>" +
        "<span style='color:#111'>% Black Children in Poverty (TANF): <b style='color:#1f2937'>" + dfy["children_poverty_pct"].map(fmt_pct) + "</b></span><br>" +
        "<span style='color:#111'>% Black Families in Poverty (TANF): <b style='color:#1f2937'>" + dfy["families_poverty_pct"].map(fmt_pct) + "</b></span>"
    )

    # Split counties with and without the chosen metric so missing ones can be shown in gray
    df_have = dfy[dfy[metric].notna()].copy()
    df_missing = dfy[dfy[metric].isna()].copy()
    # Create a simple hover for missing counties: show county name and mention missing data
    if not df_missing.empty:
        df_missing["hover_missing"] = (
            "<b style='color:#000; font-size:14px'>" + df_missing["County"].fillna("Unknown") + " County</b><br><br>" +
            "<b style='color:#dc2626'>⚠️ Data: Insufficient/Missing for this year</b>"
        )

    n_total = len(dfy)
    n_missing = dfy[metric].isna().sum()
    note = f"{n_missing}/{n_total} counties missing."

    kind = metric_fmt.get(metric, "count")
    if kind == "%":
        vmax = float(np.nanpercentile(df_have[metric], 98)) if df_have[metric].notna().any() else 100.0
        color_scale = "Viridis"
    else:
        vmax = float(np.nanpercentile(df_have[metric], 98)) if df_have[metric].notna().any() else 1.0
        color_scale = "Blues"
    vmax = max(vmax, 1.0)

    label = [k for k, v in metrics_label_map.items() if v == metric][0]

    # Keep the label text as-is for the rotated annotation
    vertical_label = str(label)

    # Main choropleth for counties with data
    fig = px.choropleth_mapbox(
        df_have if not df_have.empty else pd.DataFrame({"GEOID": []}),
        geojson=geojson,
        locations="GEOID",
        featureidkey="properties.GEOID",
        color=metric if not df_have.empty else None,
        color_continuous_scale=color_scale,
        range_color=(0, vmax),
        hover_name="County",
        hover_data=None,
        labels={metric: label},
        mapbox_style="carto-positron",
        center={"lat": 32.5, "lon": -83.3},
        zoom=5.7,
        opacity=0.86,
        
    )

    # Add a gray layer for missing counties (so they appear on the map and have hover info)
    if not df_missing.empty:
        # Use a Choroplethmapbox trace with a uniform z so it shows as gray
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=df_missing["GEOID"],
            z=[0] * len(df_missing),
            featureidkey="properties.GEOID",
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            showscale=False,
            marker_opacity=0.86,
            marker_line_width=0,
            # customdata will be set below as a 2D array for Plotly
        ))

    # Draw county/state borders as a GeoJSON line layer on the mapbox
    label = [k for k, v in metrics_label_map.items() if v == metric][0]

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title=dict(
                text=label,
                side="right",
                font=dict(size=14, family="Inter")
            ),
            titleside="right",
            title_font=dict(size=14, family="Inter"),
            tickfont=dict(size=12, family="Inter"),
            len=0.7,
            thickness=15,
            x=1.02
        ) if not df_have.empty else {},
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e5e7eb",
            font_size=13,
            font_family="Inter"
        ),
        mapbox={
            "style": "carto-positron",
            "center": {"lat": 32.5, "lon": -83.3},
            "zoom": 5.7,
            "layers": [
                {
                    "sourcetype": "geojson",
                    "source": geojson,
                    "type": "line",
                    "color": "#6b7280",
                    "line": {"width": 0.8},
                }
            ],
        },
    )

    # Set customdata and hovertemplate explicitly for both traces (main and missing)
    # Plotly expects customdata to be an array of arrays (2D). We'll set customdata accordingly.
    try:
        if not df_have.empty:
            main_custom = df_have["hover"].fillna("County: NA<br>Data: Missing").to_numpy().reshape(-1, 1).tolist()
            fig.data[0].customdata = main_custom
            fig.data[0].hovertemplate = "%{customdata[0]}<extra></extra>"
        if not df_missing.empty:
            # missing trace was appended as the last trace
            miss_custom = df_missing["hover_missing"].to_numpy().reshape(-1, 1).tolist()
            fig.data[-1].customdata = miss_custom
            fig.data[-1].hovertemplate = "%{customdata[0]}<extra></extra>"
    except Exception:
        # Fallback: do nothing if assignment fails
        pass

    return fig, note


def make_cluster_map_figure(
    geojson: dict,
    geoid_to_name: Dict[str, str],
    cluster_df: pd.DataFrame,
) -> Tuple[go.Figure, str]:
    """Build the cluster choropleth map."""
    
    # Ensure all counties appear
    base = pd.DataFrame({"GEOID": list(geoid_to_name.keys())})
    dfy = base.merge(cluster_df[["GEOID", "cluster"]], on="GEOID", how="left")
    dfy["County"] = dfy["GEOID"].map(geoid_to_name)
    
    # Create hover text for clusters
    def fmt_cluster(c):
        if pd.isna(c):
            return "No cluster data"
        else:
            return f"Cluster {int(c)}"
    
    dfy["hover"] = (
        "<b style='color:#000; font-size:14px'>" + dfy["County"].fillna("Unknown") + " County</b><br><br>" +
        "<b style='color:#1f2937'>Trend Analysis:</b><br>" +
        "<span style='color:#111'>📊 " + dfy["cluster"].map(fmt_cluster) + "</span><br>" +
        "<span style='color:#666; font-size:12px'>Based on 2017-2021 TANF coverage trends</span>"
    )
    
    # Split counties with and without cluster data
    df_have = dfy[dfy["cluster"].notna()].copy()
    df_missing = dfy[dfy["cluster"].isna()].copy()
    
    if not df_missing.empty:
        df_missing["hover_missing"] = (
            "<b style='color:#000; font-size:14px'>" + df_missing["County"].fillna("Unknown") + " County</b><br><br>" +
            "<b style='color:#dc2626'>⚠️ No cluster data available</b>"
        )
    
    n_total = len(dfy)
    n_missing = dfy["cluster"].isna().sum()
    note = f"{n_missing}/{n_total} counties missing cluster data."
    
    # Use categorical colors for clusters
    cluster_colors = {1: "#3b82f6", 2: "#ef4444", 3: "#f59e0b"}  # Blue, Red, Orange
    
    fig = go.Figure()
    
    # Add each cluster as a separate trace for better legend control
    for cluster_id in sorted(df_have["cluster"].dropna().unique()):
        cluster_data = df_have[df_have["cluster"] == cluster_id]
        if not cluster_data.empty:
            fig.add_trace(go.Choroplethmapbox(
                geojson=geojson,
                locations=cluster_data["GEOID"],
                z=[cluster_id] * len(cluster_data),  # Use cluster ID as z value
                featureidkey="properties.GEOID",
                colorscale=[[0, cluster_colors.get(cluster_id, "#666666")], 
                           [1, cluster_colors.get(cluster_id, "#666666")]],
                showscale=False,
                marker_opacity=0.86,
                marker_line_width=0.8,
                marker_line_color="#ffffff",
                name=f"Cluster {int(cluster_id)}",
                customdata=cluster_data["hover"].values.reshape(-1, 1),
                hovertemplate="%{customdata[0]}<extra></extra>",
            ))
    
    # Add missing counties in gray
    if not df_missing.empty:
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=df_missing["GEOID"],
            z=[0] * len(df_missing),
            featureidkey="properties.GEOID",
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            showscale=False,
            marker_opacity=0.86,
            marker_line_width=0.8,
            marker_line_color="#ffffff",
            name="No Data",
            customdata=df_missing["hover_missing"].values.reshape(-1, 1),
            hovertemplate="%{customdata[0]}<extra></extra>",
        ))
    
    # Optimized Georgia bounds
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=12, family="Inter")
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e5e7eb",
            font_size=13,
            font_family="Inter"
        ),
        mapbox={
            "style": "carto-positron",
            "center": {"lat": 32.5, "lon": -83.3},
            "zoom": 6.2,  # Increased zoom for better Georgia focus
            "layers": [
                {
                    "sourcetype": "geojson",
                    "source": geojson,
                    "type": "line",
                    "color": "#6b7280",
                    "line": {"width": 0.8},
                }
            ],
        },
    )
    
    return fig, note


def make_empty_trend(title_label: str) -> go.Figure:
    """Placeholder trend chart when nothing is selected."""
    fig = go.Figure()
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis_title="Year",
        yaxis_title=title_label,
        annotations=[dict(text="Click a county or choose one from the dropdown to see its trend.",
                          x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False)]
    )
    return fig


def make_trend_figure(
    df: pd.DataFrame,
    sel_geoid: str,
    sel_name: str,
    metric: str,
    metrics_label_map: Dict[str, str],
    metric_fmt: Dict[str, str],
    state_averages: pd.DataFrame,
) -> Tuple[go.Figure, str]:
    """Build the line chart for a selected county."""
    title_label = [k for k, v in metrics_label_map.items() if v == metric][0]
    if not sel_geoid:
        return make_empty_trend(title_label), ""

    dfc = df[df["GEOID"] == sel_geoid].copy().sort_values("Year_num")
    
    # Use pre-calculated state averages (passed as parameter)
    
    # Create a shorter y-axis label for better space utilization
    def shorten_label(label):
        # Create abbreviated versions of long labels
        label_mapping = {
            "Black Recipients as % of Total Recipients": "% Black Recipients",
            "TANF Black Participation Rate (% of Black Population)": "TANF Participation Rate (%)",
            "Percentage Black Children in Poverty who received TANF": "% Black Children in Poverty (TANF)",
            "Percentage Black Families in Poverty who received TANF": "% Black Families in Poverty (TANF)",
        }
        return label_mapping.get(label, label)
    
    short_label = shorten_label(title_label)
    
    # Create figure with both county and state data
    fig = go.Figure()
    
    # Add county trend line
    fig.add_trace(go.Scatter(
        x=dfc["year"],
        y=dfc[metric],
        mode='lines+markers',
        name=f'{sel_name} County',
        line=dict(width=3, color='#2563eb'),
        marker=dict(size=8, color='#2563eb', line=dict(width=2, color='#ffffff')),
    ))
    
    # Add state average baseline if we have state data
    if not state_averages.empty and metric in state_averages.columns:
        state_data = state_averages.dropna(subset=[metric])
        if not state_data.empty:
            fig.add_trace(go.Scatter(
                x=state_data["year"],
                y=state_data[metric],
                mode='lines',
                name='Georgia Average',
                line=dict(width=2, color='#dc2626', dash='dash'),
                opacity=0.7,
            ))
    # Update layout
    fig.update_layout(
        margin={"r": 20, "t": 20, "l": 80, "b": 50},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        xaxis=dict(
            title="Year",
            showgrid=True,
            gridwidth=1,
            gridcolor="#f3f4f6",
            title_font=dict(size=14, family="Inter"),
            title_standoff=25,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title=short_label,
            showgrid=True,
            gridwidth=1,
            gridcolor="#f3f4f6",
            title_font=dict(size=13, family="Inter"),
            title_standoff=40,
            tickfont=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e5e7eb",
            font_size=13,
            font_family="Inter"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, family="Inter")
        )
    )

    # Update hover templates based on metric type
    kind = metric_fmt.get(metric, "count")
    if kind == "%":
        # Update county trace hover
        fig.update_traces(
            hovertemplate="<b>📅 Year: %{x}</b><br><b>📊 %{fullData.name}: <span style='color:#2563eb'>%{y:.2f}%</span></b><extra></extra>",
            selector=dict(name=f'{sel_name} County')
        )
        # Update state average trace hover if it exists
        if len(fig.data) > 1:
            fig.update_traces(
                hovertemplate="<b>📅 Year: %{x}</b><br><b>📊 %{fullData.name}: <span style='color:#1f2937'>%{y:.2f}%</span></b><extra></extra>",
                selector=dict(name='Georgia Average')
            )
    else:
        # Update county trace hover
        fig.update_traces(
            hovertemplate="<b>📅 Year: %{x}</b><br><b>📊 %{fullData.name}: <span style='color:#2563eb'>%{y:,.0f}</span></b><extra></extra>",
            selector=dict(name=f'{sel_name} County')
        )
        # Update state average trace hover if it exists
        if len(fig.data) > 1:
            fig.update_traces(
                hovertemplate="<b>📅 Year: %{x}</b><br><b>📊 %{fullData.name}: <span style='color:#1f2937'>%{y:,.0f}</span></b><extra></extra>",
                selector=dict(name='Georgia Average')
            )

    return fig, f"Trend – {sel_name} County ({title_label})"


def make_coi_map_figure(
    df: pd.DataFrame,
    geojson: dict,
    geoid_to_name: Dict[str, str],
    year: str,
    metric: str,
    metrics_label_map: Dict[str, str],
    metric_fmt: Dict[str, str],
) -> Tuple[go.Figure, str]:
    """Build the choropleth map for COI data and the 'missing data' note."""
    dfy = df[df["year"] == str(year)].copy()

    # Ensure all counties appear, even if missing for that year
    base = pd.DataFrame({"GEOID": list(geoid_to_name.keys())})
    keep_cols = [
        "GEOID", "county_name", "pop", "z_COI_stt", "z_ED_stt", "z_HE_stt", "z_SE_stt"
    ]
    dfy = base.merge(dfy[keep_cols], on="GEOID", how="left")
    dfy["County"] = dfy["GEOID"].map(geoid_to_name)

    def fmt_count(v):
        return f"{int(v):,}" if pd.notna(v) else "N/A"

    def fmt_zscore(v):
        return f"{v:.3f}" if pd.notna(v) else "N/A"
    
    dfy["hover"] = (
        "<b style='color:#000; font-size:14px'>" + dfy["County"].fillna("Unknown") + " County</b><br><br>" +
        "<b style='color:#1f2937'>Demographics:</b><br>" +
        "<span style='color:#111'> Population: <b>" + dfy["pop"].map(fmt_count) + "</b></span><br><br>" +
        "<b style='color:#1f2937'>Child Opportunity Index Scores:</b><br>" +
        "<span style='color:#111'> Overall COI: <b style='color:#dc2626'>" + dfy["z_COI_stt"].map(fmt_zscore) + "</b></span><br>" +
        "<span style='color:#111'> Education: <b style='color:#dc2626'>" + dfy["z_ED_stt"].map(fmt_zscore) + "</b></span><br>" +
        "<span style='color:#111'> Health & Environment: <b style='color:#dc2626'>" + dfy["z_HE_stt"].map(fmt_zscore) + "</b></span><br>" +
        "<span style='color:#111'> Social & Economic: <b style='color:#dc2626'>" + dfy["z_SE_stt"].map(fmt_zscore) + "</b></span>"
    )

    # Split counties with and without the chosen metric so missing ones can be shown in gray
    df_have = dfy[dfy[metric].notna()].copy()
    df_missing = dfy[dfy[metric].isna()].copy()
    # Create a simple hover for missing counties: show county name and mention missing data
    if not df_missing.empty:
        df_missing["hover_missing"] = (
            "<b style='color:#000; font-size:14px'>" + df_missing["County"].fillna("Unknown") + " County</b><br><br>" +
            "<b style='color:#dc2626'>⚠️ Data: Insufficient/Missing for this year</b>"
        )

    n_total = len(dfy)
    n_missing = dfy[metric].isna().sum()
    note = f"{n_missing}/{n_total} counties missing."

    kind = metric_fmt.get(metric, "z-score")
    if kind == "z-score":
        # For z-scores, use a diverging color scale centered at 0
        # Determine symmetric range
        if df_have[metric].notna().any():
            vmax = float(np.nanpercentile(np.abs(df_have[metric]), 95))
            vmax = max(vmax, 0.5)  # Minimum range for visibility
        else:
            vmax = 2.0
        # Custom diverging scale: red (low) -> white (mid) -> blue (high)
        color_scale = [[0.0, 'rgb(214,39,40)'], [0.5, 'rgb(255,255,255)'], [1.0, 'rgb(31,119,180)']]
        vmin = -vmax
    elif kind == "count":
        vmin = 0
        vmax = float(np.nanpercentile(df_have[metric], 98)) if df_have[metric].notna().any() else 1.0
        vmax = max(vmax, 1.0)
        color_scale = "Blues"
    else:
        vmin = 0
        vmax = float(np.nanpercentile(df_have[metric], 98)) if df_have[metric].notna().any() else 100.0
        color_scale = "Viridis"

    label = [k for k, v in metrics_label_map.items() if v == metric][0]

    # Main choropleth for counties with data
    if not df_have.empty:
        fig = px.choropleth_mapbox(
            df_have,
            geojson=geojson,
            locations="GEOID",
            featureidkey="properties.GEOID",
            color=metric,
            color_continuous_scale=color_scale,
            range_color=(vmin, vmax) if kind == "z-score" else (0, vmax),
            hover_name="County",
            hover_data=None,
            labels={metric: label},
            mapbox_style="carto-positron",
            center={"lat": 32.5, "lon": -83.3},
            zoom=5.7,
            opacity=0.86,
        )
    else:
        # Create empty map when no data available
        fig = go.Figure(go.Choroplethmapbox())
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_center={"lat": 32.5, "lon": -83.3},
            mapbox_zoom=5.7,
        )

    # Add a gray layer for missing counties (so they appear on the map and have hover info)
    if not df_missing.empty:
        # Use a Choroplethmapbox trace with a uniform z so it shows as gray
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson,
            locations=df_missing["GEOID"],
            z=[0] * len(df_missing),
            featureidkey="properties.GEOID",
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            showscale=False,
            marker_opacity=0.86,
            marker_line_width=0,
        ))

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(
            title=dict(
                text=label,
                side="right",
                font=dict(size=14, family="Inter")
            ),
            titleside="right",
            title_font=dict(size=14, family="Inter"),
            tickfont=dict(size=12, family="Inter"),
            len=0.7,
            thickness=15,
            x=1.02
        ) if not df_have.empty else {},
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e5e7eb",
            font_size=13,
            font_family="Inter"
        ),
        mapbox={
            "style": "carto-positron",
            "center": {"lat": 32.5, "lon": -83.3},
            "zoom": 5.7,
            "layers": [
                {
                    "sourcetype": "geojson",
                    "source": geojson,
                    "type": "line",
                    "color": "#6b7280",
                    "line": {"width": 0.8},
                }
            ],
        },
    )

    # Set customdata and hovertemplate explicitly for both traces (main and missing)
    try:
        if not df_have.empty:
            main_custom = df_have["hover"].fillna("County: NA<br>Data: Missing").to_numpy().reshape(-1, 1).tolist()
            fig.data[0].customdata = main_custom
            fig.data[0].hovertemplate = "%{customdata[0]}<extra></extra>"
        if not df_missing.empty:
            # missing trace was appended as the last trace
            miss_custom = df_missing["hover_missing"].to_numpy().reshape(-1, 1).tolist()
            fig.data[-1].customdata = miss_custom
            fig.data[-1].hovertemplate = "%{customdata[0]}<extra></extra>"
    except Exception:
        # Fallback: do nothing if assignment fails
        pass

    return fig, note


def make_coi_trend_figure(
    df: pd.DataFrame,
    sel_geoid: str,
    sel_name: str,
    metric: str,
    metrics_label_map: Dict[str, str],
    metric_fmt: Dict[str, str],
) -> Tuple[go.Figure, str]:
    """Build the COI line chart for a selected county."""
    title_label = [k for k, v in metrics_label_map.items() if v == metric][0]
    if not sel_geoid:
        return make_empty_trend(title_label), ""

    dfc = df[df["GEOID"] == sel_geoid].copy().sort_values("Year_num")
    
    # Create a shorter y-axis label for better space utilization
    def shorten_coi_label(label):
        # Create abbreviated versions of long labels
        label_mapping = {
            "Overall Child Opportunity Index (COI) Z-Score": "Overall COI Z-Score",
            "COI Z-Score - Education Domain": "Education Z-Score",
            "COI Z-Score - Health & Environment Domain": "Health & Environment Z-Score",
            "COI Z-Score - Social & Economic Domain": "Social & Economic Z-Score",
        }
        return label_mapping.get(label, label)
    
    short_label = shorten_coi_label(title_label)
    
    fig = px.line(dfc, x="year", y=metric, markers=True,
                  labels={metric: short_label, "year": "Year"})
    fig.update_layout(
        margin={"r": 20, "t": 20, "l": 80, "b": 50},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="#f3f4f6",
            title_font=dict(size=14, family="Inter"),
            title_standoff=25,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="#f3f4f6",
            title_font=dict(size=13, family="Inter"),
            title_standoff=40,
            tickfont=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e5e7eb",
            font_size=13,
            font_family="Inter"
        )
    )

    kind = metric_fmt.get(metric, "count")
    if kind == "z-score":
        fig.update_traces(
            hovertemplate="<b style='color:#000'>📅 Year: %{x}</b><br>" +
                         "<b style='color:#000'>📊 Z-Score: <span style='color:#dc2626; font-weight:bold'>%{y:.3f}</span></b><extra></extra>",
            line=dict(width=3, color='#2563eb'),
            marker=dict(size=8, color='#2563eb', line=dict(width=2, color='#ffffff'))
        )
        # Add a horizontal line at y=0 for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    elif kind == "count":
        fig.update_traces(
            hovertemplate="<b style='color:#000'>📅 Year: %{x}</b><br>" +
                         "<b style='color:#000'>📊 Count: <span style='color:#dc2626; font-weight:bold'>%{y:,.0f}</span></b><extra></extra>",
            line=dict(width=3, color='#2563eb'),
            marker=dict(size=8, color='#2563eb', line=dict(width=2, color='#ffffff'))
        )
    else:
        fig.update_traces(
            hovertemplate="<b style='color:#000'>📅 Year: %{x}</b><br>" +
                         "<b style='color:#000'>📊 Value: <span style='color:#dc2626; font-weight:bold'>%{y:.2f}</span></b><extra></extra>",
            line=dict(width=3, color='#2563eb'),
            marker=dict(size=8, color='#2563eb', line=dict(width=2, color='#ffffff'))
        )

    return fig, f"Trend – {sel_name} County ({title_label})"


# ======================
# App Factory
# ======================

def build_layout(
    app: dash.Dash,
    years: List[str],
    coi_years: List[str],
    geoid_to_name: Dict[str, str],
    metrics_label_map: Dict[str, str],
    coi_metrics_label_map: Dict[str, str]
) -> html.Div:
    """Construct the static Dash layout."""
    county_options = [
        {"label": f"{name} County", "value": geoid}
        for geoid, name in sorted(geoid_to_name.items(), key=lambda kv: kv[1])
    ]

    tanf_tab = html.Div([
        html.Div([
            html.H2("Georgia TANF Enrollment Trends", className="page-title", style={"marginBottom": "2px"}),
            html.P([
                "Temporary Assistance for Needy Families (TANF) is the monthly cash assistance program, with an employment services component, for low-income families with children under age 18, children of age 18 and attending school full-time, and pregnant women. ",
                html.A("Learn more", href="https://dfcs.georgia.gov/services/temporary-assistance-needy-families", target="_blank", style={"color": "#2563eb", "textDecoration": "underline"})
            ], style={"fontSize": "14px", "color": "#4b5563", "marginBottom": "8px", "fontStyle": "italic"}),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.P("Pick a year/metric for the map. Click a county OR choose one from the dropdown to see its multi-year trend.", className="lead", style={"margin": "0 0 16px 0", "fontSize": "15px", "paddingBottom": "12px", "borderBottom": "1px solid #e5e7eb"}),
            html.Div([
                html.Label("Year", style={"fontWeight": 600, "marginRight": 8, "fontSize": "16px"}),
                dcc.Dropdown(
                    id="year-dd",
                    options=[{"label": y, "value": y} for y in years],
                    value=years[-1] if years else None,
                    clearable=False,
                    style={"width": 140, "fontSize": "15px"}
                ),
                html.Label("Metric", style={"fontWeight": 600, "marginRight": 8, "marginLeft": 20, "fontSize": "16px"}),
                dcc.Dropdown(
                    id="metric-dd",
                    options=[{"label": k, "value": v} for k, v in metrics_label_map.items()],
                    value="rate_pct",
                    clearable=False,
                    style={"width": 320, "fontSize": "15px"}
                ),
                html.Label("County", style={"fontWeight": 600, "marginRight": 8, "marginLeft": 20, "fontSize": "16px"}),
                dcc.Dropdown(
                    id="county-dd",
                    options=county_options,
                    value=None,
                    clearable=True,
                    placeholder="Select a county…",
                    style={"width": 300, "fontSize": "15px"}
                ),
                html.Button("Clear selection", id="clear-selection", n_clicks=0, style={"marginLeft": 12, "background": "#e2e8f0", "borderRadius": "6px", "border": "none", "padding": "6px 14px", "fontWeight": 500, "boxShadow": "0 1px 4px #0001", "cursor": "pointer"}),
                html.Span(id="missing-note", style={"marginLeft": 12, "color": "#555", "fontSize": "14px"})
            ], style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"}),
            
            # Cluster toggle - only show for families poverty metric
            html.Div(id="cluster-toggle-container", children=[
                html.Label("Cluster counties based on yearly trends", style={"fontWeight": 500, "marginRight": 8, "fontSize": "14px", "color": "#374151"}),
                dcc.Checklist(
                    id="cluster-toggle",
                    options=[{"label": "", "value": "show_clusters"}],
                    value=[],
                    style={"display": "inline-block"}
                )
            ], style={"marginTop": "8px", "paddingTop": "8px", "borderTop": "1px solid #e5e7eb", "display": "none"})
        ], style={"marginBottom": "12px", "background": "#fff", "borderRadius": "10px", "boxShadow": "0 2px 12px #0001", "padding": "14px 10px"}),

        html.Div([
            html.Span("ℹ️", className="description-icon"),
            html.Span(id="metric-description")
        ], id="metric-description-container"),

        dcc.Loading(
            id="map-loading",
            type="dot",
            children=dcc.Graph(id="map", style={"height": "60vh", "background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001"}),
            fullscreen=False,
        ),

        html.Hr(style={"marginTop": "24px", "marginBottom": "18px"}),
        html.Div(id="trend-title", style={"fontWeight": 600, "marginBottom": 6, "fontSize": "17px"}),
        dcc.Loading(
            id="trend-loading",
            type="default",
            children=dcc.Graph(id="trend", style={"height": "32vh", "background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001"}),
            fullscreen=False,
        ),

        html.Hr(style={"marginTop": "24px", "marginBottom": "18px"}),
        html.Div([
            html.H3("County Performance Rankings", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "16px", "color": "#1f2937"}),
            html.Div([
                html.Div(id="top-counties-tanf", style={"flex": "1", "minWidth": "280px"}),
                html.Div(id="bottom-counties-tanf", style={"flex": "1", "minWidth": "280px"})
            ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"})
        ], style={"background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001", "padding": "20px"}),

        # Citation box for TANF
        html.Div([
            html.Span("📚", className="citation-icon"),
            html.Span("Citation:", className="citation-label"),
            html.Span("[TANF data citation to be added]", className="citation-text citation-placeholder")
        ], className="citation-box"),

        dcc.Store(id="sel-geoid"),
        dcc.Store(id="sel-name"),
    ], className="tanf-content", style={"background": "#f9fafb", "borderRadius": "14px", "boxShadow": "0 4px 24px #0001", "padding": "24px 18px", "marginBottom": "24px"})

    coi_tab = html.Div([
        html.Div([
            html.H2("Georgia Child Opportunity Index", className="page-title", style={"marginBottom": "2px"}),
            html.P([
                "The Child Opportunity Index (COI) measures children's access to resources and conditions that promote healthy development across neighborhoods. ",
                html.A("Learn more", href="https://www.diversitydatakids.org/child-opportunity-index", target="_blank", style={"color": "#2563eb", "textDecoration": "underline"})
            ], style={"fontSize": "14px", "color": "#4b5563", "marginBottom": "8px", "fontStyle": "italic"}),
        ], style={"marginBottom": "10px"}),

        html.Div([
            html.P("Pick a year/metric for the map. Click a county OR choose one from the dropdown to see its multi-year trend.", className="lead", style={"margin": "0 0 16px 0", "fontSize": "15px", "paddingBottom": "12px", "borderBottom": "1px solid #e5e7eb"}),
            html.Div([
                html.Label("Year", style={"fontWeight": 600, "marginRight": 8, "fontSize": "16px"}),
                dcc.Dropdown(
                    id="coi-year-dd",
                    options=[{"label": y, "value": y} for y in coi_years],
                    value=coi_years[-1] if coi_years else None,
                    clearable=False,
                    style={"width": 140, "fontSize": "15px"}
                ),
                html.Label("Metric", style={"fontWeight": 600, "marginRight": 8, "marginLeft": 20, "fontSize": "16px"}),
                dcc.Dropdown(
                    id="coi-metric-dd",
                    options=[{"label": k, "value": v} for k, v in coi_metrics_label_map.items()],
                    value="z_COI_stt",
                    clearable=False,
                    style={"width": 320, "fontSize": "15px"}
                ),
                html.Label("County", style={"fontWeight": 600, "marginRight": 8, "marginLeft": 20, "fontSize": "16px"}),
                dcc.Dropdown(
                    id="coi-county-dd",
                    options=county_options,
                    value=None,
                    clearable=True,
                    placeholder="Select a county…",
                    style={"width": 300, "fontSize": "15px"}
                ),
                html.Button("Clear selection", id="coi-clear-selection", n_clicks=0, style={"marginLeft": 12, "background": "#e2e8f0", "borderRadius": "6px", "border": "none", "padding": "6px 14px", "fontWeight": 500, "boxShadow": "0 1px 4px #0001", "cursor": "pointer"}),
                html.Span(id="coi-missing-note", style={"marginLeft": 12, "color": "#555", "fontSize": "14px"})
            ], style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"})
        ], style={"marginBottom": "12px", "background": "#fff", "borderRadius": "10px", "boxShadow": "0 2px 12px #0001", "padding": "14px 10px"}),

        html.Div([
            html.Span("ℹ️", className="description-icon"),
            html.Span(id="coi-metric-description")
        ], id="coi-metric-description-container"),

        dcc.Loading(
            id="coi-map-loading",
            type="dot",
            children=dcc.Graph(id="coi-map", style={"height": "60vh", "background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001"}),
            fullscreen=False,
        ),

        html.Hr(style={"marginTop": "24px", "marginBottom": "18px"}),
        html.Div(id="coi-trend-title", style={"fontWeight": 600, "marginBottom": 6, "fontSize": "17px"}),
        dcc.Loading(
            id="coi-trend-loading",
            type="default",
            children=dcc.Graph(id="coi-trend", style={"height": "32vh", "background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001"}),
            fullscreen=False,
        ),

        html.Hr(style={"marginTop": "24px", "marginBottom": "18px"}),
        html.Div([
            html.H3("County Performance Rankings", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "16px", "color": "#1f2937"}),
            html.Div([
                html.Div(id="top-counties-coi", style={"flex": "1", "minWidth": "280px"}),
                html.Div(id="bottom-counties-coi", style={"flex": "1", "minWidth": "280px"})
            ], style={"display": "flex", "gap": "24px", "flexWrap": "wrap"})
        ], style={"background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001", "padding": "20px"}),

        # Citation box for COI
        html.Div([
            html.Div([
                html.Span("📚", className="citation-icon"),
                html.Span("Citation:", className="citation-label"),
            ], style={"marginBottom": "8px"}),
            html.Div([
                html.Div("diversitydatakids.org. 2025.", className="citation-text", style={"marginBottom": "4px"}),
                html.Div("\"Child Opportunity Index 3.0-2023 County-Level Data.\"", className="citation-text", style={"marginBottom": "4px", "fontStyle": "italic"}),
                html.Div([
                    html.A("https://www.diversitydatakids.org/research-library/child-opportunity-index-30-2023-county-data", 
                           href="https://www.diversitydatakids.org/research-library/child-opportunity-index-30-2023-county-data",
                           target="_blank",
                           className="citation-text",
                           style={"color": "#2563eb", "textDecoration": "underline"})
                ])
            ])
        ], className="citation-box"),

        dcc.Store(id="coi-sel-geoid"),
        dcc.Store(id="coi-sel-name"),
    ], className="coi-content", style={"background": "#f9fafb", "borderRadius": "14px", "boxShadow": "0 4px 24px #0001", "padding": "24px 18px", "marginBottom": "24px"})

    tabs = dcc.Tabs(id="top-tabs", value="tanf", children=[
        dcc.Tab(label="TANF", value="tanf", children=tanf_tab, style={"fontWeight": 600, "fontSize": "16px"}),
        dcc.Tab(label="Child Opportunity Index", value="coi", children=coi_tab, style={"fontWeight": 600, "fontSize": "16px"}),
    ], style={"marginBottom": "16px", "background": "#fff", "borderRadius": "10px", "boxShadow": "0 2px 12px #0001"})

    return html.Div(
        style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
               "margin": "0 auto", "maxWidth": "1200px", "padding": "18px", "background": "#f8fafc"},
        children=[
            html.Div([
                html.Div([
                    html.Img(src="/assets/logo_naacfrc.webp", style={"height": "50px", "marginRight": "12px", "borderRadius": "8px", "boxShadow": "0 2px 8px #0001"}),
                    html.Img(src="/assets/logo_gatech.png", style={"height": "50px", "marginRight": "16px", "borderRadius": "8px", "boxShadow": "0 2px 8px #0001"}),
                    html.Div([
                        html.Div("GA Social Analytics Dashboard", className="brand", style={"fontSize": "28px", "fontWeight": 700, "color": "#2b6cb0", "marginBottom": "2px"}),
                        html.Div("Insights and county-level trends", className="brand-sub", style={"color": "#6b7280", "fontSize": "16px"}),
                    ], style={"display": "flex", "flexDirection": "column"})
                ], style={"display": "flex", "alignItems": "center", "gap": "8px"})
            ], className="header", style={"marginBottom": "18px"}),
            tabs
        ]
    )


def register_callbacks(
    app: dash.Dash,
    df: pd.DataFrame,
    coi_df: pd.DataFrame,
    geojson: dict,
    geoid_to_name: Dict[str, str],
    metrics_label_map: Dict[str, str],
    metric_fmt: Dict[str, str],
    coi_metrics_label_map: Dict[str, str],
    coi_metric_fmt: Dict[str, str],
    cluster_df: pd.DataFrame,
    state_averages: pd.DataFrame,
):
    """Wire all Dash callbacks."""

    @app.callback(
        Output("cluster-toggle-container", "style"),
        Input("metric-dd", "value")
    )
    def show_cluster_toggle(metric):
        """Show cluster toggle only for families poverty metric."""
        if metric == "families_poverty_pct":
            return {"marginTop": "8px", "paddingTop": "8px", "borderTop": "1px solid #e5e7eb", "display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        Output("map", "figure"),
        Output("missing-note", "children"),
        Input("year-dd", "value"),
        Input("metric-dd", "value"),
        Input("cluster-toggle", "value")
    )
    def update_map(year, metric, cluster_toggle):
        show_clusters = "show_clusters" in (cluster_toggle or [])
        
        fig, note = make_map_figure(
            df=df,
            geojson=geojson,
            geoid_to_name=geoid_to_name,
            year=year,
            metric=metric,
            metrics_label_map=metrics_label_map,
            metric_fmt=metric_fmt,
            cluster_df=cluster_df,
            show_clusters=show_clusters,
        )
        return fig, note

    @app.callback(
        Output("sel-geoid", "data"),
        Output("sel-name", "data"),
        Output("county-dd", "value"),
        Input("map", "clickData"),
        Input("county-dd", "value"),
        Input("clear-selection", "n_clicks"),
        prevent_initial_call=True,
    )
    def sync_selection(clickData, dd_value, _clear_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "clear-selection":
            return None, None, None

        if trigger == "county-dd":
            if dd_value is None:
                return None, None, None
            name = geoid_to_name.get(dd_value)
            return dd_value, name, dd_value

        if trigger == "map" and clickData and "points" in clickData and clickData["points"]:
            geoid = clickData["points"][0].get("location")
            name = geoid_to_name.get(geoid)
            return geoid, name, geoid

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("trend", "figure"),
        Output("trend-title", "children"),
        Input("sel-geoid", "data"),
        Input("sel-name", "data"),
        Input("metric-dd", "value"),
    )
    def update_trend(sel_geoid, sel_name, metric):
        fig, title = make_trend_figure(
            df=df,
            sel_geoid=sel_geoid,
            sel_name=sel_name,
            metric=metric,
            metrics_label_map=metrics_label_map,
            metric_fmt=metric_fmt,
            state_averages=state_averages,
        )
        return fig, title

    @app.callback(
        Output("metric-description", "children"),
        Input("metric-dd", "value")
    )
    def update_metric_description(metric):
        """Update the TANF metric description text when a metric is selected."""
        descriptions = get_metric_descriptions()
        return descriptions.get(metric, "Select a metric to see its description.")

    @app.callback(
        [Output("top-counties-tanf", "children"),
         Output("bottom-counties-tanf", "children")],
        [Input("year-dd", "value"),
         Input("metric-dd", "value")]
    )
    def update_tanf_rankings(year, metric):
        """Update the TANF county rankings when year or metric changes."""
        if not year or not metric:
            return html.Div(), html.Div()
        
        # Get metric label for display
        metric_label = [k for k, v in metrics_label_map.items() if v == metric][0] if metric else "Unknown Metric"
        
        top_counties, bottom_counties = get_county_rankings(
            df=df,
            geoid_to_name=geoid_to_name,
            year=year,
            metric=metric,
            metric_fmt=metric_fmt,
            n_counties=5
        )
        
        top_display = create_ranking_display(
            counties=top_counties,
            title=f"Highest 5 Counties - {metric_label}",
            icon="",
            is_top=True
        )
        
        bottom_display = create_ranking_display(
            counties=bottom_counties,
            title=f"Lowest 5 Counties - {metric_label}",
            icon="",
            is_top=False
        )
        
        return top_display, bottom_display

    # COI Callbacks
    @app.callback(
        Output("coi-metric-description", "children"),
        Input("coi-metric-dd", "value")
    )
    def update_coi_metric_description(metric):
        """Update the COI metric description text when a metric is selected."""
        descriptions = get_coi_metric_descriptions()
        return descriptions.get(metric, "Select a metric to see its description.")

    @app.callback(
        [Output("top-counties-coi", "children"),
         Output("bottom-counties-coi", "children")],
        [Input("coi-year-dd", "value"),
         Input("coi-metric-dd", "value")]
    )
    def update_coi_rankings(year, metric):
        """Update the COI county rankings when year or metric changes."""
        if not year or not metric:
            return html.Div(), html.Div()
        
        # Get metric label for display
        metric_label = [k for k, v in coi_metrics_label_map.items() if v == metric][0] if metric else "Unknown Metric"
        
        top_counties, bottom_counties = get_county_rankings(
            df=coi_df,
            geoid_to_name=geoid_to_name,
            year=year,
            metric=metric,
            metric_fmt=coi_metric_fmt,
            n_counties=5
        )
        
        top_display = create_ranking_display(
            counties=top_counties,
            title=f"Highest 5 Counties - {metric_label}",
            icon="",
            is_top=True
        )
        
        bottom_display = create_ranking_display(
            counties=bottom_counties,
            title=f"Lowest 5 Counties - {metric_label}", 
            icon="",
            is_top=False
        )
        
        
        return top_display, bottom_display

    @app.callback(
        Output("coi-map", "figure"),
        Output("coi-missing-note", "children"),
        Input("coi-year-dd", "value"),
        Input("coi-metric-dd", "value")
    )
    def update_coi_map(year, metric):
        fig, note = make_coi_map_figure(
            df=coi_df,
            geojson=geojson,
            geoid_to_name=geoid_to_name,
            year=year,
            metric=metric,
            metrics_label_map=coi_metrics_label_map,
            metric_fmt=coi_metric_fmt,
        )
        return fig, note

    @app.callback(
        Output("coi-sel-geoid", "data"),
        Output("coi-sel-name", "data"),
        Output("coi-county-dd", "value"),
        Input("coi-map", "clickData"),
        Input("coi-county-dd", "value"),
        Input("coi-clear-selection", "n_clicks"),
        prevent_initial_call=True,
    )
    def sync_coi_selection(clickData, dd_value, _clear_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        trigger = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger == "coi-clear-selection":
            return None, None, None

        if trigger == "coi-county-dd":
            if dd_value is None:
                return None, None, None
            name = geoid_to_name.get(dd_value)
            return dd_value, name, dd_value

        if trigger == "coi-map" and clickData and "points" in clickData and clickData["points"]:
            geoid = clickData["points"][0].get("location")
            name = geoid_to_name.get(geoid)
            return geoid, name, geoid

        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("coi-trend", "figure"),
        Output("coi-trend-title", "children"),
        Input("coi-sel-geoid", "data"),
        Input("coi-sel-name", "data"),
        Input("coi-metric-dd", "value"),
    )
    def update_coi_trend(sel_geoid, sel_name, metric):
        fig, title = make_coi_trend_figure(
            df=coi_df,
            sel_geoid=sel_geoid,
            sel_name=sel_name,
            metric=metric,
            metrics_label_map=coi_metrics_label_map,
            metric_fmt=coi_metric_fmt,
        )
        return fig, title


def create_app(geojson_path: Path = GEOJSON_PATH,
               csv_path: Path = TANF_CSV,
               coi_csv_path: Path = COI_CSV,
               cluster_csv_path: Path = CLUSTER_CSV) -> dash.Dash:
    """
    App factory. Loads data, builds layout, and registers callbacks.
    Returns a ready-to-run Dash app.
    """
    # Load data normally for now - caching was causing issues
    geojson, name_to_geoid, geoid_to_name = load_geojson(geojson_path)
    df = load_dataset(csv_path, name_to_geoid)
    coi_df = load_coi_dataset(coi_csv_path, name_to_geoid)
    cluster_df = load_cluster_dataset(cluster_csv_path, name_to_geoid)
    
    # Pre-calculate state averages for performance
    state_averages = calculate_state_averages(df)
    
    metrics, metric_fmt = metrics_config()
    coi_metrics, coi_metric_fmt = coi_metrics_config()
    years = sorted(df["year"].dropna().unique())
    coi_years = sorted(coi_df["year"].dropna().unique())

    app = dash.Dash(__name__)
    app.title = "GA Social Analytics Dashboard"
    
    app.layout = build_layout(app, years, coi_years, geoid_to_name, metrics, coi_metrics)
    register_callbacks(
        app, df, coi_df, geojson, geoid_to_name, metrics, metric_fmt, 
        coi_metrics, coi_metric_fmt, cluster_df, state_averages
    )
    return app


# ======================
# Main
# ======================

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)