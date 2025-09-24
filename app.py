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


def metrics_config() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
        METRICS (label->column), METRIC_FMT (column->'%' or 'count')
    """
    metrics = {
        "% Black of Recipients": "rate_pct",
        "% Black Recipients of Black Pop": "black_over_blackpop_pct",
        "% Black Children in Poverty who got TANF": "children_poverty_pct",
        "% Black Families in Poverty who got TANF": "families_poverty_pct",
        "Black Recipients (count)": "Black Rec.",
        "Total Recipients (count)": "Recipients",
        "Black Population (count)": "black_population",
    }
    metric_fmt = {
        "rate_pct": "%",
        "black_over_blackpop_pct": "%",
        "children_poverty_pct": "%",
        "families_poverty_pct": "%",
    }
    return metrics, metric_fmt


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
) -> Tuple[go.Figure, str]:
    """Build the choropleth map and the 'missing data' note."""
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
        return f"{int(v):,}" if pd.notna(v) else "NA"

    def fmt_pct(v):
        return f"{v:.1f}%" if pd.notna(v) else "NA"
    dfy["hover"] = (
        "County: " + dfy["County"].fillna("NA") +
        "<br>Black Pop.: " + dfy["black_population"].map(fmt_count) +
        "<br>Recipients: " + dfy["Recipients"].map(fmt_count) +
        "<br>Black Rec.: " + dfy["Black Rec."].map(fmt_count) +
        "<br>% Black of Recipients: " + dfy["rate_pct"].map(fmt_pct) +
        "<br>% Black Rec of Black Pop: " + dfy["black_over_blackpop_pct"].map(fmt_pct) +
        "<br>% Black Children in Poverty who got TANF: " + dfy["children_poverty_pct"].map(fmt_pct) +
        "<br>% Black Families in Poverty who got TANF: " + dfy["families_poverty_pct"].map(fmt_pct)
    )

    # Split counties with and without the chosen metric so missing ones can be shown in gray
    df_have = dfy[dfy[metric].notna()].copy()
    df_missing = dfy[dfy[metric].isna()].copy()
    # Create a simple hover for missing counties: show county name and mention missing data
    if not df_missing.empty:
        df_missing["hover_missing"] = "County: " + df_missing["County"].fillna("NA") + "<br>Data: Insufficient/Missing"

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
        coloraxis_colorbar=dict(title=label) if not df_have.empty else {},
        mapbox={
            "style": "carto-positron",
            "center": {"lat": 32.5, "lon": -83.3},
            "zoom": 5.7,
            "layers": [
                {
                    "sourcetype": "geojson",
                    "source": geojson,
                    "type": "line",
                    "color": "#444",
                    "line": {"width": 0.7},
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
) -> Tuple[go.Figure, str]:
    """Build the line chart for a selected county."""
    title_label = [k for k, v in metrics_label_map.items() if v == metric][0]
    if not sel_geoid:
        return make_empty_trend(title_label), ""

    dfc = df[df["GEOID"] == sel_geoid].copy().sort_values("Year_num")
    fig = px.line(dfc, x="year", y=metric, markers=True,
                  labels={metric: title_label, "year": "Year"})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    kind = metric_fmt.get(metric, "count")
    if kind == "%":
        fig.update_traces(hovertemplate="Year %{x}<br>%{y:.2f}%<extra></extra>")
    else:
        fig.update_traces(hovertemplate="Year %{x}<br>%{y:,.0f}<extra></extra>")

    return fig, f"Trend – {sel_name} County ({title_label})"


# ======================
# App Factory
# ======================

def build_layout(
    app: dash.Dash,
    years: List[str],
    geoid_to_name: Dict[str, str],
    metrics_label_map: Dict[str, str]
) -> html.Div:
    """Construct the static Dash layout."""
    county_options = [
        {"label": f"{name} County", "value": geoid}
        for geoid, name in sorted(geoid_to_name.items(), key=lambda kv: kv[1])
    ]

    tanf_tab = html.Div([
        html.Div([
            html.Img(src="/assets/logo.png", style={"height": "44px", "marginRight": "16px", "borderRadius": "8px", "boxShadow": "0 2px 8px #0001"}),
            html.Div([
                html.H2("Georgia TANF Enrollment Trends", className="page-title", style={"marginBottom": "2px"}),
                html.P("Pick a year/metric for the map. Click a county OR choose one from the dropdown to see its multi-year trend.", className="lead"),
            ], style={"display": "flex", "flexDirection": "column"})
        ], style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "10px"}),

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
        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "12px", "flexWrap": "wrap", "background": "#fff", "borderRadius": "10px", "boxShadow": "0 2px 12px #0001", "padding": "14px 10px"}),

        dcc.Loading(
            id="map-loading",
            type="circle",
            children=dcc.Graph(id="map", style={"height": "60vh", "background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001"}),
            fullscreen=False,
        ),

        html.Hr(style={"marginTop": "24px", "marginBottom": "18px"}),
        html.Div(id="trend-title", style={"fontWeight": 600, "marginBottom": 6, "fontSize": "17px"}),
        dcc.Graph(id="trend", style={"height": "32vh", "background": "#fff", "borderRadius": "12px", "boxShadow": "0 2px 12px #0001"}),

        dcc.Store(id="sel-geoid"),
        dcc.Store(id="sel-name"),
    ], className="tanf-content", style={"background": "#f9fafb", "borderRadius": "14px", "boxShadow": "0 4px 24px #0001", "padding": "24px 18px", "marginBottom": "24px"})

    placeholder_tab = html.Div([
        html.H3("Coming soon", style={"color": "#2b6cb0"}),
        html.P("This dashboard is a placeholder for future content. Add your visualizations here.", style={"color": "#6b7280"}),
    ], className="placeholder-content", style={"background": "#fff", "borderRadius": "10px", "boxShadow": "0 2px 12px #0001", "padding": "24px 18px"})

    tabs = dcc.Tabs(id="top-tabs", value="tanf", children=[
        dcc.Tab(label="TANF", value="tanf", children=tanf_tab, style={"fontWeight": 600, "fontSize": "16px"}),
        dcc.Tab(label="Placeholder 1", value="ph1", children=placeholder_tab, style={"fontWeight": 600, "fontSize": "16px"}),
        dcc.Tab(label="Placeholder 2", value="ph2", children=placeholder_tab, style={"fontWeight": 600, "fontSize": "16px"}),
    ], style={"marginBottom": "16px", "background": "#fff", "borderRadius": "10px", "boxShadow": "0 2px 12px #0001"})

    return html.Div(
        style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
               "margin": "0 auto", "maxWidth": "1200px", "padding": "18px", "background": "#f8fafc"},
        children=[
            html.Div([
                html.Div("GA Social Safety Net Dashboards", className="brand", style={"fontSize": "28px", "fontWeight": 700, "color": "#2b6cb0", "marginBottom": "2px"}),
                html.Div("Insights and county-level trends", className="brand-sub", style={"color": "#6b7280", "fontSize": "16px"}),
            ], className="header", style={"marginBottom": "18px"}),
            tabs
        ]
    )


def register_callbacks(
    app: dash.Dash,
    df: pd.DataFrame,
    geojson: dict,
    geoid_to_name: Dict[str, str],
    metrics_label_map: Dict[str, str],
    metric_fmt: Dict[str, str],
):
    """Wire all Dash callbacks."""

    @app.callback(
        Output("map", "figure"),
        Output("missing-note", "children"),
        Input("year-dd", "value"),
        Input("metric-dd", "value")
    )
    def update_map(year, metric):
        fig, note = make_map_figure(
            df=df,
            geojson=geojson,
            geoid_to_name=geoid_to_name,
            year=year,
            metric=metric,
            metrics_label_map=metrics_label_map,
            metric_fmt=metric_fmt,
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
        )
        return fig, title


def create_app(geojson_path: Path = GEOJSON_PATH,
               csv_path: Path = TANF_CSV) -> dash.Dash:
    """
    App factory. Loads data, builds layout, and registers callbacks.
    Returns a ready-to-run Dash app.
    """
    geojson, name_to_geoid, geoid_to_name = load_geojson(geojson_path)
    df = load_dataset(csv_path, name_to_geoid)
    metrics, metric_fmt = metrics_config()
    years = sorted(df["year"].dropna().unique())

    app = dash.Dash(__name__)
    app.title = "GA TANF – County Dropdown + Click Trend"
    app.layout = build_layout(app, years, geoid_to_name, metrics)
    register_callbacks(app, df, geojson, geoid_to_name, metrics, metric_fmt)
    return app


# ======================
# Main
# ======================

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
