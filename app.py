import json
from pathlib import Path

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# -----------------------
# Paths
# -----------------------
GEOJSON_PATH = Path("data/ga_counties.geojson")
TANF_CSV     = Path("data/tanf_ga.csv")   # Columns: County, Year, Recipients, Black Rec.
POP_CSV      = Path("data/black_population_ga.csv")  # Columns: County, Year, Black Population

# -----------------------
# Helpers
# -----------------------
def clean_county_name(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
         .str.replace(r"\bCounty\b", "", regex=True)  # drop literal "County"
         .str.strip()
         .str.title()
    )

def coerce_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# -----------------------
# Load GeoJSON & build NAME→GEOID map
# -----------------------
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    GA = json.load(f)

name_to_geoid = {}
geoid_to_name = {}
for feat in GA["features"]:
    props = feat.get("properties", {})
    nm = str(props.get("NAME", "")).strip()
    geoid = str(props.get("GEOID", "")).strip()
    if nm and geoid:
        nm_clean = clean_county_name(pd.Series([nm])).iloc[0]
        name_to_geoid[nm_clean] = geoid
        geoid_to_name[geoid] = nm_clean

# -----------------------
# Load TANF
# -----------------------
tanf = pd.read_csv(TANF_CSV, dtype=str)
tanf.rename(columns={c: c.strip() for c in tanf.columns}, inplace=True)
needed = {"County", "Year", "Recipients", "Black Rec."}
if not needed.issubset(tanf.columns):
    raise ValueError(f"TANF CSV must have columns: {needed}")

tanf["County"] = clean_county_name(tanf["County"])
tanf["Year"] = tanf["Year"].astype(str)
for c in ["Recipients", "Black Rec."]:
    tanf[c] = tanf[c].map(coerce_num)

# -----------------------
# Load Black Population
# -----------------------
pop = pd.read_csv(POP_CSV, dtype=str)
pop.rename(columns={c: c.strip() for c in pop.columns}, inplace=True)
req_pop_cols = {"County", "Year", "Black Population"}
if not req_pop_cols.issubset(pop.columns):
    raise ValueError(f"Population CSV must have columns: {req_pop_cols}")

pop["County"] = clean_county_name(pop["County"])  # should already match GeoJSON NAMEs
pop["Year"] = pop["Year"].astype(str)
pop["Black Population"] = pop["Black Population"].map(coerce_num)

# -----------------------
# Merge TANF + Pop WITHOUT aggregating across years
# -----------------------
df = tanf.merge(pop, on=["County", "Year"], how="left")

# Map to GEOID
df["GEOID"] = df["County"].map(name_to_geoid)

# If duplicates exist per County+Year, keep the first to maintain one row per county-year for the map
# (Change this rule if you want different behavior.)
df = df.sort_values(["County", "Year"]).drop_duplicates(subset=["County", "Year"], keep="first")

# Core metrics (computed per-row)
df["rate_pct"] = (df["Black Rec."] / df["Recipients"]) * 100
df["black_over_blackpop_pct"] = (df["Black Rec."] / df["Black Population"]) * 100
for col in ["rate_pct", "black_over_blackpop_pct"]:
    df.loc[~np.isfinite(df[col]), col] = np.nan

# Year numeric for sorting in trend (no aggregation across years)
df["Year_num"] = pd.to_numeric(df["Year"], errors="coerce")

YEARS = sorted(df["Year"].dropna().unique())

# Metrics dictionary for dropdown (label → column)
METRICS = {
    "% Black of Recipients": "rate_pct",
    "% Black Recipients of Black Pop": "black_over_blackpop_pct",
    "Black Recipients (count)": "Black Rec.",
    "Total Recipients (count)": "Recipients",
    "Black Population (count)": "Black Population",
}

METRIC_FMT = {
    "rate_pct": "%",
    "black_over_blackpop_pct": "%",
}

# -----------------------
# Dash app
# -----------------------
app = dash.Dash(__name__)
app.title = "GA TANF – County Dropdown + Click Trend"

county_options = [
    {"label": f"{name} County", "value": geoid}
    for geoid, name in sorted(geoid_to_name.items(), key=lambda kv: kv[1])
]

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
           "margin": "0 auto", "maxWidth": "1100px", "padding": "24px"},
    children=[
        html.H2("Georgia TANF – County Metrics (with Black Population)", style={"marginBottom": "8px"}),
        html.P("Pick a year/metric for the map. Click a county OR choose one from the dropdown to see its multi-year trend."),

        html.Div([
            html.Label("Year", style={"fontWeight": 600, "marginRight": 8}),
            dcc.Dropdown(
                id="year-dd",
                options=[{"label": y, "value": y} for y in YEARS],
                value=YEARS[-1] if YEARS else None,
                clearable=False,
                style={"width": 140}
            ),
            html.Label("Metric", style={"fontWeight": 600, "marginRight": 8, "marginLeft": 20}),
            dcc.Dropdown(
                id="metric-dd",
                options=[{"label": k, "value": v} for k, v in METRICS.items()],
                value="rate_pct",
                clearable=False,
                style={"width": 320}
            ),
            html.Label("County", style={"fontWeight": 600, "marginRight": 8, "marginLeft": 20}),
            dcc.Dropdown(
                id="county-dd",
                options=county_options,
                value=None,
                clearable=True,
                placeholder="Select a county…",
                style={"width": 300}
            ),
            html.Button("Clear selection", id="clear-selection", n_clicks=0, style={"marginLeft": 12}),
            html.Span(id="missing-note", style={"marginLeft": 12, "color": "#555"})
        ], style={"display": "flex", "alignItems": "center", "gap": "8px",
                  "marginBottom": "12px", "flexWrap": "wrap"}),

        dcc.Graph(id="map", style={"height": "60vh"}),

        html.Hr(),
        html.Div(id="trend-title", style={"fontWeight": 600, "marginBottom": 6}),
        dcc.Graph(id="trend", style={"height": "32vh"}),

        dcc.Store(id="sel-geoid"),
        dcc.Store(id="sel-name"),
    ]
)

# -----------------------
# Map figure callback
# -----------------------
@app.callback(
    Output("map", "figure"),
    Output("missing-note", "children"),
    Input("year-dd", "value"),
    Input("metric-dd", "value")
)
def update_map(year, metric):
    dfy = df[df["Year"] == str(year)].copy()

    base = pd.DataFrame({"GEOID": list(geoid_to_name.keys())})
    keep_cols = ["GEOID", "County", "Recipients", "Black Rec.", "Black Population", "rate_pct", "black_over_blackpop_pct"]
    dfy = base.merge(dfy[keep_cols], on="GEOID", how="left")
    dfy["County"] = dfy["GEOID"].map(geoid_to_name)

    def fmt_count(v):
        return f"{int(v):,}" if pd.notna(v) else "—"

    def fmt_pct(v):
        return f"{v:.1f}%" if pd.notna(v) else "—"

    dfy["hover"] = (
        "County: " + dfy["County"].fillna("—") +
        "<br>Black Pop.: " + dfy["Black Population"].map(fmt_count) +
        "<br>Recipients: " + dfy["Recipients"].map(fmt_count) +
        "<br>Black Rec.: " + dfy["Black Rec."].map(fmt_count) +
        "<br>% Black of Recipients: " + dfy["rate_pct"].map(fmt_pct) +
        "<br>% Black Rec of Black Pop: " + dfy["black_over_blackpop_pct"].map(fmt_pct)
    )

    n_total = len(dfy)
    n_missing = dfy[metric].isna().sum()
    note = f"{n_missing}/{n_total} counties missing."

    kind = METRIC_FMT.get(metric, "count")
    if kind == "%":
        vmax = float(np.nanpercentile(dfy[metric], 98)) if dfy[metric].notna().any() else 100.0
        color_scale = "Viridis"
    else:
        vmax = float(np.nanpercentile(dfy[metric], 98)) if dfy[metric].notna().any() else 1.0
        color_scale = "Blues"
    vmax = max(vmax, 1.0)

    label = [k for k,v in METRICS.items() if v==metric][0]

    fig = px.choropleth_mapbox(
        dfy,
        geojson=GA,
        locations="GEOID",
        featureidkey="properties.GEOID",
        color=metric,
        color_continuous_scale=color_scale,
        range_color=(0, vmax),
        hover_name="County",
        hover_data=None,
        custom_data=[dfy["hover"]],
        labels={metric: label},
        mapbox_style="carto-positron",
        center={"lat": 32.5, "lon": -83.3},
        zoom=5.7,
        opacity=0.86,
    )

    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=label)
    )

    return fig, note

# -----------------------
# Selection handling: map click OR county dropdown → selected county
# -----------------------
@app.callback(
    Output("sel-geoid", "data"),
    Output("sel-name", "data"),
    Output("county-dd", "value"),
    Input("map", "clickData"),
    Input("county-dd", "value"),
    Input("clear-selection", "n_clicks"),
    prevent_initial_call=True,
)
def sync_selection(clickData, dd_value, clear_clicks):
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

# -----------------------
# Trend figure callback (depends on selected county + metric)
# -----------------------
@app.callback(
    Output("trend", "figure"),
    Output("trend-title", "children"),
    Input("sel-geoid", "data"),
    Input("sel-name", "data"),
    Input("metric-dd", "value"),
)
def update_trend(sel_geoid, sel_name, metric):
    title_label = [k for k, v in METRICS.items() if v == metric][0]
    if not sel_geoid:
        fig = go.Figure()
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            xaxis_title="Year",
            yaxis_title=title_label,
            annotations=[dict(text="Click a county or choose one from the dropdown to see its trend.",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False)]
        )
        return fig, ""

    dfc = df[df["GEOID"] == sel_geoid].copy().sort_values("Year_num")

    fig = px.line(
        dfc,
        x="Year",
        y=metric,
        markers=True,
        labels={metric: title_label, "Year": "Year"}
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    kind = METRIC_FMT.get(metric, "count")
    if kind == "%":
        fig.update_traces(hovertemplate="Year %{x}<br>%{y:.2f}%<extra></extra>")
    else:
        fig.update_traces(hovertemplate="Year %{x}<br>%{y:,.0f}<extra></extra>")

    return fig, f"Trend – {sel_name} County ({title_label})"

if __name__ == "__main__":
    app.run(debug=True)
