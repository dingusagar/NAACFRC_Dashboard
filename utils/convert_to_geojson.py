import geopandas as gpd
from pathlib import Path

# Path to your shapefile folder
shp_path = Path("data/Counties2018.shp")

# Read shapefile
gdf = gpd.read_file(shp_path)

# Inspect columns to confirm what’s available
print("Columns:", gdf.columns.tolist())
print(gdf.head())

# Filter to Georgia (STATEFP == '13' is Georgia in US Census shapefiles)
ga = gdf[gdf["STATEFP"] == "13"]

# Save as GeoJSON
out_path = Path("data/ga_counties.geojson")
ga.to_file(out_path, driver="GeoJSON")

print(f"Saved Georgia counties GeoJSON to {out_path}")
