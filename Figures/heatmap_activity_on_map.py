import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from pathlib import Path


plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.axisbelow": True,
})


# ============================================================
# Settings
# ============================================================

REGION_LAT_SOUTH = 55
REGION_LAT_NORTH = 82
REGION_LON_WEST = -10
REGION_LON_EAST = 45
GRID_SIZE = 0.05

MIN_HOURS_TO_PLOT = 1

effort_path = Path(
    "heatmap_of_activity/fishing_effort_grid_hours_pred_foreign_2025.parquet"
)


# ============================================================
# Load detailed aggregated grid
# ============================================================

effort = pd.read_parquet(effort_path)

n_lat = int(np.ceil((REGION_LAT_NORTH - REGION_LAT_SOUTH) / GRID_SIZE))
n_lon = int(np.ceil((REGION_LON_EAST - REGION_LON_WEST) / GRID_SIZE))

heatmap = np.full((n_lat, n_lon), np.nan)

for row in effort.itertuples(index=False):
    heatmap[int(row.lat_bin), int(row.lon_bin)] = row.fishing_hours

heatmap_plot = heatmap.copy()
heatmap_plot[heatmap_plot < MIN_HOURS_TO_PLOT] = np.nan

positive_values = heatmap_plot[np.isfinite(heatmap_plot) & (heatmap_plot > 0)]

if len(positive_values) == 0:
    raise ValueError("No grid cells above MIN_HOURS_TO_PLOT.")

vmin = MIN_HOURS_TO_PLOT
vmax = positive_values.max()


# ============================================================
# Create original 0.05 degree grid edges
# ============================================================

lon_edges = np.arange(REGION_LON_WEST, REGION_LON_EAST + GRID_SIZE, GRID_SIZE)
lat_edges = np.arange(REGION_LAT_SOUTH, REGION_LAT_NORTH + GRID_SIZE, GRID_SIZE)

lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)


# ============================================================
# Convert grid edges to Web Mercator
# ============================================================

edge_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
    crs="EPSG:4326"
).to_crs(epsg=3857)

x_grid = edge_gdf.geometry.x.to_numpy().reshape(lon_grid.shape)
y_grid = edge_gdf.geometry.y.to_numpy().reshape(lat_grid.shape)


# ============================================================
# Fixed map extent
# ============================================================

extent_gdf = gpd.GeoDataFrame(
    geometry=[
        Point(REGION_LON_WEST, REGION_LAT_SOUTH),
        Point(REGION_LON_EAST, REGION_LAT_NORTH),
    ],
    crs="EPSG:4326"
).to_crs(epsg=3857)

xmin, ymin = extent_gdf.geometry.iloc[0].x, extent_gdf.geometry.iloc[0].y
xmax, ymax = extent_gdf.geometry.iloc[1].x, extent_gdf.geometry.iloc[1].y


def lon_to_mercator_x(lon):
    return gpd.GeoSeries(
        [Point(lon, REGION_LAT_SOUTH)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).iloc[0].x


def lat_to_mercator_y(lat):
    return gpd.GeoSeries(
        [Point(REGION_LON_WEST, lat)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).iloc[0].y


def lon_formatter(lon):
    return f"{lon:.0f}°E" if lon >= 0 else f"{abs(lon):.0f}°W"


def lat_formatter(lat):
    return f"{lat:.0f}°N" if lat >= 0 else f"{abs(lat):.0f}°S"


# ============================================================
# Plot detailed effort grid on map
# ============================================================

fig, ax = plt.subplots(figsize=(12, 12))

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.PositronNoLabels,
    crs="EPSG:3857",
    attribution_size=6,
    zoom=5
)

mesh = ax.pcolormesh(
    x_grid,
    y_grid,
    heatmap_plot,
    shading="flat",
    cmap="hot",
    norm=colors.LogNorm(vmin=vmin, vmax=vmax),
    alpha=0.85,
    zorder=10,
    rasterized=True
)

cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Fishing hours", labelpad=12)

lon_ticks = [0, 20, 40]
lat_ticks = [60, 70, 80]

ax.set_xticks([lon_to_mercator_x(lon) for lon in lon_ticks])
ax.set_yticks([lat_to_mercator_y(lat) for lat in lat_ticks])

ax.set_xticklabels([lon_formatter(lon) for lon in lon_ticks])
ax.set_yticklabels([lat_formatter(lat) for lat in lat_ticks])

ax.set_aspect("equal", adjustable="box")

plt.savefig(
    "fishing_effort_2025_foreign.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
    dpi=300
)

plt.show()