import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from matplotlib.ticker import FuncFormatter


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
# Load data
# ============================================================

dfs = [
    pd.read_parquet(
        f"raw_ais/parquets_fish_only/2024-01-{day:02d}_fish_only.parquet",
        engine="pyarrow"
    )
    for day in range(1, 32)
]

df = pd.concat(dfs, ignore_index=True)


# ============================================================
# Clean and sort
# ============================================================

df["date_time_utc"] = pd.to_datetime(
    df["date_time_utc"],
    utc=True,
    errors="coerce"
)

df = (
    df.dropna(subset=["mmsi", "date_time_utc", "lon", "lat"])
      .sort_values(["mmsi", "date_time_utc"])
      .reset_index(drop=True)
)


# ============================================================
# Calculate AIS message gaps
# ============================================================

df["previous_time"] = df.groupby("mmsi")["date_time_utc"].shift()
df["gap"] = df["date_time_utc"] - df["previous_time"]

df["previous_lon"] = df.groupby("mmsi")["lon"].shift()
df["previous_lat"] = df.groupby("mmsi")["lat"].shift()

threshold = pd.Timedelta(minutes=60)
gap_messages = df.loc[df["gap"] > threshold].copy()

print(f"Number of gaps: {len(gap_messages):,}")
print(f"Vessels with at least one gap: {gap_messages['mmsi'].nunique():,}")


# ============================================================
# Combine endpoints before and after each gap
# ============================================================

gap_lons = pd.concat(
    [gap_messages["previous_lon"], gap_messages["lon"]],
    ignore_index=True
)

gap_lats = pd.concat(
    [gap_messages["previous_lat"], gap_messages["lat"]],
    ignore_index=True
)

valid = (
    gap_lons.notna()
    & gap_lats.notna()
    & gap_lons.between(-180, 180)
    & gap_lats.between(-90, 90)
)

gap_lons = gap_lons[valid].to_numpy()
gap_lats = gap_lats[valid].to_numpy()


# ============================================================
# Convert points to Web Mercator
# ============================================================

gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(gap_lons, gap_lats),
    crs="EPSG:4326"
).to_crs(epsg=3857)

x = gdf.geometry.x.to_numpy()
y = gdf.geometry.y.to_numpy()


# ============================================================
# Define fixed plotting extent
# ============================================================

lon_min, lon_max = -10, 45
lat_min, lat_max = 55, 80

extent_gdf = gpd.GeoDataFrame(
    geometry=[
        Point(lon_min, lat_min),
        Point(lon_max, lat_max)
    ],
    crs="EPSG:4326"
).to_crs(epsg=3857)

xmin, ymin = extent_gdf.geometry.iloc[0].x, extent_gdf.geometry.iloc[0].y
xmax, ymax = extent_gdf.geometry.iloc[1].x, extent_gdf.geometry.iloc[1].y

def lon_to_mercator_x(lon):
    point = gpd.GeoSeries(
        [Point(lon, lat_min)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).iloc[0]
    return point.x


def lat_to_mercator_y(lat):
    point = gpd.GeoSeries(
        [Point(lon_min, lat)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).iloc[0]
    return point.y


# ============================================================
# Create smoothed heatmap
# ============================================================

bins = 400
sigma = 1.2

H, xedges, yedges = np.histogram2d(
    x,
    y,
    bins=bins,
    range=[[xmin, xmax], [ymin, ymax]]
)

H_smooth = gaussian_filter(H, sigma=sigma)

# IMPORTANT:
# Mask low smoothed values, not only exact zeros.
# This removes the dark square patches.
min_visible = 1.0
H_smooth = np.ma.masked_less(H_smooth, min_visible)


# ============================================================
# Helpers for lon/lat tick labels
# ============================================================

def lon_to_mercator_x(lon):
    point = gpd.GeoSeries(
        [Point(lon, lat_min)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).iloc[0]
    return point.x


def lat_to_mercator_y(lat):
    point = gpd.GeoSeries(
        [Point(lon_min, lat)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).iloc[0]
    return point.y


def lon_formatter(lon):
    if lon >= 0:
        return f"{lon:.2f}°E"
    else:
        return f"{abs(lon):.2f}°W"


def lat_formatter(lat):
    if lat >= 0:
        return f"{lat:.2f}°N"
    else:
        return f"{abs(lat):.2f}°S"


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(12, 9))

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.PositronNoLabels,
    crs="EPSG:3857",
    attribution_size=6,
    zoom=5
)

im = ax.imshow(
    H_smooth.T,
    extent=[xmin, xmax, ymin, ymax],
    origin="lower",
    cmap="hot",
    norm=colors.LogNorm(vmin=min_visible, vmax=H_smooth.max()),
    alpha=0.75,
    interpolation="bilinear",
    zorder=10
)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Number of AIS gaps", labelpad=12)

lon_ticks = [0, 20, 40]
lat_ticks = [60, 70, 80]

ax.set_xticks([lon_to_mercator_x(lon) for lon in lon_ticks])
ax.set_yticks([lat_to_mercator_y(lat) for lat in lat_ticks])

ax.set_xticklabels([lon_formatter(lon) for lon in lon_ticks])
ax.set_yticklabels([lat_formatter(lat) for lat in lat_ticks])

ax.set_aspect("equal", adjustable="box")

plt.margins(x=0.02)
plt.savefig("ais_gaps_heatmap.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()