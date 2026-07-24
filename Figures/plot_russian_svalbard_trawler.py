import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import pandas as pd
from shapely.geometry import LineString
import numpy as np

R = 6378137.0  # radius used in Web Mercator

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.axisbelow": True,
})

#df = pd.read_parquet("russian_svalbard_trawler_pred.parquet")
df = pd.read_parquet("pred_russian_trawler_lstm_full.parquet", engine="pyarrow")
print(df.columns)
print(df.head())

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
#df = df.loc[df["date_time_utc"].between("2022-01-07 00:00:00", "2022-01-07 08:00:00")]
df = df.sort_values("date_time_utc")
plt.plot(df["lon"], df["lat"])
plt.show()

lon_min, lon_max = 8, 14
lat_min, lat_max = 77.75, 78.5

# Create trajectory LineString
traj = LineString(zip(df["lon"], df["lat"]))

gdf_traj = gpd.GeoDataFrame(
    geometry=[traj],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# SUBSEA CABLE PLOT
cable_lats = [77.640, 77.764, 77.771, 77.881, 77.933, 77.959, 78.030, 78.103, 78.112, 78.185, 78.209, 78.241, 78.263, 78.276, 78.290, 78.302, 78.311, 78.320, 78.321, 78.305, 78.287, 78.249, 78.187, 78.122, 78.074, 78.062, 78.062, 78.078, 78.107, 78.145, 78.141]
cable_lons = [8.556, 8.178, 8.174, 8.189, 8.295, 8.281, 8.238, 8.200, 8.203, 8.243, 8.289, 8.357, 8.488, 8.638, 8.901, 9.105, 9.307, 9.532, 9.633, 9.918, 10.155, 10.521, 11.101, 11.642, 12.042, 12.276, 12.554, 12.838, 13.289, 13.814, 14.247]
cable = LineString(zip(cable_lons, cable_lats))

gdf_cable = gpd.GeoDataFrame(
    geometry=[cable],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Fishing points
df_pred_fish = df[df["pred_fishing"] == 1]


gdf_pred_fish = gpd.GeoDataFrame(
    df_pred_fish,
    geometry=gpd.points_from_xy(
        df_pred_fish["lon"],
        df_pred_fish["lat"]
    ),
    crs="EPSG:4326"
).to_crs(epsg=3857)


bbox_poly = gpd.GeoDataFrame(
    geometry=[LineString([
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min)
    ])],
    crs="EPSG:4326"
).to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))

# Cable
gdf_cable.plot(
    ax=ax,
    color="purple",
    linewidth=1.5,
    linestyle="--",
    zorder=1,
    label="Subsea cable"
)

# Full trajectory
gdf_traj.plot(
    ax=ax,
    color="blue",
    linewidth=2,
    label="Predicted non-fishing",
    zorder=3
)


# Highlight contiguous pred_fish segments
mask = df["pred_fishing"].eq(1)
segment_id = (mask != mask.shift()).cumsum()

first = True

for _, seg in df[mask].groupby(segment_id):

    if len(seg) < 2:
        continue

    seg_line = LineString(zip(seg["lon"], seg["lat"]))

    gpd.GeoSeries(
        [seg_line],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).plot(
        ax=ax,
        color="#47d147",
        linewidth=2,
        zorder=5,
        label="Predicted fishing" if first else None
    )

    first = False

# Add basemap

minx, miny, maxx, maxy = bbox_poly.total_bounds
pad_x = (maxx - minx) * 0.02
pad_y = (maxy - miny) * 0.01
ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy + pad_y)
ax.set_aspect("equal", adjustable="box")

# Basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=8)

# ----------------------------------------------------------
# Gridlines
# ----------------------------------------------------------
def lon_to_x(lon):
    return lon * np.pi * R / 180.0

def lat_to_y(lat):
    return R * np.log(np.tan(np.pi/4 + np.radians(lat)/2))

def round_up_to_quarter(x):
    return np.ceil(x / 0.25) * 0.25

grid_lon_start = round_up_to_quarter(lon_min)
grid_lat_start = round_up_to_quarter(lat_min)

lon_ticks = np.arange(grid_lon_start, lon_max + 0.001, 1)
lat_ticks = np.arange(grid_lat_start, lat_max + 0.001, 0.25)

ax.set_xticks([lon_to_x(l) for l in lon_ticks])
ax.set_yticks([lat_to_y(l) for l in lat_ticks])

# Labels:


lon_labels = [
    rf"{l:.2f}°E" if abs((l * 100) % 100) < 1e-6 else "" 
    for l in lon_ticks
]

lat_labels = [
    rf"{l:.2f}°N"
    for l in lat_ticks
]

ax.set_xticklabels(lon_labels)
ax.set_yticklabels(lat_labels)

# Style
ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

ax.legend(loc="upper right")
plt.margins(x=0.02)

plt.savefig("russian_svalbard_trawler.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()