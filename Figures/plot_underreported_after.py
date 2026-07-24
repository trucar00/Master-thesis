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

df = pd.read_parquet("labeled/Snurrevad_2024_1_3.parquet")
print(df.columns)
print(df.head())

t_id = "257005140-5-2024-3"

df_v = df.loc[df["trajectory_id"] == t_id].copy()
df_v["date_time_utc"] = pd.to_datetime(df_v["date_time_utc"])
df_v = df_v.sort_values("date_time_utc")
plt.plot(df_v["lon"], df_v["lat"])
plt.show()

lon_min, lon_max = 5.4, 5.8
lat_min, lat_max = 62.71, 62.85

# Create trajectory LineString
traj = LineString(zip(df_v["lon"], df_v["lat"]))

gdf_traj = gpd.GeoDataFrame(
    geometry=[traj],
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Fishing points
df_snurrevad = df_v[df_v["report"] == "Snurrevad"]
df_unknown = df_v[df_v["unknown_no_fishing"] == True]
df_conf = df_v[df_v["conf_no_fishing"] == True]

gdf_snurrevad = gpd.GeoDataFrame(
    df_snurrevad,
    geometry=gpd.points_from_xy(
        df_snurrevad["lon"],
        df_snurrevad["lat"]
    ),
    crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_unknown = gpd.GeoDataFrame(
    df_unknown,
    geometry=gpd.points_from_xy(
        df_unknown["lon"],
        df_unknown["lat"]
    ),
    crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_conf = gpd.GeoDataFrame(
    df_conf,
    geometry=gpd.points_from_xy(
        df_conf["lon"],
        df_conf["lat"]
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

# Full trajectory
gdf_traj.plot(
    ax=ax,
    color="blue",
    linewidth=1.5,
    zorder=5
)

# Highlight contiguous Snurrevad segments
mask = df_v["report"].eq("Snurrevad")
segment_id = (mask != mask.shift()).cumsum()

first = True

for _, seg in df_v[mask].groupby(segment_id):

    if len(seg) < 2:
        continue

    seg_line = LineString(zip(seg["lon"], seg["lat"]))

    gpd.GeoSeries(
        [seg_line],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).plot(
        ax=ax,
        color="red",
        linewidth=5,
        zorder=4,
        label="Reported fishing" if first else None
    )

    first = False

mask_unknown = df_v["unknown_no_fishing"].eq(True)
segment_id = (mask_unknown != mask_unknown.shift()).cumsum()

first = True

for _, seg in df_v[mask_unknown].groupby(segment_id):

    if len(seg) < 2:
        continue

    seg_line = LineString(zip(seg["lon"], seg["lat"]))

    gpd.GeoSeries(
        [seg_line],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).plot(
        ax=ax,
        color="orange",
        linewidth=5,
        zorder=3,
        label="Unknown" if first else None
    )

    first = False

mask_conf = df_v["conf_no_fishing"].eq(True)
segment_id = (mask_conf != mask_conf.shift()).cumsum()

first = True

for _, seg in df_v[mask_conf].groupby(segment_id):

    if len(seg) < 2:
        continue

    seg_line = LineString(zip(seg["lon"], seg["lat"]))

    gpd.GeoSeries(
        [seg_line],
        crs="EPSG:4326"
    ).to_crs(epsg=3857).plot(
        ax=ax,
        color="lightgreen",
        linewidth=5,
        zorder=3,
        label="Confident non-fishing" if first else None
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

lon_ticks = np.arange(grid_lon_start, lon_max + 0.001, 0.25)
lat_ticks = np.arange(grid_lat_start, lat_max + 0.001, 0.25)

ax.set_xticks([lon_to_x(l) for l in lon_ticks])
ax.set_yticks([lat_to_y(l) for l in lat_ticks])

# Labels:


lon_labels = [
    rf"${l:.2f}^\circ$E"
    for l in lon_ticks
]

lat_labels = [
    rf"${l:.2f}^\circ$N"
    for l in lat_ticks
]

ax.set_xticklabels(lon_labels)
ax.set_yticklabels(lat_labels)

# Style
ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

ax.legend(loc="upper left")
plt.margins(x=0.02)

plt.savefig("scottish_after_label_clean.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()