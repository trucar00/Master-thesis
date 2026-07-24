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

df = pd.read_parquet("russian_trawler_pred.parquet")
print(df.columns)
print(df.head())

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
df = df.sort_values("date_time_utc")
plt.plot(df["lon"], df["lat"])
plt.show()

lon_min, lon_max = 13, 16
lat_min, lat_max = 68.6, 69.6

# Create trajectory LineString
traj = LineString(zip(df["lon"], df["lat"]))

gdf_traj = gpd.GeoDataFrame(
    geometry=[traj],
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

# Full trajectory
gdf_traj.plot(
    ax=ax,
    color="blue",
    linewidth=1.5,
    zorder=2
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
        color="red",
        linewidth=2,
        zorder=4,
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
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=7)

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
    rf"${l:.2f}^\circ$E" if abs((l * 100) % 50) < 1e-6 else ""
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

#plt.savefig("scottish_after_label_clean.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()