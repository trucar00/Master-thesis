import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
import contextily as ctx
from shapely.geometry import LineString
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

# PLOT MORE, because ship is supposed to back up along the set lines. So it is drifting/waiting possibly turning off AIS. Maybe try other?

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

# ----------------------------------------------------------
# 1. Load trajectory
# ----------------------------------------------------------
df = pd.read_parquet("labeled/ais_ers_krok_09_2024.parquet", engine="pyarrow")
df_v = df[df["mmsi"] == 257133000].copy()

print(df_v.head())

df_v["date_time_utc"] = pd.to_datetime(df_v["date_time_utc"])
df_v = df_v.sort_values("date_time_utc")
print(df_v["label"].unique())

df_v = df_v.loc[df_v["date_time_utc"].between("2024-09-02", "2024-09-09 18:00:00")]
df_report = df_v.loc[df_v["label"] == "Krokredskap"]

plt.scatter(df_v["lon"], df_v["lat"], s=3)
plt.scatter(df_report["lon"], df_report["lat"], s=3, color="red")
plt.show()


#print(df_v.head())

lons = df_v["lon"].values
lats = df_v["lat"].values


coords = list(zip(lons, lats))
traj = LineString(coords)

gdf_traj = gpd.GeoDataFrame(
    {"id": [1]},
    geometry=[traj],
    crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_report = gpd.GeoDataFrame(
    df_report,
    geometry=gpd.points_from_xy(df_report["lon"], df_report["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)


# ----------------------------------------------------------
# 2. Bounding box for main plot
# ----------------------------------------------------------
lon_min, lon_max = 21, 27
lat_min, lat_max = 70.25, 71.25

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

# ----------------------------------------------------------
# 3. Create figure and main axis
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))

# Plot trajectory

gdf_traj.plot(ax=ax, color="blue", linewidth=1.5)

ax.scatter(
    gdf_report.geometry.x,
    gdf_report.geometry.y,
    s=3,
    color="red",
    label="Reported fishing",
    zorder=10
)

# Main extent
minx, miny, maxx, maxy = bbox_poly.total_bounds
print(minx, maxx)
pad_x = (maxx - minx) * 0.02
pad_y = (maxy - miny) * 0.01
ax.set_xlim(minx - pad_x, maxx + pad_x)
ax.set_ylim(miny - pad_y, maxy + pad_y)
ax.set_aspect("equal", adjustable="box")

# Basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=8)

# ----------------------------------------------------------
# Add longitude/latitude gridlines
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
ax.tick_params(color="gray")
ax.grid(color="gray", linestyle="--", linewidth=0.7, alpha=0.6)



# ----------------------------------------------------------
# 4. Inset zoom map (USER-CHOSEN REGION)
# ----------------------------------------------------------
# Choose region manually (lon/lat)
zoom_lon_min, zoom_lon_max = 21.55, 21.80
zoom_lat_min, zoom_lat_max = 70.87, 70.895

# Create bounding box polygon in EPSG:4326
from shapely.geometry import Polygon

zoom_box_poly = Polygon([
    (zoom_lon_min, zoom_lat_min),
    (zoom_lon_max, zoom_lat_min),
    (zoom_lon_max, zoom_lat_max),
    (zoom_lon_min, zoom_lat_max)
])

zoom_box = gpd.GeoDataFrame(
    geometry=[zoom_box_poly],
    crs="EPSG:4326"
)

# Clip the trajectory cleanly
gdf_sub = gpd.clip(gdf_traj.to_crs(4326), zoom_box).to_crs(3857)
gdf_report_sub = gpd.clip(gdf_report.to_crs(4326), zoom_box).to_crs(3857)

zoom_bounds = zoom_box.to_crs(3857).total_bounds
minx2, miny2, maxx2, maxy2 = zoom_bounds

zoom_width = maxx2 - minx2
zoom_height = maxy2 - miny2

aspect_ratio = zoom_width / zoom_height

width_zoom_box = (maxx - minx) * 0.60
height_zoom_box = (width_zoom_box / aspect_ratio)

prc_height = height_zoom_box / (maxy - miny)

print("y%: ", prc_height)

# Choose inset width
inset_width = 0.90   # fraction of parent axes

# Compute matching height
inset_height = inset_width / aspect_ratio

# Create inset axis
in_ax = inset_axes(ax, width="60%", height=f"{prc_height*100:.1f}%", bbox_to_anchor=(-0.001, 0.1, 1, 1), bbox_transform=ax.transAxes, loc="lower right") #  bbox_to_anchor=(0.001, -0.4, 1, 1), bbox_transform=ax.transAxes,

# Plot inside inset

# Plot inside inset
gdf_sub.plot(ax=in_ax, color="blue", linewidth=1.6)

in_ax.scatter(
    gdf_report_sub.geometry.x,
    gdf_report_sub.geometry.y,
    s=8,
    color="red",
    zorder=10
)

rect = patches.Rectangle(
    (lon_to_x(zoom_lon_min), lat_to_y(zoom_lat_min)),
    lon_to_x(zoom_lon_max) - lon_to_x(zoom_lon_min),
    lat_to_y(zoom_lat_max) - lat_to_y(zoom_lat_min),
    linewidth=1.0,
    edgecolor='black',
    facecolor='none'
)
ax.add_patch(rect)
ax.legend(loc="upper right")

# Use exact zoom region extent
zoom_bounds = zoom_box.to_crs(3857).total_bounds
minx2, miny2, maxx2, maxy2 = zoom_bounds

in_ax.set_xlim(minx2, maxx2)
in_ax.set_ylim(miny2, maxy2)
in_ax.set_aspect("equal", adjustable="box")

# Inset basemap (no watermark)
ctx.add_basemap(in_ax, source=ctx.providers.CartoDB.Positron, zoom=10, attribution=False)

box_corners = [
    (lon_to_x(zoom_lon_max), lat_to_y(zoom_lat_max)),  # bottom-left
    (lon_to_x(zoom_lon_max), lat_to_y(zoom_lat_min)),  # bottom-right
]

# Inset corners (in display coords)
inset_corners = [
    (0, 1),   # bottom-left inset axes
    (0, 0),   # bottom-right inset axes
]

for (bx, by), (ix, iy) in zip(box_corners, inset_corners):
    con = ConnectionPatch(
        xyA=(ix, iy), coordsA=in_ax.transAxes,
        xyB=(bx, by), coordsB=ax.transData,
        linestyle="--", color="black", linewidth=1
    )
    fig.add_artist(con)

in_ax.set_xticks([])
in_ax.set_yticks([])
in_ax.set_title("")  # remove accidental titles

# ----------------------------------------------------------
plt.margins(x=0.02)

plt.savefig("short_duration_line_plot.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()

