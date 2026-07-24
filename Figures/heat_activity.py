import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator
from pathlib import Path


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
files = [
    "heatmap_of_activity/2025_1_3_w_lstm_full_model.parquet",
    "heatmap_of_activity/2025_4_6_w_lstm_full_model.parquet",
    #"heatmap_of_activity/2025_1_3_w_lstm_full_model.parquet",
    #"heatmap_of_activity/2025_1_3_w_lstm_full_model.parquet"
]

# Region of interest
REGION_LAT_SOUTH = 55
REGION_LAT_NORTH = 82
REGION_LON_WEST = -10
REGION_LON_EAST = 45

# Grid size in degrees
# 0.05 degrees is roughly 5.5 km in latitude.
# Longitude distance varies with latitude.
GRID_SIZE = 0.05

# Maximum allowed time gap between AIS messages.
# Larger gaps are ignored, since we do not know what happened between messages.
MAX_GAP_HOURS = 6

# If True, only count time when the vessel remains in the same grid cell
# between two consecutive AIS messages.
# If False, assign the time interval to the grid cell of the first message.
SAME_CELL_ONLY = False


# ------------------------------------------------------------
# Read fishing predictions
# ------------------------------------------------------------
dfs = []

for f in files:
    table = pq.read_table(
        f,
        columns=["mmsi", "date_time_utc", "lon", "lat", "pred_fishing"],
        filters=[("pred_fishing", "==", 1)]
    )

    df_part = table.to_pandas()
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"], utc=True)

# Keep only valid positions inside region
df = df[
    (df["lat"] >= REGION_LAT_SOUTH) &
    (df["lat"] <= REGION_LAT_NORTH) &
    (df["lon"] >= REGION_LON_WEST) &
    (df["lon"] <= REGION_LON_EAST)
].copy()

df = df.dropna(subset=["mmsi", "date_time_utc", "lon", "lat"])

print(f"Fishing AIS messages inside region: {len(df):,}")
print(f"Unique vessels: {df['mmsi'].nunique():,}")


# ------------------------------------------------------------
# Assign each message to a grid cell
# ------------------------------------------------------------
df["lon_bin"] = np.floor((df["lon"] - REGION_LON_WEST) / GRID_SIZE).astype(int)
df["lat_bin"] = np.floor((df["lat"] - REGION_LAT_SOUTH) / GRID_SIZE).astype(int)

# Grid-cell lower-left corner
df["lon_min"] = REGION_LON_WEST + df["lon_bin"] * GRID_SIZE
df["lat_min"] = REGION_LAT_SOUTH + df["lat_bin"] * GRID_SIZE

# Grid-cell center, useful for plotting with scatter/pcolormesh
df["lon_center"] = df["lon_min"] + GRID_SIZE / 2
df["lat_center"] = df["lat_min"] + GRID_SIZE / 2


# ------------------------------------------------------------
# Compute fishing time per vessel per grid cell
# ------------------------------------------------------------
df = df.sort_values(["mmsi", "date_time_utc"]).copy()

# Time until next message from same vessel
df["next_time"] = df.groupby("mmsi")["date_time_utc"].shift(-1)
df["next_lon_bin"] = df.groupby("mmsi")["lon_bin"].shift(-1)
df["next_lat_bin"] = df.groupby("mmsi")["lat_bin"].shift(-1)

df["dt_hours"] = (
    df["next_time"] - df["date_time_utc"]
).dt.total_seconds() / 3600

# Remove invalid or too-large gaps
df = df[
    (df["dt_hours"] > 0) &
    (df["dt_hours"] <= MAX_GAP_HOURS)
].copy()

# Optional: only count intervals where the vessel stays in the same grid cell
if SAME_CELL_ONLY:
    df = df[
        (df["lon_bin"] == df["next_lon_bin"]) &
        (df["lat_bin"] == df["next_lat_bin"])
    ].copy()

print(f"Valid fishing intervals: {len(df):,}")
print(f"Total estimated fishing hours: {df['dt_hours'].sum():,.1f}")


# ------------------------------------------------------------
# Aggregate total fishing hours per grid cell
# ------------------------------------------------------------
effort = (
    df.groupby(["lat_bin", "lon_bin", "lat_center", "lon_center"], as_index=False)
      .agg(
          fishing_hours=("dt_hours", "sum"),
          n_messages=("dt_hours", "size"),
          n_vessels=("mmsi", "nunique")
      )
)

print(effort.sort_values("fishing_hours", ascending=False).head())


# ------------------------------------------------------------
# Convert to 2D grid for heatmap
# ------------------------------------------------------------
n_lat = int(np.ceil((REGION_LAT_NORTH - REGION_LAT_SOUTH) / GRID_SIZE))
n_lon = int(np.ceil((REGION_LON_EAST - REGION_LON_WEST) / GRID_SIZE))

heatmap = np.full((n_lat, n_lon), np.nan)

for row in effort.itertuples(index=False):
    heatmap[int(row.lat_bin), int(row.lon_bin)] = row.fishing_hours

lon_edges = np.arange(REGION_LON_WEST, REGION_LON_EAST + GRID_SIZE, GRID_SIZE)
lat_edges = np.arange(REGION_LAT_SOUTH, REGION_LAT_NORTH + GRID_SIZE, GRID_SIZE)


# ------------------------------------------------------------
# Plot heatmap
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.axisbelow": True,
})

# Use LogNorm because fishing effort is usually highly skewed.
# Very small values are clipped to avoid log(0).
MIN_HOURS_TO_PLOT = 1   # hide cells with less than 1 fishing hour

heatmap_plot = heatmap.copy()
heatmap_plot[heatmap_plot < MIN_HOURS_TO_PLOT] = np.nan

positive_values = heatmap_plot[np.isfinite(heatmap_plot) & (heatmap_plot > 0)]

if len(positive_values) == 0:
    raise ValueError("No grid cells above MIN_HOURS_TO_PLOT. Try lowering the threshold.")

vmin = MIN_HOURS_TO_PLOT
vmax = positive_values.max()

fig, ax = plt.subplots(figsize=(12, 12))

mesh = ax.pcolormesh(
    lon_edges,
    lat_edges,
    heatmap_plot,
    shading="auto",
    norm=LogNorm(vmin=vmin, vmax=vmax)
)

cbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Estimated fishing hours")

ax.set_xlim(REGION_LON_WEST, REGION_LON_EAST)
ax.set_ylim(REGION_LAT_SOUTH, REGION_LAT_NORTH)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Estimated fishing effort by the Norwegian fishing fleet")

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))

ax.grid(True, linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Optional: save aggregated effort
# ------------------------------------------------------------
#output_path = Path("heatmap_of_activity/fishing_effort_grid_hours.parquet")
#effort.to_parquet(output_path, index=False)

#print(f"Saved grid effort to: {output_path}")