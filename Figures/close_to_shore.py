import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import pandas as pd
from shapely.geometry import LineString
import numpy as np
import random
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection

R = 6378137.0  # radius used in Web Mercator

def plot_masked_segments_fast(
    ax,
    d,
    mask,
    color,
    linewidth=2,
    linestyle="-",
    alpha=1,
    zorder=3,
    label=None,
):
    """
    Plot line segments where mask is True using one LineCollection.
    A segment from point i to i+1 is plotted if both endpoints have mask=True.
    """
    lon = d["lon"].to_numpy()
    lat = d["lat"].to_numpy()
    mask = mask.to_numpy(dtype=bool)

    if len(d) < 2:
        return

    segment_mask = mask[:-1] & mask[1:]

    if not segment_mask.any():
        return

    points = np.column_stack([lon, lat])

    segments = np.stack(
        [points[:-1][segment_mask], points[1:][segment_mask]],
        axis=1
    )

    lc = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        linestyles=linestyle,
        alpha=alpha,
        zorder=zorder,
        label=label,
    )

    ax.add_collection(lc)

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


filename = "plot_test_vessels/LSTM_2024_UNseen_test_seed0.parquet"

lon_min, lon_max = 6.10, 6.2
lat_min, lat_max = 62.445, 62.475

# Pass 1: only read tiny subset of columns
table_bbox = pq.read_table(
    filename,
    columns=["trajectory_id", "lon", "lat"]
)

df_bbox = table_bbox.to_pandas()

inside_mask = (
    df_bbox["lon"].between(lon_min, lon_max)
    & df_bbox["lat"].between(lat_min, lat_max)
)

traj_ids_inside = df_bbox.loc[inside_mask, "trajectory_id"].drop_duplicates().to_numpy()

print("Trajectories inside bbox:", len(traj_ids_inside))

columns = [
    "mmsi", "date_time_utc", "lon", "lat",
    "trajectory_id", "pred_fishing", "gear_report", "report"
]

df_lstm = pd.read_parquet(
    filename,
    engine="pyarrow",
    columns=columns,
    filters=[("trajectory_id", "in", traj_ids_inside.tolist())]
)

df_lstm["date_time_utc"] = pd.to_datetime(df_lstm["date_time_utc"])

traj_ids = df_lstm["trajectory_id"].drop_duplicates().to_numpy()

rng = np.random.default_rng(seed=42)
selected_traj_ids = rng.choice(
    traj_ids,
    size=int(0.25 * len(traj_ids)),
    replace=False
)

print("nr unique trajectories:", len(traj_ids))
print("len after:", len(selected_traj_ids))

df_plot = df_lstm[df_lstm["trajectory_id"].isin(selected_traj_ids)]
df_plot = df_plot.sort_values(["trajectory_id", "date_time_utc"])

fig, ax = plt.subplots(figsize=(10, 10))

for _, d in df_plot.groupby("trajectory_id", sort=False):
    d = d.sort_values("date_time_utc")

    # Full trajectory
    ax.plot(
        d["lon"].to_numpy(),
        d["lat"].to_numpy(),
        color="blue",
        linewidth=1,
        alpha=0.4,
        zorder=1,
    )

    pred_fishing = d["pred_fishing"].astype(bool)

    # Fast red fishing segments
    plot_masked_segments_fast(
        ax,
        d,
        pred_fishing,
        color="red",
        linewidth=2.5,
        zorder=3,
        label="Predicted fishing",
    )

plt.xlim(6.10, 6.20)
plt.ylim(62.455, 62.475)
plt.show()