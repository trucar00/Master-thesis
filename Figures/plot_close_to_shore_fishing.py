import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from matplotlib.collections import LineCollection

R = 6378137.0  # radius used in Web Mercator
GEARS = ['Snurrevad', 'Garn', 'Trål', 'Not', 'Krokredskap', 'Bur og ruser']

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.axisbelow": True,
})

# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
lon_min = 21.10
lon_max = 21.85
lat_min = 70.00
lat_max = 70.18

filename_lstm = "plot_test_vessels/LSTM_2024_UNseen_test_seed0.parquet"
filename_bilstm = "plot_test_vessels/BiLSTM_2024_UNseen_test_seed0.parquet"

columns = [
    "mmsi",
    "date_time_utc",
    "lon",
    "lat",
    "trajectory_id",
    "pred_fishing",
    "gear_report",
    "report",
]

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def lon_to_x(lon):
    return lon * np.pi * R / 180.0

def lat_to_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

def round_up_to_step(x, step):
    return np.ceil(x / step) * step

def find_traj_ids_inside_bbox(filename):
    table_bbox = pq.read_table(
        filename,
        columns=["trajectory_id", "lon", "lat"],
    )

    df_bbox = table_bbox.to_pandas()

    inside_mask = (
        df_bbox["lon"].between(lon_min, lon_max)
        & df_bbox["lat"].between(lat_min, lat_max)
    )

    return (
        df_bbox.loc[inside_mask, "trajectory_id"]
        .drop_duplicates()
        .to_numpy()
    )

def load_model_plot_df(filename, traj_ids_selected):
    df = pd.read_parquet(
        filename,
        engine="pyarrow",
        columns=columns,
        filters=[("trajectory_id", "in", traj_ids_selected.tolist())],
    )

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"]).copy()

    # Convert to Web Mercator for contextily
    df["x"] = lon_to_x(df["lon"].to_numpy())
    df["y"] = lat_to_y(df["lat"].to_numpy())

    return df

def plot_contiguous_segments(
    ax,
    d,
    mask,
    color,
    linewidth=4,
    linestyle="-",
    alpha=1,
    zorder=5,
    label=None,
):
    """
    Plot each contiguous True block in mask as one continuous line.

    This avoids plotting predicted fishing as separate tiny point-to-point
    segments and instead joins consecutive predicted-fishing messages.
    """
    mask = mask.astype(bool)

    if len(d) < 2 or not mask.any():
        return False

    # New segment every time the mask changes True/False
    segment_id = (mask != mask.shift(fill_value=False)).cumsum()

    first = True
    plotted_any = False

    for _, seg in d.loc[mask].groupby(segment_id):
        if len(seg) < 2:
            continue

        ax.plot(
            seg["x"].to_numpy(),
            seg["y"].to_numpy(),
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            zorder=zorder,
            label=label if first else None,
        )

        first = False
        plotted_any = True

    return plotted_any

def plot_model(ax, df_plot):
    first_pred_label = True
    first_nonfish_label = True

    for _, d in df_plot.groupby("trajectory_id", sort=False):
        d = d.sort_values("date_time_utc")

        # Plot full trajectory in blue
        ax.plot(
            d["x"].to_numpy(),
            d["y"].to_numpy(),
            color="blue",
            linewidth=2,
            #alpha=0.5,
            zorder=2,
            label="Predicted non-fishing" if first_nonfish_label else None,
        )
        first_nonfish_label = False

        # Plot predicted fishing as joined green segments
        pred_fishing = d["pred_fishing"].astype(bool)
        true_fishing = d["gear_report"].isin(GEARS)

        tp_mask = pred_fishing & true_fishing
        fn_mask = ~pred_fishing & true_fishing

        plotted = plot_contiguous_segments(
            ax,
            d,
            pred_fishing,
            color="#47d147",
            linewidth=2,
            zorder=5,
            label="Predicted fishing" if first_pred_label else None,
        )

        plot_contiguous_segments(
            ax,
            d,
            tp_mask,
            color="red",
            linewidth=2,
            zorder=5,
            label="TP" if first_pred_label else None,
        )

        plot_contiguous_segments(
            ax,
            d,
            fn_mask,
            color="orange",
            linewidth=2,
            zorder=5,
            label="FN" if first_pred_label else None,
        )

        if plotted:
            first_pred_label = False

# ----------------------------------------------------------
# Find trajectories inside bbox in BOTH files
# ----------------------------------------------------------
traj_ids_inside_lstm = find_traj_ids_inside_bbox(filename_lstm)
traj_ids_inside_bilstm = find_traj_ids_inside_bbox(filename_bilstm)

traj_ids_inside = np.intersect1d(
    traj_ids_inside_lstm,
    traj_ids_inside_bilstm,
)

print("Trajectories inside bbox in both files:", len(traj_ids_inside))

# Randomly keep same subset
rng = np.random.default_rng(seed=42)
selected_traj_ids = rng.choice(
    traj_ids_inside,
    size=max(1, int(0.25 * len(traj_ids_inside))),
    replace=False,
)

print("len after:", len(selected_traj_ids))

# ----------------------------------------------------------
# Load selected trajectories
# ----------------------------------------------------------
df_plot = load_model_plot_df(filename_bilstm, selected_traj_ids)

# ----------------------------------------------------------
# Shared map extent
# ----------------------------------------------------------
x_min, x_max = lon_to_x(lon_min), lon_to_x(lon_max)
y_min, y_max = lat_to_y(lat_min), lat_to_y(lat_max)

pad_x = (x_max - x_min) * 0.02
pad_y = (y_max - y_min) * 0.02

# ----------------------------------------------------------
# Ticks and gridlines
# ----------------------------------------------------------
lon_step = 0.2
lat_step = 0.05

grid_lon_start = round_up_to_step(lon_min, lon_step)
grid_lat_start = round_up_to_step(lat_min, lat_step)

lon_ticks = np.arange(grid_lon_start, lon_max + 1e-9, lon_step)
lat_ticks = np.arange(grid_lat_start, lat_max + 1e-9, lat_step)

# ----------------------------------------------------------
# Plot
# ----------------------------------------------------------
map_aspect = (y_max - y_min) / (x_max - x_min)
fig_width = 10
fig_height = fig_width * map_aspect

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

plot_model(ax, df_plot)

ax.set_xlim(x_min - pad_x, x_max + pad_x)
ax.set_ylim(y_min - pad_y, y_max + pad_y)
ax.set_aspect("equal", adjustable="box")

ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.PositronNoLabels,
    zoom=10,   # may want 11 or 12 depending on area size
)

ax.set_xticks([lon_to_x(lon) for lon in lon_ticks])
ax.set_yticks([lat_to_y(lat) for lat in lat_ticks])

ax.set_xticklabels([rf"{lon:.2f}°E" for lon in lon_ticks])
ax.set_yticklabels([rf"{lat:.2f}°N" for lat in lat_ticks])

ax.grid(
    color="gray",
    linestyle="--",
    linewidth=0.7,
    alpha=0.6,
    zorder=10,
)

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
if unique:
    ax.legend(unique.values(), unique.keys(), loc="upper right")

plt.margins(x=0.02)
""" plt.savefig(
    "close_to_shore_fishing_bilstm.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
) """

plt.show()