import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

R = 6378137.0  # radius used in Web Mercator

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 24,
    "axes.labelsize": 24,
    "legend.fontsize": 24,
    "legend.title_fontsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "axes.axisbelow": True,
})


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def lon_to_x(lon):
    return lon * np.pi * R / 180.0


def lat_to_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def round_up_to_step(x, step):
    return np.ceil(x / step) * step


def plot_contiguous_segments(
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
    Plot contiguous True segments of mask as continuous lines.
    """
    mask = pd.Series(mask, index=d.index).astype(bool)

    if len(d) < 2 or not mask.any():
        return False

    # New segment every time the mask changes
    segment_id = (mask != mask.shift(fill_value=False)).cumsum()

    plotted_any = False
    first = True

    d_true = d.loc[mask]
    seg_true = segment_id.loc[mask]

    for _, seg in d_true.groupby(seg_true):
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


def load_model_plot_df(filename, traj_ids_selected):
    df = pd.read_parquet(
        filename,
        engine="pyarrow",
        columns=columns,
        filters=[("trajectory_id", "in", traj_ids_selected.tolist())],
    )

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"]).copy()

    df["x"] = lon_to_x(df["lon"].to_numpy())
    df["y"] = lat_to_y(df["lat"].to_numpy())

    return df


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


def plot_model(ax, df_plot, title=None, show_y_ticks=True):
    first_pred_label = True
    first_traj_label = True

    for _, d in df_plot.groupby("trajectory_id", sort=False):
        d = d.sort_values("date_time_utc").copy()

        # Full trajectory
        ax.plot(
            d["x"].to_numpy(),
            d["y"].to_numpy(),
            color="blue",
            linewidth=2,
            label="Predicted non-fishing" if first_traj_label else None,
            alpha=0.5,
            zorder=2,
        )
        first_traj_label = False

        pred_fishing = d["pred_fishing"].astype(bool)

        plotted = plot_contiguous_segments(
            ax,
            d,
            pred_fishing,
            color="#47d147",
            linewidth=4,
            zorder=5,
            label="Predicted fishing" if first_pred_label else None,
        )

        if plotted:
            first_pred_label = False

    # Extent
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_aspect("equal", adjustable="box")

    # Basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.PositronNoLabels,
        zoom=16,
    )

    # Ticks and gridline positions
    ax.set_xticks([lon_to_x(lon) for lon in lon_ticks])
    ax.set_yticks([lat_to_y(lat) for lat in lat_ticks])

    ax.set_xticklabels([rf"{lon:.3f}°E" for lon in lon_ticks])

    if show_y_ticks:
        ax.set_yticklabels([rf"{lat:.3f}°N" for lat in lat_ticks])
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            labelleft=True,
        )
    else:
        ax.set_yticklabels([])
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
        )

    ax.grid(
        color="gray",
        linestyle="--",
        linewidth=0.7,
        alpha=0.6,
        zorder=10,
    )

    if title is not None:
        ax.set_title(title)


# ----------------------------------------------------------
# Settings
# ----------------------------------------------------------
filename_lstm = "plot_test_vessels/LSTM_2024_UNseen_test_seed0.parquet"
filename_bilstm = "plot_test_vessels/BiLSTM_2024_UNseen_test_seed0.parquet"

lon_min, lon_max = 6.1125, 6.135
lat_min, lat_max = 62.460, 62.475

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
# Find trajectories inside bbox in BOTH files
# ----------------------------------------------------------
traj_ids_inside_lstm = find_traj_ids_inside_bbox(filename_lstm)
traj_ids_inside_bilstm = find_traj_ids_inside_bbox(filename_bilstm)

traj_ids_inside = np.intersect1d(
    traj_ids_inside_lstm,
    traj_ids_inside_bilstm,
)

print("Trajectories inside bbox in both files:", len(traj_ids_inside))


# ----------------------------------------------------------
# Randomly keep same 50% of trajectories for both models
# ----------------------------------------------------------
rng = np.random.default_rng(seed=42)

selected_traj_ids = rng.choice(
    traj_ids_inside,
    size=max(1, int(0.5 * len(traj_ids_inside))),
    replace=False,
)

print("len after:", len(selected_traj_ids))


# ----------------------------------------------------------
# Load selected trajectories from both files
# ----------------------------------------------------------
df_lstm_plot = load_model_plot_df(filename_lstm, selected_traj_ids)
df_bilstm_plot = load_model_plot_df(filename_bilstm, selected_traj_ids)


# ----------------------------------------------------------
# Shared map extent
# ----------------------------------------------------------
x_min, x_max = lon_to_x(lon_min), lon_to_x(lon_max)
y_min, y_max = lat_to_y(lat_min), lat_to_y(lat_max)

pad_x = (x_max - x_min) * 0.02
pad_y = (y_max - y_min) * 0.01


# ----------------------------------------------------------
# Shared ticks and gridlines
# ----------------------------------------------------------
lon_step = 0.025
lat_step = 0.01

grid_lon_start = round_up_to_step(lon_min, lon_step)
grid_lat_start = round_up_to_step(lat_min, lat_step)

lon_ticks = np.arange(grid_lon_start, lon_max + 0.0001, lon_step)
lat_ticks = np.arange(grid_lat_start, lat_max + 0.0001, lat_step)


# ----------------------------------------------------------
# Plot side by side
# ----------------------------------------------------------
map_aspect = (y_max - y_min) / (x_max - x_min)

single_width = 7.0
single_height = single_width * map_aspect

fig, axes = plt.subplots(
    1,
    2,
    figsize=(2 * single_width, single_height),
    gridspec_kw={"wspace": 0.01},
)

plot_model(
    axes[0],
    df_lstm_plot,
    title=None,
    show_y_ticks=True,
)

plot_model(
    axes[1],
    df_bilstm_plot,
    title=None,
    show_y_ticks=False,
)

# Shared legend from left axis
handles, labels = axes[0].get_legend_handles_labels()
unique = dict(zip(labels, handles))

if unique:
    axes[0].legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
    )

fig.subplots_adjust(
    left=0.09,
    right=0.98,
    bottom=0.12,
    top=0.95,
    wspace=0.02,
)

plt.savefig(
    "close_to_shore_lstm_bilstm.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()