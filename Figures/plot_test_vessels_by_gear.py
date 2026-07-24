import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

tick_gap_lon = 2
tick_gap_lat = 0.5

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

GEAR = "Not"
GEARS = ['Snurrevad', 'Garn', 'Trål', 'Not', 'Krokredskap', 'Bur og ruser']
#MONTH = 2

df_lstm = pd.read_parquet("plot_test_vessels/LSTM_2024_UNseen_test_seed0.parquet", engine="pyarrow", 
                          columns=["mmsi", "trajectory_id", "date_time_utc", "y_train", "report", "gear_report", "lon", "lat", "pred_fishing"])
df_bilstm = pd.read_parquet("plot_test_vessels/BiLSTM_2024_UNseen_test_seed0.parquet", engine="pyarrow",
                            columns=["mmsi", "trajectory_id", "date_time_utc", "y_train", "report", "gear_report", "lon", "lat", "pred_fishing"])

df_lstm["date_time_utc"] = pd.to_datetime(df_lstm["date_time_utc"])
#df_lstm = df_lstm[df_lstm["date_time_utc"].between(f"2024-{MONTH:02d}-01", f"2024-{MONTH:02d}-20")]

#df_bilstm["date_time_utc"] = pd.to_datetime(df_bilstm["date_time_utc"])
#df_bilstm = df_bilstm[df_bilstm["date_time_utc"].between(f"2024-{MONTH:02d}-01", f"2024-{MONTH:02d}-20")]

#TRAJ_FILTER = "257219000-1-2024-2"
#df_lstm = df_lstm[df_lstm["trajectory_id"] == TRAJ_FILTER]
#df_bilstm = df_bilstm[df_bilstm["trajectory_id"] == TRAJ_FILTER]

LEGEND = False


def filter_for_gear_vs_no_fishing(
    df,
    gear_type,
    no_gear="no_fishing",
    gear_col="gear_report",
    time_col="date_time_utc",
):
    print("getting trajectories that has a message reported as ", GEAR)
    allowed_gear = [gear_type, no_gear]

    allowed_mask = df[gear_col].isin(allowed_gear)
    has_gear_mask = df[gear_col].eq(gear_type)

    valid_by_traj = (
        allowed_mask.groupby(df["trajectory_id"]).all()
        &
        has_gear_mask.groupby(df["trajectory_id"]).any()
    )

    valid_ids = valid_by_traj[valid_by_traj].index

    df_out = df[df["trajectory_id"].isin(valid_ids)].copy()
    df_out[time_col] = pd.to_datetime(df_out[time_col])

    df_out = (
        df_out
        .sort_values(["trajectory_id", time_col])
        .reset_index(drop=True)
    )

    df_out["row_id"] = np.arange(len(df_out))
    print("Done getting trajectories!")
    return df_out

df_lstm = filter_for_gear_vs_no_fishing(df_lstm, GEAR)
df_bilstm = filter_for_gear_vs_no_fishing(df_bilstm, GEAR)

print("NR OF VESSELS: ", df_lstm["mmsi"].nunique())
print("NR OF trajectories: ", df_lstm["trajectory_id"].nunique())

def lon_formatter(x, pos):
    if x >= 0:
        return f"{x:.2f}°E"
    else:
        return f"{abs(x):.2f}°W"


def lat_formatter(y, pos):
    if y >= 0:
        return f"{y:.2f}°N"
    else:
        return f"{abs(y):.2f}°S"


def get_limits(d, pad_fraction=0.05):
    lon_min, lon_max = d["lon"].min(), d["lon"].max()
    lat_min, lat_max = d["lat"].min(), d["lat"].max()

    lon_pad = pad_fraction * (lon_max - lon_min)
    lat_pad = pad_fraction * (lat_max - lat_min)

    if lon_pad == 0:
        lon_pad = 0.01
    if lat_pad == 0:
        lat_pad = 0.01

    return (
        (lon_min - lon_pad, lon_max + lon_pad),
        (lat_min - lat_pad, lat_max + lat_pad),
    )


def style_axis(ax, mean_lat, xlim, ylim, show_ticks=True):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect(
        1 / np.cos(np.deg2rad(mean_lat)),
        adjustable="datalim"
    )

    ax.xaxis.set_major_locator(MultipleLocator(tick_gap_lon))
    ax.yaxis.set_major_locator(MultipleLocator(tick_gap_lat))

    # Show two decimals on longitude and latitude ticks
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.6,
    )

    ax.set_facecolor("#D4DADC")

    if not show_ticks:
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )
        ax.set_ylabel("")


def plot_contiguous_segments(
    ax,
    d,
    mask,
    color,
    linewidth,
    label,
    linestyle,
    zorder=3,
    alpha=1,
):
    """
    Plot contiguous True segments of mask as lines.
    """
    segment_id = (mask != mask.shift()).cumsum()

    first = True

    for _, seg in d[mask].groupby(segment_id):
        if len(seg) < 2:
            continue

        ax.plot(
            seg["lon"],
            seg["lat"],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zorder,
            label=label if first else None,
        )

        first = False

def plot_prediction_comparison(
    ax,
    d,
    gear,
    show_ylabel=True,
):
    d = d.sort_values("date_time_utc").copy()
    print(d["report"])

    pred_fishing = d["pred_fishing"].astype(bool)
    true_fishing = d["gear_report"].eq(gear)
    true_non_fishing = d["report"].eq("conf_no_fishing")

    tp_mask = pred_fishing & true_fishing
    fn_mask = ~pred_fishing & true_fishing
    fp_mask = pred_fishing & true_non_fishing

    # Full trajectory
    ax.plot(
        d["lon"],
        d["lat"],
        color="dimgray",
        linewidth=1,
        linestyle="--",
        zorder=1,
        alpha=0.5,
    )

    # Predicted non-fishing
    plot_contiguous_segments(
        ax,
        d,
        ~pred_fishing,
        color="blue",
        linewidth=2,
        linestyle="-",
        label="Predicted non-fishing",
        zorder=2,
        alpha=1,
    )

    # Predicted positives
    plot_contiguous_segments(
        ax,
        d,
        pred_fishing,
        color="#47d147", # 
        linewidth=2,
        linestyle="-",
        label="Predicted fishing",
        zorder=4,
    )

    # False positives
    """plot_contiguous_segments(
        ax,
        d,
        fp_mask,
        color="black", # 
        linewidth=2,
        linestyle="-",
        label="False positive",
        zorder=10,
    ) """

    # True positives
    plot_contiguous_segments(
        ax,
        d,
        tp_mask,
        color="red",
        linewidth=2,
        linestyle="-",
        label="TP",
        zorder=6,
    )

    # False negatives
    plot_contiguous_segments(
        ax,
        d,
        fn_mask,
        color="#ffcc00", # gold best  , #e6e600 
        linewidth=2,
        linestyle="-",
        label="FN",
        zorder=5,
    )


def link_axes(ax1, ax2):
    syncing = {"active": False}

    def sync_xlim(changed_ax):
        if syncing["active"]:
            return

        syncing["active"] = True
        other_ax = ax2 if changed_ax is ax1 else ax1
        other_ax.set_xlim(changed_ax.get_xlim())
        other_ax.figure.canvas.draw_idle()
        syncing["active"] = False

    def sync_ylim(changed_ax):
        if syncing["active"]:
            return

        syncing["active"] = True
        other_ax = ax2 if changed_ax is ax1 else ax1
        other_ax.set_ylim(changed_ax.get_ylim())
        other_ax.figure.canvas.draw_idle()
        syncing["active"] = False

    ax1.callbacks.connect("xlim_changed", sync_xlim)
    ax2.callbacks.connect("xlim_changed", sync_xlim)
    ax1.callbacks.connect("ylim_changed", sync_ylim)
    ax2.callbacks.connect("ylim_changed", sync_ylim)


eligible_traj_ids = []

for traj_id, d in df_lstm.groupby("trajectory_id"):
    report_mask = d["report"].isin(GEARS)

    if report_mask.any():
        eligible_traj_ids.append(traj_id)

eligible_traj_ids = sorted(
    set(df_lstm["trajectory_id"]).intersection(df_bilstm["trajectory_id"])
)

rng = np.random.default_rng(seed=5)
rng.shuffle(eligible_traj_ids)

#eligible_traj_ids = ["259563000-3-2024-8"]

for traj_id in eligible_traj_ids:
    d_lstm = df_lstm[df_lstm["trajectory_id"] == traj_id].copy()
    d_bilstm = df_bilstm[df_bilstm["trajectory_id"] == traj_id].copy()

    print("Reported in this trajectory: ", d_lstm["gear_report"].unique())

    if d_lstm.empty or d_bilstm.empty:
        continue

    # Use combined limits so the maps are directly comparable
    d_combined = pd.concat([d_lstm, d_bilstm], ignore_index=True)

    mean_lat = d_combined["lat"].mean()
    xlim, ylim = get_limits(d_combined)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

    lstm_counts = plot_prediction_comparison(
        axes[0],
        d_lstm,
        gear=GEAR,
        show_ylabel=True,
    )

    bilstm_counts = plot_prediction_comparison(
        axes[1],
        d_bilstm,
        gear=GEAR,
        show_ylabel=False,
    )

    style_axis(
        axes[0],
        mean_lat,
        xlim,
        ylim,
        show_ticks=True,
    )

    style_axis(
        axes[1],
        mean_lat,
        xlim,
        ylim,
        show_ticks=False,
    )

    link_axes(axes[0], axes[1])

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    if LEGEND:
        leg = axes[0].legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            frameon=True,
            facecolor="white",
            framealpha=0.85,
        )
        leg.set_zorder(100)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.12,
        top=0.95,
        wspace=0.02,
    )

    plt.show(block=False)

    choice = input(
        f"Trajectory {traj_id}: zoom/pan, then type 's' to save, "
        "Enter to skip, or 'q' to quit: "
    ).strip().lower()

    if choice == "s":
        fig.savefig(
            f"plot_test_vessels/{GEAR}_trajectory_{traj_id}_new_colors.pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"Saved trajectory {traj_id}")

    elif choice == "q":
        plt.close(fig)
        break

    else:
        print(f"Skipped trajectory {traj_id}")

    plt.close(fig)
