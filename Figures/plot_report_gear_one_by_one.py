import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

# ============================================================
# Settings
# ============================================================
# snurrevad example: 257149000-2-2023-4

tick_gap_lon = 0.1
tick_gap_lat = 0.05

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

GEAR = "Snurrevad"
GEARS = ["Snurrevad", "Garn", "Trål", "Not", "Krokredskap", "Bur og ruser"]

LEGEND = False
SHOW_UNKNOWN = False
SHOW_CONF = False

# ============================================================
# Load data
# ============================================================

df = pd.read_parquet(
    "labeled_gear/Snurrevad_2023_1_3.parquet",
    engine="pyarrow"
)

print("Number of vessels:", df["mmsi"].nunique())

# ============================================================
# Formatters
# ============================================================

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


def style_axis(ax, mean_lat, xlim, ylim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_aspect(
        1 / np.cos(np.deg2rad(mean_lat)),
        adjustable="datalim"
    )

    ax.xaxis.set_major_locator(MultipleLocator(tick_gap_lon))
    ax.yaxis.set_major_locator(MultipleLocator(tick_gap_lat))

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


def plot_contiguous_segments(
    ax,
    d,
    mask,
    color,
    linewidth,
    label,
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
            zorder=zorder,
            label=label if first else None,
        )

        first = False


def plot_trajectory(
    ax,
    d,
    report_mask,
    unknown_mask=None,
    conf_mask=None,
    show_unknown=True,
    show_conf=True,
):
    # Full trajectory
    ax.plot(
        d["lon"],
        d["lat"],
        color="blue",
        linewidth=2,
        label="Trajectory",
        zorder=1,
    )

    # Confident non-fishing segments
    if show_conf and conf_mask is not None:
        plot_contiguous_segments(
            ax,
            d,
            conf_mask,
            color="lime",
            linewidth=2,
            label="Confident non-fishing",
            zorder=2,
        )

    # Unknown segments
    if show_unknown and unknown_mask is not None:
        plot_contiguous_segments(
            ax,
            d,
            unknown_mask,
            color="black",
            linewidth=2,
            label="Unknown",
            zorder=2,
        )

    # Reported fishing segments
    plot_contiguous_segments(
        ax,
        d,
        report_mask,
        color="red",
        linewidth=2,
        label="Reported fishing",
        zorder=3,
    )


# ============================================================
# Find eligible trajectories and shuffle randomly
# ============================================================

eligible_traj_ids = []

for traj_id, d_traj in df.groupby("trajectory_id"):
    report_mask = d_traj["report"].isin(GEARS)

    if report_mask.any():
        eligible_traj_ids.append(traj_id)

rng = np.random.default_rng(seed=42)
rng.shuffle(eligible_traj_ids)

#eligible_traj_ids = ["258874000-3-2023-6"]

print("Eligible trajectories:", len(eligible_traj_ids))

# ============================================================
# Loop through random trajectories
# ============================================================

for traj_id in eligible_traj_ids:
    d = df[df["trajectory_id"] == traj_id].copy()
    d = d.sort_values("date_time_utc")

    report_mask = d["report"].isin(GEARS)
    unknown_mask = d["unknown_no_fishing"]
    conf_mask = d["conf_no_fishing"]

    if not report_mask.any():
        continue

    mean_lat = d["lat"].mean()
    xlim, ylim = get_limits(d)

    fig, ax = plt.subplots(figsize=(8, 7.5))

    plot_trajectory(
        ax,
        d,
        report_mask,
        unknown_mask=unknown_mask,
        conf_mask=conf_mask,
        show_unknown=SHOW_UNKNOWN,
        show_conf=SHOW_CONF,
    )

    style_axis(ax, mean_lat, xlim, ylim)

    if LEGEND:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))

        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            markerscale=2,
        )

    fig.subplots_adjust(
        left=0.2,
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
            f"labeled_gear/bef_unknown/{GEAR}_trajectory_{traj_id}.pdf",
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