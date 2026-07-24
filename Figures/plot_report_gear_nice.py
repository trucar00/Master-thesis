import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter

tick_gap_lon = 0.1
tick_gap_lat = 0.02

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

df = pd.read_parquet(f"labeled_gear/Not_2023_4_6.parquet", engine="pyarrow")
print(df["mmsi"].nunique())
df = df[df["trajectory_id"] == "257021970-0-2023-6"]
LEGEND = True

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
    # Full trajectory as one line
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
            color= "lime",
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
        zorder=2,
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

for traj_id, d in df.groupby("trajectory_id"):
    report_mask = d["report"].isin(GEARS)

    if report_mask.any():
        eligible_traj_ids.append(traj_id)

#eligible_traj_ids = ["259102000-12-2023-6"]

rng = np.random.default_rng(seed=42)
rng.shuffle(eligible_traj_ids)

for traj_id in eligible_traj_ids:
    d = df[df["trajectory_id"] == traj_id].copy()
    report_mask = d["report"].isin(GEARS)
    unknown_mask = d["unknown_no_fishing"]
    conf_mask = d["conf_no_fishing"]

    if not report_mask.any():
        continue

    mean_lat = d["lat"].mean()
    xlim, ylim = get_limits(d)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))

    # Left: without unknown highlighted
    plot_trajectory(
        axes[0],
        d,
        report_mask,
        unknown_mask=unknown_mask,
        conf_mask=conf_mask,
        show_unknown=False,
        show_conf=False,
    )
    style_axis(
        axes[0],
        mean_lat,
        xlim,
        ylim,
        show_ticks=True,
    )

    # Right: with unknown highlighted
    plot_trajectory(
        axes[1],
        d,
        report_mask,
        unknown_mask=unknown_mask,
        conf_mask=conf_mask,
        show_unknown=True,
        show_conf=True,
    )

    style_axis(
        axes[1],
        mean_lat,
        xlim,
        ylim,
        show_ticks=False,
    )

    link_axes(axes[0], axes[1])

    handles0, labels0 = axes[0].get_legend_handles_labels()
    handles1, labels1 = axes[1].get_legend_handles_labels()

    handles = handles0 + handles1
    labels = labels0 + labels1

    unique = dict(zip(labels, handles))

    if LEGEND:
        axes[0].legend(
            unique.values(),
            unique.keys(),
            loc="upper left",
            markerscale=2,
        )

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.12,
        top=0.95,
        wspace=0.02,
    )

    # For thesis export:
    # fig.savefig(f"trajectory_{traj_id}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.show(block=False)

    choice = input(
        f"Trajectory {traj_id}: zoom/pan, then type 's' to save, "
        "Enter to skip, or 'q' to quit: "
    ).strip().lower()

    if choice == "s":
        fig.savefig(
            f"labeled_gear/new/{GEAR}_trajectory_{traj_id}.pdf",
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
