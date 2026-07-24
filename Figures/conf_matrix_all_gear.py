import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.axisbelow": True,
})

# ------------------------------------------------------------
# Confusion matrices
# ------------------------------------------------------------

cm_unseen = np.array([
    [3209,  299,  155,  204,    30],
    [ 233, 4181,   11,  111,   611],
    [  60,   32, 1658,  463,     8],
    [ 251,  223,  561, 7923,    54],
    [  94, 1065,  218,  574, 26686]
])

cm_seen = np.array([
    [4000,  439,   91,   290,    51],
    [ 506, 8258,   79,   213,  1672],
    [ 150,   95, 3007,  1279,    80],
    [ 480,  369,  787, 14812,    83],
    [  99, 2368,  203,   564, 69212]
])

# Add seen and unseen matrices
cm = cm_seen #cm_unseen #+ cm_seen

classes = ["Gillnet", "Hooked gear", "Purse seine", "Scottish seine", "Trawl"]

# ------------------------------------------------------------
# Row-normalize
# ------------------------------------------------------------

cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 7))

# Heatmap
im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)

# Colorbar
cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.set_ylabel("Fraction of samples", rotation=90, labelpad=15)

# Axis labels and ticks
ax.set(
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="Predicted label",
    ylabel="True label",
)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# Add normalized values
fmt = ".2f"
thresh = cm_norm.max() / 2.

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        ax.text(
            j,
            i,
            format(cm_norm[i, j], fmt),
            ha="center",
            va="center",
            color="white" if cm_norm[i, j] > thresh else "#2f4f6f",
        )

# White grid lines between cells
ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)

ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

plt.margins(x=0.02)

plt.savefig(f"conf_matrix_gear_class_seen-all.pdf", bbox_inches="tight", pad_inches=0.05)

plt.show()