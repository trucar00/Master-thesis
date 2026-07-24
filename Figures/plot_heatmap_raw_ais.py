import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 20,
    "axes.labelsize": 20,
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "axes.axisbelow": True,
})


lon_bins = np.linspace(-10, 45, 111)   # 0.5 degree bins
lat_bins = np.linspace(55, 90, 70)
heatmap = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

#df1 = pd.read_parquet(f"raw_ais/parquets/01-01_fish_only.parquet", engine="pyarrow")
#df2 = pd.read_parquet(f"raw_ais/parquets/01-01.parquet", engine="pyarrow")
#print(df1.shape, df2.shape)

year = 2024

for day in range(1, 31+1):
    df = pd.read_parquet(f"raw_ais/parquets/{year}-01-{day:02d}_fish_only.parquet", columns=["lon", "lat"], engine="pyarrow")
    #df = pd.read_parquet(f"cleaned_parquets/{year}-01.parquet", columns=["lon", "lat"], engine="pyarrow")
    
    # Remove invalid coordinates
    df = df.dropna(subset=["lon", "lat"])
    df = df[
        (df["lon"].between(-10, 45)) &
        (df["lat"].between(55, 90))
    ]

    H, _, _ = np.histogram2d(
        df["lat"],
        df["lon"],
        bins=[lat_bins, lon_bins]
    )

    heatmap += H
fig, ax = plt.subplots(figsize=(14, 7))

im = ax.imshow(
    heatmap,
    extent=[-10, 45, 55, 90],
    origin="lower",
    aspect="auto",
    norm=LogNorm(vmin=1, vmax=heatmap.max())
)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Global density of raw AIS messages, January {year}")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Number of AIS messages")

plt.tight_layout()
#plt.savefig("raw_ais_global_heatmap_january_2025.png", dpi=300)
plt.show()