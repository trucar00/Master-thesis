import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import geopandas as gpd
import contextily as ctx
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "legend.title_fontsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.axisbelow": True,
})

def make_heatmap(files, lon_min, lon_max, lat_min, lat_max, x_edges, y_edges):
    Z = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.float32)

    for filename in files:
        print(f"Reading {filename}")

        df = pd.read_parquet(
            filename,
            columns=["lon", "lat"],
            engine="pyarrow"
        )

        df = df.dropna(subset=["lon", "lat"])
        df = df[
            (df["lon"].between(lon_min, lon_max)) &
            (df["lat"].between(lat_min, lat_max))
        ]

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        x = gdf.geometry.x.to_numpy()
        y = gdf.geometry.y.to_numpy()

        H, _, _ = np.histogram2d(
            x,
            y,
            bins=[x_edges, y_edges]
        )

        Z += H.astype(np.float32)

    Z = gaussian_filter(Z, sigma=0.4)
    Z = np.ma.masked_where(Z < 1, Z)

    return Z


def plot_two_ais_heatmaps():
    year = 2024

    lat_min, lat_max = 55, 90
    lon_min, lon_max = -10, 45

    # Convert plot extent to Web Mercator
    extent_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [lon_min, lon_max],
            [lat_min, lat_max]
        ),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    x_min = extent_gdf.geometry.iloc[0].x
    y_min = extent_gdf.geometry.iloc[0].y
    x_max = extent_gdf.geometry.iloc[1].x
    y_max = extent_gdf.geometry.iloc[1].y

    n_bins_x = 221
    n_bins_y = 140

    x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)

    files_1 = [
        f"raw_ais/parquets/{year}-01-{day:02d}_fish_only.parquet"
        for day in range(1, 32)
    ]

    files_2 = [
        f"cleaned_parquets/{year}-01.parquet"
    ]

    Z1 = make_heatmap(files_1, lon_min, lon_max, lat_min, lat_max, x_edges, y_edges)
    Z2 = make_heatmap(files_2, lon_min, lon_max, lat_min, lat_max, x_edges, y_edges)

    vmax = max(Z1.max(), Z2.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)

    titles = [
        "Raw AIS, fishing vessels only",
        "Cleaned AIS"
    ]

    cmaps = []
    for _ in range(2):
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(alpha=0)
        cmaps.append(cmap)

    imgs = []

    for ax, Z, title, cmap in zip(axes, [Z1, Z2], titles, cmaps):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.PositronNoLabels,
            zoom=5,
            zorder=0
        )

        img = ax.imshow(
            Z.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap=cmap,
            norm=LogNorm(vmin=1, vmax=vmax),
            alpha=0.75,
            interpolation="none",
            zorder=10
        )

        ax.set_title(title)
        ax.set_axis_off()
        imgs.append(img)

    cbar = fig.colorbar(
        imgs[0],
        ax=axes,
        fraction=0.035,
        pad=0.02
    )
    cbar.set_label("Number of AIS messages")

    plt.tight_layout()
    plt.show()

    # fig.savefig("ais_heatmaps_side_by_side.pdf", bbox_inches="tight")


plot_two_ais_heatmaps()