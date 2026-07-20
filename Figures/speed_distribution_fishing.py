import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

GEARS = {
    "Trål": "Trawl",
    "Krokredskap": "Hooked gear",
    "Bur og ruser": "Traps",
    "Garn": "Gillnet",
    "Not": "Purse seine",
    "Snurrevad": "Scottish seine",
}

MONTHS = range(1, 3)  # January-February

plt.figure(figsize=(12, 6))

for gear_label, gear_translation in GEARS.items():
    dfs = []

    for month in MONTHS:
        path = Path(f"Stat/ais_ers_labels_{month:02d}_2024.parquet")

        if path.exists():
            print("Reading", path)
            df = pd.read_parquet(
                    path,
                    engine="pyarrow",
                    filters=[("label", "==", gear_label)]
                )
            dfs.append(df)
        else:
            print(path, "does not exist.")

    if not dfs:
        print(f"No data found for {gear_label}")
        continue

    all_df = pd.concat(dfs, ignore_index=True)

    all_df = all_df.dropna(subset=["speed"])
    #all_df = all_df
    all_df = all_df[(all_df["speed"] > 0) & (all_df["speed"] < 20)]

    sns.kdeplot(
        all_df["speed"],
        fill=False,
        label=gear_translation,
        linewidth=2.5,
        cut=0
    )

plt.xlabel("Speed [knots]")
plt.ylabel("Density")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.margins(x=0.02)

plt.savefig("speed_distribution_fishing_stat.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()