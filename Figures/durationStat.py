import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq 
import seaborn as sns
import numpy as np


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

colors = plt.get_cmap("tab10").colors

files = [
    "Data/ers-fangstmelding-nonan-2023.csv",
    "Data/ers-fangstmelding-nonan-2024.csv",
    "Data/ers-fangstmelding-nonan-2025.csv"
]

dfs = [pd.read_csv(f) for f in files]
df_ers = pd.concat(dfs, ignore_index=True)

df_ers = df_ers[
    df_ers["Starttidspunkt"].str.contains(" ", na=False) &
    df_ers["Stopptidspunkt"].str.contains(" ", na=False)
]

nr_callsigns_ers = df_ers["Radiokallesignal (ERS)"].nunique()

print(df_ers["Redskap - gruppe"].unique())

df_ers = df_ers.dropna(subset=["Starttidspunkt", "Stopptidspunkt", "Radiokallesignal (ERS)", "Redskap - gruppe", "Varighet"])
df_ers = df_ers.drop_duplicates(keep="first")

fmt = "%d.%m.%Y %H:%M:%S"
df_ers["Starttidspunkt"] = pd.to_datetime(df_ers["Starttidspunkt"], format=fmt)
df_ers["Stopptidspunkt"] = pd.to_datetime(df_ers["Stopptidspunkt"], format=fmt)

#df_ers = df_ers.loc[df_ers["Starttidspunkt"].between("2024-01-01", "2024-01-31 23:59:59")] # CHANGE for month

df_ers["Radiokallesignal (ERS)"] = df_ers["Radiokallesignal (ERS)"].astype("string").str.strip().str.upper()
df_ers["Redskap - gruppe"] = df_ers["Redskap - gruppe"].astype("string").str.strip()
df_ers["Varighet"] = pd.to_numeric(df_ers["Varighet"], errors="coerce")
df_ers = df_ers.loc[df_ers["Varighet"] < 2880].copy()

gears = ["Trål", "Krokredskap", "Bur og ruser", "Garn", "Not", "Snurrevad"]
#gears = ["Bur og ruser"]
activity_flags = ["I fiske"]

gear_translation = {
    "Trål": "Trawl",
    "Krokredskap": "Hooked gear",
    "Bur og ruser": "Traps",
    "Garn": "Gillnet",
    "Not": "Purse seine",
    "Snurrevad": "Scottish seine",
    
}

df_fishing = df_ers.loc[
    df_ers["Aktivitet"].isin(activity_flags) &
    df_ers["Redskap - gruppe"].isin(gears)
].copy()

duration_span_90 = (
    df_fishing
    .groupby("Redskap - gruppe")["Varighet"]
    .quantile([0.05, 0.95])
    .unstack()
    .rename(columns={0.05: "p5", 0.95: "p95"})
)

duration_span_90["span"] = duration_span_90["p95"] - duration_span_90["p5"]
duration_span_90["gear"] = duration_span_90.index.map(gear_translation)

print(duration_span_90[["gear", "p5", "p95", "span"]])

plt.figure(figsize=(12,6))

for i, gear in enumerate(gears):

    gear_specific = df_ers.loc[df_ers["Redskap - gruppe"] == gear].copy()
    reported_gear_fishing = gear_specific.loc[
        gear_specific["Aktivitet"].isin(activity_flags)
    ].copy()
    
    sns.kdeplot(
        reported_gear_fishing["Varighet"].dropna(),
        label=gear_translation[gear],
        clip=(0, 1700),
        linewidth=2.5,
        color=colors[i]
        
    )

plt.xlabel("Minutes")
plt.ylabel("Density")
#plt.title("Duration by gear type")
plt.xticks(np.arange(0, df_ers["Varighet"].max(), 200))
plt.legend()
plt.xlim(0, 1700)
plt.margins(x=0.02)

plt.savefig("duration_stat.pdf", bbox_inches="tight", pad_inches=0.05)
plt.show()