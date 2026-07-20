import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("redskap_data.csv", sep=";", encoding="utf-8", decimal=",")

df = df[~df["Redskap - gruppe"].isin(["Harpun/kanon", "Oppdrett/uspesifisert"])]

gear_stat = []

gear_translation = {
    "Trål": "Trawl",
    "Not": "Purse seine",
    "Krokredskap": "Hooked gear",
    "Snurrevad": "Scottish seine",
    "Garn": "Gillnet",
    "Bur og ruser": "Traps",
    "Andre redskap": "Other gear"

}

for year in range(2023, 2025+1):
    df_year = df.loc[df["Fangstår"] == year].copy()
    for gear, d in df_year.groupby("Redskap - gruppe"):

        total_catch = d["Measure Values"].sum()

        gear_stat.append({
            "Year": year,
            "gear": gear_translation[gear],
            "total": round(total_catch)
        })


print(gear_stat)
gear_df = pd.DataFrame(gear_stat)

# Pivot for plotting
pivot_df = gear_df.pivot(index="gear", columns="Year", values="total")
pivot_df = pivot_df / 1e6

# Sort gears by total catch across all years
order = gear_df.groupby("gear")["total"].sum().sort_values(ascending=False).index
pivot_df = pivot_df.loc[order]

colors = ["#2066a8", "#3594cc", "#8cc5e3"]
pivot_df.index.name = None

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern look
    "font.size": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.axisbelow": True,
})

pivot_df.plot(kind="bar", color=colors, figsize=(10, 6), width=0.8)

plt.ylabel("Total catch [million t]")
plt.legend(title="Year", loc="upper right", frameon=True, facecolor="white", edgecolor="lightgray")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.margins(x=0.02)

plt.savefig("catch_per_gear.pdf", bbox_inches="tight")
plt.show()