import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("../Data/AIS/whole_month/01clean2.parquet", columns=["mmsi", "date_time_utc", "lon", "lat"], engine="pyarrow")

first_mmsis = df["mmsi"].drop_duplicates()

df_small = df[df["mmsi"].isin(first_mmsis)]

df_small["mmsi"] = df_small["mmsi"].astype(str)
df_small["mmsi"] = df_small["mmsi"].str.strip()

df_small["landcode"] = df_small["mmsi"].str.slice(stop=3)
df_small["landcode"] = df_small["landcode"].astype(int)

country_code = pd.read_csv("../Data/mmsi_landcodes.csv", sep=";", encoding="utf-8")

df_small = df_small.join(country_code.set_index("Digit"), on="landcode")
df_small = df_small.rename(columns={"Allocated to": "country"})
df_small["country"] = df_small["country"].str.slice(stop=15)

df_small_dropped = df_small.drop_duplicates(subset=["mmsi", "country"], keep="first")
country_counts = df_small_dropped["country"].value_counts()
plt.figure(figsize=(5, 6))
ax = country_counts.sort_values().plot(kind="barh")

plt.xlabel("Nr of fishing vessels")
plt.ylabel("Country")

# Add value labels
for i, v in enumerate(country_counts.sort_values()):
    ax.text(v, i, f" {v}", va='center')

plt.tight_layout()
plt.show()

#print(df_small["country"].unique())

threshold = pd.Timedelta(hours=1)

df_small["date_time_utc"] = pd.to_datetime(df_small["date_time_utc"])

df_small = df_small.sort_values(by=["mmsi", "date_time_utc"])
df_small["gap"] = df_small.groupby("mmsi")["date_time_utc"].diff()
df_small["large_gap"] = df_small["gap"] > threshold

gap_per_mmsi = (
    df_small
    .groupby(["country", "mmsi"])["large_gap"]
    .sum()
    .reset_index(name="n_large_gaps")
)

# Then average per country
avg_gap_country = (
    gap_per_mmsi
    .groupby("country")["n_large_gaps"]
    .mean()
    .sort_values(ascending=False)
)

avg_gap_country = avg_gap_country[avg_gap_country > 5]
print(avg_gap_country)

plt.figure(figsize=(5, 6))

avg_gap_country.sort_values().plot(kind="barh")

plt.xlabel("Average number of AIS gaps (>1h) per vessel")
plt.ylabel("Country")
plt.tight_layout()
plt.show()