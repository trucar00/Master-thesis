import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

df = pd.read_parquet("../Data/AIS/whole_month/01clean2.parquet", columns=["mmsi", "date_time_utc", "lon", "lat"], engine="pyarrow")

#df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

#first_mmsis = df["mmsi"].drop_duplicates().head(20000)

#df_small = df[df["mmsi"].isin(first_mmsis)]

plt.figure(figsize=(10,8))

plt.hist2d(df["lon"], df["lat"], bins=400, cmap="hot", norm=colors.LogNorm())
plt.colorbar(label="Number of AIS messages", shrink=0.5)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heatmap of AIS messages")
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
    