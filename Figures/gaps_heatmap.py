import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as colors
import pyarrow.parquet as pq

REGION_LAT = 55 # We want all vessels north of 62 degrees north
REGION_LON_EAST = 45
REGION_LON_WEST = -10

def readParquetFile(filename):
    df = pd.read_parquet(filename, engine='pyarrow')
    return df

def readParquetFile_onlyFish(filename):
    table = pq.read_table(
        filename,
        columns=["mmsi", "date_time_utc", "lon", "lat"],
        filters=[
            ("ship_type", "==", 30),
            ("lat", ">=", REGION_LAT),
            ("lon", ">=", REGION_LON_WEST),
            ("lon", "<=", REGION_LON_EAST),
        ]
    )
    return table.to_pandas()


df = readParquetFile_onlyFish("ais_gaps/raw_2024_01_01.parquet")

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])


threshold = pd.Timedelta(hours=1)

lons = []
lats = []

#fig, ax = plt.subplots(figsize=(10, 8))

for mmsi, d in df.groupby("mmsi"):
    d = d.sort_values(by="date_time_utc")
    d["gap"] = d["date_time_utc"].diff()
    d["large_gap"] = (d["gap"] > threshold)
    nr_gaps = d["large_gap"].sum()
    gap_messages = d.loc[d["large_gap"] == True].copy()
    lons.extend(gap_messages["lon"].values)
    lats.extend(gap_messages["lat"].values)
    #print(d.shape)
    #print(gap_messages.shape)
    #print(f"{mmsi}, nr of gaps: {nr_gaps}")

    #plt.plot(d["lon"], d["lat"], linewidth=1, label="Trajectory")
    #ax.scatter(gap_messages["lon"], gap_messages["lat"], s=2, color="red")

#plt.legend()

fig = plt.figure(figsize=(10,8))

plt.hist2d(lons, lats, bins=100, cmap="hot", norm=colors.LogNorm())
plt.colorbar(label="Number of AIS gaps", shrink=0.5)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Heatmap of AIS Signal Gaps")

plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
    