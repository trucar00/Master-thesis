import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as colors

df = pd.read_parquet("../Data/AIS/whole_month/01clean2.parquet", columns=["mmsi", "date_time_utc", "lon", "lat"], engine="pyarrow")

df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

#df = df.loc[df["lat"] > 62].copy()

#df = df.loc[df["lon"] > 10].copy()

df = df.loc[df["mmsi"] == 257079000]

threshold = pd.Timedelta(hours=1)



for mmsi, d in df.groupby("mmsi"):
    fig, ax = plt.subplots(figsize=(10, 8))
    d = d.sort_values(by="date_time_utc")
    print(d.shape)
    d = d.iloc[91000:92000]
    min_time = d["date_time_utc"].min()
    max_time = d["date_time_utc"].max()
    d["gap"] = d["date_time_utc"].diff()
    d["large_gap"] = (d["gap"] > threshold)
    nr_gaps = d["large_gap"].sum()
    gap_messages = d.loc[d["large_gap"] == True].copy()
    before_gap = d.loc[d["large_gap"].shift(-1, fill_value=False)].copy()
    #print(d.shape)
    #print(gap_messages.shape)
    #print(f"{mmsi}, nr of gaps: {nr_gaps}")

    ax.plot(d["lon"], d["lat"], linewidth=1.5, label="Linear interpolated trajectory", zorder=1)
    ax.scatter(before_gap["lon"], before_gap["lat"], s=15, color="black", label="AIS message before gap", zorder=3, marker="x")
    ax.scatter(gap_messages["lon"], gap_messages["lat"], s=15, color="red", label="AIS message after gap", zorder=2)

    plt.title(f"MMSI: {mmsi} with {nr_gaps} gaps > 1h. {min_time} to {max_time}.")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude") 
    plt.gca().set_aspect('equal', adjustable='box') 
    plt.tight_layout()
    plt.legend()
    plt.show()


    