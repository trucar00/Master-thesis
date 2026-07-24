import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("labeled/ais_ers_sub_labels_01_2024.parquet", engine="pyarrow")

traj_ids = df.loc[df["label"] == "Not", "trajectory_id"].unique()

df_gear = df[df["trajectory_id"].isin(traj_ids)]

print(df_gear.shape)

print(df_gear["mmsi"].nunique())


for traj_id, d in df_gear.groupby("mmsi"):
    fig, ax = plt.subplots(figsize=(10,8))
    d["date_time_utc"] = pd.to_datetime(d["date_time_utc"])
    d = d.sort_values(by="date_time_utc")
    ax.scatter(d["lon"], d["lat"], s=3)
    plt.title(traj_id)
    plt.xlim(20.83, 22.13)
    plt.ylim(69.87, 70.16)
    plt.show()
