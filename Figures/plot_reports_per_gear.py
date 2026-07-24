import pandas as pd
import matplotlib.pyplot as plt

def load_ais_w_labels(df, allowed_report, gear):
    
    df = df.fillna(value={"label": "no_fishing"})

    df = df.rename(columns={"label": "report"})

    allowed_mask = df["report"].isin(allowed_report)
    gear_mask = df["report"].isin(gear)

    valid_by_traj = (
        allowed_mask.groupby(df["trajectory_id"]).all() &
        gear_mask.groupby(df["trajectory_id"]).any()
    )

    valid_ids = valid_by_traj[valid_by_traj].index
    df = df[df["trajectory_id"].isin(valid_ids)].reset_index(drop=True)

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    df = df.sort_values(["trajectory_id", "date_time_utc"]).reset_index(drop=True)

    return df


df = pd.read_parquet("2024_4_6.parquet", engine="pyarrow")
print(df["report"].unique())
#print(df.head())
df = df[df["trajectory_id"] == "257006840-10-2024-4"]
print(df.shape)

GEAR_TO_PLOT = {"Snurrevad"}

ALLOWED_LIST = ["Snurrevad", "no_fishing"]

df = load_ais_w_labels(df, ALLOWED_LIST, GEAR_TO_PLOT)

print(df.shape)

for traj_id, d in df.groupby("trajectory_id"):
    report_mask = (d["report"] == list(GEAR_TO_PLOT)[0])
    print(d.head())
    plt.figure()
    plt.scatter(
            d["lon"],
            d["lat"],
            s=4,
            c="blue",
            alpha=0.7,
            label="Reported fishing",
        )
    plt.scatter(
            d.loc[report_mask, "lon"],
            d.loc[report_mask, "lat"],
            s=4,
            c="red",
            alpha=0.7,
            label="Reported fishing",
        )

    plt.show()



