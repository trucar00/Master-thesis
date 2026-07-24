import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import gc

GEARS = ["Trål", "Krokredskap", "Bur og ruser", "Not", "Snurrevad", "Garn"]

FOLDER = "Gear_type_classification"
RESAMPLE_PATH = F"{FOLDER}/Resampled"

DIRECT_LABEL_PATH = "Labeling/Direct_labels"
RESAMPLE_FILES = [
    f"{DIRECT_LABEL_PATH}/2024_1_3.parquet",     # Q1 2024
    f"{DIRECT_LABEL_PATH}/2024_4_6.parquet",     # Q2 2024
    f"{DIRECT_LABEL_PATH}/2024_7_9.parquet",     # Q3 2024
    f"{DIRECT_LABEL_PATH}/2024_10_12.parquet",   # Q4 2024
    f"{DIRECT_LABEL_PATH}/2023_1_3.parquet",     # Q1 2023
    f"{DIRECT_LABEL_PATH}/2023_4_6.parquet",     # Q2 2023
    f"{DIRECT_LABEL_PATH}/2023_7_9.parquet",     # Q3 2023
    f"{DIRECT_LABEL_PATH}/2023_10_12.parquet",   # Q4 2023

    f"{DIRECT_LABEL_PATH}/2025_1_3.parquet",     # Q1 2025
    f"{DIRECT_LABEL_PATH}/2025_4_6.parquet",     # Q2 2025
    f"{DIRECT_LABEL_PATH}/2025_7_9.parquet",     # Q3 2025
    f"{DIRECT_LABEL_PATH}/2025_10_12.parquet",   # Q4 2025
]

def get_file_name(filepath):
    split_str = filepath.split("/")
    return split_str[2] # add + onl_ ---


def downsample(df, step):
    df = df.copy()

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])
    df = df.sort_values(["trajectory_id", "date_time_utc"]).set_index("date_time_utc")
    
    theta = np.deg2rad(df["cog"].astype(float))
    df["cog_x"] = np.cos(theta)
    df["cog_y"] = np.sin(theta)

    def resample_and_interpolate(g):
        traj = g.name  # <-- this group's trajectory_id string

         # regular grid from first to last date_time_utc of this trajectory
        regular_index = pd.date_range(
            start=g.index.min(),
            end=g.index.max(),
            freq=step
        )

        # union original timestamps + target grid, so interpolation uses real points
        g_union = g.reindex(g.index.union(regular_index)).sort_index()

        # linear spatial interpolation in time
        interp_cols = [c for c in ["lon", "lat"] if c in g_union.columns]
        g_union[interp_cols] = g_union[interp_cols].interpolate(
            method="time",
            limit_area="inside"
        )

        # optional interpolation of speed and cog representation
        extra_interp_cols = [c for c in ["speed", "cog_x", "cog_y"] if c in g_union.columns]
        g_union[extra_interp_cols] = g_union[extra_interp_cols].interpolate(
            method="time",
            limit_area="inside"
        )

        # keep only the regular timestamps
        g_res = g_union.loc[regular_index].copy()
        
        r = np.hypot(g_res["cog_x"], g_res["cog_y"])
        valid = r > 0
        g_res.loc[valid, "cog_x"] /= r[valid]
        g_res.loc[valid, "cog_y"] /= r[valid]
        g_res["theta"] = np.arctan2(g_res["cog_y"], g_res["cog_x"])
        g_res["cog_interp"] = np.rad2deg(g_res["theta"]) % 360

        # Fill identifiers that exist
        for c in ["mmsi", "callsign"]:
            if c in g_res.columns:
                g_res[c] = g[c].iloc[0]

       
        lab = g[["report"]].copy()
        lab = lab[~lab.index.duplicated(keep="first")]

        g_res["report"] = (
            lab.reindex(regular_index, method="ffill")["report"].values
        )

        # Re-add trajectory_id as a string column
        g_res["trajectory_id"] = traj

        return g_res

    resampled = (
        df.groupby("trajectory_id", group_keys=False)
          .apply(resample_and_interpolate)
          .reset_index()
          .rename(columns={"index": "date_time_utc"})
    )

    # Ensure trajectory_id is string dtype
    resampled["trajectory_id"] = resampled["trajectory_id"].astype("string")
    resampled["mmsi"] = resampled["mmsi"].astype("int64")

    resampled = resampled.drop(columns=["cog", "cog_x", "cog_y", "theta"])
    return resampled

def resample_all_files():
    for file in RESAMPLE_FILES:
        save_file = get_file_name(file)
        df = pd.read_parquet(file, engine="pyarrow")
        print("Downsampling ", save_file)
        df = downsample(df, step="30s")
        df.to_parquet(f"{RESAMPLE_PATH}/{save_file}", engine="pyarrow")
        del df
        gc.collect()
        print("Saved downsampled!")


resample_all_files()