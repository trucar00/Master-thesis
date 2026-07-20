import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import rasterio
import matplotlib.pyplot as plt
from rasterio.transform import rowcol
from pyproj import Transformer
import gc

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of the earth in meters

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1


    # apply formulae
    a = (pow(np.sin(dlat / 2), 2) +  
             np.cos(lat1) * np.cos(lat2) * pow(np.sin(dlon / 2), 2))
    
    c = 2 * np.arcsin(np.sqrt(a))

    dist = R * c
    #speed = (dist/dt) * 1.94384 # Convert m/s to knots

    return dist #, speed

def concat_year(months, path):
    print("Concating full year.")
    for y in range(2022, 2024+1):
        dfs = []
        print("Concating all of ", y)
        for m in range(1, months+1):
            df = pd.read_parquet(f"{path}{m:02d}_{y}.parquet")
            df["trajectory_id"] = df["trajectory_id"].astype(str) + "-" + str(y) + "-" + str(m) # new unique traj_id
            dfs.append(df)
        year_df = pd.concat(dfs, ignore_index=True)

        year_df.to_parquet(f"ais_ers_labels_full_{y}.parquet", index=False)

    return "Done concating full year!"

def concat_month(month, path):
    print("Concating all years month ", month)
    dfs = []
    for y in range(2022, 2024+1):
        df = pd.read_parquet(f"{path}{month:02d}_{y}.parquet")
        df["trajectory_id"] = df["trajectory_id"].astype(str) + "-" + str(y) + "-" + str(month) # new unique traj_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def concat_3_months(year, path):
    for start in range(1, 12+1, 3):   # starts at 1 and 
        dfs = []
        for i in range(start, start + 3):
            df = pd.read_parquet(f"{path}{i:02d}_{year}.parquet")
            df["trajectory_id"] = df["trajectory_id"].astype(str) + "-" + str(year) + "-" + str(i) # new unique traj_id
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

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

    df["row_id"] = np.arange(len(df))

    # Columns for deciding if reported no_fishing is acutally true no fishing
    df["high_speed"] = 0
    df["no_fish_cl"] = 0
    df["close_to_shore"] = 0
    return df

# RULE 1: HIGH SPEED

def speed_rule(df, speed_threshold, window_len):
    print(f"Checking speed rule. Threshold: {speed_threshold} knots.")
    df = df.copy()

    window_seconds = window_len.total_seconds()

    t0 = df.groupby("trajectory_id")["date_time_utc"].transform("min")

    df["time_bin"] = (
        (df["date_time_utc"] - t0)
        .dt.total_seconds()
        .floordiv(window_seconds)
        .astype(int)
    )

    stats = (
        df.groupby(["trajectory_id", "time_bin"])["speed"]
        .agg(["mean", "size"])
        .reset_index()
    )

    bad_bins = stats[
        (stats["size"] >= 5) &
        (stats["mean"] > speed_threshold)
    ][["trajectory_id", "time_bin"]]

    df = df.merge(
        bad_bins.assign(high_speed_new=1),
        on=["trajectory_id", "time_bin"],
        how="left"
    )

    df["high_speed"] = df["high_speed_new"].fillna(0).astype(int)

    return df.drop(columns=["time_bin", "high_speed_new"])


def close_to_shore(df, threshold_km, raster_path="distance-from-shore.tif"):
    print(f"Checking distance to shore. Shore threshold: {threshold_km}")
    df = df.copy()
    print("Rounding coordinates...")
    df["lon_r"] = df["lon"].round(4)
    df["lat_r"] = df["lat"].round(4)

    print("Finding unique coordinates...")
    unique_pts = df[["lon_r", "lat_r"]].drop_duplicates().copy()

    with rasterio.open(raster_path) as src:
        print("Reading raster...")
        band = src.read(1)
        transform = src.transform
        nodata = src.nodata

        print("Preparing coordinate arrays...")
        lon = unique_pts["lon_r"].to_numpy()
        lat = unique_pts["lat_r"].to_numpy()

        print("Converting to raster indices...")
        cols, rows = ~transform * (lon, lat)
        rows = rows.astype(int)
        cols = cols.astype(int)

        print("Filtering valid indices...")
        valid = (
            (rows >= 0) & (rows < band.shape[0]) &
            (cols >= 0) & (cols < band.shape[1])
        )

        print("Sampling raster (vectorized)...")
        dist = np.full(len(unique_pts), np.nan, dtype="float32")

        # tqdm here to track assignment progress (optional but visible)
        dist[valid] = band[rows[valid], cols[valid]]

        if nodata is not None:
            dist[dist == nodata] = np.nan

        unique_pts["dist_to_shore_km"] = dist

    print("Merging back...")
    df = df.merge(unique_pts, on=["lon_r", "lat_r"], how="left")

    df["close_to_shore"] = (df["dist_to_shore_km"] < threshold_km).astype(int)

    print("Cleaning up...")
    df = df.drop(columns=["lon_r", "lat_r"])

    return df

def features_for_clustering(df, half_window, min_messages):
    print("Building features for clustering")

    feat_names = [
        "mean_speed", "std_speed", "min_speed", "max_speed",
        "mean_acc", "std_acc", "mean_abs_acc",
        "mean_abs_dcog", "std_dcog", "cum_abs_turn",
        "path_length", "net_displacement", "straightness"
    ]

    def window_sum(x, lo, hi):
        cs = np.r_[0, np.cumsum(x)]
        return cs[hi] - cs[lo]

    def features_for_trip(d):
        d = d.sort_values("date_time_utc").reset_index(drop=True)

        if len(d) < min_messages:
            return None

        d["dt"] = d["date_time_utc"].diff().dt.total_seconds()

        lon = d["lon"].to_numpy()
        lat = d["lat"].to_numpy()

        dist = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        d["dist_to_prev"] = np.r_[np.nan, dist]

        d["speed_calc_ms"] = d["dist_to_prev"] / d["dt"]
        d["accel"] = d["speed_calc_ms"].diff() / d["dt"]
        d["jerk"] = d["accel"].diff() / d["dt"]

        # Faster than .diff().apply(ff.angle_wrap)
        dcog_raw = d["cog"].diff().to_numpy()
        dcog_wrapped = ((dcog_raw + 180) % 360) - 180
        d["dcog"] = dcog_wrapped / d["dt"].to_numpy()

        no_nan_cols = ["dt", "dist_to_prev", "speed_calc_ms", "accel", "jerk", "dcog"]
        d = d.dropna(subset=no_nan_cols).reset_index(drop=True)

        n = len(d)
        if n < min_messages:
            return None

        times = d["date_time_utc"].to_numpy()
        times_ns = times.astype("datetime64[ns]").astype("int64")
        half_ns = int(half_window.value)

        lat = d["lat"].to_numpy()
        lon = d["lon"].to_numpy()
        sog = d["speed_calc_ms"].to_numpy()
        acc = d["accel"].to_numpy()
        dcog = d["dcog"].to_numpy()
        dist = d["dist_to_prev"].to_numpy()

        abs_acc = np.abs(acc)
        abs_dcog = np.abs(dcog)

        lo_idx = np.searchsorted(times_ns, times_ns - half_ns, side="left")
        hi_idx = np.searchsorted(times_ns, times_ns + half_ns, side="left")

        counts = hi_idx - lo_idx
        keep = counts >= min_messages

        # Remove edge windows that are not fully inside trajectory
        keep &= (times_ns - half_ns >= times_ns[0])
        keep &= (times_ns + half_ns <= times_ns[-1])

        if not keep.any():
            return None

        out = np.full((n, len(feat_names)), np.nan)

        cnt = counts.astype(float)

        sum_sog = window_sum(sog, lo_idx, hi_idx)
        sum_sog2 = window_sum(sog ** 2, lo_idx, hi_idx)

        sum_acc = window_sum(acc, lo_idx, hi_idx)
        sum_acc2 = window_sum(acc ** 2, lo_idx, hi_idx)

        sum_abs_acc = window_sum(abs_acc, lo_idx, hi_idx)

        sum_dcog = window_sum(dcog, lo_idx, hi_idx)
        sum_dcog2 = window_sum(dcog ** 2, lo_idx, hi_idx)
        sum_abs_dcog = window_sum(abs_dcog, lo_idx, hi_idx)

        path = window_sum(dist, lo_idx, hi_idx)

        def fast_std(sum_x, sum_x2, cnt):
            std = np.full_like(cnt, np.nan, dtype="float64")

            valid = cnt > 1

            var = (sum_x2[valid] - (sum_x[valid] ** 2) / cnt[valid]) / (cnt[valid] - 1)
            var = np.maximum(var, 0)

            std[valid] = np.sqrt(var)

            return std

        mean_sog = sum_sog / cnt
        std_sog = fast_std(sum_sog, sum_sog2, cnt)

        mean_acc = sum_acc / cnt
        std_acc = fast_std(sum_acc, sum_acc2, cnt)

        mean_dcog = sum_dcog / cnt
        std_dcog = fast_std(sum_dcog, sum_dcog2, cnt)

        net = haversine(
            lat[lo_idx],
            lon[lo_idx],
            lat[hi_idx - 1],
            lon[hi_idx - 1]
        )

        # min/max are harder to fully vectorize for variable windows.
        # This loop is much cheaper now because only min/max remain.
        min_sog = np.full(n, np.nan)
        max_sog = np.full(n, np.nan)

        valid_i = np.where(keep)[0]
        for i in valid_i:
            s = sog[lo_idx[i]:hi_idx[i]]
            min_sog[i] = s.min()
            max_sog[i] = s.max()

        out[:, 0] = mean_sog
        out[:, 1] = std_sog
        out[:, 2] = min_sog
        out[:, 3] = max_sog
        out[:, 4] = mean_acc
        out[:, 5] = std_acc
        out[:, 6] = sum_abs_acc / cnt
        out[:, 7] = sum_abs_dcog / cnt
        out[:, 8] = std_dcog
        out[:, 9] = sum_abs_dcog
        out[:, 10] = path
        out[:, 11] = net
        out[:, 12] = np.where(path > 0, net / path, np.nan)

        feat_df = pd.DataFrame(out, columns=feat_names)

        small_d = d[["row_id", "report"]].reset_index(drop=True)
        return pd.concat([small_d, feat_df], axis=1).loc[keep].reset_index(drop=True)

    results = []

    for traj_id, d in df.groupby("trajectory_id"):
        feats = features_for_trip(d)
        if feats is not None:
            results.append(feats)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)

def cluster_no_fishing(df, df_cluster_feats, n_clusters):
    print(f"Clustering with K-means into {n_clusters} clusters.")
    df = df.copy()

    feature_cols = [
        "mean_speed", "std_speed", "min_speed", "max_speed",
        "mean_acc", "std_acc", "mean_abs_acc",
        "mean_abs_dcog", "std_dcog", "cum_abs_turn",
        "path_length", "net_displacement", "straightness",
    ]

    if df_cluster_feats.empty:
        print("No clustering features found.")
        return df

    cl = df_cluster_feats[
            df_cluster_feats["report"].eq("no_fishing")
        ].replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols).copy()

    X = cl[feature_cols].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cl["cluster"] = km.fit_predict(X_scaled)

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=feature_cols
    )

    print("\nCluster sizes:")
    print(cl.groupby("cluster").size())

    print("\nCluster centroids:")
    print(centroids.T)

    # Heuristic: confident no-fishing is fast, stable and straight
    score = (
        centroids["mean_speed"].rank(ascending=True) +
        centroids["straightness"].rank(ascending=True) +
        centroids["std_speed"].rank(ascending=False)
    )

    no_fishing_cluster = score.idxmax()

    print(f"\nChosen confident no_fishing cluster: {no_fishing_cluster}")

    confident_row_ids = cl.loc[cl["cluster"] == no_fishing_cluster, "row_id"]

    df.loc[df["row_id"].isin(confident_row_ids), "no_fish_cl"] = 1

    return df

def add_confidence_flags(df):
    df = df.copy()

    df["passed_any_rule"] = (
        (df["high_speed"] == 1) |
        (df["close_to_shore"] == 1) |
        (df["no_fish_cl"] == 1)
    ).astype(int)

    df["conf_no_fishing"] = df["passed_any_rule"] & (df["report"] == "no_fishing")
    df["unknown_no_fishing"] = (df["report"] == "no_fishing") & (~df["passed_any_rule"])

    return df

SPEED_THRESHOLD = 10  
SPEED_WINDOW = pd.Timedelta(minutes=20)

SHORE_THRESHOLD_KM = 5

HALF_WINDOW = pd.Timedelta(minutes=20) # looks 40 minutes in total
MIN_MESSAGES = 10

LIST = [{"Not"}, {"Trål"}, {"Krokredskap"}, {"Snurrevad"}, {"Garn"}, {"Bur og ruser"}]

ALLOWED_LIST = [["Not", "no_fishing"], ["Trål", "no_fishing"], ["Krokredskap", "no_fishing"], ["Snurrevad", "no_fishing"], ["Garn", "no_fishing"], ["Bur og ruser", "no_fishing"]]

N_CLUSTERS = 2

def main(direct_labels_path, confident_labels_path, year):
    for year in range(year, year+1): # can do multiple years at the same time
        for start in range(1, 12+1, 3):   # starts at 1 and increases by 3 for each step. 1, 4, 7
            dfs = []
            for i in range(start, start + 3):
                df = pd.read_parquet(f"{direct_labels_path}/{i:02d}_{year}.parquet")
                df["trajectory_id"] = df["trajectory_id"].astype(str) + "-" + str(year) + "-" + str(i) # new unique traj_id
                dfs.append(df)

            base_df = pd.concat(dfs, ignore_index=True) # concat three and three months
    
            for i in range(6):  # ALL gear
                g = LIST[i]
                allowed = ALLOWED_LIST[i]

                print(f"Making for {g} year {year} months {start} to {start+3}")

                df = load_ais_w_labels(base_df, allowed_report=allowed, gear=g)
    
                df = speed_rule(df, speed_threshold=SPEED_THRESHOLD, window_len=SPEED_WINDOW)
        
                df = close_to_shore(df, threshold_km=SHORE_THRESHOLD_KM)
        
                feats_df_for_clustering = features_for_clustering(df, half_window=HALF_WINDOW, min_messages=MIN_MESSAGES)
                df = cluster_no_fishing(df, df_cluster_feats=feats_df_for_clustering, n_clusters=N_CLUSTERS)

                print(df.head())

                df = add_confidence_flags(df)
                
                gear_name = next(iter(g))
                if gear_name == "Bur og ruser": # Does not change the values in "report" == Bur og ruser ...
                    gear_name = "Traps"
                df.to_parquet(f"{confident_labels_path}/{gear_name}_{year}_{start}_{start+2}.parquet", index=False)

                del df, feats_df_for_clustering
                gc.collect()

            del base_df
            gc.collect()
    return


def plot(df, gear_set): # plotting function to see the messages and the unknown, confident non fishing and reported fishing. 

    for traj_id, d in df.groupby("trajectory_id"):
        
        reported = d[d["report"] == "no_fishing"]
        confident = d[d["conf_no_fishing"]]
        unknown = d[d["unknown_no_fishing"]]
        gear = d[d["report"].isin(gear_set)]

        if len(reported) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(d["lon"], d["lat"], s=2, alpha=0.15, label="All points")

        ax.scatter(
            confident["lon"], confident["lat"],
            s=8, alpha=0.8, color="green",
            label="Confident no_fishing"
        )

        ax.scatter(
            unknown["lon"], unknown["lat"],
            s=10, color="red", alpha=0.9,
            label="Unknown"
        )

        ax.scatter(
            gear["lon"], gear["lat"],
            s=10, color="black", alpha=0.9,
            label="Reported gear"
        )

        reported_gear = set(d["report"]).intersection(gear_set)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"Trajectory {traj_id} | "
            f"reported gear: {reported_gear}, "
            f"confident={len(confident)}, "
            f"unknown={len(unknown)}"
        )
        ax.legend()

        plt.show(block=False)
        plt.pause(0.001)

        inp = input("Press Enter to continue, type 'exit' to skip month: ")

        if inp.strip().lower() == "exit":
            plt.close('all')
            return True


if __name__ == "__main__":
    main()


    #df = pd.read_parquet("conf_labels/Not_2024_1_3.parquet")
    #print(df.head())
    #print(df.tail())
    #print(df["no_fish_cl"].value_counts())
    #plot(df, gear_set={"Krokredskap", "Trål", "Not", "Garn", "Snurrevad", "Bur og ruser"})

    """ for i in range(5):
        g = LIST[i]
        allowed = ALLOWED_LIST[i]
        
        for m in range(1, 5): # should maybe do the clustering across all months .
            print(f"Plotting for {g} month: {m}")
            df = pd.read_parquet(f"conf_labels/{g}_confident_no_fishing_ais_{m:02d}.parquet")
  
            if plot(df, g):  
                continue """