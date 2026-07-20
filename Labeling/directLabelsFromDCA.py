import pandas as pd
import pyarrow.parquet as pq

def get_registered_callsigns(df_dca):
    return df_dca["Radiokallesignal (dca)"].unique()

def read_ais_parquet(parquet_path):
    columns = ["mmsi", "trajectory_id", "callsign", "date_time_utc", "lon", "lat", "speed", "cog"]

    df_ais = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")

    df_ais["callsign"] = (
        df_ais["callsign"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df_ais["date_time_utc"] = pd.to_datetime(df_ais["date_time_utc"], errors="coerce")
    df_ais = df_ais.dropna(subset=["callsign", "date_time_utc"])

    return df_ais

def assign_ais_message_to_label(df_ais, df_dca):

    df_ais["label"] = pd.NA
    df_ais["label_sub1"] = pd.NA  # Redskap FAO
    df_ais["label_sub2"] = pd.NA  # Redskap FDIR

    dca_groups = {
        callsign: d.sort_values("Starttidspunkt").reset_index(drop=True)
        for callsign, d in df_dca.groupby("Radiokallesignal (dca)", sort=False)
    }

    labeled_parts = []

    for callsign, d_ais in df_ais.groupby("callsign", sort=False):
        d_ais = d_ais.sort_values("date_time_utc").copy()
        if callsign not in dca_groups:
            labeled_parts.append(d_ais)
            continue

        d_dca = dca_groups[callsign]
        for _, row in d_dca.iterrows():
            mask = (
                (d_ais["date_time_utc"] >= row["Starttidspunkt"]) &
                (d_ais["date_time_utc"] <= row["Stopptidspunkt"])
            )
            d_ais.loc[mask, "label"] = row["Redskap - gruppe"]
            d_ais.loc[mask, "label_sub1"] = row["Redskap FAO"]
            d_ais.loc[mask, "label_sub2"] = row["Redskap FDIR"]

        labeled_parts.append(d_ais)

    df_labeled = pd.concat(labeled_parts, ignore_index=True)
    return df_labeled


def main(clean_dca_path, clean_path, direct_labels_path, year):

    df_dca = pd.read_csv(clean_dca_path)
    registered_callsigns = get_registered_callsigns(df_dca) # All unique callsigns in the DCA data
    print("Nr of vessels in dca ", len(registered_callsigns))

    for month in range(1, 13):
        filepath = f"{clean_path}/{month:02d}.parquet"

        df_ais = read_ais_parquet(parquet_path=filepath)

        df_ais_with_labels = assign_ais_message_to_label(df_ais, df_dca)
        df_ais_with_labels.to_parquet(f"{direct_labels_path}/{month:02d}_{year}.parquet", index=False)


if __name__ == "__main__":
    main()
    #local_main()
