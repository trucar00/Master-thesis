import pandas as pd

YEAR = "2024"

def main(raw_dca_path, clean_dca_path, activities, gear_types, duration_limits):

    dca_df = pd.read_csv(raw_dca_path, sep=";", encoding="utf-8", decimal=",")

    dca_df = dca_df[["Fartøynavn (dca)", "Fartøynasjonalitet (kode)", "Meldingstidspunkt", "Radiokallesignal (dca)", "Aktivitet", "Starttidspunkt",
                 "Stopptidspunkt", "Varighet", "Startposisjon bredde", "Startposisjon lengde", "Stopposisjon bredde", 
                 "Stopposisjon lengde", "Hovedområde start (kode)", "Redskap - gruppe", "Redskap FAO","Redskap FDIR",  "Hovedart FAO"]]

    before = len(dca_df)
    dca_df = dca_df.dropna(
        subset=[
            "Starttidspunkt",
            "Stopptidspunkt",
            "Radiokallesignal (dca)",
            "Redskap - gruppe",
            "Redskap FDIR",
            "Redskap FAO",
            "Varighet",
            "Aktivitet",
        ]
    )
    dca_df = dca_df[dca_df["Varighet"] > 0]
    dca_df = dca_df.drop_duplicates(keep="first")

    print(dca_df["Redskap - gruppe"].unique())

    # Some reports did not include a timestamp, but only a date. Remove those. 
    dca_df = dca_df[
        dca_df["Starttidspunkt"].str.contains(" ", na=False) &
        dca_df["Stopptidspunkt"].str.contains(" ", na=False)
    ]

    fmt = "%d.%m.%Y %H:%M:%S"
    dca_df["Starttidspunkt"] = pd.to_datetime(dca_df["Starttidspunkt"], format=fmt)
    dca_df["Stopptidspunkt"] = pd.to_datetime(dca_df["Stopptidspunkt"], format=fmt)

    dca_df = dca_df.loc[dca_df["Stopptidspunkt"] >= dca_df["Starttidspunkt"]].copy() # Stop time has to be after start time...
    dca_df["Varighet"] = pd.to_numeric(dca_df["Varighet"], errors="coerce")

    dca_df["Radiokallesignal (dca)"] = ( # Make sure that radio callsign does not have whitespaces / lowercase letters.
        dca_df["Radiokallesignal (dca)"].astype("string").str.strip().str.upper()
    )

    dca_df["Redskap - gruppe"] = (
        dca_df["Redskap - gruppe"].astype("string").str.strip()
    )

    dca_df["Redskap FDIR"] = (
        dca_df["Redskap FDIR"].astype("string").str.strip()
    )

    dca_df["Redskap FAO"] = (
        dca_df["Redskap FAO"].astype("string").str.strip()
    )

    dca_df = dca_df.loc[dca_df["Redskap - gruppe"].isin(gear_types)].copy() # Only reports with gear types that are in the main catagories
    dca_df = dca_df.loc[dca_df["Aktivitet"].isin(activities)].copy() # Only reports with activity = I fiske

    # Apply duration limits for each gear type
    dca_df["min_duration"] = dca_df["Redskap - gruppe"].map(lambda g: duration_limits[g][0])
    dca_df["max_duration"] = dca_df["Redskap - gruppe"].map(lambda g: duration_limits[g][1])

    dca_df = dca_df.loc[
        (dca_df["Varighet"] >= dca_df["min_duration"]) &
        (dca_df["Varighet"] <= dca_df["max_duration"])
    ].copy()

    dca_df = dca_df.drop(columns=["min_duration", "max_duration"])

    dca_df = dca_df.reset_index(drop=True)

    after = len(dca_df)
    print(f"Dropped {before - after} rows ({(before-after)/before:.1%})")

    dca_df.to_csv(clean_dca_path, index=False) # Save a no nan version of the DCA
    return dca_df



if __name__ == "__main__":
    #main()
    print("Run from labeling_main.py")