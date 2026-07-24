import pandas as pd
from time import time
import glob
import os
import pyarrow.parquet as pq

def readParquetFile(filename):
    df = pd.read_parquet(filename, columns=["lon", "lat"], engine='pyarrow')
    return df

REGION_LAT = 55 # We want all vessels north of 62 degrees north
REGION_LON_EAST = 45
REGION_LON_WEST = -10

def readParquetFile_onlyFish(filename):
    table = pq.read_table(
        filename,
        columns = ["mmsi","ship_name", "date_time_utc", "lon", "lat", "speed", "cog"],
        filters=[
            ("ship_type", "==", 30),
            ("lat", ">=", REGION_LAT),
            ("lon", ">=", REGION_LON_WEST),
            ("lon", "<=", REGION_LON_EAST),
        ]
    )
    return table.to_pandas()

def saveToParquet(outfile, df):
    df.to_parquet(outfile, engine="pyarrow", compression="snappy")
    print("Successfully saved to ", outfile)

def readFilterSave(filename, saveFilename):
    df = readParquetFile(filename)
    saveToParquet(saveFilename, df)

def readFilterSave_onlyFish(filename, saveFilename):
    df = readParquetFile_onlyFish(filename)
    saveToParquet(saveFilename, df)

def main(months, year=2024):
    start = time()
    print("Getting data from NTNUs copy of AIS-data from Kystverket.")
    for month in range(1, months+1):
        pattern = f"raw_ais/date_utc={year}-{month:02d}-*" # ADD Y: for running locally
        folders = sorted(glob.glob(pattern))
        if not folders:
            print("No folders for month:", month)
            continue
        day = 0
        for folder in folders:
            for entry in os.scandir(folder):
                if entry.is_file() and entry.name.endswith(".parquet"):
                    print("Processing file: ", entry.path)
                    day += 1
                    readFilterSave(entry.path, f"raw_ais/parquets_raw/{month:02d}-{day:02d}.parquet") #use readFilterSave2 for STS
                    #readFilterSave_onlyFish(entry.path, f"raw_ais/parquets/{year}-{month:02d}-{day:02d}_fish_only.parquet")

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")


if __name__ == "__main__":
    main(months=12)