import os
from . import dataProcessing
from time import time
import glob

# --- Gets the copy of NTNUs AIS-data from Kystverket ---
# --- readFilterSave() reads the parquet files, and filters out fishing vessels within region ---

def main(months, filtered_path, year):
    start = time()
    print("Getting data from NTNUs copy of AIS-data from Kystverket.")
    for month in range(1, months+1):
        pattern = f"../../Data/{year}/date_utc={year}-{month:02d}-*" # ADD Y: for running locally
        folders = sorted(glob.glob(pattern))
        if not folders:
            print("No folders for month:", month)
            continue
        for folder in folders:
            for entry in os.scandir(folder):
                if entry.is_file() and entry.name.endswith(".parquet"):
                    print("Processing file: ", entry.path)
                    day += 1
                    dataProcessing.readFilterSave(entry.path, f"{filtered_path}{month:02d}-{day:02d}.parquet") #use readFilterSave2 for STS

    end = time()
    print("Done! It took: ", (end-start)/60, " minutes.")


# If it for some reason failed underway, use this function to restart from where it failed.
def main3(months, filtered_path, year):
    start = time()

    print("Getting data from NTNUs copy of AIS-data from Kystverket.")

    restart_from = f"{year}-08-28"

    started = False

    for month in range(1, months + 1):

        pattern = f"../../Data/{year}/date_utc={year}-{month:02d}-*"
        folders = sorted(glob.glob(pattern))

        if not folders:
            print("No folders for month:", month)
            continue

        for folder in folders:

            # Extract date from folder name
            folder_date = folder.split("date_utc=")[-1]

            # Skip until we reach restart date
            if not started:
                if folder_date < restart_from:
                    continue
                started = True

            for entry in os.scandir(folder):
                if entry.is_file() and entry.name.endswith(".parquet"):

                    print("Processing file:", entry.path)

                    month_day = folder_date[5:]   # gets "08-28" from "2025-08-28"

                    output_name = f"{filtered_path}{month_day}.parquet"

                    dataProcessing.readFilterSave(
                        entry.path,
                        output_name
                    )

    end = time()

    print("Done! It took:", (end - start) / 60, "minutes.")


if __name__ == "__main__":
    main(months=12, filtered_path="2024", year=2024)
