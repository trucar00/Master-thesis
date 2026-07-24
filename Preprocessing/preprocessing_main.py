from pathlib import Path
import getData, concatParquets, cleaningOfAIS, downSample


# Choose year and nr of months
YEAR = 2025
MONTHS = 12

RESAMPLE_STEP = "30s"

RAW_AIS_PATH = f"RAW_AIS/{YEAR}/date_utc={YEAR}" # NTNUs copy of AIS data are in folders of years with raw data on a daily basis
FILTERED_PATH = f"Preprocessing/Processed_AIS_{YEAR}/Filtered_parquets" # Folder for filtered AIS data (fishing vessels within region of interest (ROI))
CONCAT_PATH = f"Preprocessing/Processed_AIS_{YEAR}/Concatenated/" # The daily AIS files are concatenated on a monthly basis

CLEAN_PATH = f"Preprocessing/Processed_AIS_{YEAR}/Cleaned"

RESAMPLE_PATH = f"Preprocessing/Processed_AIS_{YEAR}/Resampled"

def main():
    folder_paths = [FILTERED_PATH, CONCAT_PATH, CLEAN_PATH, RESAMPLE_PATH]

    for p in folder_paths:
        path = Path(p)

        if path.exists():
            print(f"[EXISTS]  {path}")
        else:
            path.mkdir(parents=True)
            print(f"[CREATED] {path}")
    
    getData.main(months=MONTHS, raw_ais_path=RAW_AIS_PATH, filtered_path=FILTERED_PATH)
    
    concatParquets.main(months=MONTHS, filtered_path=FILTERED_PATH, concat_path=CONCAT_PATH)

    cleaningOfAIS.main(months=MONTHS, concat_path=CONCAT_PATH, cleaned_path=CLEAN_PATH)

    #downSample.main(cleaned_path=CLEAN_PATH, resampled_path=RESAMPLE_PATH, step=RESAMPLE_STEP, months=MONTHS) # if one want to resample


if __name__ == "__main__":
    main()