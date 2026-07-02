from pathlib import Path
import processingDCA, directLabelsFromDCA


# Choose year and nr of months
YEAR = 2023

GEAR_TYPES = ["Trål", "Not", "Krokredskap", "Snurrevad", "Garn", "Bur og ruser"] # All main gear types used by the NDF (except other gear)
ACTIVITIES = ["I fiske"] # What activity type to consider. If vessels start reporting other activity consisently, add them here. 

DURATION_LIMITS = {
    "Trål": (30, 600),
    "Not": (10, 250),
    "Snurrevad": (10, 250),
    "Krokredskap": (500, 1500), 
    "Garn": (150, 1000),
    "Bur og ruser": (10, 300)
}

CLEAN_AIS_PATH = f"Preprocessing/Processed_AIS_{YEAR}/Cleaned"
RAW_DCA_PATH = f"Labeling/DCA_data/elektronisk-rapportering-ers-{YEAR}-fangstmelding-dca.csv" # UPLOAD DCA data from fdir.no to the folder
CLEAN_DCA_PATH = f"Labeling/DCA_data/dca-clean-{YEAR}.csv"

DIRECT_LABELS_PATH = f"Labeling/Direct_labels"

def main():
    folder_paths = []

    for p in folder_paths:
        path = Path(p)

        if path.exists():
            print(f"[EXISTS]  {path}")
        else:
            path.mkdir(parents=True)
            print(f"[CREATED] {path}")
    
    
    processingDCA.main(raw_dca_path=RAW_DCA_PATH, clean_dca_path=CLEAN_DCA_PATH, activities=ACTIVITIES, 
                       gear_types=GEAR_TYPES, duration_limits=DURATION_LIMITS)
    
    directLabelsFromDCA.main(clean_dca_path=CLEAN_DCA_PATH, clean_path=CLEAN_AIS_PATH, year=YEAR)
    

    


if __name__ == "__main__":
    main()