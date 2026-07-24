# Master Thesis: ML-Based Classification of Fishing Vessel Activity from AIS Data

Code for my master's thesis at NTNU, which uses machine learning to analyze
AIS (Automatic Identification System) data from Norwegian waters. The thesis
covers two classification tasks:

1. **Binary fishing / non-fishing detection** — is a vessel fishing at a given point in its trajectory?
2. **Gear-type classification** — for segments identified as fishing, which gear type is in use
   (trawl, purse seine, Scottish seine, longline, gillnet, hooked gear, traps)?

Training labels are derived from DCA (daily catch report) data from the Norwegian
Directorate of Fisheries (NDF), matched against AIS messages from the Norwegian Coastal Administration (NCA).

## Data

Raw AIS and DCA data are **not included** in this repository due to
size and data-sharing restrictions. To reproduce the pipeline you will need:

- **AIS data**: from the Norwegian Coastal Administration (NCA), organized as daily
  parquet files under `RAW_AIS/{year}/date_utc={year}`.
- **DCA data**: daily catch reports from the Directorate of Fisheries
  (`elektronisk-rapportering-ers-{year}-fangstmelding-dca.csv`, downloaded from
  NDF and placed in `Labeling/DCA_data/`).

## Repository structure

```
Preprocessing/                       # Raw AIS filtering, cleaning, resampling
Labeling/                            # DCA processing + direct/confident label extraction
Fishing_no_fishing_classification/   # Binary fishing detection (XGBoost, LSTM, BiLSTM)
Gear_type_classification/            # Gear-type classification (1D CNN)
Figures/                             # Standalone plotting/analysis scripts used in the thesis
```

## Requirements
Install the requirements defined in `requirements.txt`.


## Pipeline

### 1. Preprocessing
Filters raw AIS data to fishing vessels within the region of interest, concatenates daily
files into monthly batches, cleans trajectories.
```
python Preprocessing/preprocessing_main.py
```

Set `YEAR` and `MONTHS`

### 2. Labeling
Cleans the DCA catch reports, matches them to AIS trajectories to produce direct labels,
then derives the confident-label set (the confident non-fishing / unknown split described
in the thesis).

```
python Labeling/labeling_main.py
```

Set `YEAR` and adjust `GEAR_TYPES` / `DURATION_LIMITS` as needed. Requires the raw DCA CSV
to already be placed in `Labeling/DCA_data/`.

### 3. Fishing vs. no-fishing classification
```
Fishing_no_fishing_classification/
├── features_for_labeled.py       # Build features from confident labels
├── features_for_unlabeled.py     # Build features for unlabeled (deployment) data
├── get_mmsi_split.py             # Train/validation/test split by MMSI
├── XGBoost/
│   ├── xgb_tuning.py              # Hyperparameter tuning (Optuna)
│   └── xgb_seeds.py               # Multi-seed evaluation
├── LSTM/
│   ├── lstm_optuna.py             # Hyperparameter tuning
│   ├── lstm_train_val_test.py     # Train/val/test run
│   ├── lstm_train_full.py         # Deployment model, trained on full 2023+2024
│   ├── predict_unlabeled_2025.py  # Inference on unlabeled 2025 data
│   ├── predict_labeled_2025.py    # Inference on labeled 2025 data
│   ├── predict_russian_trawler.py # Case-study inference (foreign vessel)
│   ├── metrics.py / metrics_2025.py
└── BiLSTM/
    ├── bilstm_optuna.py           # Hyperparameter tuning
    └── bilstm_train_val_test.py   # Train/val/test run
```

Order of operations:
1. Run `features_for_labeled.py` and `features_for_unlabeled.py`.
2. Run `get_mmsi_split.py` to obtain the MMSI-level train/validation/test split
   (note: sort the tuning MMSI set before shuffling, for reproducibility).
3. Run `xgb_tuning.py` and `xgb_seeds.py` for the XGBoost baseline.
4. Run `lstm_optuna.py` and `bilstm_optuna.py` for hyperparameter tuning.
5. Run `lstm_train_val_test.py` and `bilstm_train_val_test.py` for evaluation.
6. Run `lstm_train_full.py` to train the deployment model on the full 2023–2024 data.
7. Run the `predict_*.py` scripts for inference on unlabeled/deployment data.

### 4. Gear-type classification
```
Gear_type_classification/
├── extract_only_gear_reports.py   # Keep only segments with a reported fishing gear
├── features.py                    # Feature extraction
├── create_dataset.py              # Assemble the training dataset
└── cnn_model.py                   # 1D CNN training and evaluation
```

Order of operations:
1. Run `extract_only_gear_reports.py`.
2. Run `features.py`.
3. Run `create_dataset.py`.
4. Run `cnn_model.py`.

### 5. Figures
Standalone scripts used to generate the plots and maps in the thesis (traffic heatmaps,
speed distributions, gap analysis, gear-report statistics, case-study vessel tracks, etc.).
Each script can be run independently once the relevant upstream data/models exist.


The README file was generated with assistance from Claude. As of now the repo is not ready to run with one shared main and the filepaths in the different scripts will likely lead to some crashes and needs to be updated as one runs the scripts. I will improve the structure and readability of the repo later.