# Master Thesis
Code for my master's thesis where I used different machine learning methods to detect fishing in Norwegian waters.

## Preprocessing
Run the preprocessing_main.py function

## Labeling
Run the labeling_main.py

## Fishing vs no fishing classification
1. Run the features scripts.
2. Obtain the split of train, validation and test mmsis with get_mmsi_split.py
3. Run the XGB_tuning.py and xgb_seeds.py
4. Run the tuning scripts for LSTM and BiLSTM
5. Run the lstm_train_val_test.py and bilstm_train_val_test.py 
6. For the deployment model, run the lstm_train_full.py. This trains a model on the whole of 2023 and 2024.
7. Run the predict scripts.

## Gear_type_classification
1. Run the extract_only_gear_reports.py to get only segments where fishing has been reported
2. Create the features with features.py
3. Create the dataset.
4. Run the cnn_model.py.

## Figures