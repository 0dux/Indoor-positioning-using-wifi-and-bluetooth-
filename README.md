# Indoor Positioning Project (Beginner-Friendly Guide)

This project predicts indoor locations using WiFi and Bluetooth (BLE) signal strengths. It downloads the public "RSSI Dataset for Indoor Localization Fingerprinting", prepares the data, trains machine learning models, and plots results.

For the final demo, the project focuses on Scenario 1 and Scenario 2 and compares WiFi-only positioning against WiFi+BLE fused positioning using mean distance error in meters. Lower error means better localization.

The goal of this README is to explain what each folder and file does in very simple language.

## Folder Guide

### data/

- Purpose: Local data storage (optional).
- The code can download data directly from GitHub, but if you save CSVs, they go here.
- Subfolders:
  - data/wifi/ : place to store WiFi CSV files if you save them.
  - data/ble/ : place to store BLE CSV files if you save them.

### functions/

- Purpose: All the helper code (data loading, fusion, models, plots).
- This is the core logic used by the notebook.

### notebooks/

- Purpose: Jupyter notebook(s) that run the full workflow step-by-step.
- demo.ipynb is the main entry point for most users.

### outputs/

- Purpose: Saved plots (images) produced by the notebook.
- Examples: confusion matrix images and accuracy/error comparison charts.

### indoor-env/

- Purpose: Local Python virtual environment (dependencies live here).
- You usually do not edit anything in this folder.

### .ipynb_checkpoints/

- Purpose: Auto-saved notebook backups created by Jupyter.
- You can ignore this folder.

### requirements.txt

- Purpose: List of Python packages you need to run the project.

## File-by-File Guide (Beginner Level)

### requirements.txt

- Contains the packages needed:
  - numpy, pandas: data handling
  - scikit-learn: machine learning
  - matplotlib, seaborn: plotting
  - jupyter, ipykernel: notebook support
  - requests: download dataset files

### functions/**init**.py

- Purpose: Makes it easier to import common functions in one line.
- It re-exports:
  - get_clean_data, load_scenario_data, RSSI_COLUMNS
  - prepare_both_datasets
  - run_full_evaluation, compare_results
  - plot_all_confusion_matrices, plot_accuracy_comparison
  - run_scenario, run_demo_scenarios

### functions/load_data.py

- Purpose: Download data and prepare it for modeling.
- Main ideas:
  - The dataset is stored on GitHub as Excel files.
  - Each sheet is a different technology (WiFi, BLE, Zigbee).
  - The data includes coordinates (x, y) and signal strengths (RSSI A/B/C).

Functions inside:

- download_xlsx(url)
  - Downloads an Excel file from the web and loads it into memory.

- read_sheet(xls, sheet_name)
  - Reads one sheet from the Excel file.
  - Removes empty columns and a "Point" ID column if it exists.

- create_location_label(df)
  - Creates a label like P_0.5_1.0 from x and y.
  - This is used for classification (predicting a label).

- load_scenario_data(scenario=1, technology="WiFi", save_dir=None)
  - Downloads one scenario (1, 2, or 3) for WiFi/BLE/Zigbee.
  - Returns two DataFrames: database (training) and tests (testing).
  - Optionally saves CSV files to a folder.

- load_all_scenarios(technology="WiFi")
  - Loads scenarios 1, 2, and 3 and combines them.
  - Adds a "scenario" column and prefixes labels so they stay unique.

- handle_missing_rssi(df, fill_value=-100)
  - Fills missing signal values with -100 (means "no signal").

- normalize_rssi(df)
  - Scales RSSI values to the range [0, 1].
  - Returns the scaled data and the scaler used.

- apply_normalization(df, scaler)
  - Uses the same scaler on new data (important for test data).

- encode_labels(db_df, test_df)
  - Converts location labels into numbers for ML models.
  - Fits the encoder on both train and test labels to avoid unknown labels.

- preprocess_pipeline(db_df, test_df)
  - Runs: missing-value handling -> normalization -> label encoding.
  - Returns cleaned data plus the label encoder and scaler.

- get_clean_data(scenario=1, technology="WiFi", save_dir=None)
  - One-call shortcut for loading and preprocessing.
  - Returns cleaned database, cleaned tests, encoder, and scaler.

### functions/fusion.py

- Purpose: Build two datasets:
  - WiFi-only (3 features)
  - WiFi + BLE fused (6 features)

Functions inside:

- create_wifi_only(wifi_db, wifi_test)
  - Preprocesses WiFi data only.
  - Returns cleaned data, encoder, scaler, and feature list.

- create_fused_dataset(wifi_db, wifi_test, ble_db, ble_test)
  - Merges WiFi and BLE data using the same (x, y, location) rows.
  - Renames columns to keep WiFi and BLE separate.
  - Normalizes all 6 features and encodes labels.

- prepare_both_datasets(scenario=1)
  - Convenience function used by the notebook.
  - Downloads WiFi and BLE, prepares both datasets, and returns them.

### functions/models.py

- Purpose: Train ML models and evaluate results.
- Important note: If almost every location is unique (one sample per class),
  the code switches to regression (predict x,y coordinates) instead of
  classification (predict label).

Functions inside:

- \_should_use_regression(db)
  - Checks if there are too many unique classes.
  - If yes, uses regression so results make sense.

- \_extract_features_and_targets(db, test, feature_cols, task)
  - Creates X (features) and y (targets).
  - For classification: y = label numbers.
  - For regression: y = (x, y) coordinates.

- train_and_evaluate_classification(X_train, X_test, y_train, y_test, label_names=None)
  - Trains KNN, SVM, Random Forest for classification.
  - Returns accuracy and confusion matrices.

- train_and_evaluate_regression(X_train, X_test, y_train, y_test)
  - Trains KNN regressor, SVR, Random Forest regressor.
  - Returns mean distance error and RMSE.

- run_full_evaluation(data, dataset_name="Dataset")
  - Full evaluation for one dataset (WiFi-only or fused).
  - Chooses classification or regression automatically.
  - Returns results and task type.

- compare_results(wifi_eval, fused_eval)
  - Creates a comparison table.
  - For classification: accuracy comparison.
  - For regression: mean distance error comparison.

### functions/frontend_backend.py

- Purpose: Clean backend wrapper for the Streamlit/frontend demo.
- It exposes only Scenario 1 and Scenario 2 for the final presentation.
- It returns:
  - model-wise WiFi-only and WiFi+BLE mean distance error
  - improvement in meters and percent
  - normalized positioning score for presentation cards
  - actual vs predicted coordinates for split-screen visualization

### functions/plots.py

- Purpose: Create and save plots in outputs/.

Functions inside:

- plot_confusion_matrix(cm, model_name, dataset_name, output_dir="outputs", label_names=None, ax=None)
  - Draws a heatmap confusion matrix for one model.

- plot_all_confusion_matrices(results, dataset_name, output_dir="outputs", encoder=None)
  - Creates a 3-panel figure (KNN, SVM, Random Forest).

- plot_accuracy_comparison(wifi_results, fused_results, output_dir="outputs")
  - Bar chart comparing accuracy (classification).

- plot_error_comparison(wifi_results, fused_results, output_dir="outputs", metric="mean_error")
  - Bar chart comparing error in meters (regression).

### notebooks/demo.ipynb

- Purpose: The full, step-by-step walkthrough.
- It does the following:
  1. Imports helper functions.
  2. Downloads WiFi and BLE data.
  3. Prepares WiFi-only and fused datasets.
  4. Trains models.
  5. Compares results.
  6. Saves plots into outputs/.

## How to Run (Simple Steps)

1. Create/activate your Python environment.
2. Install packages:
   - pip install -r requirements.txt
3. Open the notebook:
   - notebooks/demo.ipynb
4. Run each cell from top to bottom.

## Notes for Beginners

- RSSI means "Received Signal Strength Indicator". It is just a signal number.
- "Fingerprinting" means each location has a unique pattern of signals.
- WiFi-only uses 3 signal features; fused uses 6.
- The project switches to regression if classification does not make sense.
