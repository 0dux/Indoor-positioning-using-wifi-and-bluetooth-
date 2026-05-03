# Indoor Positioning — Development Notes
> **Purpose:** A plain-English reference for understanding, running, and explaining this project to your invigilator.  
> Covers the 20% of the code that does 80% of the work.

---

## 1. What This Project Does (The Big Picture)

This project answers one question:

> **"Can we locate a person inside a building more accurately by combining WiFi and Bluetooth signals than by using WiFi alone?"**

It does this by:
1. Downloading a public dataset of indoor signal measurements (RSSI = signal strength readings)
2. Training three machine learning models on those readings
3. Comparing how far off each model's predicted position is from the actual position
4. Displaying the results as an interactive Streamlit web app

The core finding the demo shows is: **WiFi + BLE fusion produces lower localization error than WiFi-only** in most test scenarios.

---

## 2. Key Concepts to Know

| Term | What it means |
|---|---|
| **RSSI** | Received Signal Strength Indicator — a number (in dBm) that measures how strong a WiFi or BLE signal is. More negative = weaker. −100 dBm ≈ no signal. |
| **Fingerprinting** | Each physical location has a unique "fingerprint" — a pattern of RSSI values from multiple transmitters. We record these during training and match them during prediction. |
| **Database / Fingerprint Map** | The training data — a grid of known locations with their RSSI fingerprints recorded. |
| **Test set** | New RSSI measurements at the same locations. The model predicts where you are; we check how far off it is. |
| **Mean Distance Error** | The average physical distance (in metres) between where the model thought you were and where you actually were. **Lower is better.** |
| **Sensor Fusion** | Combining data from two different sensors (WiFi + BLE) to get a richer picture and improve accuracy. |

---

## 3. Project Structure — Every File Explained

```
indoor-positioning/
│
├── streamlit_app.py          ← The web app (front-end)
│
├── functions/                ← The backend logic
│   ├── __init__.py           ← Makes the folder importable as a Python package
│   ├── load_data.py          ← Downloads & cleans the dataset
│   ├── fusion.py             ← Combines WiFi + BLE into one dataset
│   ├── models.py             ← Trains & evaluates the 3 ML models
│   ├── plots.py              ← Generates saved chart images (used by notebook)
│   └── frontend_backend.py  ← Bridge layer between models and Streamlit
│
├── notebooks/
│   └── demo.ipynb            ← Jupyter notebook: step-by-step walkthrough
│
├── outputs/                  ← Saved chart images (confusion matrices, etc.)
├── data/                     ← Optional local CSV storage
├── indoor-env/               ← Python virtual environment (local packages)
└── requirements.txt          ← List of all Python packages needed
```

---

## 4. How to Run the Project

### Prerequisites
- Python 3.10+ installed
- Internet connection (data is downloaded from GitHub on first run)

### Commands (run these in PowerShell inside the project folder)

```powershell
# Step 1 — Activate the virtual environment (do this every time)
.\indoor-env\Scripts\Activate.ps1

# Step 2a — Run the Streamlit web app (main demo)
streamlit run streamlit_app.py

# Step 2b — OR open the Jupyter notebook instead
jupyter notebook notebooks/demo.ipynb

# To install packages from scratch (only if indoor-env is missing)
pip install -r requirements.txt
```

The Streamlit app opens automatically at **http://localhost:8501** in your browser.

> **First run is slow** — it downloads the dataset from GitHub (~30–60 seconds). After that it's cached and fast.

---

## 5. The Data Pipeline — How Data Flows Through the System

```
GitHub (xlsx files)
        │
        ▼
  load_data.py         ← Downloads, cleans, normalises raw RSSI data
        │
        ▼
  fusion.py            ← Builds two datasets:
        │                   (a) WiFi-only  [3 signal features]
        │                   (b) WiFi+BLE   [6 signal features]
        ▼
  models.py            ← Trains KNN, SVR, Random Forest on each dataset
        │                 Measures mean distance error per model
        ▼
  frontend_backend.py  ← Packages results into dictionaries Streamlit can use
        │
        ▼
  streamlit_app.py     ← Renders the interactive web UI
```

---

## 6. File-by-File Deep Dive

---

### `functions/load_data.py` — The Data Loader

**What it does:** Downloads the dataset from GitHub and prepares it for machine learning.

#### The Dataset
The dataset comes from this public GitHub repo:  
`pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting`

Each scenario is an Excel file with sheets for WiFi, BLE, and Zigbee. Each row is one measurement:

| x | y | RSSI A | RSSI B | RSSI C |
|---|---|--------|--------|--------|
| 0.5 | 1.0 | −62 | −75 | −88 |

- `x, y` = physical coordinates in the room (in metres)
- `RSSI A/B/C` = signal strength from 3 transmitters

#### Key Functions

**`download_xlsx(url)`**
- Downloads an Excel file from a URL and keeps it in memory
- Uses `@lru_cache` — meaning it only downloads each file once per run, even if called multiple times. Like a "remember the result" shortcut.

**`handle_missing_rssi(df, fill_value=-100)`**
- Replaces any missing RSSI readings with −100 dBm
- −100 is the standard "no signal" convention in RF engineering
- This dataset has no missing values, but the function makes the code robust

**`normalize_rssi(df)`** ← **One of the most important functions**
- Scales all RSSI values from their raw dBm range (e.g., −100 to −40) to a 0–1 range
- Uses `MinMaxScaler` from scikit-learn
- **Critical rule:** The scaler is **fit (trained) only on the database/training data**, then applied to the test data using `.transform()`. This prevents the model from "peeking" at test statistics — a common mistake called **data leakage**.

**`encode_labels(db_df, test_df)`**
- Converts text location labels like `"P_0.5_1.0"` into numbers like `0, 1, 2, 3…`
- ML models need numbers, not strings
- Fitted on the union of train+test labels to ensure no "unseen label" errors

**`preprocess_pipeline(db_df, test_df)`**
- One function that chains together: fill missing → normalize → encode labels
- Returns cleaned database, cleaned test, encoder, and scaler objects

---

### `functions/fusion.py` — The Signal Combiner

**What it does:** Creates two versions of the dataset — WiFi-only (3 features) and WiFi+BLE fused (6 features) — so we can compare them fairly.

#### Key Functions

**`create_wifi_only(wifi_db, wifi_test)`**
- Just runs the standard preprocessing pipeline on WiFi data
- Returns 3 feature columns: `RSSI A`, `RSSI B`, `RSSI C`

**`create_fused_dataset(wifi_db, wifi_test, ble_db, ble_test)`** ← **Core of the fusion idea**

This is where sensor fusion actually happens. Here's the logic:

1. Rename columns so they don't clash:
   - `RSSI A` → `WiFi_RSSI_A` and `BLE_RSSI_A`
2. **Merge** the WiFi and BLE DataFrames on `(x, y, location)` — since both technologies were measured at the same physical grid points, this is a 1-to-1 join
3. The result has **6 RSSI columns** instead of 3 — a richer fingerprint
4. Normalize all 6 features and encode labels

```python
# Before fusion:  3 features per location
#   WiFi_RSSI_A, WiFi_RSSI_B, WiFi_RSSI_C

# After fusion:   6 features per location  
#   WiFi_RSSI_A, WiFi_RSSI_B, WiFi_RSSI_C,
#   BLE_RSSI_A,  BLE_RSSI_B,  BLE_RSSI_C
```

**`prepare_both_datasets(scenario)`**
- One-stop function: loads WiFi and BLE data, calls both `create_wifi_only` and `create_fused_dataset`, returns two ready-to-train dicts

---

### `functions/models.py` — The Machine Learning Engine

**What it does:** Trains three ML models and measures how accurately each one locates you.

#### Why Regression, Not Classification?

The natural instinct is classification: "predict which location label this fingerprint belongs to." But if almost every location appears only once in the training data (which is common in indoor fingerprinting), classification is meaningless — there's nothing to generalize from a single example.

**`_should_use_regression(db)`** checks this:
```python
n_classes / n_samples > 0.5  →  switch to regression
```
If more than half the classes have just one sample, we instead predict `(x, y)` coordinates directly (regression).

**This project uses regression** for both scenarios.

#### The Three Models

**`train_and_evaluate_regression(X_train, X_test, y_train, y_test)`** ← **The main training function**

Trains and evaluates three regressors:

| Model | How it works (simple) | Key hyperparameters |
|---|---|---|
| **KNN (K-Nearest Neighbors)** | Find the 3 most similar fingerprints in the database, average their positions | `n_neighbors=3, weights="distance"` |
| **SVR (Support Vector Regression)** | Find a function that fits the data with maximum margin | `kernel="rbf", C=10` via `MultiOutputRegressor` |
| **Random Forest** | Build 200 decision trees, average their predictions | `n_estimators=200, random_state=42` |

> **Why `MultiOutputRegressor` for SVR?**  
> scikit-learn's `SVR` can only predict one number at a time. We need to predict both `x` and `y`. `MultiOutputRegressor` trains two separate SVR models (one for x, one for y) and combines their outputs.

**Distance error calculation:**
```python
errors = np.linalg.norm(y_test - y_pred, axis=1)
```
This is just Pythagoras — `√((x_actual - x_pred)² + (y_actual - y_pred)²)` — the straight-line distance between actual and predicted position, in metres.

**`compare_results(wifi_eval, fused_eval)`**
- Creates a summary table: for each model, shows WiFi-only error, Fused error, and the improvement

---

### `functions/frontend_backend.py` — The Bridge Layer

**What it does:** Runs the full pipeline and packages everything into simple Python dictionaries that Streamlit can display without knowing anything about ML.

#### Key Design Decisions

**`@lru_cache`** on `run_scenario(scenario)`:
- The full pipeline (download + train + evaluate) takes ~30 seconds
- Once run, the result is stored in memory
- Every Streamlit rerun (triggered by clicking widgets) reuses the cached result — the model only trains **once** per scenario per session

**`_positioning_score(mean_error, room_diagonal)`**
```python
score = 100 × (1 − error / room_diagonal)
```
- Converts raw error into a 0–100 score relative to room size
- A 1 m error in a 2 m room is terrible (score: 50); a 1 m error in a 20 m room is fine (score: 95)
- **Not classification accuracy** — this is a custom presentation metric

**`run_scenario(scenario)` → returns:**
```python
{
  "metrics":     [...],  # per-model WiFi error, Fused error, improvement
  "predictions": [...],  # per test-point: actual_x, actual_y, predicted_x, predicted_y, error_m
  "room":        {...},  # min/max x and y coordinates for drawing the floor plan
}
```

---

### `streamlit_app.py` — The Web App

**What it does:** Renders the interactive demo in the browser. It only reads from `frontend_backend.py` — it does no ML itself.

#### App Structure

```
Header (title + subtitle)
│
├── ⚙️ Configure Demo card
│       • Scenario dropdown (1 or 2)
│       • Model dropdown (KNN / SVR / Random Forest)
│       • Test Point slider
│
├── 4 Metric cards
│       • WiFi Mean Error
│       • Fused Mean Error
│       • Error Reduction %
│       • Fused Positioning Score
│
└── 3 Tabs
        ├── 📊 Dashboard
        │       • Side-by-side floor plans (WiFi vs Fused)
        │       • Bar chart (model error comparison)
        │       • Scenario insight panel
        │       • Evaluation table
        │
        ├── 🎬 Live Demo
        │       • ▶ Play / ⏸ Pause auto-animation
        │       • Floor plan that updates each step
        │       • Green trajectory trail of visited points
        │       • Per-step scoreboard
        │
        └── 🏆 Model Race
                • All 3 models on one floor plan
                • Medal leaderboard (🥇🥈🥉)
                • Win-count bar chart across all test points
```

#### Key Streamlit Concepts

**`st.session_state`** — Streamlit reruns the entire script from top to bottom on every user interaction. `session_state` is a dictionary that persists between reruns. The live demo's play/pause button and current frame index use this.

**`st.cache_data`** — Wraps `run_scenario()` so the model result is cached by Streamlit across reruns (on top of the `lru_cache` in the function itself — double protection).

**`st.tabs()`** — Creates the three tab sections. Code inside `with tab1:` only renders when that tab is active.

**`st.rerun()`** — Forces the script to re-execute. The live demo calls this after each frame advance with a `time.sleep()` delay to create the animation effect.

---

## 7. How the Files Connect — The Full Call Chain

```
streamlit_app.py
    │
    └── cached_demo_data(scenario)
            │
            └── frontend_backend.run_scenario(scenario)   [cached]
                    │
                    ├── fusion.prepare_both_datasets(scenario)
                    │       │
                    │       ├── load_data.load_scenario_data(scenario, "WiFi")
                    │       │       └── download_xlsx() → read_sheet() → create_location_label()
                    │       │
                    │       ├── load_data.load_scenario_data(scenario, "BLE")
                    │       │
                    │       ├── fusion.create_wifi_only(wifi_db, wifi_test)
                    │       │       └── normalize → encode → 3 features
                    │       │
                    │       └── fusion.create_fused_dataset(wifi_db, wifi_test, ble_db, ble_test)
                    │               └── merge → normalize → encode → 6 features
                    │
                    ├── models.run_full_evaluation(wifi_data)
                    │       └── _should_use_regression() → True
                    │           └── train_and_evaluate_regression()
                    │                   → KNN, SVR, Random Forest fitted & evaluated
                    │
                    ├── models.run_full_evaluation(fused_data)
                    │       └── same as above but with 6 features
                    │
                    └── returns dict → metrics, predictions, room bounds
```

---

## 8. Common Invigilator Questions & Answers

**Q: Why use regression instead of classification?**  
A: Each physical location in the dataset appears only once or twice in the training set. With so few samples per class, classification cannot generalize. Regression directly predicts `(x, y)` coordinates, which is more meaningful for a positioning system anyway — and mean distance error in metres is a more interpretable metric than accuracy percentage.

**Q: How does WiFi+BLE fusion improve accuracy?**  
A: WiFi and BLE signals have different propagation characteristics. WiFi signals are stronger and travel further; BLE signals are shorter-range and more sensitive to the immediate environment. By combining both sets of RSSI readings, we give the model 6 features instead of 3, creating a more unique fingerprint per location. This reduces the chance that two different locations look identical to the model.

**Q: Why MinMaxScaler specifically?**  
A: RSSI values vary between about −100 and −30 dBm. Without normalization, models like KNN would treat an RSSI difference of 30 (e.g., −70 to −40) the same as a coordinate difference of 30 metres. MinMaxScaler brings all features into [0, 1] so the model treats them equally. We use MinMax (not StandardScaler) because RSSI doesn't follow a normal distribution.

**Q: What is the scaler fitted on? Why does it matter?**  
A: The scaler is fitted **only on the training (database) data**, then applied to test data using `.transform()` (not `.fit_transform()`). If we fitted the scaler on both train and test together, the model would have indirect knowledge of test statistics during training — this is called **data leakage** and would artificially inflate performance metrics.

**Q: What does KNN actually do at prediction time?**  
A: It searches the entire training database for the 3 fingerprints most similar (by Euclidean distance in RSSI space) to the test fingerprint. It then averages the `(x, y)` coordinates of those 3 nearest neighbors, weighted by their distance — closer matches have more influence.

**Q: Why is Random Forest generally more accurate than KNN?**  
A: Random Forest builds many decision trees on random subsets of data and features, then averages their predictions. This reduces overfitting. KNN is sensitive to irrelevant features and noise because it uses raw distance in the full feature space. Random Forest implicitly selects which RSSI readings matter most.

**Q: What is the Positioning Score?**  
A: It's a custom presentation metric, not a standard ML metric. It normalizes mean distance error relative to room size: `Score = 100 × (1 − error / room_diagonal)`. A score of 70/100 means the average error is 30% of the room's diagonal length. The primary technical metric is always mean distance error in metres.

**Q: Where does the dataset come from?**  
A: The public "RSSI Dataset for Indoor Localization Fingerprinting" by P. Spachos, hosted on GitHub. It contains 3 scenarios (different rooms/environments) each with a database (training fingerprints) and test set, measured for WiFi, BLE, and Zigbee technologies. This project uses Scenarios 1 and 2 with WiFi and BLE.

**Q: What does the `lru_cache` do?**  
A: `@lru_cache` is Python's built-in "remember the last N results" decorator. When `run_scenario(1)` is called the first time, it runs the full pipeline (~30s). The second time it's called with the same argument, it immediately returns the stored result without recomputing. Streamlit reruns the entire script on every click, so without caching, the models would retrain on every button press.

---

## 9. Packages Used & Why

| Package | Purpose |
|---|---|
| `pandas` | DataFrame operations — the core data structure for all tabular data |
| `numpy` | Numerical operations — distance calculations, array math |
| `scikit-learn` | All ML models (KNN, SVR, RF), preprocessing (MinMaxScaler, LabelEncoder) |
| `streamlit` | The web app framework — turns Python scripts into interactive UIs |
| `matplotlib` | Chart generation (bar charts, floor plan plots) |
| `requests` | HTTP downloads of the dataset from GitHub |
| `openpyxl` | Reading `.xlsx` Excel files (used internally by pandas) |
