"""
load_data.py — Downloads and preprocesses Wi-Fi and BLE RSSI data
from the RSSI-Dataset-for-Indoor-Localization-Fingerprinting GitHub repo.

The dataset uses xlsx files with separate sheets for each wireless technology.
Each sheet has columns: x, y, RSSI A, RSSI B, RSSI C
- x, y are the physical coordinates where the reading was taken
- RSSI A/B/C are signal strength readings from three transmitters

We combine (x, y) into a single location label for classification.
"""

import os
import io
from functools import lru_cache
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ──────────────────────────────────────────────
# GitHub raw URLs for all three scenarios
# ──────────────────────────────────────────────
BASE_URL = (
    "https://github.com/pspachos/"
    "RSSI-Dataset-for-Indoor-Localization-Fingerprinting/raw/master"
)

SCENARIOS = {
    1: {
        "database": f"{BASE_URL}/Scenario1/Database_Scenario1.xlsx",
        "tests":    f"{BASE_URL}/Scenario1/Tests_Scenario1.xlsx",
    },
    2: {
        "database": f"{BASE_URL}/Scenario2/Database_Scenario2.xlsx",
        "tests":    f"{BASE_URL}/Scenario2/Tests_Scenario2.xlsx",
    },
    3: {
        "database": f"{BASE_URL}/Scenario3/Database_Scenario3.xlsx",
        "tests":    f"{BASE_URL}/Scenario3/Tests_Scenario3.xlsx",
    },
}


@lru_cache(maxsize=8)
def download_xlsx(url: str) -> pd.ExcelFile:
    """
    Downloads an xlsx file from a URL and returns a pandas ExcelFile object.
    Raises an error if the download fails.
    """
    print(f"  ↳ Downloading {url.split('/')[-1]} …")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()  # stop immediately if download fails
    return pd.ExcelFile(io.BytesIO(resp.content), engine="openpyxl")


def read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    Reads a specific sheet from an ExcelFile.
    Drops any fully-empty unnamed columns that appear in the raw data.
    """
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Drop unnamed placeholder columns (they are all NaN in the raw file)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols)

    # Also drop 'Point' column if it exists (only in Tests files — it's just a row id)
    if "Point" in df.columns:
        df = df.drop(columns=["Point"])

    return df


def create_location_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a human-readable location label from (x, y) coordinates.
    Example:  x=0.5, y=1.0  →  label "P_0.5_1.0"

    This gives us a categorical target variable for classification.
    """
    df = df.copy()
    df["location"] = df.apply(
        lambda row: f"P_{row['x']}_{row['y']}", axis=1
    )
    return df


def load_scenario_data(
    scenario: int = 1,
    technology: str = "WiFi",
    save_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads and returns (database_df, tests_df) for a given scenario & technology.

    Parameters
    ----------
    scenario   : int — 1, 2, or 3
    technology : str — "WiFi", "BLE", or "Zigbee"
    save_dir   : str or None — if provided, saves the raw data as CSV here

    Returns
    -------
    db_df   : DataFrame with columns [x, y, RSSI A, RSSI B, RSSI C, location]
    test_df : DataFrame with same columns
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Scenario must be 1, 2, or 3. Got {scenario}.")

    urls = SCENARIOS[scenario]

    # Download both xlsx files
    db_xls   = download_xlsx(urls["database"])
    test_xls = download_xlsx(urls["tests"])

    # Read the requested technology sheet
    db_df   = read_sheet(db_xls,   technology)
    test_df = read_sheet(test_xls, technology)

    # Create location labels from (x, y) coordinates
    db_df   = create_location_label(db_df)
    test_df = create_location_label(test_df)

    # Optionally save as CSV for offline use
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        db_path   = os.path.join(save_dir, f"db_scenario{scenario}_{technology}.csv")
        test_path = os.path.join(save_dir, f"test_scenario{scenario}_{technology}.csv")
        db_df.to_csv(db_path, index=False)
        test_df.to_csv(test_path, index=False)
        print(f"  ✓ Saved {db_path}")
        print(f"  ✓ Saved {test_path}")

    return db_df, test_df


def load_all_scenarios(
    technology: str = "WiFi",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data from ALL three scenarios and concatenates them.
    This gives us more training data (49+16+40 = 105 database points).

    We add a 'scenario' column so we can tell them apart if needed,
    and we prefix each location label with the scenario number so that
    locations from different rooms don't get mixed up.
    """
    all_db   = []
    all_test = []

    for scn in [1, 2, 3]:
        db, test = load_scenario_data(scenario=scn, technology=technology)

        # Prefix location with scenario number to keep them unique
        db["location"]   = db["location"].apply(lambda s, s_=scn: f"S{s_}_{s}")
        test["location"] = test["location"].apply(lambda s, s_=scn: f"S{s_}_{s}")

        db["scenario"]   = scn
        test["scenario"] = scn

        all_db.append(db)
        all_test.append(test)

    return pd.concat(all_db, ignore_index=True), pd.concat(all_test, ignore_index=True)


# ──────────────────────────────────────────────
# Preprocessing utilities
# ──────────────────────────────────────────────

RSSI_COLUMNS = ["RSSI A", "RSSI B", "RSSI C"]  # columns holding signal strengths


def handle_missing_rssi(df: pd.DataFrame, fill_value: int = -100) -> pd.DataFrame:
    """
    Replaces any missing RSSI values with a sensible default.
    -100 dBm means "essentially no signal" — a standard convention.

    (In this particular dataset there are no missing values, but this
    function makes the pipeline robust for other datasets.)
    """
    df = df.copy()
    df[RSSI_COLUMNS] = df[RSSI_COLUMNS].fillna(fill_value)
    return df


def normalize_rssi(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scales RSSI values to the [0, 1] range using Min-Max normalization.
    Returns the transformed DataFrame AND the fitted scaler (so we can
    apply the same transform to test data later).
    """
    df = df.copy()
    scaler = MinMaxScaler()
    df[RSSI_COLUMNS] = scaler.fit_transform(df[RSSI_COLUMNS])
    return df, scaler


def apply_normalization(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """
    Applies a previously fitted scaler to new data (e.g., the test set).
    """
    df = df.copy()
    df[RSSI_COLUMNS] = scaler.transform(df[RSSI_COLUMNS])
    return df


def encode_labels(db_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Converts the string 'location' column into integer class labels.
    Fits the encoder on the COMBINED locations from both database and tests
    to prevent 'unseen label' errors during prediction.
    """
    db_df = db_df.copy()
    test_df = test_df.copy()
    
    encoder = LabelEncoder()
    all_locations = pd.concat([db_df["location"], test_df["location"]])
    encoder.fit(all_locations)
    
    db_df["label"] = encoder.transform(db_df["location"])
    test_df["label"] = encoder.transform(test_df["location"])
    
    return db_df, test_df, encoder


def preprocess_pipeline(
    db_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, MinMaxScaler]:
    """
    Full preprocessing pipeline:
      1. Fill missing RSSI values with -100
      2. Normalize RSSI to [0, 1]
      3. Encode location labels as integers

    Returns preprocessed (db, test) DataFrames + the fitted encoder and scaler.
    """
    # Step 1 — Handle missing values
    db_df   = handle_missing_rssi(db_df)
    test_df = handle_missing_rssi(test_df)

    # Step 2 — Normalize RSSI (fit on database, transform both)
    db_df, scaler   = normalize_rssi(db_df)
    test_df         = apply_normalization(test_df, scaler)

    # Step 3 — Encode labels (fit on both to capture all unique locations)
    db_df, test_df, encoder = encode_labels(db_df, test_df)

    return db_df, test_df, encoder, scaler


# ──────────────────────────────────────────────
# Convenience: load + preprocess in one call
# ──────────────────────────────────────────────

def get_clean_data(
    scenario: int = 1,
    technology: str = "WiFi",
    save_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, LabelEncoder, MinMaxScaler]:
    """
    One-liner to load data from GitHub and return it fully preprocessed.

    Returns
    -------
    db_clean, test_clean, label_encoder, rssi_scaler
    """
    print(f"\n📡 Loading {technology} data for Scenario {scenario}…")
    db_df, test_df = load_scenario_data(scenario=scenario, technology=technology, save_dir=save_dir)
    print(f"  ✓ Database: {db_df.shape[0]} points | Tests: {test_df.shape[0]} points")

    db_clean, test_clean, encoder, scaler = preprocess_pipeline(db_df, test_df)
    print(f"  ✓ Preprocessing complete ({len(encoder.classes_)} unique locations)")

    return db_clean, test_clean, encoder, scaler
