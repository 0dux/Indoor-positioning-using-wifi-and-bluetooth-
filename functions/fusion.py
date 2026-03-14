"""
fusion.py — Merges Wi-Fi and BLE RSSI fingerprints into a unified dataset.

The idea: Wi-Fi and BLE capture different signal characteristics.
By combining both, we create a richer fingerprint that should improve
location prediction accuracy compared to using Wi-Fi alone.

We also keep a Wi-Fi-only version so we can compare the two approaches.
"""

import pandas as pd
from functions.load_data import (
    RSSI_COLUMNS,
    load_scenario_data,
    handle_missing_rssi,
    normalize_rssi,
    apply_normalization,
    encode_labels,
)


def create_wifi_only(
    wifi_db: pd.DataFrame,
    wifi_test: pd.DataFrame,
):
    """
    Prepares a Wi-Fi-only dataset ready for ML.

    Takes raw (already location-labelled) DataFrames and returns
    preprocessed versions with features and labels.

    Returns
    -------
    db_clean   : preprocessed database DataFrame
    test_clean : preprocessed test DataFrame
    encoder    : fitted LabelEncoder
    scaler     : fitted MinMaxScaler
    feature_cols : list of feature column names
    """
    # Handle missing values
    db   = handle_missing_rssi(wifi_db)
    test = handle_missing_rssi(wifi_test)

    # Normalize RSSI
    db, scaler = normalize_rssi(db)
    test       = apply_normalization(test, scaler)

    # Encode labels
    db, test, encoder = encode_labels(db, test)

    # Feature columns are just the three RSSI readings
    feature_cols = RSSI_COLUMNS.copy()

    return db, test, encoder, scaler, feature_cols


def create_fused_dataset(
    wifi_db: pd.DataFrame,
    wifi_test: pd.DataFrame,
    ble_db: pd.DataFrame,
    ble_test: pd.DataFrame,
):
    """
    Merges Wi-Fi and BLE RSSI readings into a single fingerprint per location.

    Both datasets share the same (x, y) grid, so we merge on (x, y)
    and rename columns to distinguish WiFi vs BLE signals:
      - WiFi_RSSI_A, WiFi_RSSI_B, WiFi_RSSI_C
      - BLE_RSSI_A,  BLE_RSSI_B,  BLE_RSSI_C

    This gives us 6 features instead of 3 → richer fingerprint.

    Returns
    -------
    db_clean     : preprocessed fused database DataFrame
    test_clean   : preprocessed fused test DataFrame
    encoder      : fitted LabelEncoder
    scaler       : fitted MinMaxScaler (for the 6 fused columns)
    feature_cols : list of 6 feature column names
    """
    # ── Rename RSSI columns to distinguish WiFi vs BLE ──
    wifi_rename = {c: f"WiFi_{c.replace(' ', '_')}" for c in RSSI_COLUMNS}
    ble_rename  = {c: f"BLE_{c.replace(' ', '_')}"  for c in RSSI_COLUMNS}

    # Work on copies so we don't modify the originals
    w_db   = wifi_db.copy().rename(columns=wifi_rename)
    w_test = wifi_test.copy().rename(columns=wifi_rename)
    b_db   = ble_db.copy().rename(columns=ble_rename)
    b_test = ble_test.copy().rename(columns=ble_rename)

    # ── Merge on (x, y) coordinates ──
    # Both datasets share the same measurement grid, so this is a 1-to-1 join
    merge_keys = ["x", "y", "location"]  # location is derived from (x, y)

    fused_db = w_db.merge(
        b_db[[*merge_keys, *ble_rename.values()]],
        on=merge_keys,
        how="inner",
    )
    fused_test = w_test.merge(
        b_test[[*merge_keys, *ble_rename.values()]],
        on=merge_keys,
        how="inner",
    )

    print(f"  ✓ Fused database: {fused_db.shape[0]} points × {fused_db.shape[1]} cols")
    print(f"  ✓ Fused tests:    {fused_test.shape[0]} points × {fused_test.shape[1]} cols")

    # ── Define the 6 feature columns ──
    feature_cols = list(wifi_rename.values()) + list(ble_rename.values())

    # ── Handle missing values ──
    fused_db[feature_cols]   = fused_db[feature_cols].fillna(-100)
    fused_test[feature_cols] = fused_test[feature_cols].fillna(-100)

    # ── Normalize all 6 RSSI features to [0, 1] ──
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    fused_db[feature_cols]   = scaler.fit_transform(fused_db[feature_cols])
    fused_test[feature_cols] = scaler.transform(fused_test[feature_cols])

    # ── Encode labels ──
    fused_db, fused_test, encoder = encode_labels(fused_db, fused_test)

    return fused_db, fused_test, encoder, scaler, feature_cols


def prepare_both_datasets(scenario: int = 1):
    """
    Convenience function: loads WiFi + BLE data for a scenario and returns
    BOTH the WiFi-only and Fused datasets, ready for ML.

    Returns
    -------
    wifi_data : dict with keys 'db', 'test', 'encoder', 'scaler', 'features'
    fused_data: dict with same keys
    """
    print(f"\n📡 Loading Wi-Fi data for Scenario {scenario}…")
    wifi_db, wifi_test = load_scenario_data(scenario=scenario, technology="WiFi")
    print(f"  ✓ WiFi — Database: {wifi_db.shape[0]} | Tests: {wifi_test.shape[0]}")

    print(f"\n📡 Loading BLE data for Scenario {scenario}…")
    ble_db, ble_test = load_scenario_data(scenario=scenario, technology="BLE")
    print(f"  ✓ BLE  — Database: {ble_db.shape[0]} | Tests: {ble_test.shape[0]}")

    # Wi-Fi only
    print("\n🔧 Preparing Wi-Fi-only dataset…")
    w_db, w_test, w_enc, w_scl, w_feats = create_wifi_only(wifi_db, wifi_test)

    # Fused (WiFi + BLE)
    print("\n🔧 Preparing Fused (WiFi + BLE) dataset…")
    f_db, f_test, f_enc, f_scl, f_feats = create_fused_dataset(
        wifi_db, wifi_test, ble_db, ble_test
    )

    wifi_data = {
        "db": w_db, "test": w_test,
        "encoder": w_enc, "scaler": w_scl, "features": w_feats,
    }
    fused_data = {
        "db": f_db, "test": f_test,
        "encoder": f_enc, "scaler": f_scl, "features": f_feats,
    }

    return wifi_data, fused_data
