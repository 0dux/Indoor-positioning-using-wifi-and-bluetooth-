"""
Quick script to download and inspect the RSSI dataset from GitHub.
We need to understand the actual structure of the xlsx files.
"""
import requests
import io
import sys
import subprocess

# Install openpyxl
subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl", "-q"])

import pandas as pd

# URLs for Scenario 1 xlsx files
base = "https://github.com/pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting/raw/master/Scenario1"
files = {
    "Database": f"{base}/Database_Scenario1.xlsx",
    "Tests": f"{base}/Tests_Scenario1.xlsx",
}

for name, url in files.items():
    print(f"\n{'='*60}")
    print(f"FILE: {name} ({url.split('/')[-1]})")
    print(f"{'='*60}")
    
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"  FAILED to download (status {resp.status_code})")
        continue
    
    xls = pd.ExcelFile(io.BytesIO(resp.content), engine='openpyxl')
    print(f"  Sheet names: {xls.sheet_names}")
    
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        print(f"\n  --- Sheet: '{sheet}' ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:")
        print(df.dtypes.to_string())
        print(f"  First 5 rows:")
        print(df.head().to_string())
        print(f"\n  Last 5 rows:")
        print(df.tail().to_string())
        print(f"\n  Null count:")
        print(df.isnull().sum().to_string())
