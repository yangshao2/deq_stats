#!/usr/bin/env python3
import pandas as pd
import os
import sys

# 1. Edit these paths
input_file = 'SMLDEQdata_v5.csv'
output_dir = 'station_csvs'

# 2. Load the data (Excel first, then CSV)
try:
    df = pd.read_excel(input_file, engine='openpyxl')
except Exception:
    df = pd.read_csv(input_file)

# 3. Make sure output folder exists
os.makedirs(output_dir, exist_ok=True)

# 4. Split, write, and collect counts
summary = []
for sta_id, group in df.groupby('FDT_STA_ID'):
    # sanitize station ID for a filename
    safe_id = str(sta_id).strip().replace('/', '_').replace(' ', '_')
    out_path = os.path.join(output_dir, f"{safe_id}.csv")
    group.to_csv(out_path, index=False)
    count = len(group)
    summary.append((safe_id, count))
    print(f"• Wrote {count} rows to {out_path}")

# 5. Print a neat summary
print("\nRows per CSV file:")
for safe_id, count in summary:
    print(f"  {safe_id}.csv: {count} rows")
