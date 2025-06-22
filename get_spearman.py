#!/usr/bin/env python3
import os
import glob
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────────────────────
input_dir  = 'station_csvs'   # folder containing your per‐station CSVs
output_dir = 'spearman_corr'  # folder to save correlation matrices
os.makedirs(output_dir, exist_ok=True)

# List of variables to include in the correlation
variables = [
    'FDT_FIELD_PH','FDT_TEMP_CELCIUS','DO_mg_L','NITROGEN_mg_L',
    'AMMONIA_mg_L','NOX_mg_L','NITROGEN_KJELDAHL_TOTAL_00625_mg_L',
    'PHOSPHORUS_TOTAL_00665_mg_L','PHOSPHORUS_TOTAL_ORTHOPHOSPHATE_70507_mg_L',
    'HARDNESS_TOTAL_00900_mg_L','CHLORIDE_mg_L','SULFATE_mg_L',
    'ECOLI','FECAL_COLI','CHLOROPHYLL_A_ug_L',
    'TSS_mg_L','SECCHI_DEPTH_M','NOx_TKN_Sum'
]

# Process each station file
for csv_path in glob.glob(os.path.join(input_dir, '*.csv')):
    print(csv_path)
    station_id = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)

    # Determine which variables exist
    existing = [v for v in variables if v in df.columns]
    missing  = set(variables) - set(existing)
    if missing:
        print(f"[{station_id}] Skipping missing vars: {sorted(missing)}")

    # Compute Spearman correlation on pairwise complete observations
    corr = df[existing].corr(method='spearman', numeric_only=True)

    # Save to CSV
    out_path = os.path.join(output_dir, f"{station_id}_spearman.csv")
    corr.to_csv(out_path)
    print(f"[{station_id}] Spearman correlation saved to {out_path}")
