#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, kendalltau

# ─── CONFIG ────────────────────────────────────────────────────────────────
input_dir    = 'station_csvs'    # folder containing your per‐station CSVs
output_plots = 'station_plots'   # where to save PNG trend plots
os.makedirs(output_plots, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────

def analyze_station(csv_path):
    station_id = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)

    # 1) Parse date strings like "6/30/94 11:00"
    df['Date'] = (
        pd.to_datetime(
            df['FDT_DATE_TIME'],
            format='%m/%d/%y %H:%M',
            errors='coerce'
        )
        .dt.normalize()
    )
    df = df.dropna(subset=['Date', 'FDT_DEPTH', 'FDT_FIELD_PH']).copy()
    if df.empty:
        print(f"Station {station_id}: no valid records, skipping.")
        return

    # 2) Depth classification & raw sample counts
    df['Depth_cat'] = np.where(df['FDT_DEPTH'] <= 1, '0-1 m', '>1 m')
    raw_counts = df['Depth_cat'].value_counts().to_dict()

    # 3) Calendar‐month medians (month‐end)
    monthly = (
        df
        .groupby([
            'Depth_cat',
            pd.Grouper(key='Date', freq='ME')
        ])['FDT_FIELD_PH']
        .median()
        .reset_index()
    )

    results = {}
    month_counts = {}
    for cat in ['0-1 m', '>1 m']:
        sub = monthly[monthly['Depth_cat'] == cat]
        if sub.empty:
            continue

        month_counts[cat] = len(sub)

        # build continuous monthly series & interpolate gaps
        s = sub.set_index('Date')['FDT_FIELD_PH']
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq='ME')
        s = s.reindex(full_idx).interpolate(method='time')

        # Theil–Sen slope
        x = s.index.map(pd.Timestamp.toordinal).values
        y = s.values
        slope, intercept, low, high = theilslopes(y, x, 0.95)
        slope_yr = slope * 365
        ci_lower = low * 365
        ci_upper = high * 365

        # Mann–Kendall test
        tau, p_val = kendalltau(np.arange(len(y)), y)

        results[cat] = {
            'slope_yr': slope_yr,
            'ci': (ci_lower, ci_upper),
            'tau': tau,
            'p_val': p_val,
            'series': s,
            'intercept': intercept,
            'slope': slope
        }

    # 4) Print summary
    print(f"\n=== Station {station_id} (Median monthly pH) ===")
    print("Raw samples by depth band:")
    for cat in ['0-1 m', '>1 m']:
        print(f"  {cat}: {raw_counts.get(cat, 0)} rows")
    print("Months of data (pre‐interpolation):")
    for cat, mc in month_counts.items():
        print(f"  {cat}: {mc} months")
    print("Trend results (Theil–Sen + Mann–Kendall):")
    for cat, r in results.items():
        print(
            f"  {cat} → Sen’s slope = {r['slope_yr']:.4f} pH/yr "
            f"(95% CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]), "
            f"τ = {r['tau']:.3f}, p = {r['p_val']:.4g}"
        )

    # 5) Plot, show, and save
    plt.figure(figsize=(10, 4))
    for cat, r in results.items():
        s = r['series']
        trend = r['intercept'] + r['slope'] * s.index.map(pd.Timestamp.toordinal)
        plt.plot(s.index, s, label=f"{cat} monthly median pH")
        plt.plot(s.index, trend, '--', label=f"{cat} Sen’s trend")
    plt.title(f"pH Trend — Station {station_id} (Median)")
    plt.xlabel("Date")
    plt.ylabel("pH")
    plt.legend()
    plt.tight_layout()

    # display interactively
    plt.show()

    # save to disk
    out_png = os.path.join(output_plots, f"{station_id}_median_trend.png")
    plt.savefig(out_png, dpi=200)
    plt.close()

# ─── MAIN LOOP ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for csv_file in glob.glob(os.path.join(input_dir, '*.csv')):
        analyze_station(csv_file)
