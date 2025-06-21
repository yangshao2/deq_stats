#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, kendalltau

def main(input_dir, variable_col, plot_folder):
    os.makedirs(plot_folder, exist_ok=True)
    summary_csv = f"{variable_col}_summary.csv"
    summary = []

    for csv_path in glob.glob(os.path.join(input_dir, '*.csv')):
        station_id = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)

        # Parse dates (e.g. "6/30/94 11:00")
        df['Date'] = (pd.to_datetime(
                          df['FDT_DATE_TIME'],
                          format='%m/%d/%y %H:%M',
                          errors='coerce')
                      .dt.normalize())
        df = df.dropna(subset=['Date', 'FDT_DEPTH', variable_col])
        if df.empty:
            print(f"[{station_id}] no valid {variable_col} data, skipping.")
            continue

        # Depth bands
        df['Depth_band'] = np.where(df['FDT_DEPTH'] <= 1, '0-1 m', '>1 m')

        # Monthly means (month-end)
        monthly = (df
                   .groupby(['Depth_band', pd.Grouper(key='Date', freq='ME')])[variable_col]
                   .mean()
                   .reset_index())

        # Plot setup
        plt.figure(figsize=(10,5))
        ax = plt.gca()

        for band in ['0-1 m', '>1 m']:
            sub = monthly[monthly['Depth_band'] == band]
            if sub.empty:
                continue

            # months of raw data
            months = len(sub)
            # starting and ending year
            start_year = sub['Date'].min().year
            end_year   = sub['Date'].max().year

            # build & interpolate continuous series
            s = sub.set_index('Date')[variable_col]
            idx = pd.date_range(s.index.min(), s.index.max(), freq='ME')
            s = s.reindex(idx).interpolate(method='time')

            # Theil–Sen
            x = s.index.map(pd.Timestamp.toordinal).values
            y = s.values
            slope, intercept, low, high = theilslopes(y, x, 0.95)
            slope_yr = slope * 365

            # Mann–Kendall
            tau, pval = kendalltau(np.arange(len(y)), y)
            signif = 'sig' if pval < 0.05 else 'ns'

            # record summary
            summary.append({
                'station_id': station_id,
                'depth_band': band,
                'start_year': start_year,
                'end_year': end_year,
                'months': months,
                f'{variable_col}_sen_slope_per_year': slope_yr,
                f'{variable_col}_kendall_sig': signif
            })

            # plot series + trend
            ax.plot(s.index, s, label=f"{band} monthly mean")
            ax.plot(s.index,
                    intercept + slope * x,
                    '--',
                    label=f"{band} trend")

            # annotate slope & signif
            x0 = s.index[0]
            y0 = ax.get_ylim()[1] - 0.05*(ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(x0, y0 - 0.05*(ax.get_ylim()[1] - ax.get_ylim()[0]) * ['0-1 m','>1 m'].index(band),
                    f"{band}: {slope_yr:.3f} ({signif})",
                    va='top')

        # finalize plot
        ax.set_title(f"{variable_col} Trend — {station_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel(variable_col)
        ax.legend()
        plt.tight_layout()

        # save & show
        out_png = os.path.join(plot_folder, f"{station_id}_trend.png")
        plt.savefig(out_png, dpi=200)
        plt.show()
        plt.close()

    # write summary CSV
    pd.DataFrame(summary).to_csv(summary_csv, index=False)
    print(f"Summary written to {summary_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trend analysis for a given variable across stations")
    parser.add_argument(
        '-i', '--input_dir', default='station_csvs',
        help='Folder with station CSV files')
    parser.add_argument(
        '-v', '--variable', required=True,
        help='Column name to analyze (e.g., FDT_FIELD_PH, FDT_TEMP_CELCIUS)')
    parser.add_argument(
        '-p', '--plot_folder', default=None,
        help='Output folder for plots (defaults to variable name)')
    args = parser.parse_args()

    plot_folder = args.plot_folder or args.variable
    main(args.input_dir, args.variable, plot_folder)
