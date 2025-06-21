import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, kendalltau

# 1. Load and parse data
df = pd.read_excel('./station1.xlsx')
# Convert Excel serial date to datetime; adjust origin if needed
df['Date'] = pd.to_datetime(df['FDT_DATE_TIME'], unit='D', origin='1899-12-30')

# 2. Clean and categorize by fixed depth thresholds (0–1 m vs >1 m)
df_clean = df.dropna(subset=['FDT_DEPTH', 'FDT_FIELD_PH']).copy()
df_clean['Depth_cat'] = np.where(df_clean['FDT_DEPTH'] <= 1, '0-1 m', '>1 m')

# 3. Compute monthly mean pH for each depth category
df_clean.set_index('Date', inplace=True)
monthly = (
    df_clean
    .groupby('Depth_cat')['FDT_FIELD_PH']
    .resample('M')
    .mean()
    .reset_index()
)

# 4. Interpolate missing months and compute trends
results = {}
for cat in ['0-1 m', '>1 m']:
    # a) Build continuous monthly series
    series = monthly[monthly['Depth_cat'] == cat].set_index('Date')['FDT_FIELD_PH']
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq='M')
    series = series.reindex(full_idx).interpolate(method='time')
    
    # b) Theil–Sen slope (pH change per year)
    x = series.index.map(pd.Timestamp.toordinal).values
    y = series.values
    slope, intercept, lower, upper = theilslopes(y, x, 0.95)
    slope_year = slope * 365
    lower_year = lower * 365
    upper_year = upper * 365
    
    # c) Mann–Kendall test for significance
    tau, p_value = kendalltau(np.arange(len(y)), y)
    
    results[cat] = {
        'slope_year': slope_year,
        'ci_lower': lower_year,
        'ci_upper': upper_year,
        'tau': tau,
        'p_value': p_value,
        'series': series,
        'intercept': intercept,
        'slope': slope
    }

# 5. Print summary
for cat, res in results.items():
    print(f"{cat}:\n"
          f"  Sen’s slope: {res['slope_year']:.4f} pH/yr "
          f"(95% CI [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}])\n"
          f"  Kendall’s tau: {res['tau']:.3f}, p-value = {res['p_value']:.4g}\n")

# 6. Plot monthly series with Sen’s trend line
plt.figure(figsize=(10, 6))
for cat, res in results.items():
    series = res['series']
    trend = res['intercept'] + res['slope'] * series.index.map(pd.Timestamp.toordinal).values
    plt.plot(series.index, series, label=f'{cat} monthly pH')
    plt.plot(series.index, trend, linewidth=2, label=f'{cat} Sen’s trend')
plt.title('Monthly Mean pH and Theil–Sen Trend by Depth Category')
plt.xlabel('Date')
plt.ylabel('pH')
plt.legend()
plt.tight_layout()
plt.show()
