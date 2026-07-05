# Milestone 1 — Data Collection & Preprocessing
**Project:** Stock Market Trend Analysis | **Week:** 6 | **Weight:** 7.5% of course | **Seed:** 42

> **How to use this file:** You (any AI model — Antigravity, Claude, GPT, a local 7B model) are the data scientist completing this milestone. Read every section top to bottom. Do not skip the **Justify** lines — the rubric grades process adherence, not just code. Produce every file listed in *Expected Outputs*. When done, fill in the *Self-Audit* table at the bottom and hand off to Milestone 2 using the *Hand-off* block.

---

## 0. Prerequisites

- Python 3.10+, `pandas pyarrow numpy scipy scikit-learn matplotlib seaborn statsmodels yfinance jupyter`
- Internet access on the first run (to fetch from Yahoo Finance). Subsequent runs read the snapshot CSVs in `data/raw/`.
- Empty working directory; create the repo layout below before anything else.

```
Stock-Market-Trend-Analysis/
├── data/{raw,processed}/
├── notebooks/
├── models/
├── reports/{figures,milestones}/
├── Plans/
├── References/
├── requirements.txt
└── README.md
```

## 1. Problem Definition & Business Context

Write a `reports/milestones/M1.md` opening section that answers:
- **Primary question:** "Given 10 years of daily price data for the S&P 500 (`^GSPC`) and three tech stocks (`AAPL`, `AMZN`, `NVDA`), can we forecast next-day log returns better than naive baselines?"
- **Stakeholders:** retail investors (decision context), portfolio analysts (signal generation), academic supervisors (rubric grading).
- **Why time-series:** prices exhibit a strong upward drift, volatility clustering, and regime shifts (COVID, rate hikes, AI boom). A pure tabular regression that ignored temporal order would leak the future via random splits.
- **Course-fit:** applies Weeks 4–6 (cleaning, feature engineering, encoding) and sets up Weeks 7–11 (EDA + forecasting with ARIMA + LSTM).

## 2. Data Acquisition

Run from the M1 notebook:
```python
import yfinance as yf
TICKERS = ["^GSPC", "AAPL", "AMZN", "NVDA"]
END   = pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)
raw = yf.download(TICKERS, start=START, end=END, auto_adjust=False, group_by="ticker")
```

For each ticker, save a snapshot CSV to `data/raw/<TICKER>_10y.csv`. Commit these — they are the reproducibility anchor (re-runs read from CSV, not the live API).

Record in `M1.md`: per-ticker file SHA256, row count, date range, and the licence note ("public Yahoo Finance data, no PII"). Typical expected row count: ~2,520 trading days (252 × 10), but trim with a `± 50` tolerance because of holiday calendar variation.

**Failure modes to document:**
- **HTTP 429 / rate limit:** retry once with a 5-second backoff. If still fails, abort with a clear message asking the user to re-run later.
- **`^GSPC` sparse Adj Close:** the cash index doesn't pay dividends, so `Adj Close` ≈ `Close`. State this explicitly; don't treat the equality as a bug.

## 3. Initial Inspection

In `notebooks/01_data_collection_preprocessing.ipynb`:
```python
import pandas as pd, numpy as np
SEED = 42; np.random.seed(SEED)
df = pd.read_csv("data/raw/AAPL_10y.csv", parse_dates=["Date"]).rename(columns={"Date": "date"})
df["ticker"] = "AAPL"
# ... concat all 4 tickers into one long-form DataFrame
print(df.info()); print(df.describe()); print(df.isnull().sum())
print("dupes:", df.duplicated(["date","ticker"]).sum())
print("date span:", df.date.min(), "→", df.date.max())
```

**Document** every output in `M1.md`. Confirm: 0 nulls in `Close`/`Adj Close`, 0 duplicates on `(date, ticker)`, trading-day calendar consistent across tickers (forward-fill any rare halt day per ticker — document the count).

## 4. Missing-Value Handling

Yahoo data on the S&P 500 and three large-cap tech stocks is essentially gap-free across trading days. **Check explicitly** with `df.isnull().sum()`. State the policy you would apply for any future missing day:

- **Trading-day holes (halts):** forward-fill per ticker (a halt means "last known price still holds").
- **Non-trading days (weekends / market holidays):** leave absent — there is no price to impute.

Silent assumption = grade reduction.

## 5. Outlier Detection

Compute daily simple returns `r_t = (close_t - close_{t-1}) / close_{t-1}` per ticker, then:

```python
g = df.groupby("ticker")["daily_return"]
q1, q3 = g.transform("quantile", 0.25), g.transform("quantile", 0.75)
iqr = q3 - q1
hi  = q3 + 1.5 * iqr
lo  = q1 - 1.5 * iqr
df["is_outlier_hi"]    = (df["daily_return"] > hi).astype("int8")
df["is_outlier_lo"]    = (df["daily_return"] < lo).astype("int8")
df["is_extreme_return"] = (df["daily_return"].abs() > 0.05).astype("int8")
```

**Decision (justify in M1.md):** *retain* outliers and *flag* them. Reasons:
1. Crashes (e.g. COVID March 2020) and rallies are real, predictable-after-the-fact events that the model should learn the *context* of, not be blinded to.
2. Capping returns would destroy the very tail behavior we want to characterise in M2.
3. Flags let downstream models (e.g. LightGBM if used) split on them; they preserve information.

## 6. Feature Engineering (≥ 5 groups)

Add all of the following to a single function `make_features(df, train_stats=None)` so it can be reused on val/holdout without leakage:

| Group | Columns | Notes |
|---|---|---|
| Datetime parts | `year, month, day, dayofweek, weekofyear, quarter, is_month_start, is_month_end, is_quarter_end` | `df.date.dt.*` |
| Cyclical | `month_sin, month_cos, dow_sin, dow_cos` | `sin/cos(2π·x/period)` |
| Returns | `daily_return, log_return` | `pct_change` and `log(close/close.shift(1))` |
| Lags | `close_lag_1, _5, _21, _63`, `log_return_lag_1, _5, _21` | `groupby("ticker").shift(k)`; 5 ≈ week, 21 ≈ month, 63 ≈ quarter (trading days) |
| Rolling | `sma_20, sma_50, sma_200`, `rolling_std_20` (volatility) | `.shift(1).rolling(k).mean()` — shift first to prevent leakage |
| Target encoding | `ticker_mean_log_return, ticker_std_log_return` | computed **only from rows with date ≤ train cutoff**, passed in via `train_stats` |

Write a one-paragraph **leakage warning** in `M1.md`: any feature using the target must be computed strictly from past rows. The `.shift(1).rolling()` order is critical.

## 7. Target Diagnostics

The forecasting target for M3 will be **`log_return`** (stationary), not raw `Close` (non-stationary). Justify this with the ADF test:

```python
from statsmodels.tsa.stattools import adfuller
print("ADF p-value, Close   :", adfuller(df_aapl.Close.dropna())[1])
print("ADF p-value, log_ret :", adfuller(df_aapl.log_return.dropna())[1])
```

Expected: `Close` p-value ≫ 0.05 (non-stationary, unit root), `log_return` p-value ≈ 0 (stationary). Tabulate per ticker in `M1.md`. Plot the log-return histogram → `reports/figures/M1_target_hist.png`. Note: distribution is leptokurtic (fat tails) — this is well-known in finance.

## 8. Time-Based Split (no shuffle, ever)

```
train_fe:    date ≤ 2023-12-31   (~7.5 yr)
val_fe:      2024-01-01 ≤ date ≤ 2024-12-31   (~1 yr)
holdout_fe:  date ≥ 2025-01-01   (~1.4 yr, to today)
```
Recompute lag/rolling features **only after** assigning rows to splits to keep windows valid (or compute on the full data, then split — both work as long as no future-target encoding bleeds across splits). Save as parquet:
```python
train_fe.to_parquet("data/processed/train_fe.parquet")
val_fe.to_parquet("data/processed/val_fe.parquet")
holdout_fe.to_parquet("data/processed/holdout_fe.parquet")
```

## 9. Feature Dictionary

Write `data/processed/feature_dictionary.md` with one row per column: name, dtype, description, how computed, leakage-safe (yes/no).

## Expected Outputs

- `notebooks/01_data_collection_preprocessing.ipynb` (runs top-to-bottom, seed=42)
- `data/raw/{GSPC, AAPL, AMZN, NVDA}_10y.csv` (snapshots, committed)
- `data/processed/{train_fe, val_fe, holdout_fe}.parquet`
- `data/processed/feature_dictionary.md`
- `reports/figures/M1_target_hist.png`, `M1_price_trends.png`, `M1_before_after_cleaning.png`
- `reports/milestones/M1.md` with 9 sections: Problem → Sources → Inspection → Missing → Outliers → Features → Target Diagnostics → Final Dataset → Limitations → Self-Audit

## Self-Audit Table (paste into M1.md, all must be ✅)

| Criterion | Status | Evidence |
|---|---|---|
| Problem clearly defined with business relevance | ☐ | … |
| Data source documented (rows, dates, SHA per ticker) | ☐ | … |
| Missing-value check + policy stated | ☐ | … |
| Outlier strategy justified (retain + flag) | ☐ | … |
| ≥ 5 engineered feature groups, leakage-safe | ☐ | … |
| ADF test reported for prices vs log returns | ☐ | … |
| Time-based split (no shuffle) | ☐ | … |
| Notebook reproducible (seed=42, pinned deps) | ☐ | … |

## Forbidden

- Random train/test split, K-fold shuffle.
- Computing target-derived features (e.g. ticker mean) on the full dataset before splitting.
- Dropping rows without a logged justification.
- Treating `^GSPC` like a tradeable security (it's an index — not directly investable; document this).
- Un-commented code cells.

## Hand-off to Milestone 2

> **State of project after M1:** Cleaned, leakage-safe feature dataset is at `data/processed/{train_fe,val_fe,holdout_fe}.parquet`. Target = `log_return`. ~25 engineered features documented in `feature_dictionary.md`. Tickers: ^GSPC, AAPL, AMZN, NVDA. ADF confirms `Close` non-stationary, `log_return` stationary → ARIMA will be fit on returns in M3. Next: deep EDA on `train_fe` to surface seasonality, autocorrelation, cross-ticker correlations, and inform ARIMA orders.
