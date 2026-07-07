# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Milestone 1 — Data Collection & Preprocessing
# **Project:** Stock Market Trend Analysis | **Tickers:** ^GSPC, AAPL, AMZN, NVDA | **Window:** 10 years (pinned) | **Seed:** 42
#
# ## What this notebook does
#
# This is the **first** step of the capstone. By the end of this notebook you will have:
#
# 1. Fetched **10 years** of daily price data from Yahoo Finance for **four tickers** (S&P 500 + AAPL + AMZN + NVDA).
# 2. Saved a **CSV snapshot** per ticker so future runs are reproducible (no network needed).
# 3. Concatenated everything into one **long-form** DataFrame with a fixed, enforced column schema.
# 4. Run statistical tests (**ADF** for the mean, **Ljung-Box on squared returns** for the variance) to justify
#    modelling **log returns** and to flag volatility clustering for M3.
# 5. Engineered several feature groups (datetime, cyclical, returns, **stationary momentum / volatility**, lags,
#    rolling windows, expanding target-encoding) — all leakage-safe.
# 6. Built a proper **next-day forecasting target** (`target_log_return`).
# 7. Split the data **chronologically** (never randomly!) into train / val / holdout.
# 8. Saved three **parquet** files + a **feature dictionary** + a **NaN warm-up report** ready for EDA (M2) and
#    modelling (M3).
#
# `np.random` is seeded to `42`. (M1 itself does no random sampling — the seed matters in M3, where the LSTM
# must additionally call `tf.random.set_seed(42)` / `random.seed(42)` / set `PYTHONHASHSEED`.)
#
# ## Why time-series data is special
#
# In a normal supervised-learning problem (e.g. predicting house prices from features), you can shuffle the rows
# freely and split them randomly into train and test. In time-series, **you cannot do that** — the rows have
# an inherent order, and shuffling lets the model peek at the future. We will split by date.
#
# ---
# ### Revision history
# - **v1** — initial M1.
# - **v2 (single-pass audit)** — Adj Close everywhere, side-by-side kurtosis, expanding-mean target encoding,
#   rolling-MAD outlier flag.
# - **v3 (multi-agent audit — this version)** — pinned date window; enforced schema (Volume dtype);
#   **added a real next-day target** (`target_log_return`); explicit `MODEL_FEATURES` role separation so the
#   same-day return/price columns can't leak; warm-up NaN report + `is_warmup` flag; `tail_event` flag NaN on
#   warm-up rows; Ljung-Box(r²) heteroskedasticity test + corrected "stationary" wording; stationary
#   momentum / realized-vol / Parkinson / RSI / volume-z features; safer forward-fill; Colab bootstrap +
#   requirements install; partial-ticker-failure handling.

# %% [markdown]
# ## 0a. (Colab only) Bootstrap — get the project files onto the runtime
#
# A fresh Colab session is an **empty machine**. Before anything else, the project folder (including the
# committed `data/raw/*.csv` snapshots) must exist at `/content/Stock-Market-Trend-Analysis`. Pick ONE:
#
# - **Option A — clone from GitHub** (if you pushed the repo):
#   ```python
#   !git clone https://github.com/<you>/<repo>.git /content/Stock-Market-Trend-Analysis
#   ```
# - **Option B — mount Google Drive** (if the folder lives in your Drive):
#   ```python
#   from google.colab import drive; drive.mount('/content/drive')
#   !cp -r "/content/drive/MyDrive/Stock-Market-Trend-Analysis" /content/
#   ```
# - **Option C — upload the 4 snapshot CSVs** manually into `/content/Stock-Market-Trend-Analysis/data/raw/`.
#
# If none of these is done, the notebook will fall back to a **live** Yahoo fetch of the pinned window
# (works only if Yahoo is reachable from Colab). The cell below is a no-op outside Colab.

# %%
import sys
import subprocess

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    # Install pinned dependencies so Colab matches the local environment as closely as possible.
    # (Colab ships its own numpy/pandas; this aligns the analysis stack. Restart runtime if prompted.)
    from pathlib import Path as _P

    _req = _P("/content/Stock-Market-Trend-Analysis/requirements.txt")
    if _req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(_req)], check=False)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yfinance"], check=False)

# %% [markdown]
# ## 0b. Setup — imports, paths, seed

# %%
# Install yfinance silently if it's missing (covers a local kernel without it).
try:
    import yfinance  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "yfinance"])

# %%
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted")

# Resolve project root (IN_COLAB already set in the bootstrap cell).
if IN_COLAB:
    ROOT = Path("/content/Stock-Market-Trend-Analysis")
elif "__file__" in globals():
    ROOT = Path(__file__).resolve().parent.parent
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
MILE = ROOT / "reports" / "milestones"
for d in (RAW, PROC, FIG, MILE):
    d.mkdir(parents=True, exist_ok=True)

print("Project root :", ROOT)
print("IN_COLAB     :", IN_COLAB)
print("Raw dir      :", RAW)
print("Processed dir:", PROC)

# %% [markdown]
# ## 1. Problem definition & business context
#
# **Primary question:**
# > Given 10 years of daily price data for the S&P 500 (`^GSPC`) and three tech stocks
# > (`AAPL`, `AMZN`, `NVDA`), can we forecast the **next-day** log return better than naive baselines?
#
# The phrase *next-day* is load-bearing: the model sees information up to and including the close of day `t`
# and must predict the return of day `t+1`. The target column we build for that is `target_log_return`
# (= `log_return` shifted **one day into the past** so tomorrow's return sits on today's feature row).
#
# **Stakeholders:** retail investors, portfolio analysts, academic supervisors.
#
# **Why time-series methods:** prices have drift, volatility clustering, and regime shifts (COVID 2020,
# 2022 rate hikes, 2023 AI boom). A random split would leak the future into training.
#
# **Disclaimer:** educational project only. NOT investment advice. `^GSPC` is an index (not directly
# tradeable — you'd use `SPY`), and its series is a price return while the single stocks' `Adj Close` are
# total returns (dividend-adjusted). Immaterial for per-ticker forecasting; relevant only if returns are
# pooled or compared cross-ticker.

# %% [markdown]
# ## 2. Data acquisition
#
# We use `yfinance` to download 10 years of daily OHLCV data per ticker, with a **direct Yahoo chart-API
# fallback** for networks that block `fc.yahoo.com` (yfinance's consent endpoint).
#
# **The date window is PINNED** (`START`/`END` constants below), not anchored to "today". This is essential
# for reproducibility: a run on any date — or a fresh Colab/clone with no cached CSVs — fetches the *same*
# window and produces the *same* splits. The committed snapshot CSVs are the primary anchor; the pinned
# window is the backup anchor when the cache is absent.
#
# We use **`Adj Close`** for every return / feature: raw `Close` jumps artificially on stock-split days
# (NVDA's 10-for-1 split on 2024-06-10 would otherwise look like a 90% crash).

# %%
TICKERS = ["^GSPC", "AAPL", "AMZN", "NVDA"]
# Exogenous market-wide series (same value for every ticker on a given date), fetched the same way.
# These are the only break from price-history-only data (M4 limitation #3). All daily + market-priced —
# values are known within minutes of the equity close (VIX settles ~4:15pm ET, FX/bonds trade later on their
# own calendars), which is immaterial when predicting the NEXT day's return. So no meaningful release-lag
# lookahead (unlike monthly CPI, which has a multi-week reporting lag). Free + reproducible via the pipeline.
EXOG_TICKERS = {
    "^VIX": "vix",        # CBOE volatility index = market fear / sentiment + regime proxy
    "^TNX": "tnx",        # 10-year Treasury yield
    "^IRX": "irx",        # 13-week T-bill yield (short rate)
    "DX-Y.NYB": "dxy",    # US dollar index
}
START = pd.Timestamp("2016-05-23")   # pinned — matches committed snapshot CSVs
END = pd.Timestamp("2026-05-21")     # pinned — exclusive upper bound; snapshots end 2026-05-20
MIN_ROWS = 2000                      # ~8 yr of trading days; a healthy 10-yr pull is ~2,500

print(f"Pinned date range: {START.date()} -> {END.date()} (exclusive)")
print(f"Tickers          : {TICKERS}")
print(f"Exogenous        : {list(EXOG_TICKERS)}")

# Canonical column schema every fetch path must conform to (prevents int/float dtype drift).
SCHEMA = {
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Adj Close": "float64",
    "Volume": "float64",  # float, not int: halts produce NaN volume; one stable dtype across paths
}


def safe_filename(ticker: str) -> str:
    """`^GSPC` has a caret that makes a messy filename; strip it."""
    return ticker.lstrip("^")


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a freshly-fetched frame to the canonical dtypes so all fetch paths are identical."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    for col, dtype in SCHEMA.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
    keep = ["Date"] + [c for c in SCHEMA if c in df.columns]
    return df[keep]


def _fetch_yfinance(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Primary path: yfinance. Works on Colab and most networks."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError("yfinance returned an empty DataFrame")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return enforce_schema(df.reset_index())


def _fetch_yahoo_direct(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fallback: Yahoo public chart API (bypasses the blocked consent endpoint)."""
    period1, period2 = int(start.timestamp()), int(end.timestamp())
    enc = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
    chart = data.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo API error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo API returned no results")
    result = results[0]
    ts = result.get("timestamp") or []
    if not ts:
        raise RuntimeError("Yahoo API returned no timestamps")
    quote = result["indicators"]["quote"][0]
    adjclose_block = result["indicators"].get("adjclose") or [{"adjclose": [None] * len(ts)}]
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(ts, unit="s"),
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Adj Close": adjclose_block[0]["adjclose"],
            "Volume": quote["volume"],
        }
    )
    return enforce_schema(df)


def fetch_with_snapshot(ticker, start, end, out_dir, retries=2, backoff_sec=5.0):
    """
    Download one ticker (yfinance → direct-API fallback), enforce schema, snapshot to CSV.
    Cached CSV (if present) is read and re-coerced to the schema so cache/network paths match.
    """
    out_path = out_dir / f"{safe_filename(ticker)}_10y.csv"
    if out_path.exists():
        print(f"  [cached] {ticker}: reading {out_path.name}")
        return enforce_schema(pd.read_csv(out_path, parse_dates=["Date"]))

    last_err = None
    for attempt in range(retries):
        try:
            print(f"  [yf    ] {ticker}: attempt {attempt + 1}/{retries}")
            df = _fetch_yfinance(ticker, start, end)
            break
        except Exception as e:
            last_err = e
            print(f"  [warn  ] yfinance failed: {e}")
            if attempt < retries - 1:
                time.sleep(backoff_sec)
    else:
        print(f"  [direct] {ticker}: falling back to Yahoo chart API")
        try:
            df = _fetch_yahoo_direct(ticker, start, end)
        except Exception as e:
            raise RuntimeError(
                f"Both yfinance and direct Yahoo API failed for {ticker}. "
                f"yfinance error: {last_err}. Direct API error: {e}"
            )
    df.to_csv(out_path, index=False)
    print(f"  [saved ] {ticker}: {len(df):>5d} rows -> {out_path.name}")
    return df


# %%
# Fetch all tickers with explicit partial-failure handling: collect successes/failures,
# then validate row counts. One bad ticker reports clearly instead of aborting opaquely.
per_ticker, failures = {}, {}
for t in TICKERS:
    try:
        df_t = fetch_with_snapshot(t, START, END, RAW)
        if len(df_t) < MIN_ROWS:
            raise RuntimeError(f"only {len(df_t)} rows (< MIN_ROWS={MIN_ROWS}); suspect partial download")
        per_ticker[t] = df_t
    except Exception as e:
        failures[t] = str(e)
        print(f"  [FAIL  ] {t}: {e}")

if failures:
    raise RuntimeError(
        "Data acquisition failed for: "
        + ", ".join(f"{k} ({v})" for k, v in failures.items())
        + f". Succeeded: {sorted(per_ticker)}."
    )

for t, df_t in per_ticker.items():
    print(f"{t:6s}  rows={len(df_t):>5d}  dates {df_t.Date.min().date()} -> {df_t.Date.max().date()}")

# %% [markdown]
# ### Exogenous series (VIX, yields, dollar)
#
# Fetched with the same snapshot+fallback machinery. We keep each one's `Adj Close` (for these indices,
# `Adj Close` == `Close`), build a date-indexed wide frame, forward-fill any calendar gaps, and merge it onto
# the long-form stock frame by date. Each ticker row on date D therefore sees the same market-wide values for D
# — all known at the close of D, so they are valid predictors of D+1.

# %%
exog_frames, exog_failures = {}, {}
for sym, name in EXOG_TICKERS.items():
    try:
        e = fetch_with_snapshot(sym, START, END, RAW)
        exog_frames[name] = e.rename(columns={"Date": "date"})[["date", "Adj Close"]].rename(
            columns={"Adj Close": name}
        )
    except Exception as ex:
        exog_failures[sym] = str(ex)
        print(f"  [FAIL  ] {sym}: {ex}")

if exog_failures:
    raise RuntimeError(f"Exogenous acquisition failed: {exog_failures}")

# Merge all exogenous into one date-indexed frame; ffill calendar gaps (markets occasionally differ by holiday).
exog_wide = None
for name, f in exog_frames.items():
    exog_wide = f if exog_wide is None else exog_wide.merge(f, on="date", how="outer")
exog_wide = exog_wide.sort_values("date").reset_index(drop=True)
exog_wide[list(EXOG_TICKERS.values())] = exog_wide[list(EXOG_TICKERS.values())].ffill()
print("exog_wide:", exog_wide.shape, "cols:", list(exog_wide.columns))
print(exog_wide.tail(3).to_string(index=False))

# %% [markdown]
# ### File hashes (for the M1.md report)

# %%
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


snapshot_hashes = {
    "_window": {"start": str(START.date()), "end_exclusive": str(END.date())},
}
for t in TICKERS:
    p = RAW / f"{safe_filename(t)}_10y.csv"
    snapshot_hashes[t] = {"sha256": sha256(p), "rows": int(len(per_ticker[t]))}
    print(f"{t:6s}  SHA256: {snapshot_hashes[t]['sha256']}")
for sym in EXOG_TICKERS:
    p = RAW / f"{safe_filename(sym)}_10y.csv"
    snapshot_hashes[sym] = {"sha256": sha256(p), "rows": int(sum(1 for _ in open(p))) - 1}
    print(f"{sym:10s}  SHA256: {snapshot_hashes[sym]['sha256']}")
(RAW / "snapshot_hashes.json").write_text(json.dumps(snapshot_hashes, indent=2))

# %% [markdown]
# ## 3. Concatenate to long-form

# %%
frames = []
for t, raw_df in per_ticker.items():
    frame = raw_df.rename(columns={"Date": "date"}).copy()
    frame["ticker"] = t
    cols = ["date", "ticker"] + [c for c in frame.columns if c not in ("date", "ticker")]
    frames.append(frame[cols])

df = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

# Merge the market-wide exogenous values by date (left join — each ticker row gets that date's macro values).
df = df.merge(exog_wide, on="date", how="left")
df[list(EXOG_TICKERS.values())] = df.groupby("ticker")[list(EXOG_TICKERS.values())].ffill()

# Schema assertion — fail loudly if any fetch path drifted the dtypes.
assert df["Volume"].dtype == np.float64, f"Volume dtype drift: {df['Volume'].dtype}"
assert str(df["date"].dtype) == "datetime64[ns]", f"date dtype drift: {df['date'].dtype}"
print("long-form shape:", df.shape, "| dtypes OK | exog merged:", list(EXOG_TICKERS.values()))
print(df.head())

# %% [markdown]
# ## 4. Initial inspection

# %%
print("--- df.info() ---")
df.info()

# %%
print("--- df.describe() ---")
df.describe()

# %%
print("nulls per column:")
print(df.isnull().sum())
print("duplicate (date, ticker) rows:", df.duplicated(["date", "ticker"]).sum())
print("\nper-ticker span:")
print(df.groupby("ticker").agg(rows=("date", "count"), first=("date", "min"), last=("date", "max")))

# %% [markdown]
# ## 5. Missing-value handling
#
# Yahoo data on these four highly-liquid tickers is gap-free across trading days (verified above). We still
# document and apply a defensive policy:
#
# - **Price columns** (`Open, High, Low, Close, Adj Close`) → forward-fill within the ticker. A halt means
#   "the last price still holds."
# - **`Volume`** → fill missing with **0**, NOT forward-fill. Zero volume is the correct meaning of a
#   no-trade day; carrying yesterday's volume forward would fabricate activity.
# - Non-trading days (weekends/holidays) → left absent (no row exists; nothing to impute).
#
# We count how many cells are actually filled so silent imputation can't hide.

# %%
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
price_cols = ["Open", "High", "Low", "Close", "Adj Close"]

n_price_missing = int(df[price_cols].isna().sum().sum())
n_vol_missing = int(df["Volume"].isna().sum())

df[price_cols] = df.groupby("ticker")[price_cols].ffill()
gspc = df.ticker == "^GSPC"
df.loc[gspc, "Adj Close"] = df.loc[gspc, "Adj Close"].fillna(df.loc[gspc, "Close"])
df["Volume"] = df["Volume"].fillna(0.0)

print(f"price cells forward-filled: {n_price_missing}")
print(f"volume cells set to 0     : {n_vol_missing}")
print("remaining nulls:\n", df.isnull().sum())

# %% [markdown]
# ## 6. Returns + tail-event detection (rolling MAD)
#
# **Returns from `Adj Close`:**
# - `daily_return = (AdjClose_t - AdjClose_{t-1}) / AdjClose_{t-1}` — simple percent change.
# - `log_return = ln(AdjClose_t / AdjClose_{t-1})` — additive across time, the finance standard.
#
# **Naming note (audit correction):** what we flag below are **tail events**, not data errors. For daily
# equity returns ~6% of observations exceed a 3.5·MAD band — that *is* the fat tail of a leptokurtic
# process, an intrinsic property we must model, not noise to remove. We **retain** every row and use the
# flag only as an informative feature. (A genuine data-error screen — zero/negative prices, >50% one-day
# moves not matching a known split — is a separate, rarer check; none triggered here.)
#
# **Rolling MAD** (60-day window) adapts to the local volatility regime: a 5% move in calm 2017 is extreme;
# the same move in March 2020 is barely above the local median.

# %%
df["daily_return"] = df.groupby("ticker")["Adj Close"].pct_change()
df["log_return"] = np.log(df["Adj Close"] / df.groupby("ticker")["Adj Close"].shift(1))

# NVDA split sanity check (2024-06-10 10-for-1): adjusted returns must NOT show a ~90% crash.
nvda_split = df[(df.ticker == "NVDA") & (df.date == pd.Timestamp("2024-06-10"))]
if not nvda_split.empty:
    r = float(nvda_split["daily_return"].iloc[0])
    print(f"NVDA 2024-06-10 daily_return = {r:+.4f}  ({'OK split handled' if abs(r) < 0.20 else 'FAIL'})")


def _rolling_mad(x: pd.Series, window: int) -> pd.Series:
    med = x.rolling(window, min_periods=20).median()
    return (x - med).abs().rolling(window, min_periods=20).median()


WINDOW = 60
df["roll_median_60"] = df.groupby("ticker")["log_return"].transform(
    lambda s: s.rolling(WINDOW, min_periods=20).median()
)
df["roll_mad_60"] = df.groupby("ticker")["log_return"].transform(lambda s: _rolling_mad(s, WINDOW))
df["mad_z_score"] = (df["log_return"] - df["roll_median_60"]) / df["roll_mad_60"]

# tail-event flag: NaN where the MAD window hasn't warmed up (status genuinely unknown), else 0/1.
df["is_tail_event"] = np.where(df["mad_z_score"].notna(), (df["mad_z_score"].abs() > 3.5).astype(float), np.nan)
# Interpretable reference flag (NOT the primary definition): |daily move| > 5%.
df["is_extreme_5pct"] = (df["daily_return"].abs() > 0.05).astype("int8")

defined = df[df["is_tail_event"].notna()]
print("tail-event flag rate per ticker (rolling-MAD z>3.5, warm-up rows excluded):")
print((defined.groupby("ticker")["is_tail_event"].mean() * 100).round(2).to_string())
print("\n|daily move|>5% reference counts:")
print(df.groupby("ticker")["is_extreme_5pct"].sum().to_string())

# %% [markdown]
# ## 7. Target diagnostics — stationarity (ADF) AND volatility clustering (Ljung-Box on r²)
#
# **ADF** tests only the *mean* (unit root). Stock prices fail it (trending, non-stationary); log returns
# pass it (mean-stationary). But ADF says nothing about the *variance*. Financial returns are famously
# **conditionally heteroskedastic** — volatility clusters (calm follows calm, turbulent follows turbulent).
# We test that explicitly with **Ljung-Box on squared returns**: a tiny p-value means today's squared return
# predicts tomorrow's → an ARCH/GARCH signature.
#
# **Bottom line for M3:** returns are *stationary in mean* (so ARIMA can fit them with `d=0`) but
# *conditionally heteroskedastic in variance* (so plain ARIMA's constant-variance Gaussian errors are
# misspecified — motivates ARIMA+GARCH, or at least variance-aware interval forecasts).

# %%
print(f"{'ticker':<8}{'ADF Close p':>14}{'ADF logret p':>16}{'LB(r^2) lag1 p':>18}{'ACF(r^2) lag1':>16}")
diag = {}
for t in TICKERS:
    sub = df[df.ticker == t].set_index("date")
    r = sub["log_return"].dropna()
    p_close = adfuller(sub["Adj Close"].dropna())[1]
    p_logret = adfuller(r)[1]
    lb_p = float(acorr_ljungbox(r**2, lags=[1], return_df=True)["lb_pvalue"].iloc[0])
    acf1 = float(pd.Series(r.values).autocorr(lag=1))            # ACF of returns lag1 (mean signal)
    acf1_sq = float(pd.Series((r**2).values).autocorr(lag=1))     # ACF of squared returns lag1 (vol cluster)
    diag[t] = dict(adf_close=p_close, adf_logret=p_logret, lb_sq_p=lb_p, acf1_ret=acf1, acf1_sq=acf1_sq)
    print(f"{t:<8}{p_close:>14.4f}{p_logret:>16.4f}{lb_p:>18.4f}{acf1_sq:>16.3f}")

print("\nInterpretation: Close non-stationary (p>>0.05); log_return stationary in mean (p~0);")
print("LB(r^2) p~0 + positive ACF(r^2) => volatility clustering (heteroskedastic) => flag GARCH for M3.")

# %% [markdown]
# ## 8. Feature engineering
#
# **Leakage rule:** features use information available *through the close of day `t`*. The forecasting target
# (built in section 9) is day `t+1`'s return, so every same-day column (`log_return`, `daily_return`, OHLC,
# tail flags) is a *legitimate predictor* — it is known when the day-`t` decision is made. Rolling/expanding
# features additionally `.shift(1)` (use through `t-1`), which is simply more conservative.
#
# Feature groups:
#
# | Group | Columns | Stationary? | Notes |
# |---|---|---|---|
# | Datetime | `year, month, day, dayofweek, weekofyear, quarter, is_month_start/end, is_quarter_end` | — | calendar |
# | Cyclical | `month_sin/cos, dow_sin/cos` | — | smooth month encoding (period 12). dow uses period 5 as a smooth weekday encoder — NOT a true wrap (no weekend), `dayofweek` is also kept raw |
# | Returns | `daily_return, log_return` | yes | known at `t` |
# | Momentum (stationary) | `momentum_5/21/63` = `ln(C_t / C_{t-k})` | yes | scale-free, replaces raw price levels for ARIMA |
# | Realized vol | `realized_vol_5/21/63`, `rolling_std_20`, `parkinson_vol_21` | yes | the predictable quantity given clustering; Parkinson uses High/Low |
# | Momentum osc. | `rsi_14` | yes | bounded 0-100 |
# | Volume | `volume_z_20` | yes | volume z-score vs 20-day window |
# | Price levels (RAW) | `close_lag_1/5/21/63, sma_20/50/200` | **NO** | non-stationary; kept for reference/trees, EXCLUDED from default `MODEL_FEATURES` |
# | Price-relative | `price_to_sma_20/50/200` = `C_t/sma_k - 1` | yes | stationary substitute for the raw SMAs |
# | Target encoding | `ticker_expanding_mean/std` (of `log_return`, shifted 1) | yes | per-ticker, strictly past-only |

# %%
def make_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["year"] = d.dt.year.astype("int16")
    df["month"] = d.dt.month.astype("int8")
    df["day"] = d.dt.day.astype("int8")
    df["dayofweek"] = d.dt.dayofweek.astype("int8")
    df["weekofyear"] = d.dt.isocalendar().week.astype("int16")
    df["quarter"] = d.dt.quarter.astype("int8")
    df["is_month_start"] = d.dt.is_month_start.astype("int8")
    df["is_month_end"] = d.dt.is_month_end.astype("int8")
    df["is_quarter_end"] = d.dt.is_quarter_end.astype("int8")
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Smooth weekday encoder (period 5). Not a real cyclic wrap — there is no weekend trading.
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 5)
    return df


def make_price_and_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    g_close = df.groupby("ticker")["Adj Close"]
    g_logret = df.groupby("ticker")["log_return"]

    # Raw price lags (NON-stationary — reference only)
    for k in (1, 5, 21, 63):
        df[f"close_lag_{k}"] = g_close.shift(k)
    # Lagged returns (stationary)
    for k in (1, 5, 21):
        df[f"log_return_lag_{k}"] = g_logret.shift(k)

    # Stationary momentum = log price ratio over k trading days
    for k in (5, 21, 63):
        df[f"momentum_{k}"] = np.log(df["Adj Close"] / g_close.shift(k))

    # SMAs (raw, non-stationary, reference) — shift(1) to avoid using day t in day t's average
    for k in (20, 50, 200):
        df[f"sma_{k}"] = g_close.shift(1).rolling(k).mean().reset_index(level=0, drop=True)
        # Stationary price-relative version
        df[f"price_to_sma_{k}"] = df["Adj Close"] / df[f"sma_{k}"] - 1.0

    # Realized volatility (rolling std of log returns) — the clustering signal
    for k in (5, 21, 63):
        df[f"realized_vol_{k}"] = (
            g_logret.shift(1).rolling(k).std().reset_index(level=0, drop=True)
        )
    df["rolling_std_20"] = g_logret.shift(1).rolling(20).std().reset_index(level=0, drop=True)

    # Parkinson volatility (range-based, uses High/Low) over 21 days
    hl = (np.log(df["High"] / df["Low"]) ** 2)
    park = hl.groupby(df["ticker"]).transform(
        lambda s: np.sqrt(s.shift(1).rolling(21).mean() / (4 * np.log(2)))
    )
    df["parkinson_vol_21"] = park

    # RSI(14) on Adj Close — TRUE Wilder smoothing (EMA alpha=1/n), then shift(1) so the window is strictly
    # past-only (consistent with every other rolling feature). Previously a simple rolling mean mislabeled
    # "Wilder"; fixed in the M3.5 hardening pass.
    def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - 100 / (1 + rs)).shift(1)

    df["rsi_14"] = df.groupby("ticker")["Adj Close"].transform(lambda s: _rsi(s, 14))

    # Volume z-score over 20 days (shifted)
    def _vol_z(v: pd.Series, n: int = 20) -> pd.Series:
        m = v.shift(1).rolling(n).mean()
        sd = v.shift(1).rolling(n).std()
        return (v - m) / sd

    df["volume_z_20"] = df.groupby("ticker")["Volume"].transform(lambda s: _vol_z(s, 20))

    # --- CANDIDATE variance-axis features (M3.5 ablation pool) ------------------------------------------
    # Persisted in the parquet but NOT auto-added to MODEL_FEATURES: a train-only ablation in notebook 03
    # decides which survive (MI vs noise floor + walk-forward permutation importance). All use the same
    # shift(1)-before-rolling + per-ticker discipline as the features above, so they are leakage-safe as X.
    # Rationale: M2 showed the predictable signal is VARIANCE (parkinson_vol MI 0.16 vs 0.002 floor), not
    # direction (|r|<0.075). These sharpen the variance/higher-moment axis; direction indicators were rejected.
    def _semidev(s: pd.Series, n: int) -> pd.Series:
        """Downside vol proxy: rolling std of the downside-clipped series (positives set to 0)."""
        return s.where(s < 0, 0.0).shift(1).rolling(n).std()

    df["semi_vol_21"] = g_logret.transform(lambda s: _semidev(s, 21))
    df["semi_vol_63"] = g_logret.transform(lambda s: _semidev(s, 63))
    # Rolling higher moments (time-varying asymmetry / tail-thickness) — motivated by confirmed fat tails.
    df["ret_skew_21"] = g_logret.transform(lambda s: s.shift(1).rolling(21).skew())
    df["ret_kurt_21"] = g_logret.transform(lambda s: s.shift(1).rolling(21).kurt())
    # Normalized ATR(14): true range includes the overnight gap terms that Parkinson (High/Low only) misses.
    # H/L are RAW (auto_adjust=False) while Adj Close is split/dividend-adjusted; TR uses cross-day DIFFERENCES,
    # so the two bases must be reconciled (a ratio like Parkinson's ln(H/L) is immune, but differences are not).
    # Put H/L on the adjusted basis via adj = Adj Close / Close, then difference against the adjusted prev close.
    adj = df["Adj Close"] / df["Close"]
    h_adj, l_adj = df["High"] * adj, df["Low"] * adj
    prev_c = g_close.shift(1)   # per-ticker adjusted previous close
    tr = np.maximum(h_adj - l_adj,
                    np.maximum((h_adj - prev_c).abs(), (l_adj - prev_c).abs()))
    df["atr_14"] = tr.groupby(df["ticker"]).transform(lambda s: s.shift(1).rolling(14).mean()) / df["Adj Close"]
    return df


def make_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker EXPANDING mean/std of log_return, shifted 1 → each row sees only its strict past."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = df.groupby("ticker")["log_return"]
    df["ticker_expanding_mean"] = g.transform(
        lambda s: s.expanding(min_periods=20).mean().shift(1)
    ).astype("float32")
    df["ticker_expanding_std"] = g.transform(
        lambda s: s.expanding(min_periods=20).std().shift(1)
    ).astype("float32")
    return df


def make_exog_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exogenous (market-wide) features from VIX / yields / dollar. All values are known at the close of day t,
    so they are valid predictors of the day-(t+1) target. Per-ticker grouping is only so the diff/rolling
    don't bleed across the ticker stacking — the underlying series is identical across tickers per date.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = lambda col: df.groupby("ticker")[col]

    # VIX (fear / regime). Level, daily log change, 60d z-score, and a high-vol regime flag (above own 60d median).
    df["vix_level"] = df["vix"]
    df["vix_log_change"] = np.log(df["vix"] / g("vix").shift(1))
    vix_med = g("vix").transform(lambda s: s.rolling(60, min_periods=20).median())
    vix_sd = g("vix").transform(lambda s: s.rolling(60, min_periods=20).std())
    df["vix_z_60"] = (df["vix"] - vix_med) / vix_sd
    df["is_high_vol_regime"] = (df["vix"] > vix_med).astype("float32")  # 1 = fear above its 60d norm

    # Rates: 10y level, daily change, term spread (10y - 3m) and its change.
    df["tnx_level"] = df["tnx"]
    df["tnx_change"] = g("tnx").diff()
    df["term_spread"] = df["tnx"] - df["irx"]          # >0 normal curve, <0 inverted (recession signal)
    df["term_spread_change"] = g("term_spread").diff()

    # Dollar index daily log return.
    df["dxy_log_return"] = np.log(df["dxy"] / g("dxy").shift(1))
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = make_datetime_features(df)
    df = make_price_and_return_features(df)
    df = make_exog_features(df)
    df = make_target_encoding(df)
    return df


# %% [markdown]
# ## 9. Feature computation, chronological split, and next-day target
#
# 1. Compute features on the **full** chronological frame (so the expanding window sees history across the
#    split boundary — a val row correctly sees cumulative train history).
# 2. Split by date (no shuffle).
# 3. Build the next-day target **per split** via `groupby(ticker).log_return.shift(-1)`. Doing it *after*
#    splitting means the last row of each split has `NaN` target (its "tomorrow" lives in the next split) —
#    so no label leaks forward across a boundary. M3 drops NaN-target rows.
# 4. Build the **M3.6 volatility targets** the same way: `fwd_rv_{h}` = sqrt(sum of the next `h` squared
#    log returns) per ticker per split, via `shift(-h)`. Last `h` rows/split are NaN (forward window would
#    cross the split edge). Kept alongside `target_log_return`; notebook 04 forecasts these, M3 ignores them.
#
# Splits: train ≤ 2023-12-31 · val = 2024 · holdout ≥ 2025-01-01.

# %%
df_fe = make_features(df)


def add_next_day_target(split_df: pd.DataFrame) -> pd.DataFrame:
    split_df = split_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    split_df["target_log_return"] = split_df.groupby("ticker")["log_return"].shift(-1)
    return split_df


# M3.6 volatility targets. h-day forward realized volatility per ticker =
# sqrt(sum of the NEXT h squared log returns). Built WITHIN one split via shift(-h), exactly
# like the next-day target above: the last h rows of each split become NaN (their forward
# window lives in the next split), so no volatility label leaks across a split boundary. These
# columns live ALONGSIDE target_log_return; M3 is unaffected (it drops NaN on its own target).
VOL_HORIZONS = [5, 20]


def add_forward_vol_targets(split_df: pd.DataFrame, horizons=VOL_HORIZONS) -> pd.DataFrame:
    d = split_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    d["_sq_ret"] = d.groupby("ticker")["log_return"].transform(lambda s: s.pow(2))
    for h in horizons:
        d[f"fwd_rv_{h}"] = np.sqrt(
            d.groupby("ticker")["_sq_ret"].transform(lambda s: s.rolling(h).sum().shift(-h))
        )
    return d.drop(columns="_sq_ret")


TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")

train_fe = add_forward_vol_targets(add_next_day_target(df_fe[df_fe.date <= TRAIN_END].copy()))
val_fe = add_forward_vol_targets(add_next_day_target(df_fe[(df_fe.date > TRAIN_END) & (df_fe.date <= VAL_END)].copy()))
holdout_fe = add_forward_vol_targets(add_next_day_target(df_fe[df_fe.date > VAL_END].copy()))

for name, sdf in [("train", train_fe), ("val", val_fe), ("holdout", holdout_fe)]:
    print(f"{name:7s}  rows={len(sdf):>5d}  dates {sdf.date.min().date()} -> {sdf.date.max().date()}")

assert train_fe.date.max() < val_fe.date.min(), "train/val overlap!"
assert val_fe.date.max() < holdout_fe.date.min(), "val/holdout overlap!"
print("\nLeakage assertion passed: train.max < val.min < val.max < holdout.min")

# M3.6 volatility-target leakage guard. The last h rows of each ticker in each split MUST be NaN
# (a forward h-day window there would reach past the split edge into the next split). We assert the
# stronger property that EXACTLY h rows/ticker are NaN — catching both (a) a too-short ticker where a
# plain tail(h).all() would pass vacuously, and (b) an interior NaN log_return silently voiding extra
# target rows and shrinking the eval sample without error (audit finding, M3.6 Phase 1).
for _sdf, _name in [(train_fe, "train"), (val_fe, "val"), (holdout_fe, "holdout")]:
    for _h in VOL_HORIZONS:
        _na_per_ticker = _sdf.groupby("ticker")[f"fwd_rv_{_h}"].apply(lambda s: int(s.isna().sum()))
        assert (_na_per_ticker == _h).all(), (
            f"{_name}: fwd_rv_{_h} NaN count per ticker is {_na_per_ticker.to_dict()}, expected exactly {_h} "
            f"(tail-only). >h ⇒ interior gap voided targets; <h ⇒ boundary leak.")
        _tail_all_nan = _sdf.groupby("ticker").tail(_h)[f"fwd_rv_{_h}"].isna().all()
        assert _tail_all_nan, f"{_name}: fwd_rv_{_h} NaNs are not the tail rows — vol label leak!"
print(f"Forward-vol leakage guard passed: exactly h tail rows/ticker NaN for horizons {VOL_HORIZONS}")

# Verify target encoding varies per row (not a constant per-ticker global mean).
print("\nTarget-encoded unique values per ticker (train) — should be ~#rows, not 1:")
print(train_fe.groupby("ticker")[["ticker_expanding_mean"]].nunique().to_string())

# %% [markdown]
# ## 9b. Feature roles — what M3 may use as X (prevents same-day target leakage)
#
# The forecasting target is `target_log_return`. Everything else splits into roles. The **same-day**
# `log_return`/`daily_return`/tail-flag columns are valid predictors *because* the target is next-day — but
# the raw **price-level** columns are non-stationary and are excluded from the default model feature set
# (kept in the parquet for reference / tree models). M3 should train on `MODEL_FEATURES`.

# %%
ID_COLS = ["date", "ticker"]
TARGET = "target_log_return"
SOURCE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
EXOG_RAW = ["vix", "tnx", "irx", "dxy"]  # raw exogenous levels (kept; engineered versions go in MODEL_FEATURES)
RAW_PRICE_LEVEL = ["close_lag_1", "close_lag_5", "close_lag_21", "close_lag_63",
                   "sma_20", "sma_50", "sma_200"]  # non-stationary; excluded from MODEL_FEATURES
DIAGNOSTIC_COLS = ["roll_median_60", "roll_mad_60", "mad_z_score", "is_tail_event",
                   "is_extreme_5pct", "is_warmup"]

MODEL_FEATURES = [
    # calendar
    "month", "dayofweek", "quarter", "is_month_start", "is_month_end", "is_quarter_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    # same-day returns (known at t)
    "daily_return", "log_return",
    "log_return_lag_1", "log_return_lag_5", "log_return_lag_21",
    # stationary momentum
    "momentum_5", "momentum_21", "momentum_63",
    # stationary price-relative
    "price_to_sma_20", "price_to_sma_50", "price_to_sma_200",
    # volatility
    "realized_vol_5", "realized_vol_21", "realized_vol_63", "rolling_std_20", "parkinson_vol_21",
    # oscillator / volume
    "rsi_14", "volume_z_20",
    # target encoding
    "ticker_expanding_mean", "ticker_expanding_std",
    # EXOGENOUS (new in the enhancement pass) — VIX/sentiment, rates, dollar
    "vix_level", "vix_log_change", "vix_z_60", "is_high_vol_regime",
    "tnx_level", "tnx_change", "term_spread", "term_spread_change",
    "dxy_log_return",
]
# CANDIDATE features (M3.5): persisted in the parquet, NOT in MODEL_FEATURES yet. Notebook 03 runs a
# train-only ablation (MI vs noise floor + walk-forward permutation importance) and promotes only survivors.
CANDIDATE_FEATURES = ["semi_vol_21", "semi_vol_63", "ret_skew_21", "ret_kurt_21", "atr_14"]
print(f"MODEL_FEATURES: {len(MODEL_FEATURES)} columns ({len(MODEL_FEATURES) - 9} base + 9 exogenous)")
print(f"CANDIDATE_FEATURES (ablation pool, gated into X by notebook 03): {CANDIDATE_FEATURES}")
print("Excluded from X (non-stationary price levels):", RAW_PRICE_LEVEL)

# %% [markdown]
# ## 9c. Warm-up NaN report + `is_warmup` flag
#
# Lag/rolling/expanding features are `NaN` during their warm-up window (e.g. `sma_200` for the first 200
# rows/ticker, `momentum_63` for 63, expanding stats for 20). These rows are **retained** but flagged with
# `is_warmup` (any MODEL_FEATURE is NaN). M3 must `dropna(subset=MODEL_FEATURES)` (or filter `is_warmup==0`)
# before fitting — LSTMs and scalers crash on NaN. We quantify the NaNs explicitly so nothing is silent.

# %%
def add_warmup_flag(sdf: pd.DataFrame) -> pd.DataFrame:
    sdf = sdf.copy()
    sdf["is_warmup"] = sdf[MODEL_FEATURES].isna().any(axis=1).astype("int8")
    return sdf


train_fe = add_warmup_flag(train_fe)
val_fe = add_warmup_flag(val_fe)
holdout_fe = add_warmup_flag(holdout_fe)

nan_lines = ["# Feature NaN / warm-up report — M1", "",
             "Per-split NaN counts for MODEL_FEATURES + target. Rows with any NaN feature are flagged "
             "`is_warmup=1`; M3 must drop them before fitting.", ""]
for name, sdf in [("train", train_fe), ("val", val_fe), ("holdout", holdout_fe)]:
    n_warm = int(sdf["is_warmup"].sum())
    n_tgt = int(sdf[TARGET].isna().sum())
    nan_lines.append(f"## {name}  (rows={len(sdf)}, is_warmup=1: {n_warm}, target NaN: {n_tgt})")
    nz = sdf[MODEL_FEATURES].isna().sum()
    nz = nz[nz > 0].sort_values(ascending=False)
    if len(nz):
        nan_lines.append("")
        nan_lines.append("| column | NaN count |")
        nan_lines.append("|---|---|")
        nan_lines += [f"| `{c}` | {int(n)} |" for c, n in nz.items()]
    nan_lines.append("")
    print(f"{name}: is_warmup={n_warm}, target NaN={n_tgt}")
(PROC / "feature_nan_report.md").write_text("\n".join(nan_lines), encoding="utf-8")
print("saved:", PROC / "feature_nan_report.md")

# %% [markdown]
# ## 10. Visualisations

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, t in zip(axes.ravel(), TICKERS):
    sub = df[df.ticker == t]
    ax.plot(sub.date, sub["Adj Close"], lw=1.2)
    ax.set_title(f"{t} — Adj Close")
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)
fig.suptitle("10-year price trends (Yahoo Finance, daily, pinned window)", fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "M1_price_trends.png", dpi=120)
plt.close(fig)
print("saved:", FIG / "M1_price_trends.png")

# %%
fig, ax = plt.subplots(figsize=(10, 5))
for t in TICKERS:
    sub = df[(df.ticker == t) & df.log_return.notna()]
    ax.hist(sub.log_return, bins=60, alpha=0.4, label=t, density=True)
ax.set_title("Daily log-return distribution per ticker (10 yr)")
ax.set_xlabel("log return")
ax.set_ylabel("density")
ax.set_xlim(-0.12, 0.12)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "M1_target_hist.png", dpi=120)
plt.close(fig)
print("saved:", FIG / "M1_target_hist.png")

# Side-by-side skew + kurtosis for Close vs log_return.
print("\nSkewness and excess kurtosis side-by-side:")
print(f"{'ticker':<8}{'skew(Close)':>13}{'kurt(Close)':>14}  |  {'skew(logret)':>14}{'kurt(logret)':>14}")
print("-" * 78)
for t in TICKERS:
    close = df.loc[df.ticker == t, "Adj Close"].dropna()
    logret = df.loc[(df.ticker == t) & df.log_return.notna(), "log_return"]
    print(f"{t:<8}{skew(close):>13.3f}{kurtosis(close):>14.3f}  |  "
          f"{skew(logret):>14.3f}{kurtosis(logret):>14.3f}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, t in zip(axes.ravel(), TICKERS):
    sub = df[(df.ticker == t) & df.log_return.notna()].copy()
    normal = sub[sub.is_tail_event == 0]
    flagged = sub[sub.is_tail_event == 1]
    ax.scatter(normal.date, normal.log_return, s=4, alpha=0.45, color="steelblue", label="normal")
    ax.scatter(flagged.date, flagged.log_return, s=18, alpha=0.85, color="crimson",
               label="tail event (MAD z>3.5, retained)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"{t} — log_return (tail-event flags)")
    ax.set_ylabel("log_return")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("Tail-event policy: retain + flag (no rows deleted)", fontsize=14)
fig.tight_layout()
fig.savefig(FIG / "M1_before_after_cleaning.png", dpi=120)
plt.close(fig)
print("saved:", FIG / "M1_before_after_cleaning.png")

# %% [markdown]
# ## 11. Save parquet splits + feature dictionary

# %%
train_fe.to_parquet(PROC / "train_fe.parquet", index=False)
val_fe.to_parquet(PROC / "val_fe.parquet", index=False)
holdout_fe.to_parquet(PROC / "holdout_fe.parquet", index=False)
print("saved parquet files to:", PROC)
for p in PROC.glob("*.parquet"):
    print(f"  {p.name:30s}  {p.stat().st_size / 1024:>8.1f} KB")

# Persist the machine-readable feature roles so M3 can't guess wrong.
roles = {
    "target": TARGET,
    "id_cols": ID_COLS,
    "model_features": MODEL_FEATURES,
    "candidate_features": CANDIDATE_FEATURES,  # M3.5 ablation pool — notebook 03 promotes survivors
    "vol_targets": {f"fwd_rv_{h}": h for h in VOL_HORIZONS},  # M3.6 — h-day forward realized vol
    "raw_price_level_excluded": RAW_PRICE_LEVEL,
    "diagnostic_cols": DIAGNOSTIC_COLS,
    "source_cols": SOURCE_COLS,
    "exog_raw": EXOG_RAW,
}
(PROC / "feature_roles.json").write_text(json.dumps(roles, indent=2))
print("saved:", PROC / "feature_roles.json")

# %%
def write_feature_dictionary(df: pd.DataFrame, out_path: Path) -> None:
    """One row per column: dtype, role, description, how computed, leakage-safe-as-X."""
    # role, description, how, leakage-safe-as-X (relative to the NEXT-day target)
    meta = {
        "date": ("id", "Trading date", "from source", "yes"),
        "ticker": ("id", "Ticker symbol", "constant per row", "yes"),
        "Open": ("source", "Opening price (raw)", "from source", "yes"),
        "High": ("source", "Daily high", "from source", "yes"),
        "Low": ("source", "Daily low", "from source", "yes"),
        "Close": ("source", "Closing price (raw, split-jumpy)", "from source", "yes"),
        "Adj Close": ("source", "Split/dividend-adjusted close (used for all returns)", "from source", "yes"),
        "Volume": ("source", "Daily share volume", "from source; NaN→0", "yes"),
        "daily_return": ("feature", "Simple daily return", "pct_change(Adj Close) per ticker", "yes (known at t)"),
        "log_return": ("feature", "Log return (today's; a PREDICTOR, not the target)", "ln(AdjC_t/AdjC_{t-1})", "yes (known at t)"),
        "target_log_return": ("TARGET", "NEXT-day log return (what M3 predicts)", "groupby(ticker).log_return.shift(-1) per split", "n/a (label)"),
        "roll_median_60": ("diagnostic", "60d rolling median of log_return", "rolling(60).median()", "diagnostic — not in X"),
        "roll_mad_60": ("diagnostic", "60d rolling MAD of log_return", "median(|x-rollmed|)", "diagnostic — not in X"),
        "mad_z_score": ("diagnostic", "Robust z = (logret-rollmed)/rollmad", "see above", "diagnostic — not in X"),
        "is_tail_event": ("diagnostic", "Tail-event flag |z|>3.5 (NaN during warm-up)", "abs(mad_z)>3.5", "diagnostic — not in X"),
        "is_extreme_5pct": ("diagnostic", "Reference flag |daily move|>5%", "abs threshold", "diagnostic — not in X"),
        "is_warmup": ("diagnostic", "1 if any MODEL_FEATURE is NaN (drop in M3)", "isna().any() over MODEL_FEATURES", "diagnostic — not in X"),
        "year": ("feature", "Calendar year", "date.dt.year", "yes"),
        "month": ("feature", "Month 1-12", "date.dt.month", "yes"),
        "day": ("feature", "Day of month", "date.dt.day", "yes"),
        "dayofweek": ("feature", "Weekday 0=Mon", "date.dt.dayofweek", "yes"),
        "weekofyear": ("feature", "ISO week", "isocalendar().week", "yes"),
        "quarter": ("feature", "Quarter 1-4", "date.dt.quarter", "yes"),
        "is_month_start": ("feature", "First trading day of month", "dt.is_month_start", "yes"),
        "is_month_end": ("feature", "Last trading day of month", "dt.is_month_end", "yes"),
        "is_quarter_end": ("feature", "Last trading day of quarter", "dt.is_quarter_end", "yes"),
        "month_sin": ("feature", "Cyclical month", "sin(2π·month/12)", "yes"),
        "month_cos": ("feature", "Cyclical month", "cos(2π·month/12)", "yes"),
        "dow_sin": ("feature", "Smooth weekday encoder (period 5)", "sin(2π·dow/5)", "yes"),
        "dow_cos": ("feature", "Smooth weekday encoder (period 5)", "cos(2π·dow/5)", "yes"),
        "close_lag_1": ("raw_price_level", "Adj Close 1d ago (NON-stationary)", "shift(1)", "excluded from X"),
        "close_lag_5": ("raw_price_level", "Adj Close 5d ago (NON-stationary)", "shift(5)", "excluded from X"),
        "close_lag_21": ("raw_price_level", "Adj Close 21d ago (NON-stationary)", "shift(21)", "excluded from X"),
        "close_lag_63": ("raw_price_level", "Adj Close 63d ago (NON-stationary)", "shift(63)", "excluded from X"),
        "log_return_lag_1": ("feature", "log_return 1d ago", "shift(1)", "yes"),
        "log_return_lag_5": ("feature", "log_return 5d ago", "shift(5)", "yes"),
        "log_return_lag_21": ("feature", "log_return 21d ago", "shift(21)", "yes"),
        "momentum_5": ("feature", "5d log momentum (stationary)", "ln(AdjC_t/AdjC_{t-5})", "yes"),
        "momentum_21": ("feature", "21d log momentum (stationary)", "ln(AdjC_t/AdjC_{t-21})", "yes"),
        "momentum_63": ("feature", "63d log momentum (stationary)", "ln(AdjC_t/AdjC_{t-63})", "yes"),
        "sma_20": ("raw_price_level", "20d SMA of Adj Close (NON-stationary)", "shift(1).rolling(20).mean()", "excluded from X"),
        "sma_50": ("raw_price_level", "50d SMA (NON-stationary)", "shift(1).rolling(50).mean()", "excluded from X"),
        "sma_200": ("raw_price_level", "200d SMA (NON-stationary)", "shift(1).rolling(200).mean()", "excluded from X"),
        "price_to_sma_20": ("feature", "Adj Close / SMA20 - 1 (stationary)", "C_t/sma_20-1", "yes"),
        "price_to_sma_50": ("feature", "Adj Close / SMA50 - 1 (stationary)", "C_t/sma_50-1", "yes"),
        "price_to_sma_200": ("feature", "Adj Close / SMA200 - 1 (stationary)", "C_t/sma_200-1", "yes"),
        "realized_vol_5": ("feature", "5d realized vol of log_return", "shift(1).rolling(5).std()", "yes"),
        "realized_vol_21": ("feature", "21d realized vol", "shift(1).rolling(21).std()", "yes"),
        "realized_vol_63": ("feature", "63d realized vol", "shift(1).rolling(63).std()", "yes"),
        "rolling_std_20": ("feature", "20d rolling std of log_return", "shift(1).rolling(20).std()", "yes"),
        "parkinson_vol_21": ("feature", "21d Parkinson range vol (High/Low)", "sqrt(mean(ln(H/L)^2)/(4ln2))", "yes"),
        "rsi_14": ("feature", "14d RSI on Adj Close (true Wilder EMA, shifted 1)", "Wilder ewm(alpha=1/14).mean() gain/loss, .shift(1)", "yes"),
        "volume_z_20": ("feature", "20d volume z-score", "(V-rollmean)/rollstd", "yes"),
        # CANDIDATE variance-axis features (M3.5 ablation pool — promoted into X only if they survive notebook 03's ablation)
        "semi_vol_21": ("candidate", "21d downside semi-deviation (leverage asymmetry)", "std(min(logret,0)).shift(1).rolling(21)", "yes"),
        "semi_vol_63": ("candidate", "63d downside semi-deviation", "std(min(logret,0)).shift(1).rolling(63)", "yes"),
        "ret_skew_21": ("candidate", "21d rolling skewness of log_return", "shift(1).rolling(21).skew()", "yes"),
        "ret_kurt_21": ("candidate", "21d rolling kurtosis of log_return", "shift(1).rolling(21).kurt()", "yes"),
        "atr_14": ("candidate", "14d normalized ATR (true range incl. overnight gap / close)", "mean(TR).shift(1).rolling(14)/AdjC", "yes"),
        "ticker_expanding_mean": ("feature", "Expanding mean log_return, shifted 1 (past-only)", "expanding(min20).mean().shift(1)", "yes"),
        "ticker_expanding_std": ("feature", "Expanding std log_return, shifted 1", "expanding(min20).std().shift(1)", "yes"),
        # raw exogenous levels (kept for reference; engineered versions are the model features)
        "vix": ("exog_raw", "CBOE VIX close (raw level)", "Yahoo ^VIX Adj Close, ffill", "yes (known at t)"),
        "tnx": ("exog_raw", "10y Treasury yield (raw)", "Yahoo ^TNX, ffill", "yes (known at t)"),
        "irx": ("exog_raw", "13wk T-bill yield (raw)", "Yahoo ^IRX, ffill", "yes (known at t)"),
        "dxy": ("exog_raw", "US dollar index (raw)", "Yahoo DX-Y.NYB, ffill", "yes (known at t)"),
        # engineered exogenous features (in MODEL_FEATURES)
        "vix_level": ("feature", "VIX level (fear/sentiment proxy)", "= vix", "yes (known at t)"),
        "vix_log_change": ("feature", "Daily VIX log change", "ln(vix_t/vix_{t-1})", "yes"),
        "vix_z_60": ("feature", "VIX 60d z-score", "(vix-roll_median)/roll_std", "yes"),
        "is_high_vol_regime": ("feature", "1 if VIX above its 60d median (fear regime)", "vix > rolling_median_60(vix)", "yes"),
        "tnx_level": ("feature", "10y Treasury yield level", "= tnx", "yes (known at t)"),
        "tnx_change": ("feature", "Daily change in 10y yield", "tnx.diff()", "yes"),
        "term_spread": ("feature", "Yield curve slope 10y-3m", "tnx - irx", "yes (known at t)"),
        "term_spread_change": ("feature", "Daily change in term spread", "term_spread.diff()", "yes"),
        "dxy_log_return": ("feature", "US dollar index daily log return", "ln(dxy_t/dxy_{t-1})", "yes"),
    }
    lines = [
        "# Feature dictionary — Stock Market Trend Analysis M1",
        "",
        "Generated by `notebooks/01_data_collection_preprocessing.ipynb` (seed=42).",
        "",
        "**Target:** `target_log_return` (next-day log return). **Model features:** see `feature_roles.json` "
        "(`MODEL_FEATURES`) — raw price-level columns are NON-stationary and excluded from X.",
        "",
        "'Leakage-safe-as-X' is judged relative to the NEXT-day target: same-day columns are valid predictors "
        "(known at decision time t).",
        "",
        "| Column | dtype | Role | Description | How computed | Leakage-safe as X |",
        "|---|---|---|---|---|---|",
    ]
    for col in df.columns:
        m = meta.get(col)
        if m is None:
            lines.append(f"| `{col}` | {df[col].dtype} | ? | _undocumented_ | _undocumented_ | ? |")
        else:
            role, desc, how, safe = m
            lines.append(f"| `{col}` | {df[col].dtype} | {role} | {desc} | {how} | {safe} |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


write_feature_dictionary(train_fe, PROC / "feature_dictionary.md")
print("saved:", PROC / "feature_dictionary.md")

# %% [markdown]
# ## 12. Self-audit
#
# Mechanical checks the rubric requires. All must pass before handing off to M2.

# %%
audit = {
    "snapshot_csv_per_ticker": all((RAW / f"{safe_filename(t)}_10y.csv").exists() for t in TICKERS),
    "pinned_window_recorded": json.loads((RAW / "snapshot_hashes.json").read_text()).get("_window") is not None,
    "volume_dtype_float64": train_fe["Volume"].dtype == np.float64,
    "train_fe_exists": (PROC / "train_fe.parquet").exists(),
    "val_fe_exists": (PROC / "val_fe.parquet").exists(),
    "holdout_fe_exists": (PROC / "holdout_fe.parquet").exists(),
    "feature_dictionary_exists": (PROC / "feature_dictionary.md").exists(),
    "feature_roles_exists": (PROC / "feature_roles.json").exists(),
    "nan_report_exists": (PROC / "feature_nan_report.md").exists(),
    "no_split_overlap": (train_fe.date.max() < val_fe.date.min()
                         and val_fe.date.max() < holdout_fe.date.min()),
    "next_day_target_exists": "target_log_return" in train_fe.columns,
    "target_is_next_day": bool(
        # last train row per ticker has NaN target (its tomorrow is in val) → no forward leak
        train_fe.groupby("ticker")["target_log_return"].apply(lambda s: pd.isna(s.iloc[-1])).all()
    ),
    "model_features_present": all(c in train_fe.columns for c in MODEL_FEATURES),
    "raw_price_excluded_from_features": not any(c in MODEL_FEATURES for c in RAW_PRICE_LEVEL),
    "target_encoding_varies_per_row": all(
        train_fe.groupby("ticker")["ticker_expanding_mean"].nunique().gt(1)
    ),
    "tail_flag_rate_reasonable": (
        train_fe.loc[train_fe.is_tail_event.notna()].groupby("ticker")["is_tail_event"].mean().max() < 0.10
    ),
    "target_dist_fig_exists": (FIG / "M1_target_hist.png").exists(),
    "price_trends_fig_exists": (FIG / "M1_price_trends.png").exists(),
    "before_after_fig_exists": (FIG / "M1_before_after_cleaning.png").exists(),
}
for k, v in audit.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

assert all(audit.values()), "M1 self-audit failed!"
print(f"\nAll {len(audit)} M1 self-audit checks passed. Ready for M2.")
