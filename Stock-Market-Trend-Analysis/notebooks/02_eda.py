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
# # Milestone 2 — Exploratory Data Analysis
# **Project:** Stock Market Trend Analysis | **Tickers:** ^GSPC, AAPL, AMZN, NVDA | **Seed:** 42
#
# ## What this notebook does
#
# M1 produced a cleaned, leakage-safe dataset. M2 **explores** it to answer: *what patterns exist, and which
# models should M3 try?* Every figure gets a caption and a one-line takeaway.
#
# **Rule we never break:** we read **`train_fe.parquet` only**. `val_fe`/`holdout_fe` stay sealed — looking at
# them during EDA is a form of leakage (we'd unconsciously design features/models to fit the test set).
#
# Sections: summary stats → distributions → time-series views → seasonal decomposition → autocorrelation
# (returns AND squared returns) → stationarity (ADF) + heteroskedasticity (ARCH-LM) → correlation →
# hypotheses → modelling implications → self-audit.
#
# **Carried over from M1 (v3):** the forecast target is `target_log_return` (next-day). Returns are
# stationary in *mean* but conditionally heteroskedastic (volatility clusters) — M2 quantifies this to
# justify the M3 model choice (ARIMA on returns + a GARCH/vol-aware component).

# %% [markdown]
# ## 0. Setup (Colab-aware) — load ONLY train_fe

# %%
import sys
import subprocess

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from pathlib import Path as _P

    _req = _P("/content/Stock-Market-Trend-Analysis/requirements.txt")
    if _req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(_req)], check=False)

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="muted")

if IN_COLAB:
    ROOT = Path("/content/Stock-Market-Trend-Analysis")
elif "__file__" in globals():
    ROOT = Path(__file__).resolve().parent.parent
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TICKERS = ["^GSPC", "AAPL", "AMZN", "NVDA"]

# Load ONLY train_fe. We log every parquet path opened so the self-audit can PROVE the val/holdout
# files were never touched (a real sealing check, not a vacuous one).
_OPENED_PARQUET = []
# Stash the TRUE original on the module the first time only. Re-running this cell must NOT capture the
# already-patched wrapper as "original" (that self-wraps -> infinite recursion). The wrapper always calls
# the stashed true function, so re-running is safe.
if not hasattr(pd, "_eda_true_read_parquet"):
    pd._eda_true_read_parquet = pd.read_parquet


def read_parquet_logged(path, *a, **k):
    _OPENED_PARQUET.append(str(path))
    return pd._eda_true_read_parquet(path, *a, **k)


pd.read_parquet = read_parquet_logged

train = pd.read_parquet(PROC / "train_fe.parquet")
train["date"] = pd.to_datetime(train["date"])
roles = json.loads((PROC / "feature_roles.json").read_text())
MODEL_FEATURES = roles["model_features"]
TARGET = roles["target"]

print("train_fe shape:", train.shape)
print("date range    :", train.date.min().date(), "->", train.date.max().date())
print("tickers       :", sorted(train.ticker.unique()))
print("n MODEL_FEATURES:", len(MODEL_FEATURES), "| target:", TARGET)

# %% [markdown]
# ## 1. Summary statistics
#
# Three views: overall, per-ticker (annualised), per-year. Annualisation uses the finance convention of
# 252 trading days: annual mean return ≈ daily mean × 252; annual volatility ≈ daily std × √252.
# The **Sharpe-like ratio** here is `mean/std × √252` (excess-return-over-zero, no risk-free adjustment —
# a simplification for comparison only).

# %%
def max_drawdown(log_returns: pd.Series) -> float:
    """Max peak-to-trough drawdown from a log-return series (as a negative fraction)."""
    cum = np.exp(log_returns.cumsum())
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return float(dd.min())


# Overall
lr_all = train["log_return"].dropna()
print("=== Overall ===")
print(f"rows={len(train)}  tickers={train.ticker.nunique()}  "
      f"dates {train.date.min().date()}..{train.date.max().date()}")
print(f"Adj Close: mean={train['Adj Close'].mean():.2f} min={train['Adj Close'].min():.2f} "
      f"max={train['Adj Close'].max():.2f}")
print(f"log_return: mean={lr_all.mean():.5f} std={lr_all.std():.5f} "
      f"min={lr_all.min():.4f} max={lr_all.max():.4f}")

# Per ticker (annualised)
rows = []
for t in TICKERS:
    r = train.loc[train.ticker == t, "log_return"].dropna()
    rows.append({
        "ticker": t,
        "days": len(r),
        "ann_return_%": round(r.mean() * 252 * 100, 1),
        "ann_vol_%": round(r.std() * np.sqrt(252) * 100, 1),
        "sharpe_like": round(r.mean() / r.std() * np.sqrt(252), 2),
        "min_day_%": round(r.min() * 100, 1),
        "max_day_%": round(r.max() * 100, 1),
        "max_drawdown_%": round(max_drawdown(r) * 100, 1),
    })
per_ticker = pd.DataFrame(rows).set_index("ticker")
print("\n=== Per ticker (annualised) ===")
print(per_ticker.to_string())

# Per year x ticker
per_year = (
    train.assign(year=train.date.dt.year)
    .groupby(["ticker", "year"])["log_return"]
    .agg(ann_ret=lambda s: s.mean() * 252, ann_vol=lambda s: s.std() * np.sqrt(252))
    .round(3)
)
print("\n=== Per year x ticker (annualised return, vol) — head ===")
print(per_year.head(12).to_string())

# %% [markdown]
# **Takeaways:**
# - NVDA carries by far the highest annualised volatility (single high-beta stock) vs ^GSPC (a 500-name
#   basket whose idiosyncratic moves diversify away) — quantified in the table above.
# - Sharpe-like ratios rank the risk-adjusted performance; tech names beat the index on raw return but at
#   materially higher volatility and deeper drawdowns.

# %% [markdown]
# ## 2. Univariate distributions

# %%
# Fig 1 — log-return histogram per ticker with a fitted normal overlay.
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, t in zip(axes.ravel(), TICKERS):
    r = train.loc[train.ticker == t, "log_return"].dropna()
    ax.hist(r, bins=80, density=True, alpha=0.55, color="steelblue")
    xs = np.linspace(r.min(), r.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), "r-", lw=1.5, label="normal fit")
    ax.set_title(f"{t}  (excess kurtosis={stats.kurtosis(r):.1f})")
    ax.set_xlabel("log return")
    ax.legend(fontsize=8)
fig.suptitle("Fig 1 — Log-return distribution vs normal (fat tails visible)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "M2_fig01_logret_hist.png", dpi=120)
plt.close(fig)

# Fig 2 — Q-Q plot vs normal (4 panels).
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, t in zip(axes.ravel(), TICKERS):
    r = train.loc[train.ticker == t, "log_return"].dropna()
    stats.probplot(r, dist="norm", plot=ax)
    ax.set_title(f"{t} Q-Q vs normal")
fig.suptitle("Fig 2 — Q-Q plots: points leave the line in the tails (leptokurtic)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "M2_fig02_qq.png", dpi=120)
plt.close(fig)

# Fig 3 — log-return boxplot by ticker.
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=train, x="ticker", y="log_return", order=TICKERS, ax=ax, showfliers=True)
ax.set_title("Fig 3 — Log-return spread by ticker (whiskers/outliers = tail risk)")
fig.tight_layout()
fig.savefig(FIG / "M2_fig03_logret_box.png", dpi=120)
plt.close(fig)

# Fig 4 — log-return boxplot by month (pooled).
fig, ax = plt.subplots(figsize=(11, 5))
sns.boxplot(data=train.assign(month=train.date.dt.month), x="month", y="log_return", ax=ax, showfliers=False)
ax.set_title("Fig 4 — Log-return by calendar month (pooled across tickers, fliers hidden)")
fig.tight_layout()
fig.savefig(FIG / "M2_fig04_month_box.png", dpi=120)
plt.close(fig)
print("saved fig 1-4")

# %% [markdown]
# **Takeaways:** all four return series are visibly **leptokurtic** — the histogram peak is taller and the
# tails fatter than the normal overlay, and the Q-Q points bend away from the line at both ends. The per-month
# boxplot shows no dramatic calendar pattern in the *median*, but spread varies (a volatility, not mean, effect).

# %% [markdown]
# ## 3. Time-series visualizations

# %%
# Fig 5 — price facets.
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, t in zip(axes.ravel(), TICKERS):
    s = train[train.ticker == t]
    ax.plot(s.date, s["Adj Close"], lw=1.1)
    ax.set_title(t)
    ax.set_ylabel("Adj Close")
fig.suptitle("Fig 5 — Adjusted close, 2016 → 2023 (train only)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG / "M2_fig05_price_facets.png", dpi=120)
plt.close(fig)

# Fig 6 — normalised price (start = 1.0).
fig, ax = plt.subplots(figsize=(11, 6))
for t in TICKERS:
    s = train[train.ticker == t].sort_values("date")
    ax.plot(s.date, s["Adj Close"] / s["Adj Close"].iloc[0], lw=1.3, label=t)
ax.set_title("Fig 6 — Growth of $1 invested 2016-05 (train period)")
ax.set_ylabel("normalised price")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "M2_fig06_normalised.png", dpi=120)
plt.close(fig)

# Fig 7 — 21-day rolling volatility (annualised).
fig, ax = plt.subplots(figsize=(11, 6))
for t in TICKERS:
    s = train[train.ticker == t].sort_values("date")
    rv = s["log_return"].rolling(21).std() * np.sqrt(252)
    ax.plot(s.date, rv, lw=1.0, label=t)
ax.set_title("Fig 7 — 21-day rolling annualised volatility (clustering visible)")
ax.set_ylabel("annualised vol")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "M2_fig07_rolling_vol.png", dpi=120)
plt.close(fig)

# Fig 8 — NVDA zoom, last 2 yr of train.
fig, ax = plt.subplots(figsize=(11, 5))
s = train[train.ticker == "NVDA"].sort_values("date")
s2 = s[s.date >= s.date.max() - pd.DateOffset(years=2)]
ax.plot(s2.date, s2["Adj Close"], lw=1.3, color="darkgreen")
ax.set_title("Fig 8 — NVDA Adj Close, last 2 yr of train (AI-cycle run-up)")
fig.tight_layout()
fig.savefig(FIG / "M2_fig08_nvda_zoom.png", dpi=120)
plt.close(fig)
print("saved fig 5-8")

# %% [markdown]
# **Takeaways:** the COVID-2020 drawdown and recovery are visible across all four series; volatility clearly
# **clusters** (calm 2017 vs turbulent 2020/2022 in Fig 7). NVDA's 2023 AI-cycle run-up dominates the
# normalised-growth chart.

# %% [markdown]
# ## 4. Seasonality decomposition (^GSPC)
#
# `seasonal_decompose` needs a regular index. Trading days are irregular (gaps for weekends/holidays), so we
# decompose on the **ordinal trading-day sequence** with the stated period (5 ≈ week, 252 ≈ year) and label
# the x-axis as trading-day index, not calendar date.

# %%
g = train[train.ticker == "^GSPC"].sort_values("date").reset_index(drop=True)
s_price = pd.Series(g["Adj Close"].values)  # integer index = trading-day ordinal

weekly = seasonal_decompose(s_price, model="additive", period=5)
annual = seasonal_decompose(s_price, model="multiplicative", period=252)

for tag, dec, title in [("weekly", weekly, "period=5 (trading week)"),
                        ("annual", annual, "period=252 (trading year)")]:
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(dec.observed, lw=0.8); axes[0].set_ylabel("observed")
    axes[1].plot(dec.trend, lw=0.8, color="orange"); axes[1].set_ylabel("trend")
    axes[2].plot(dec.seasonal, lw=0.8, color="green"); axes[2].set_ylabel("seasonal")
    axes[3].plot(dec.resid, lw=0.5, color="grey"); axes[3].set_ylabel("resid")
    axes[3].set_xlabel("trading-day index")
    fig.suptitle(f"Fig {'09' if tag=='weekly' else '10'} — ^GSPC decomposition, {title}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / f"M2_fig{'09' if tag=='weekly' else '10'}_decompose_{tag}.png", dpi=120)
    plt.close(fig)

# Quantify how small the seasonal component is vs the trend.
wk_seas_amp = float(np.nanmax(weekly.seasonal) - np.nanmin(weekly.seasonal))
an_seas_factors = (float(np.nanmin(annual.seasonal)), float(np.nanmax(annual.seasonal)))
price_range = float(s_price.max() - s_price.min())
print(f"weekly period-5 seasonal amplitude: {wk_seas_amp:.2f} pts ({100*wk_seas_amp/price_range:.2f}% of range)")
print(f"annual seasonal factors span: {an_seas_factors[0]:.3f}..{an_seas_factors[1]:.3f} "
      f"(i.e. ~+-{100*max(abs(an_seas_factors[0]-1), abs(an_seas_factors[1]-1)):.1f}% around 1.0)")

# CAVEAT (audit): decomposing a strongly-trending PRICE level is fragile — the centered moving-average trend
# mechanically absorbs almost all variance, so "no seasonality" is partly a method artifact. Confirm on the
# stationary RETURN series, where a seasonal component would actually show up if present.
r_gspc = g["log_return"].dropna().reset_index(drop=True)
ret_dec = seasonal_decompose(r_gspc, model="additive", period=5)
ret_seas_var = float(np.nanvar(ret_dec.seasonal))
ret_obs_var = float(np.nanvar(ret_dec.observed))
print(f"returns period-5 decomposition: seasonal/observed variance ratio = "
      f"{100*ret_seas_var/ret_obs_var:.2f}% (negligible -> confirms no real seasonality)")
print("saved fig 9-10")

# %% [markdown]
# **Takeaways (with audit caveats):** the **trend dominates**. Weekly period-5 seasonal amplitude is ~0.11% of
# the price range; annual multiplicative factors span ~0.97–1.03 (±3%). **Two caveats from review:**
# (1) `period=5` on the *ordinal* trading-day index is **not** a calendar-weekday cycle — holiday/gap rows
# desync the 5-cycle from Mon–Fri, so this is a 5-row arithmetic cycle with no weekday meaning (a genuine
# weekday effect is tested separately in §7 via `dayofweek`). (2) Decomposing the non-stationary **price level**
# is fragile (the trend absorbs ~all variance). Re-running the decomposition on **returns** (stationary) gives a
# seasonal/observed variance ratio of ~0.3% — independently confirming **no material calendar seasonality**.
# Implication for M3: a SARIMA seasonal term adds little; the predictable structure is the **variance of
# returns** (§5–6), not a price season.

# %% [markdown]
# ## 5. Autocorrelation — returns vs squared returns
#
# The key EDA result for this project. If the **return** ACF is ~0 at all lags, the return level is
# (near) unpredictable from its own past — the efficient-market expectation. If the **squared-return** ACF
# is strongly positive, volatility is predictable — the GARCH signature.

# %%
r = g["log_return"].dropna().reset_index(drop=True)

fig = plot_acf(r, lags=60, alpha=0.05)
fig.set_size_inches(10, 4)
fig.suptitle("Fig 11 — ACF of ^GSPC log returns (expect ~0: weak mean predictability)")
fig.tight_layout(); fig.savefig(FIG / "M2_fig11_acf_returns.png", dpi=120); plt.close(fig)

fig = plot_pacf(r, lags=60, alpha=0.05, method="ywm")
fig.set_size_inches(10, 4)
fig.suptitle("Fig 12 — PACF of ^GSPC log returns")
fig.tight_layout(); fig.savefig(FIG / "M2_fig12_pacf_returns.png", dpi=120); plt.close(fig)

fig = plot_acf(r**2, lags=60, alpha=0.05)
fig.set_size_inches(10, 4)
fig.suptitle("Fig 13 — ACF of SQUARED ^GSPC returns (expect strong: volatility clustering)")
fig.tight_layout(); fig.savefig(FIG / "M2_fig13_acf_sq_returns.png", dpi=120); plt.close(fig)

print(f"^GSPC ACF(return)  lag1={r.autocorr(1):+.3f} lag2={r.autocorr(2):+.3f} lag5={r.autocorr(5):+.3f}")
print(f"^GSPC ACF(return^2)lag1={(r**2).autocorr(1):+.3f} lag2={(r**2).autocorr(2):+.3f} lag5={(r**2).autocorr(5):+.3f}")
print("saved fig 11-13")

# %% [markdown]
# **Takeaways (corrected after review):** the lag-1 return ACF is **not** ~0 — it is **−0.162** for ^GSPC
# (≈4σ past the ±0.045 significance band; also significant & negative for AAPL −0.087, NVDA −0.078). The
# **negative sign** is the textbook signature of **1-day mean-reversion / bid-ask bounce / non-synchronous
# trading microstructure**, not genuine return alpha — and for an index (^GSPC, no bid-ask) it points to
# stale-component effects. So: a low-order MA(1)/AR(1) term *will* fit (justifies trying ARIMA(1,0,0) /
# (0,0,1)), but it is unlikely to beat a naive baseline net of trading costs. Beyond lag-1 the return ACF is
# small. Meanwhile the **squared-return ACF is large and slow-decaying** (lag-1 = 0.487) — **volatility
# clustering confirmed**. The mean is near-unpredictable; the variance is highly predictable → GARCH-family
# variance model in M3.

# %% [markdown]
# ## 6. Stationarity (ADF) + heteroskedasticity (ARCH-LM, Ljung-Box)
#
# ADF tests the mean (unit root). ARCH-LM and Ljung-Box(r²) test for time-varying variance. Together they
# pin down exactly what kind of model M3 needs.

# %%
print(f"{'ticker':<8}{'ADF Close p':>13}{'ADF dClose p':>14}{'ADF logret p':>14}"
      f"{'ARCH-LM p':>12}{'LB(r^2,10) p':>14}")
diag_rows = {}
for t in TICKERS:
    s = train[train.ticker == t].sort_values("date")
    close = s["Adj Close"].dropna()
    rr = s["log_return"].dropna()
    p_close = adfuller(close)[1]
    p_dclose = adfuller(close.diff().dropna())[1]
    p_lr = adfuller(rr)[1]
    arch_p = float(het_arch(rr, nlags=10)[1])
    lb_p = float(acorr_ljungbox(rr**2, lags=[10], return_df=True)["lb_pvalue"].iloc[0])
    diag_rows[t] = dict(adf_close=p_close, adf_dclose=p_dclose, adf_lr=p_lr, arch_p=arch_p, lb_sq_p=lb_p)
    print(f"{t:<8}{p_close:>13.4f}{p_dclose:>14.4f}{p_lr:>14.4f}{arch_p:>12.4f}{lb_p:>14.4f}")

print("\nReading: Close non-stationary (p>>0.05); dClose & log_return stationary (p~0);")
print("ARCH-LM & LB(r^2) p~0 => significant conditional heteroskedasticity (ARCH effects) for every ticker.")

# %% [markdown]
# ## 7. Correlation analysis

# %%
# Wide matrix of daily log returns, aligned on date.
wide = train.pivot_table(index="date", columns="ticker", values="log_return")[TICKERS].dropna()

# Fig 14 — cross-ticker return correlation heatmap.
fig, ax = plt.subplots(figsize=(6.5, 5.5))
corr = wide.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, square=True, ax=ax)
ax.set_title("Fig 14 — Daily log-return correlation (train)")
fig.tight_layout()
fig.savefig(FIG / "M2_fig14_corr_heatmap.png", dpi=120)
plt.close(fig)

# Fig 15 — rolling 63-day correlation of each stock with ^GSPC.
fig, ax = plt.subplots(figsize=(11, 6))
for t in ["AAPL", "AMZN", "NVDA"]:
    rc = wide[t].rolling(63).corr(wide["^GSPC"])
    ax.plot(wide.index, rc, lw=1.0, label=f"{t} vs ^GSPC")
ax.axhline(0.6, color="k", ls="--", lw=0.7, alpha=0.6)
ax.set_title("Fig 15 — Rolling 63-day correlation with ^GSPC (rises in crises)")
ax.set_ylabel("corr")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "M2_fig15_rolling_corr.png", dpi=120)
plt.close(fig)

# Fig 16 — correlation of MODEL_FEATURES with the NEXT-DAY target (the actual predictive signal for M3).
feat_corr = (
    train[MODEL_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
)
fig, ax = plt.subplots(figsize=(8, 10))
feat_corr.plot(kind="barh", ax=ax, color=np.where(feat_corr >= 0, "steelblue", "crimson"))
ax.set_title("Fig 16 — MODEL_FEATURES vs NEXT-DAY target_log_return (predictive signal)")
ax.set_xlabel("Pearson r")
fig.tight_layout()
fig.savefig(FIG / "M2_fig16_feature_corr.png", dpi=120)
plt.close(fig)

print("cross-ticker corr matrix:")
print(corr.round(2).to_string())
print("\nTop-5 |Pearson| of features vs next-day target:")
print(feat_corr.reindex(feat_corr.abs().sort_values(ascending=False).index).head(5).round(4).to_string())
print("saved fig 14-16")

# %% [markdown]
# ### Fig 17 — Pearson is the wrong lens: rank features by mutual information (nonlinear)
#
# Pearson only sees *linear* association. The whole reason to bring an LSTM (nonlinear) to M3 is that there may
# be nonlinear signal Pearson can't see. We rank the same features by **mutual information** and **Spearman**
# against the next-day target. (MI computed on warm-up-dropped rows, standardized features.)

# %%
from sklearn.feature_selection import mutual_info_regression

mi_df = train[MODEL_FEATURES + [TARGET]].dropna()
X_mi = mi_df[MODEL_FEATURES].values
y_mi = mi_df[TARGET].values
mi = mutual_info_regression(X_mi, y_mi, random_state=SEED)
spear = mi_df[MODEL_FEATURES].corrwith(mi_df[TARGET], method="spearman")
rank = (
    pd.DataFrame({"mutual_info": mi, "spearman": spear.values,
                  "pearson": feat_corr.reindex(MODEL_FEATURES).values}, index=MODEL_FEATURES)
    .sort_values("mutual_info", ascending=False)
)
# Shuffle control: MI of a permuted target = noise floor.
rng = np.random.default_rng(SEED)
mi_shuffled = mutual_info_regression(X_mi, rng.permutation(y_mi), random_state=SEED).mean()

fig, ax = plt.subplots(figsize=(8, 10))
rank["mutual_info"].iloc[::-1].plot(kind="barh", ax=ax, color="darkviolet")
ax.axvline(mi_shuffled, color="k", ls="--", lw=0.8, label=f"shuffled noise floor ~{mi_shuffled:.4f}")
ax.set_title("Fig 17 — MODEL_FEATURES ranked by mutual information vs next-day target")
ax.set_xlabel("mutual information (nats)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "M2_fig17_feature_mi.png", dpi=120)
plt.close(fig)

print("Top-6 features by mutual information (MI vs Pearson — note vol features dominate MI, hide in Pearson):")
print(rank.head(6).round(4).to_string())
print(f"shuffled-target MI noise floor: {mi_shuffled:.4f}")

# %% [markdown]
# ### Fig 18 — Leverage effect: volatility is asymmetric (down days → more next-day vol)
#
# Symmetric GARCH(1,1) assumes a +x% and −x% shock raise tomorrow's variance equally. Equities usually show a
# **leverage effect** (down moves raise vol more). We test it: mean |next-day return| conditioned on today's
# sign. If asymmetric, M3 should use **EGARCH / GJR-GARCH**, not plain GARCH.

# %%
lev_rows = []
fig, ax = plt.subplots(figsize=(8, 5))
for t in TICKERS:
    s = train[train.ticker == t].sort_values("date")
    rr = s["log_return"]
    nxt_abs = rr.abs().shift(-1)
    up = nxt_abs[rr > 0].dropna()
    dn = nxt_abs[rr < 0].dropna()
    tstat, p = stats.ttest_ind(dn, up, equal_var=False)
    lev_rows.append({"ticker": t, "vol_after_up_%": round(up.mean() * 100, 3),
                     "vol_after_down_%": round(dn.mean() * 100, 3), "t": round(tstat, 2), "p": p})
lev = pd.DataFrame(lev_rows).set_index("ticker")
lev[["vol_after_up_%", "vol_after_down_%"]].plot(kind="bar", ax=ax, color=["seagreen", "indianred"])
ax.set_title("Fig 18 — Next-day |return| after up vs down days (leverage effect)")
ax.set_ylabel("mean |next-day return| %")
fig.tight_layout()
fig.savefig(FIG / "M2_fig18_leverage.png", dpi=120)
plt.close(fig)
print("Leverage-effect test (vol after down vs up days):")
print(lev.to_string())

# %% [markdown]
# ### Weekday mean gradient (genuine day-of-week check, not the desynced decomposition)

# %%
wkday = (train.assign(dow=train.date.dt.dayofweek)
         .groupby("dow")["log_return"].mean() * 100)
wkday.index = ["Mon", "Tue", "Wed", "Thu", "Fri"]
print("Mean daily log-return by weekday (%):")
print(wkday.round(3).to_string())

# %% [markdown]
# **Takeaways:** all four return series are positively correlated; tech names correlate most with ^GSPC
# (0.65–0.78). Rolling correlation spikes toward 1 in 2020 — diversification fails exactly in crises.
#
# **Pearson vs MI (the key correction):** every feature's *Pearson* correlation with the next-day target is
# tiny (<0.075) — but that is a linear lens. By **mutual information**, the **volatility features dominate**
# (`parkinson_vol_21` ≈ 0.16, `realized_vol_21` ≈ 0.14, `rolling_std_20` ≈ 0.12) — roughly **3× the best
# linear feature and ~40× the shuffled-target noise floor** (Fig 17). This is the quantitative justification
# for a nonlinear model (LSTM) in M3: there is real, mostly-nonlinear, volatility-based signal that ARIMA-on-
# mean and a Pearson screen both miss.
#
# **Leverage effect (Fig 18):** next-day volatility is significantly higher after **down** days than up days
# for every ticker (e.g. ^GSPC ~0.86% vs ~0.67%, p≈1e-5) → the volatility response is **asymmetric** → M3's
# variance model should be **EGARCH or GJR-GARCH**, not symmetric GARCH(1,1).
#
# **Weekday:** a weak but real mean gradient (Mon/Tue slightly positive, Fri ≈ flat/negative) — supports
# keeping `dow_*` features even though calendar *seasonality* in the price level is negligible.

# %% [markdown]
# ## 8. Hypothesis tests (state -> test -> verdict -> implication)

# %%
results = []

# H1 — returns are normally distributed (Jarque-Bera per ticker).
for t in TICKERS:
    rr = train.loc[train.ticker == t, "log_return"].dropna()
    jb, p = stats.jarque_bera(rr)
    results.append(("H1 normal?", t, f"JB={jb:.0f}", f"p={p:.2e}", "REJECT normal" if p < 0.05 else "fail to reject"))

# H2 — volatility clusters (ARCH-LM lag 10).
for t in TICKERS:
    rr = train.loc[train.ticker == t, "log_return"].dropna()
    lm, p, _, _ = het_arch(rr, nlags=10)
    results.append(("H2 ARCH?", t, f"LM={lm:.0f}", f"p={p:.2e}", "ARCH present" if p < 0.05 else "none"))

# H3 — each tech stock corr with ^GSPC > 0.6.
for t in ["AAPL", "AMZN", "NVDA"]:
    c = wide[t].corr(wide["^GSPC"])
    results.append(("H3 corr>0.6?", t, f"r={c:.2f}", "", "YES" if c > 0.6 else "NO"))

# H4 — January effect, PER TICKER (audit fix: pooling 4 correlated tickers, r~0.65, violates t-test
# independence and inflates effective-n ~3x; pooled "no effect" also masked a per-ticker signal).
for t in TICKERS:
    s = train[train.ticker == t]
    jan = s[s.date.dt.month == 1]["log_return"].dropna()
    rest = s[s.date.dt.month != 1]["log_return"].dropna()
    tstat, p = stats.ttest_ind(jan, rest, equal_var=False)
    results.append(("H4 January effect?", t, f"t={tstat:.2f}", f"p={p:.3f}",
                    "different" if p < 0.05 else "no diff"))

# H5 — leverage effect (asymmetric volatility): vol after down days > after up days (per ticker).
for row in lev_rows:
    results.append(("H5 leverage?", row["ticker"],
                    f"dn={row['vol_after_down_%']}% up={row['vol_after_up_%']}%",
                    f"p={row['p']:.1e}", "ASYMMETRIC" if row["p"] < 0.05 else "symmetric"))

hyp = pd.DataFrame(results, columns=["hypothesis", "ticker", "stat", "p", "verdict"])
print(hyp.to_string(index=False))

# %% [markdown]
# **Verdicts:**
# - **H1 — normality:** REJECTED every ticker (JB p ≈ 0) → fat-tailed; ARIMA Gaussian errors misspecified
#   (consider Student-t errors in M3).
# - **H2 — volatility clustering:** ARCH present every ticker (p ≈ 0) → **GARCH-family variance model**.
# - **H3 — co-movement:** all tech names > 0.6 corr with ^GSPC → global model can share cross-series signal.
# - **H4 — January effect (per ticker, corrected):** pooling was invalid (cross-ticker corr ~0.65 →
#   non-independent obs, ~3× variance inflation). Tested per ticker: most show no January effect, but
#   **AMZN is individually significant** (the pooled test masked this). Net: no robust market-wide January
#   mean effect; `month` stays in MODEL_FEATURES for the LSTM regardless.
# - **H5 — leverage effect (new):** next-day volatility is significantly higher after down days for **every**
#   ticker → asymmetric → **EGARCH / GJR-GARCH** is the right M3 variance model, not symmetric GARCH(1,1).

# %% [markdown]
# ## 9. Modelling implications (hand-off to M3)
#
# - **Target & transform:** forecast `target_log_return` (next-day). ARIMA fits the **return** series with
#   `d=0` (returns already stationary in mean — ADF p≈0).
# - **ARIMA order:** return ACF/PACF are near-zero, so expect **small p, q** (try (1,0,1), (2,0,2), and a
#   pure-MA/AR around lag 1). A high-order ARIMA will overfit noise.
# - **ARIMA order (corrected):** the only non-trivial mean structure is a **negative lag-1** autocorrelation
#   (−0.162 ^GSPC), a mean-reversion/microstructure signature → try **ARIMA(1,0,0) / (0,0,1)**, but expect it
#   to roughly match the naive baseline net of costs. Do not over-order.
# - **Variance is the real signal → asymmetric GARCH:** ARCH-LM + ACF(r²) confirm strong volatility
#   clustering, and the **leverage effect** (H5: down days raise next-day vol more, p≈1e-5 all tickers) means
#   the variance model should be **EGARCH or GJR-GARCH**, not symmetric GARCH(1,1).
# - **LSTM justification = mutual information, not Pearson:** Pearson vs the next-day target is ~0 for all
#   features, but **MI ranks the volatility features ~3× higher** (`parkinson_vol_21` ≈0.16) and ~40× the
#   noise floor (Fig 17). That nonlinear, vol-based signal is exactly what a **global LSTM** (ticker one-hot +
#   30 MODEL_FEATURES) can exploit and ARIMA-on-mean cannot.
# - **Per-ticker vs global:** per-ticker ARIMA + asymmetric GARCH (GJR) (4 sets) for the classical track; one global LSTM for the
#   ML track (exploits cross-series co-movement, H3: corr 0.65–0.78).
# - **Suggested extra features for M3 (from this EDA):** a **downside/semi-deviation vol** (given the leverage
#   effect) and a **volatility-regime flag** (Fig 7 shows clear calm vs turbulent regimes) — neither is in the
#   current 30 MODEL_FEATURES.
# - **Metric:** RMSE/MAE on returns AND **directional accuracy** — a 52–55% hit rate is the realistic target.
#   Report a naive baseline first.
# - **Feature hygiene:** train on `MODEL_FEATURES` only; `dropna(subset=MODEL_FEATURES)`; never touch
#   val/holdout until M3's final step.

# %% [markdown]
# ## 10. Self-audit

# %%
present = [p.name for p in FIG.glob("M2_fig*.png")]
M2_REPORT = ROOT / "reports" / "milestones" / "M2.md"
audit = {
    # REAL sealing check: prove only train_fe was opened, never val/holdout (tracked via read_parquet logger).
    "sealed_train_only": (
        any("train_fe" in p for p in _OPENED_PARQUET)
        and not any(("val_fe" in p) or ("holdout_fe" in p) for p in _OPENED_PARQUET)
    ),
    "ge_10_visualizations": len(present) >= 10,
    "decompose_weekly_and_annual": (FIG / "M2_fig09_decompose_weekly.png").exists()
                                   and (FIG / "M2_fig10_decompose_annual.png").exists(),
    "decompose_confirmed_on_returns": ret_seas_var / ret_obs_var < 0.05,  # returns-based cross-check
    "adf_table_done": all(k in diag_rows["^GSPC"] for k in ("adf_close", "adf_dclose", "adf_lr")),
    "acf_returns_and_squared": (FIG / "M2_fig11_acf_returns.png").exists()
                               and (FIG / "M2_fig13_acf_sq_returns.png").exists(),
    "heteroskedasticity_tested": all("arch_p" in diag_rows[t] for t in TICKERS),
    "leverage_effect_tested": (FIG / "M2_fig18_leverage.png").exists() and len(lev_rows) == 4,
    "mutual_info_ranking": (FIG / "M2_fig17_feature_mi.png").exists(),
    "cross_ticker_corr": (FIG / "M2_fig14_corr_heatmap.png").exists(),
    "feature_vs_next_day_target": (FIG / "M2_fig16_feature_corr.png").exists(),
    "ge_5_hypotheses": hyp["hypothesis"].nunique() >= 5,
    "january_tested_per_ticker": ((hyp.hypothesis == "H4 January effect?") & (hyp.ticker != "pooled")).sum() == 4,
}
for k, v in audit.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print(f"opened parquet files: {_OPENED_PARQUET}")
assert all(audit.values()), "M2 self-audit failed!"
print(f"\nAll {len(audit)} M2 self-audit checks passed. {len(present)} figures saved. Ready for M3.")

# Restore the original read_parquet (clean up the logging wrapper).
pd.read_parquet = _orig_read_parquet
