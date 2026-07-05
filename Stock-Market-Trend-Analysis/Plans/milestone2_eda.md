# Milestone 2 — Exploratory Data Analysis (EDA)
**Project:** Stock Market Trend Analysis | **Week:** 9 | **Weight:** 7.5% of course | **Seed:** 42

> **How to use this file:** You are completing Milestone 2 of the Stock Market Trend Analysis capstone. M1 has already produced cleaned, leakage-safe parquet files. Your job: surface patterns, generate hypotheses, and decide which models M3 should try. Every figure needs a caption AND a one-sentence takeaway, or it doesn't count toward the rubric.

---

## 0. Prerequisites

- M1 outputs exist: `data/processed/{train_fe, val_fe, holdout_fe}.parquet`, `data/processed/feature_dictionary.md`.
- Libraries: `pandas pyarrow numpy matplotlib seaborn statsmodels scipy`.
- Open `notebooks/02_eda.ipynb`. Set `SEED=42`, `np.random.seed(SEED)`. Load only `train_fe.parquet` — never peek at val/holdout in EDA.

## 1. Summary Statistics

Produce three tables in `M2.md`:

1. **Overall:** rows, date range, n_tickers, mean/median/std/min/max of `Close` and `log_return`.
2. **Per ticker** (4 rows): total trading days, mean daily log return (annualised: `× 252`), std (annualised: `× √252`), min/max single-day return, Sharpe-like ratio.
3. **Per year** per ticker: mean log return, vol, max drawdown.

**Takeaway sentence required** under each table — e.g., "NVDA has 2.3× the annualised volatility of ^GSPC, consistent with single-stock vs index behaviour."

## 2. Univariate Distributions

- Histogram of `log_return` linear + log scale (overlaid normal curve for reference) → `M2_fig01_logret_hist.png`.
- Q-Q plot of `log_return` vs normal → `M2_fig02_qq.png` (expect heavy tails).
- Boxplot of `log_return` by ticker → `M2_fig03_logret_box.png`.
- Boxplot of `log_return` by `month` (pooled across tickers) → `M2_fig04_month_box.png`.

## 3. Time-Series Visualizations

- Closing-price plots for all 4 tickers (faceted small-multiples) → `M2_fig05_price_facets.png`.
- Normalised price (`close / close[0]`) overlaid on one axis → `M2_fig06_normalised.png` (shows relative growth).
- 21-day rolling volatility per ticker → `M2_fig07_rolling_vol.png` (volatility clustering should be visible).
- Single (ticker=NVDA) zoomed plot, last 2 yr only → `M2_fig08_nvda_zoom.png`.

## 4. Seasonality Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose
s = train_fe.query("ticker == '^GSPC'").set_index("date")["Close"]
weekly = seasonal_decompose(s, model="additive", period=5)       # 5 trading days
annual = seasonal_decompose(s, model="multiplicative", period=252)  # 252 trading days
```
Plot trend / seasonal / resid for both periods → `M2_fig09_decompose_weekly.png`, `M2_fig10_decompose_annual.png`. Caption each with the dominant pattern found. (Likely finding: weekly seasonality is weak; annual has a discernible trend with mild cyclicality.)

## 5. Autocorrelation

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
r = train_fe.query("ticker == '^GSPC'").set_index("date")["log_return"]
plot_acf(r.dropna(), lags=60)
plot_pacf(r.dropna(), lags=60)
```
Save → `M2_fig11_acf_returns.png`, `M2_fig12_pacf_returns.png`. **Expected finding:** log-return ACF should be near zero at all lags (efficient market). However, **squared returns** (volatility proxy) should show strong autocorrelation (volatility clustering). Also plot `plot_acf(r**2, lags=60)` → `M2_fig13_acf_sq_returns.png`.

State which lags (if any) justify lag features for ARIMA in M3. Likely: `r` itself is hard to predict from its own lags, but `|r|` and `r²` are highly predictable — motivates **GARCH** as a stretch goal in M3.

## 6. Stationarity (ADF)

```python
from statsmodels.tsa.stattools import adfuller
for ticker in TICKERS:
    s = train_fe.query("ticker == @ticker").set_index("date")
    print(f"{ticker:6s}  Close p-value:    {adfuller(s.Close.dropna())[1]:.4f}")
    print(f"{ticker:6s}  Δ Close p-value:  {adfuller(s.Close.diff().dropna())[1]:.4f}")
    print(f"{ticker:6s}  log_return p-val: {adfuller(s.log_return.dropna())[1]:.4f}")
```
Tabulate p-values in `M2.md`. Interpret: `Close` should be ≫ 0.05 (non-stationary, unit root); `Δ Close` and `log_return` should be ≈ 0 (stationary). This is the textbook result that justifies fitting ARIMA on returns, not prices.

## 7. Correlation Analysis

- Pearson correlation matrix of the 4 tickers' log returns → `M2_fig14_corr_heatmap.png`. **Expected:** all positive (0.4–0.8), strongest within tech stocks.
- Rolling 63-day correlation of (AAPL, ^GSPC), (AMZN, ^GSPC), (NVDA, ^GSPC) → `M2_fig15_rolling_corr.png`. Expected: correlation rises in crisis periods (COVID March 2020, late 2022 rate-hike volatility).
- Pearson heatmap of engineered numerical features vs `log_return` → `M2_fig16_feature_corr.png`. Most lagged returns will have near-zero correlation; that's the EMH at work.

## 8. Hypothesis Generation (≥ 4)

For each hypothesis: **state → test → verdict → implication**.

Mandatory four (add more if interesting):

| # | Hypothesis | Test | Verdict | Implication for M3 |
|---|---|---|---|---|
| H1 | Daily log returns are normally distributed | Jarque-Bera test on each ticker's returns | ✅/❌ + effect size | If ❌ (expected), ARIMA Gaussian-error assumption is mis-specified — note as a limitation |
| H2 | Volatility clusters: today's `|r|` predicts tomorrow's `|r|` | ACF of `r²` at lag 1 + Ljung-Box | ✅/❌ | Justifies GARCH as stretch in M3 |
| H3 | Tech stocks have correlation > 0.6 with ^GSPC | Pearson r on overlapping date range | ✅/❌ | If yes, a global model can leverage cross-series signal |
| H4 | January effect: returns in January differ from other months | Welch t-test on `log_return ~ is_january` | ✅/❌ | If yes, include `month` as a categorical feature |

## 9. Model Recommendation for M3

End `M2.md` with a 4–6 line paragraph titled **"Modeling implications"**:
- Confirms ARIMA should be fit on **log returns**, not prices.
- Suggests ARIMA orders `(p, d, q)` based on ACF/PACF (likely small p and q, d=0 because returns are already stationary).
- Recommends LSTM input: 60-day sequence of `log_return` + engineered features → predict next-day `log_return`.
- States whether a per-ticker model (4 ARIMA models) or a global model (1 LSTM across all 4 tickers) is appropriate — likely **per-ticker ARIMA** + **global LSTM**.
- Notes the directional-accuracy metric will be more interpretable than raw RMSE for this domain.

## Expected Outputs

- `notebooks/02_eda.ipynb` (runs top-to-bottom, seed=42)
- `reports/figures/M2_fig01..16_*.png` (≥ 13 figures, each with caption + takeaway)
- `reports/milestones/M2.md` with sections matching this file (Summary → Univariate → Time-series → Decomposition → Autocorrelation → Stationarity → Correlation → Hypotheses → Modeling implications → Self-Audit)

## Self-Audit Table (paste into M2.md, all must be ✅)

| Criterion | Status | Evidence |
|---|---|---|
| ≥ 10 distinct visualizations | ☐ | … |
| Decomposition for both weekly (5) and annual (252) periods | ☐ | … |
| ADF table for `Close`, `Δ Close`, `log_return` per ticker | ☐ | … |
| ACF of returns + ACF of squared returns (volatility clustering) | ☐ | … |
| Cross-ticker correlation heatmap | ☐ | … |
| ≥ 4 hypotheses with state/test/verdict | ☐ | … |
| Every figure has caption + takeaway | ☐ | … |
| Modeling implications paragraph present | ☐ | … |
| Notebook reproducible (seed=42) | ☐ | … |

## Forbidden

- Touching `val_fe.parquet` or `holdout_fe.parquet` (peeking = leakage).
- Figures without captions or without a one-line takeaway.
- "Interesting!" / "Looks normal!" — every observation must be quantified.
- Reporting ACF of *prices* as meaningful (prices are non-stationary; ACF is trivially ≈ 1 for many lags).

## Hand-off to Milestone 3

> **State of project after M2:** EDA confirms log returns are stationary, weakly autocorrelated, but `|r|` is strongly autocorrelated (volatility clustering). Tech stocks correlate 0.5–0.8 with ^GSPC. Recommended modeling track: per-ticker ARIMA on log returns + a global LSTM trained on engineered features. Optional GARCH stretch if volatility prediction is in scope. All artifacts in `reports/figures/M2_*` and `reports/milestones/M2.md`.
