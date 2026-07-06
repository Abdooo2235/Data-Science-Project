# Project Decisions & Architecture Report

**Project:** Stock Market Trend Analysis — next-day log-return forecasting
**Tickers:** `^GSPC`, `AAPL`, `AMZN`, `NVDA` · **Window:** 2016-05-23 → 2026-05-20 (pinned, 2,513 rows/ticker)
**Owner:** Ali Agela · **Seed:** 42 · **Status:** M1 + M2 + M3 + M3.5 + M4 complete, all multi-agent-audited. The attention LSTM was trained on a Colab GPU (2026-07-06); its predictions are in `holdout_predictions.parquet`. The exec-summary table below and the whole project are current as of the M4 evaluation.
**Sources:** `Plans/progress_checklist.md` (Decisions Log), `reports/milestones/M1–M3.md`, `Plans/milestone1–4_*.md`, the three notebooks, `data/processed/feature_roles.json`, `feature_dictionary.md`, `data/raw/snapshot_hashes.json`

---

## 1. Executive summary

This capstone asks a deliberately falsifiable question: *given 10 years of daily price data for the S&P 500 index and three large-cap tech stocks, can we forecast the next-day log return better than a naive baseline?* The pipeline spans four milestone specs — data collection and preprocessing (M1), exploratory data analysis (M2), model building (M3), and evaluation/presentation (M4) — executed as Jupytext `.py` notebooks synced to `.ipynb` via `scripts/py_to_nb.py`, snapshotted to committed CSVs for reproducibility, and reviewed at every milestone by independent expert agents.

The honest headline result is that **no model demonstrates economically usable out-of-sample skill.** On the sealed holdout (2025-01-02 → 2026-05-20, opened exactly once), pooled across tickers:

| Model | Holdout RMSE | vs naive (Diebold-Mariano p) | DirAcc | DirAcc p |
|---|---|---|---|---|
| `naive_zero` | 0.02109 | n/a | n/a | n/a |
| ARIMA | **0.02107** | 0.783 (tie) | 0.514 | 0.294 (coin flip) |
| GJR-GARCH-t (mean) | 0.02106 | 0.589 (tie) | **0.543** | **0.002 (significant)** |
| LightGBM (baseline, un-tuned) | 0.02168 | **0.003 (WORSE)** | 0.483 | 0.226 (coin flip) |
| LightGBM (tuned, M3.5) | **0.02107** | 0.533 (tie) | **0.542** | **0.002 (significant)** |
| LSTM+Attention (real GPU) | 0.02031 | **0.004 (WORSE)** | 0.491 | 0.574 (coin flip) |

Updated after the M3.5 hardening pass and the real Colab GPU run (2026-07-06). ARIMA ties naive on RMSE and is a directional coin flip. The un-tuned LightGBM overfits (holdout worse than naive, DM p = 0.003). The M3.5 Optuna tuning on a purged walk-forward CV fixed that overfit: the tuned LightGBM was promoted because it went from worse-than-naive to naive-tying (DM vs baseline p = 0.0025). Two models keep a statistically significant directional edge, GJR-GARCH at 0.543 and the tuned LightGBM at 0.542 (both p = 0.002), but both predict up on 99 to 100 percent of days, so that accuracy is the holdout up-day rate (0.542), the market's drift in a calm period rather than timing. The attention LSTM edge did not reproduce on the real GPU run (0.542 on an earlier forced-CPU run, 0.491 on the GPU run, a coin flip). No model beats naive on RMSE, and the one edge that survives is drift that transaction costs erase (an illustrative ^GSPC timer switches 174 times and collapses to Sharpe 0.33 at 10 bps, below buy-and-hold's 0.99, daily returns not significant at t = 1.38, p = 0.168). The efficient-market ceiling holds, and this negative result is the deliverable. See `reports/milestones/M4.md` for the full evaluation.

---

## 2. Milestone-by-milestone decisions

### M1 — Data collection & preprocessing *(complete; self-audit 19/19)*

**What was decided.** Forecast **log return, not `Close`**; source from **Yahoo Finance via `yfinance` with a direct chart-API fallback**, snapshotted to committed CSVs; **retain and flag** outliers rather than cap them; split **chronologically** (train ≤ 2023-12-29, val = 2024, holdout = 2025+); build a **next-day target** and an explicit 30→39-column `MODEL_FEATURES` set (`feature_roles.json`).

**Why (from the Decisions Log).** ADF confirms `Close` is non-stationary (p ≥ 0.86) while `log_return` is stationary (p = 0.0000), so ARIMA can fit returns with `d = 0` — "standard finance practice." A random split would leak the future; a chronological split with holdout = 2025+ reserves the AI cycle for an unbiased final test. `yfinance` + a committed CSV snapshot is the "reproducibility anchor: re-runs read from CSV not the live API." Outliers encode "real regime events — capping would destroy the signal we need to learn." The direct `query1.finance.yahoo.com` chart API is a fallback because the consent endpoint `fc.yahoo.com` was blocked from the local sandbox.

**What was produced.** 2,513 rows/ticker over 2016-05-23 → 2026-05-20 (SHA256-recorded in `snapshot_hashes.json`); a 55-column leakage-safe frame written to `train_fe.parquet` (7,660 rows), `val_fe.parquet` (1,008 rows), `holdout_fe.parquet` (1,384 rows); plus `feature_dictionary.md`, `feature_roles.json`, and `feature_nan_report.md`.

**Key metrics.** ADF `log_return` p = 0.0000 all tickers; `Adj Close` p = 0.86–0.998. Ljung-Box(r²) lag-1 p ≈ 0 all with ACF(r²) up to 0.453 (^GSPC) — i.e. **stationary in mean but conditionally heteroskedastic**. Log-return kurtosis 5.1–16.8 (^GSPC fattest, COVID-driven; ex-COVID 6.46). Rolling-MAD tail-event flag rate 5.85%–7.16%.

### M2 — Exploratory data analysis *(complete; self-audit 13/13, 18 figures, 5 hypotheses)*

**What was decided.** EDA reads **`train_fe.parquet` only** (val/holdout sealed); rank features by **mutual information**, not just Pearson; test the **January effect per ticker**, not pooled; add a **leverage-effect test**; recommend **asymmetric GARCH (EGARCH/GJR)** for the M3 variance model.

**Why (from the Decisions Log).** Pearson is ~0 for every feature and "hid the signal"; MI shows the volatility features carry ~67× the noise floor. Pooling four tickers correlated at ~0.65 "inflated effective-n ~3×," masking AMZN's January significance. Down days raise next-day volatility more, so "symmetric GARCH would miss the asymmetry."

**What was produced.** 18 figures (`M2_fig01..18`), including the MI ranking (Fig 17) and leverage test (Fig 18); five hypotheses with verdicts; a modelling-implications hand-off.

**Key metrics.** Annualised vol ^GSPC 18.8% → NVDA 49.6% (max drawdown −33.9% → −66.3%). Cross-ticker correlation ^GSPC–AAPL 0.78 (range 0.58–0.78). ACF(return) lag-1 = −0.162; ACF(return²) lag-1 = +0.487. MI: `parkinson_vol_21` 0.160 vs shuffle floor 0.0024 (~67×) vs its own Pearson ~0.01. Hypotheses: H1 normality **REJECTED** (JB 2,129–21,958, p ≈ 0); H2 ARCH **CONFIRMED** (LM 65–757, p ≈ 0); H3 tech corr > 0.6 **YES**; H4 January significant **AMZN only** (t = 2.24, p = 0.027; the invalid pooled test gave p = 0.103); H5 leverage **ASYMMETRIC all** (p 7e-6 … 0.035).

### M3 — Model building *(core complete; self-audit 19/19 post-enhancement; LSTM full-train pending Colab)*

**What was decided.** Score **baselines first**, then per-ticker **ARIMA with order selected by BIC**, per-ticker **GJR-GARCH(1,1) Student-t** (evaluated as a variance model), a **global attention-LSTM** (Colab), and — in the enhancement pass — **LightGBM on 39 features + 9 exogenous macro series** with a z-scored directional ensemble; the **holdout is opened exactly once**.

**Why (from the Decisions Log).** BIC over AIC because "AIC over-selected (2,0,2); BIC gives parsimonious (1,0,0)/(0,0,1)." GJR-GARCH because M2 proved the leverage effect. Significance tests (binomial + Diebold-Mariano) because "a 55% DirAcc read as skill without a test." Exogenous data is "the only break from price-history-only," fetched leakage-safe (market-priced daily, known at close *t*, no CPI-style release lag). LightGBM was left "deliberately un-tuned … so the overfit is the lesson."

**What was produced.** `models/{arima,garch}_*.pkl`, `lgbm_global{,_final}.txt`, `lstm_attention_smoke.keras`, `val_scores.csv`, and `holdout_predictions.parquet` (with `y_pred_arima`, `y_pred_garch`, `y_pred_lgb`, `y_pred_lstm` placeholder).

**Key metrics.** Baselines (val): `naive_zero` RMSE 0.02041; `persistence` DirAcc 0.512; `moving_avg_20` 0.520. ARIMA orders: ^GSPC/AMZN/NVDA (1,0,0), AAPL (0,0,1). GJR-GARCH γ > 0 all (^GSPC 0.233), ν ≈ 5. Val leader was GJR-GARCH-mean (DirAcc 0.555, p = 0.0006) — but selection-biased, so only the holdout (Section 1) counts.

### Enhancement pass *(2026-06-17)*

Added the nine exogenous features (`vix_level`, `vix_log_change`, `vix_z_60`, `is_high_vol_regime`, `tnx_level`, `tnx_change`, `term_spread`, `term_spread_change`, `dxy_log_return`), LightGBM, a Bahdanau-style attention layer on the LSTM, and a z-scored ensemble. **The instructive finding is the overfit:** exogenous data is genuinely informative in-sample (8 of LightGBM's top-15 importances are exogenous) and lifts val DirAcc to 0.539 (p = 0.015), but does **not** generalize — holdout RMSE 0.02165 > naive, DM p = 0.003, DirAcc 0.499. More data and more model capacity bought no out-of-sample lift, sharpening the EMH conclusion.

---

## 3. The multi-agent audit process

Every milestone was reviewed by **independent expert agents** — an **Investment Researcher** and a **Model QA Specialist** on all three, plus a **Data Engineer** on M1 — who reproduced each headline number before flagging issues by severity. This adversarial discipline is a core part of the story: several audit findings were BLOCKERs that invalidated the stated task until fixed.

**M1 (three agents).** Material findings and fixes:
- **BLOCKER — no next-day target / off-by-one leak (Model QA):** the target was the same-row `log_return`, making the stated "next-day forecast" unbuildable and same-day columns leaky. Fix: `target_log_return = groupby(ticker).log_return.shift(-1)`, built per split, verified `target[t] == log_return[t+1]`.
- **BLOCKER — non-deterministic date window (Data Eng + Model QA):** a `today`-anchored window made a fresh clone/Colab non-reproducible. Fix: pinned `START=2016-05-23`, `END=2026-05-21` (exclusive), recorded in `snapshot_hashes.json`.
- **BLOCKER — `Volume` dtype drift (Data Eng + Model QA):** `yfinance` returned int64, the fallback float64, the dictionary claimed float64. Fix: a single `enforce_schema()` on all fetch paths + post-concat assertion.
- **HIGH — global-mean target-encoding leakage (M1 v2 audit):** `ticker_mean_log_return` was a global `groupby(ticker).mean()`, so every early-train row saw the whole-train mean. Fix: `expanding(min_periods=20).mean().shift(1)` (~1,894 unique values/ticker, was 1).
- **HIGH — IQR outliers over-flagged NVDA (M1 v2 audit):** fixed-quantile IQR flagged 14% of NVDA. Fix: rolling 60-day MAD, threshold 3.5 (flag rates 5.9%–7.2%).
- **HIGH — "stationary" overclaim (Investment Researcher):** returns are conditionally heteroskedastic. Fix: added Ljung-Box(r²) + ACF(r²), reworded to "stationary in mean," flagged ARIMA+GARCH for M3.
- Plus: `MODEL_FEATURES`/`feature_roles.json` role separation, `is_warmup` flag + NaN report, tail-event renaming with NaN warm-up, Colab bootstrap, and partial-ticker-failure handling.

**M2 (two agents).** `pooled January invalid → per-ticker` (unmasked AMZN p = 0.027); `Pearson hides the signal → MI ranking` added (Fig 17); `vacuous sealing check → real read_parquet-path tracking`; leverage test added (H5/Fig 18) driving the EGARCH/GJR recommendation; ACF lag-1 reframed as negative microstructure; returns-based decomposition cross-check.

**M3 (two agents).** `off-by-one in the ARIMA one-step (BLOCKER)` — the forecast was taken before appending today's return, predicting the same day but scored against the next (val ARIMA DirAcc 0.536 → 0.521, the old edge was the artifact; holdout RMSE 0.02129 → 0.02107); `BIC not AIC`; binomial + Diebold-Mariano significance tests added; GARCH evaluated by QLIKE + Mincer-Zarnowitz on variance (not point RMSE) and persisted via `pickle`; costed backtest with Sharpe/maxDD/t-stat. **Enhancement re-audit:** confirmed exogenous leakage-clean (VIX same-day corr −0.70 vs next-day +0.05 = no label contamination), **re-tested GARCH out-of-sample** (yielding the 54.3% p = 0.002 survivor), and corrected the **"EGARCH" mislabel to GJR-GARCH (o=1)** everywhere, softened the VIX release-timing wording, and documented LightGBM as deliberately un-tuned.

---

## 4. Leakage-prevention invariants

The pipeline enforces five guards, each agent-verified:

1. **Expanding-mean target encoding.** `ticker_expanding_mean/std = expanding(min_periods=20).{mean,std}().shift(1)` — each row sees only its strict past (`np.allclose` reproduced independently by the agents).
2. **`.shift(1).rolling(...)` on every rolling/lag feature.** SMAs, realized vols, `rolling_std_20`, Parkinson vol, RSI, `volume_z_20`, and all lags shift *before* rolling — no same-day value enters its own window.
3. **Next-day target built per split.** `target_log_return = groupby(ticker).log_return.shift(-1)` is computed *after* splitting, so the last row of each split per ticker is NaN (4 NaN targets/split = 1/ticker) — no label leaks across a boundary.
4. **Fit on train only.** The LSTM `StandardScaler` is fit on train exclusively; GARCH uses `last_obs = first_val` so parameters come only from train; ARIMA one-step is causal (append-then-forecast); the target is never in the feature matrix.
5. **Holdout opened exactly once**, in the final M3 step; a logged `read_parquet` wrapper + self-audit check *prove* M2 touched only `train_fe.parquet`.

**Exogenous checks (enhancement pass).** VIX matches the raw snapshot exactly; same-day VIX correlation −0.70 vs next-day +0.05 proves no label contamination; forward-fill is backward-safe and ticker-scoped; `term_spread` (10y − 3m) is economically correct, not a unit artifact; the attention layer pools over the time axis correctly; all features are market-priced daily and known at close *t* (no release-lag lookahead).

---

## 5. Key quantitative findings

**Significance-tested val→holdout skill decay** is the central quantitative story: the only significant val edge (GJR-GARCH-mean DirAcc 0.555, p = 0.0006) was selected on val (selection bias), and the sealed holdout shrinks it to a marginal 0.543 (p = 0.002) while ARIMA (0.521 → 0.514) and LightGBM (0.539 → 0.499) decay to coin flips. Every model ties naive on RMSE (DM p ≥ 0.07 on val; ≥ 0.59 on holdout except LightGBM, which is *worse*).

**Volatility clustering** is confirmed everywhere: ARCH-LM p ≈ 0 for all four tickers, Ljung-Box(r²) p ≈ 0, and ACF(r²) lag-1 = 0.487 (^GSPC) — the single most important result for model selection. **The leverage effect** is significant for all four (down-day next-day vol > up-day, p 7e-6 … 0.035), and the fitted GJR-GARCH recovers it internally (γ > 0 all; ^GSPC 0.233).

**MI vs Pearson** quantifies why a nonlinear model was warranted at all: `parkinson_vol_21` carries MI 0.160 (~67× the 0.0024 noise floor, ~3× the best linear feature) while its Pearson is ~0.01 — real, mostly-nonlinear, volatility-based structure invisible to a linear screen. **Fat tails** are pervasive (log-return kurtosis 5.1–16.8; GARCH ν ≈ 5), justifying Student-t errors over Gaussian.

Taken together, these are consistent, textbook **EMH evidence**: next-day *mean* is near-unpredictable while next-day *variance* is strongly predictable — and even the predictable variance, converted to a directional signal, is uneconomic after costs.

---

## 6. Limitations & outstanding work

- **LSTM trained on Colab GPU (2026-07-06).** The full 2-layer/64-unit/20-epoch attention-LSTM ran; its holdout directional accuracy is 0.491 (p = 0.574, a coin flip), and the earlier forced-CPU 0.542 did not reproduce. So the deep arm exists and its honest verdict is in.
- **M4 complete** (11/11): final performance table vs a perfect-foresight bound, top-10-worst error narratives, confusion matrix, ethics/misuse audit with the explicit "not investment advice" disclaimer, and a 12-slide HTML deck are all delivered. See `reports/milestones/M4.md` and `reports/M4_presentation.html`.
- **Finance-specific limits.** The EMH bounds predictability from prices alone; the holdout is a **single calm regime** (2025–26, lower vol than train), so no claim generalizes across bull/bear; transaction costs erase the marginal GARCH edge; the four tickers are **survivors** (survivorship bias) and `^GSPC` is not directly tradeable; daily granularity hides intraday signal; multiple comparisons (4 tickers × orders × models) mean any single "win" should be read skeptically. The GARCH variance was scored against a noisy squared-return proxy; M4 could add a range-based realized variance and a VaR/Kupiec exceedance backtest.
