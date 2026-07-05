# Stock Market Trend Analysis — Progress Checklist
**Project:** Stock Market Trend Analysis | **Dataset:** Yahoo Finance via `yfinance` (`^GSPC`, `AAPL`, `AMZN`, `NVDA`) | **Seed:** 42

> **How to use this file with ANY AI model (Antigravity, Claude, GPT, local 7B, etc.):**
> 1. Paste this file + the relevant `milestoneN_*.md` into the model's context.
> 2. Ask: *"Execute the next unchecked item in Milestone N. When done, return this checklist file with that box ticked ([x]) and add one line to the Decisions Log."*
> 3. Review the diff, accept or revise, repeat.
>
> Boxes: `[ ]` = not started, `[~]` = in progress, `[x]` = done. Update `Overall status` at the top of each milestone section as you go.

---

## Project metadata

| Field | Value |
|---|---|
| Owner | Ali Agela |
| Start date | 2026-05-21 |
| Current week (1–14) | 6 |
| Current milestone | M3 (M1+M2 complete) |
| Last updated | 2026-05-21 (M1 complete) |

## Overall progress

- [x] **Milestone 1 — Data Collection & Preprocessing** (Week 6) — 13 / 13 (v3 multi-agent audit 2026-06-17: 19/19 self-audit PASS; v2 was 12/12, v1 10/10)
- [x] **Milestone 2 — Exploratory Data Analysis** (Week 9) — 11 / 11 (2026-06-17: multi-agent reviewed + fixed → 18 figures, 5 hypotheses, 13/13 self-audit PASS)
- [~] **Milestone 3 — Model Building (ARIMA+GARCH vs LSTM)** (Week 11) — 12 / 13 (core done + multi-agent-audited 2026-06-17, 15/15 self-audit; LSTM full-train pending Colab)
- [ ] **Milestone 4 — Evaluation & Presentation** (Week 13) — 0 / 11

---

## Milestone 1 — Data Collection & Preprocessing  *(status: ☑ complete)*

**File:** `milestone1_data_collection_preprocessing.md` | **Report:** [`reports/milestones/M1.md`](../reports/milestones/M1.md)

- [x] Repo layout created (`data/`, `notebooks/`, `models/`, `reports/`, `Plans/`, `References/`, `scripts/`)
- [x] `requirements.txt`, `README.md`, `.gitignore` written
- [x] Notebook `01_data_collection_preprocessing.ipynb` runs top-to-bottom with seed=42 (verified locally; converted from `.py` source via `scripts/py_to_nb.py`)
- [x] Snapshot CSVs saved: `data/raw/{GSPC, AAPL, AMZN, NVDA}_10y.csv` — 2,513 rows each, 2016-05-23 → 2026-05-20, SHA256 recorded in `data/raw/snapshot_hashes.json`
- [x] Problem definition written (primary question + 3 stakeholder groups)
- [x] Initial inspection logged: 0 nulls, 0 dupes, identical date span across all 4 tickers
- [x] Missing-value policy stated with justification (forward-fill within ticker on rare halts; documented even though current data has zero gaps)
- [x] Outlier strategy documented (IQR on daily returns + |r|>5% flag; retain + flag — 13 extreme days on ^GSPC vs 232 on NVDA)
- [x] 6 engineered feature groups (datetime, cyclical, returns, lag, rolling, target-encoding); 39 total columns in `feature_dictionary.md`
- [x] ADF p-values reported per ticker — all 4 `Close` series non-stationary (p ≥ 0.86); all 4 `log_return` series stationary (p = 0.0000)
- [x] Time-based split saved as parquet — `train_fe` 7,660 rows ≤ 2023-12-29, `val_fe` 1,008 rows 2024, `holdout_fe` 1,384 rows 2025-01-02 → 2026-05-20
- [x] Leakage warning + assertion in notebook (assertion passes)
- [x] `reports/milestones/M1.md` complete (9 sections + 10/10 self-audit checks PASS)

## Milestone 2 — Exploratory Data Analysis  *(status: ☑ complete)*

**File:** `milestone2_eda.md` | **Report:** [`reports/milestones/M2.md`](../reports/milestones/M2.md) | **Reads:** `train_fe.parquet` only.

- [x] Notebook `02_eda.ipynb` runs top-to-bottom with seed=42 (16 figures, 9/9 self-audit PASS)
- [x] Summary stats: overall + per-ticker (annualised return/vol/Sharpe-like/drawdown) + per-year
- [x] Univariate distributions: hist + normal overlay, Q-Q, boxplots by ticker and month (Figs 1-4)
- [x] Time-series: price facets + normalised growth + 21d rolling vol + NVDA zoom (Figs 5-8)
- [x] Seasonal decomposition period=5 AND period=252 on ^GSPC (Figs 9-10) — seasonality negligible (weekly 0.11% of range)
- [x] ACF + PACF returns AND ACF squared returns (Figs 11-13) — ACF(r) lag1 -0.162, ACF(r²) lag1 +0.487
- [x] ADF (Close, ΔClose, log_return) + ARCH-LM + Ljung-Box(r²) per ticker — heteroskedasticity confirmed all
- [x] Cross-ticker correlation heatmap (^GSPC-AAPL 0.78) + rolling 63d correlation (Figs 14-15)
- [x] 4 hypotheses tested: H1 normality REJECTED, H2 ARCH CONFIRMED, H3 corr>0.6 YES, H4 January no-effect (p=0.103)
- [x] Modeling-implications paragraph (per-ticker ARIMA d=0 small p/q + GARCH(1,1) + global LSTM; directional-accuracy metric)
- [x] `reports/milestones/M2.md` complete (every figure captioned + takeaway)
- [x] **Improvement over spec:** feature-correlation computed vs NEXT-DAY target (Fig 16), not same-day — max |corr| 0.075 (EMH evidence)
- [x] **Multi-agent audit applied** (Investment Researcher + Model QA): added MI ranking (Fig 17), leverage test (H5/Fig 18), per-ticker January, real sealing check, returns-decomp cross-check; EGARCH/GJR recommendation

## Milestone 3 — Model Building (ARIMA+GARCH vs LSTM)  *(status: ☑ core complete; LSTM full-train pending Colab)*

**File:** `milestone3_model_building.md` | **Report:** [`reports/milestones/M3.md`](../reports/milestones/M3.md) | **Reads:** train+val; holdout opened once at final step.

- [x] Notebook `03_model_building.ipynb` runs top-to-bottom, seed=42 + tf.random.set_seed(42); 10/10 self-audit
- [x] RMSE / MAE / directional-accuracy helpers unit-tested on toy arrays
- [x] Baseline `naive_zero` (RMSE 0.02041), `persistence` (DirAcc 0.512), `moving_avg_20` (0.520) scored on val
- [x] 4 per-ticker ARIMA fit (order by AIC; ^GSPC/AAPL (2,0,2), AMZN/NVDA (1,0,0)) + residual Ljung-Box; one-step rolling val (RMSE 0.02039, DirAcc 0.536)
- [x] **GJR-GARCH(1,1) Student-t** per ticker (asymmetric, M2 leverage effect): γ>0 all (^GSPC 0.233), ν≈5; OOS via `last_obs` (no leak)
- [x] Global LSTM: 60-day sequences, ticker one-hot, scaler train-only, shuffle=False — **2-epoch local smoke-test PASS**; full 20-epoch config documented for Colab (real number pending)
- [x] Model comparison table in `M3.md` (baselines + ARIMA + GARCH + LSTM-smoke); saved `models/val_scores.csv`
- [x] Diagnostics: DirAcc-by-model fig, ARIMA residual ACF, backtest fig
- [x] Final retrain ARIMA on train+val, one-step holdout, save `holdout_predictions.parquet` (pooled RMSE **0.02107 ≈ naive 0.02109, DM p=0.78**; DirAcc 0.514, binomial p=0.29 — **no significant OOS skill**)
- [x] Costed long-only backtest: B&H Sharpe 0.99 vs strategy 0.33 @10bps (175 switches); daily t-stat p=0.168 — honest underperformance
- [x] Leakage assertions pass (agent-verified: split order + scaler train-only + target not in features + GARCH last_obs)
- [x] `reports/milestones/M3.md` complete — **honest EMH write-up: no significant out-of-sample skill**
- [x] **Audit fixes applied:** off-by-one one-step alignment (BLOCKER), BIC order selection, binomial+Diebold-Mariano significance, GARCH QLIKE/MZ + std-resid LB, GARCH pkls persisted, error-by-ticker/month figs, real audit checks (15/15)
- [x] **LSTM full training done** (FULL_TRAIN flag auto-on Colab GPU; 20-epoch attention; CPU-verified locally): val DirAcc 0.539 (p=0.036), holdout DirAcc 0.542 (p=0.005) — weak-significant like GARCH, RMSE worse than naive. Artifacts: lstm_attention_final.keras, lstm_val_metrics.json, y_pred_lstm filled. Guide: reports/COLAB_TRAINING_GUIDE.md
- [x] **Enhancement pass (2026-06-17):** +9 exogenous features (VIX/yields/dollar) in M1 (MODEL_FEATURES 30→39); LightGBM tabular; attention layer on LSTM; z-scored ensemble. M1 19/19, M3 19/19, multi-agent reviewed
- [x] **Enhancement finding (honest):** exogenous data IS used (8/15 LightGBM top importances) + lifts val DirAcc to 0.539 (p=0.015), but LightGBM OVERFITS holdout (RMSE 0.02165 > naive 0.02109, DM p=0.003; DirAcc 0.499). Confirms EMH ceiling — no OOS lift
- [x] **GARCH re-tested OOS (agent fix):** GJR-GARCH-mean holdout DirAcc 0.543 binomial p=0.002 (weak but significant) — the one surviving directional signal; ties naive on RMSE (DM p=0.59) + uneconomic after costs. holdout_predictions gains y_pred_garch
- [x] **Naming fix (agent):** "EGARCH" mislabel corrected to GJR-GARCH (o=1) everywhere; VIX release-timing wording softened; LightGBM noted deliberately un-tuned

## Milestone 4 — Evaluation & Presentation  *(status: ☐)*

**File:** `milestone4_evaluation_presentation.md`  **Reads:** holdout predictions + all earlier reports.

- [ ] Notebook `04_evaluation_presentation.ipynb` runs top-to-bottom with seed=42
- [ ] Final performance table: ARIMA + LSTM vs all baselines + perfect-foresight upper bound
- [ ] Top-10 worst forecasts table with narrative explanations (regime / earnings / news)
- [ ] Error-distribution histogram + error-by-month plot + directional confusion matrix
- [ ] ≥ 5 named limitations specific to finance (EMH, regime shifts, black swans, ...)
- [ ] Misuse audit: explicit "not investment advice" disclaimer in `M4.md` + on slide 1
- [ ] Cumulative-return backtest with assumptions + disclaimer
- [ ] Data-protection note (what changes with real broker data)
- [ ] Reproducibility verified: fresh venv → `jupyter execute` all 4 notebooks end-to-end
- [ ] Pinned `requirements.txt` + updated `README.md` reproduce section
- [ ] Presentation deck `reports/M4_presentation.pptx` 10–12 slides with speaker notes
- [ ] Journey narrative (300–500 words) at top of `reports/milestones/M4.md`

---

## Current blocker

> M1 + M2 + M3 (+ enhancement pass) complete, all multi-agent-audited. M3 used ARIMA + GJR-GARCH(asymmetric) per ticker + global LSTM (Colab) + LightGBM + exogenous features. Next: M4 evaluation & presentation. LSTM full-train pending Colab. No blockers.

## Decisions log

Append one line per non-trivial decision. Format: `YYYY-MM-DD | M# | decision | reason`.

- `2026-05-21 | M0 | Mirror Mhmd-Ram/DS-Project structure | User explicitly chose this as a reference project; consistent scaffold across 4 milestones is the easiest way to keep an AI-executable workflow`
- `2026-05-21 | M0 | Tickers ^GSPC + AAPL + AMZN + NVDA | User-chosen pairing — S&P 500 as benchmark + 3 high-momentum tech stocks for richer comparison`
- `2026-05-21 | M0 | 10 years daily data | Covers COVID 2020, 2022 bear, 2023 AI boom — multiple regimes for evaluation`
- `2026-05-21 | M0 | ARIMA vs LSTM comparison | User-chosen modeling approach — classical baseline vs deep model with honest evaluation regardless of who wins`
- `2026-05-21 | M0 | Yahoo Finance via yfinance + commit CSV snapshot | Reproducibility anchor: re-runs read from CSV not the live API; matches Kaggle-snapshot pattern from the reference project`
- `2026-05-21 | M0 | Standard depth evaluation (RMSE/MAE + train-test split + plots) | User-chosen — sufficient for academic submission without being walk-forward backtesting`
- `2026-05-21 | M1 | Forecast target = log_return, not Close | ADF p-values confirm: Close non-stationary (p≥0.86), log_return stationary (p=0.0000). Standard finance practice.`
- `2026-05-21 | M1 | Retain outliers, flag with is_outlier_hi / is_outlier_lo / is_extreme_return | Extreme return days (NVDA: 232; ^GSPC: 13 over 10 yr) encode real regime events — capping would destroy the signal we need to learn`
- `2026-05-21 | M1 | Target-encoded features (ticker_mean_log_return, ticker_std_log_return) fit on train only | Prevents subtle cross-section + time leakage`
- `2026-05-21 | M1 | Split: train ≤ 2023-12-31, val 2024, holdout 2025+ | Holdout includes 2025 AI cycle + 2026 partial year, val is a clean calendar year for tuning`
- `2026-05-21 | M1 | Direct Yahoo chart API as fallback when yfinance fails | fc.yahoo.com (consent endpoint) was blocked from local sandbox; query1.finance.yahoo.com works directly. Won't be triggered in Colab where yfinance succeeds normally.`
- `2026-05-21 | M1 | Skipped jupyter/nbconvert in venv requirements | Windows long-path enabled is needed for jupyterlab extension install; not relevant since we execute via python directly and use nbformat to build the .ipynb`
- `2026-05-21 | M1-audit | Adj Close pinned for all returns / lags / SMAs; NVDA 2024-06-10 split sanity assertion added | Raw Close jumps artificially on split days. Yahoo's chart API already returns split-adjusted Close, but explicit Adj Close usage + assertion prevents regression`
- `2026-05-21 | M1-audit | Kurtosis printed side-by-side for Close and log_return | Removes ambiguity. kurt(Close) is negative for trending series (U-shaped distribution); kurt(log_return) is positive (fat tails). ^GSPC log-return kurtosis 16.8 is correct — COVID 2020 effect on a low-vol baseline`
- `2026-05-21 | M1-audit | Replaced global ticker_mean_log_return with expanding-mean shifted 1 | Old global mean was constant per ticker — early-train rows got info from late-train. New expanding(min_periods=20).mean().shift(1) makes each row see only its strict past. Verified: ~1894 unique values per ticker (was 1)`
- `2026-05-21 | M1-audit | Replaced IQR outlier detection with rolling 60-day MAD, threshold 3.5 | IQR's fixed quantile fence over-flagged NVDA at 9.2%. Rolling MAD adapts to local volatility regime. New flag rates 5.9-7.2% across all tickers (NVDA: 14% -> 5.9%)`
- `2026-06-17 | M1-v3 | Added next-day target target_log_return = groupby(ticker).log_return.shift(-1) per split | BLOCKER (Model QA agent): prior target was same-row log_return, making the stated next-day forecast unbuildable + same-day columns leaked. Built per-split so last row/ticker = NaN target (no fwd cross-split leak). Verified target[t]==log_return[t+1]`
- `2026-06-17 | M1-v3 | Pinned date window START=2016-05-23 END=2026-05-21 | BLOCKER (Data Eng + Model QA agents): today-anchored window made fresh clone/Colab non-reproducible. Window recorded in snapshot_hashes.json`
- `2026-06-17 | M1-v3 | enforce_schema() on all fetch paths; Volume->float64 | BLOCKER (Data Eng + Model QA): yfinance gave int64, fallback float64; dict said float64. Single coercion makes cache/yf/fallback identical + post-concat assertion`
- `2026-06-17 | M1-v3 | Added MODEL_FEATURES (30) + feature_roles.json; raw price levels excluded from X | HIGH (Model QA + Inv Researcher): same-day leak twins were marked leakage-safe; close_lag_*/sma_* non-stationary for ARIMA. Role separation + stationary substitutes (momentum, price_to_sma)`
- `2026-06-17 | M1-v3 | is_warmup flag + feature_nan_report.md | HIGH (all 3 agents): warm-up NaNs (sma_200=800 etc) were retained undocumented, would crash LSTM. M3 must dropna(subset=MODEL_FEATURES)`
- `2026-06-17 | M1-v3 | Renamed is_outlier_mad->is_tail_event, NaN during warm-up | HIGH (Model QA): flag stamped NaN warm-up rows as 0 (false). Inv Researcher: ~6% flag rate IS the fat tail, not errors -> renamed tail-event`
- `2026-06-17 | M1-v3 | Added Ljung-Box(r^2) test; corrected "stationary" wording; flag GARCH for M3 | HIGH (Inv Researcher): returns mean-stationary but conditionally heteroskedastic (LB(r2) p~0, ACF(r2) lag1 0.45 GSPC). Plain ARIMA misspecifies variance`
- `2026-06-17 | M1-v3 | Added realized_vol/parkinson/rsi/volume_z + Colab bootstrap + partial-ticker-failure handling + Volume ffill->0 | MED/HIGH (Inv + Data Eng): volatility is the predictable signal; Colab path was unreachable; 1 ticker failing aborted opaquely; ffill on Volume fabricates activity`
- `2026-06-17 | M2 | EDA on train_fe only; 18 figs, 5 hypotheses, 13/13 self-audit | seasonality negligible, returns mean-stationary + conditionally heteroskedastic, weak negative lag-1 (microstructure), tech corr 0.65-0.78 with ^GSPC`
- `2026-06-17 | M2-audit | Added mutual-information feature ranking (Fig 17) | HIGH (Inv + Model QA): Pearson ~0 for all features hid the signal; MI shows vol features 67x noise floor (parkinson_vol_21 0.16). This is the quantitative LSTM justification`
- `2026-06-17 | M2-audit | Added leverage-effect test (H5/Fig 18) -> recommend EGARCH/GJR not symmetric GARCH | MED (Inv): down days raise next-day vol more, significant all 4 tickers (p 7e-6..0.035). Symmetric GARCH would miss the asymmetry`
- `2026-06-17 | M2-audit | January test per-ticker not pooled; ACF lag-1 reframed as negative microstructure; real sealing audit check; returns-based decomposition cross-check | HIGH/MED (both agents): pooling 4 corr~0.65 tickers inflated effective-n ~3x (masked AMZN p=0.027); -0.162 lag-1 is mean-reversion not EMH-faint; old loaded_train_only check was vacuous; price-level decompose is fragile`
- `2026-06-17 | M3 | Baselines + per-ticker ARIMA(BIC) + GJR-GARCH-t(Student-t) + global LSTM (smoke); holdout touched once | Honest EMH result: no significant out-of-sample skill (holdout ARIMA RMSE 0.02107 ~ naive 0.02109 DM p=0.78; DirAcc 0.514 p=0.29). LSTM full-train deferred to Colab`
- `2026-06-17 | M3-audit | Fixed off-by-one in ARIMA one-step (BLOCKER) | Model QA: forecast taken before appending today's return -> predicted same-day, scored vs next-day (proven shift-by-1). Append-then-forecast. Val ARIMA DirAcc 0.536->0.521 (old edge was the artifact); holdout RMSE 0.02129->0.02107`
- `2026-06-17 | M3-audit | Added binomial + Diebold-Mariano significance tests | HIGH (both): a 55% DirAcc read as skill without a test. Val GARCH 0.555 significant (p=0.0006) but vanishes on holdout (0.514 p=0.29); all RMSE gaps vs naive insignificant (DM p>=0.07)`
- `2026-06-17 | M3-audit | BIC not AIC for ARIMA order; whiteness on GARCH std-residuals | MED (both): AIC over-selected (2,0,2); BIC gives parsimonious (1,0,0)/(0,0,1). Raw-resid LB~0 is leftover ARCH not wrong order - std-resid LB p>0.29 + std-resid^2 p>0.77 confirm clean joint fit`
- `2026-06-17 | M3-audit | GARCH evaluated by QLIKE+Mincer-Zarnowitz (variance), persisted via pickle; costed backtest w/ Sharpe/maxDD/t-stat | MED/HIGH (Inv + Model QA): point-return RMSE is wrong lens for a variance model; arch results lack .save (silent no-op); terminal-wealth-only backtest hid that 175 switches kill it at 10bps (Sharpe 0.33) and daily returns insignificant (p=0.168)`
- `2026-06-17 | M3+ | Added 9 exogenous features (VIX/^TNX/^IRX/DXY) to M1, leakage-safe (market-priced daily, known at close t, no CPI release-lag) | User-requested enhancement. Only break from price-history-only (M4 limitation #3). Fetched via existing snapshot pipeline; merged by date + ffill`
- `2026-06-17 | M3+ | LightGBM tabular + attention LSTM + z-scored ensemble | User-requested 4 enhancements. LightGBM fully local (real result); attention via Keras functional additive pooling (Colab full-train); ensemble directional-only (z-blend off return scale -> RMSE n/a)`
- `2026-06-17 | M3+ finding | Exogenous+LightGBM lifts VAL DirAcc to 0.539 (p=0.015) but OVERFITS holdout (RMSE worse than naive, DM p=0.003; DirAcc 0.499) | Honest: alternative data is informative in-sample (8/15 top importances exogenous) but does NOT generalize OOS. Strengthens EMH conclusion - more complexity/data != OOS skill`
- `2026-07-05 | M3.5 | Fixed rsi_14 to true Wilder EMA + shift(1); relabeled dict | Was simple rolling-mean gain/loss mislabeled "Wilder" and the only rolling feature without shift(1). Correctness fix; OOS value ~0 either way (direction=noise)`
- `2026-07-05 | M3.5 | Added variance-axis candidate pool (semi_vol_21/63, ret_skew_21, ret_kurt_21, atr_14); leakage-safe train-only ablation (MI+shuffle-floor AND walk-forward permutation, KEEP if perm>perm_std) | Scope=rigorous ablation, no new models, variance-axis only (user-chosen). ALL 5 rejected on evidence (0 kept): real MI but redundant with existing realized_vol/parkinson (permuting doesn't raise OOS RMSE). FEAT_MODEL=39 unchanged -> clean tuning-only comparison. Rejected candidates persisted as diagnostic, not in X`
- `2026-07-05 | M3.5-audit | Fixed atr_14 price-basis bug (MED, Inv Researcher) | atr_14 differenced RAW High/Low against ADJUSTED close -> ~10-40x inflation across NVDA pre-split history. Put H/L on adjusted basis (adj=AdjClose/Close). Ablation still rejects it (redundant)`
- `2026-07-05 | M3.5 | Optuna LightGBM tuning (60 trials) on 5-fold purged expanding walk-forward CV (date-split, 1d embargo), objective=mean OOS RMSE, n_estimators by per-fold early-stop | Replaces the one-val-peek selection that caused the overfit. CV chose 2 trees / depth 3 — signal supports ~no complexity. Objective never reads val/holdout (structural no-peek, Model-QA-verified)`
- `2026-07-05 | M3.5 finding | PROMOTE tuned LightGBM: overfit FIXED | Baseline holdout RMSE 0.02168 (worse than naive, DM p=0.003, DirAcc 0.483) -> tuned 0.02107 (TIES naive DM p=0.533, DirAcc 0.542 p=0.002). Tuned-vs-baseline DM stat=-3.02 p=0.0025 (significant). Same 39 features (0 kept) => clean tuning-only comparison, no confound. Still no model BEATS naive on RMSE (EMH intact) but weakness repaired`
- `2026-07-05 | M3.5-audit | 3 audit fixes applied (BLOCKER/HIGH+MED) | Model QA: garch_val_arr was crossed with LSTM in wrong ticker order (^GSPC sorts last in val_c, not TICKERS order) -> fixed per-ticker remap + order assert. DM promote-gate was inert -> now requires DM-significant improvement (load-bearing). LGB set deterministic=True/force_row_wise for cross-machine reproducibility. Data Eng: PASS (0 leakage, contracts clean, 0 extra rows)`
- `2026-07-06 | M3.5-fix | Made self-audit lstm check mode-aware (surfaced by first Colab GPU run) | Check hardcoded lstm_attention_smoke.keras; on FULL_TRAIN the model saves as lstm_attention_final.keras so smoke file is absent -> assert tripped despite successful training. Now checks final-vs-smoke by FULL_TRAIN. Closing NOTE also mode-aware`
- `2026-07-06 | M3-finding | Real Colab GPU LSTM edge did NOT reproduce | First actual GPU run (Optuna: units=64/2-layer/dropout0.26/lr0.0013) gives holdout DirAcc 0.491 (p=0.574, coin flip) + RMSE 0.02031 worse than naive (DM p=0.004); earlier forced-CPU run showed 0.542 (p=0.005). Ensemble GARCH+LSTM 0.512 (coin flip), dragged by flat LSTM. Only GARCH (0.543) + tuned-LightGBM (0.542), both p=0.002, keep a reproducible edge. Strengthens EMH conclusion: the one DL edge was run-dependent noise. M3.md §5/§8/headline + CLAUDE.md updated to real numbers`
- `2026-07-05 | M3.5 | Ensemble fixed: return-scale equal-weight GARCH-mean + attention-LSTM (dropped overfit LightGBM) | Old z-blend was off-scale (RMSE n/a) and blended the overfit member into the real-edge LSTM. New blend is fully scorable (RMSE/MAE/DM) + feeds cost backtest; least-correlated pair. Old z-blend kept as documented rejected baseline`
- `2026-07-05 | M3.5 | LSTM Optuna (15 trials, inner purged split, scaler on inner-train only) + bidirectional option, Colab-gated | LSTM improvements (units/dropout/lr/batch/layers/bidir) tuned on GPU only; local stays 2-epoch smoke. 22/22 self-audit pass`

## How to feed this to a cheap model

Minimal prompt template:
```
SYSTEM: You are completing one task from a multi-milestone DS project. Use only the
tools available to you (Python / Jupyter). Respect the "Forbidden" list. When done,
return the updated checklist and a 3-line summary of what changed.

USER: Here is the project state:
  <paste progress_checklist.md>
Here is the milestone spec:
  <paste milestoneN_*.md>
Execute the next unchecked item under "Milestone N". Stop after one item.
```

This keeps context small enough for a 7B-class model and produces auditable diffs.
