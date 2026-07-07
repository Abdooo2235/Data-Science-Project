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
| Current week (1–14) | 13 |
| Current milestone | M5 done (M1+M2+M3+M3.5+M4+M5 complete) |
| Last updated | 2026-07-06 (M5 deployment complete, multi-agent audited) |

## Overall progress

- [x] **Milestone 1 — Data Collection & Preprocessing** (Week 6) — 13 / 13 (v3 multi-agent audit 2026-06-17: 19/19 self-audit PASS; v2 was 12/12, v1 10/10)
- [x] **Milestone 2 — Exploratory Data Analysis** (Week 9) — 11 / 11 (2026-06-17: multi-agent reviewed + fixed → 18 figures, 5 hypotheses, 13/13 self-audit PASS)
- [~] **Milestone 3 — Model Building (ARIMA+GARCH vs LSTM)** (Week 11) — 12 / 13 (core done + multi-agent-audited 2026-06-17, 15/15 self-audit; LSTM full-train pending Colab)
- [x] **Milestone 4 — Evaluation & Presentation** (Week 13) — 11 / 11 (2026-07-06: 12/12 self-audit PASS; M4.md + notebook 04 + 4 figures + 12-slide HTML deck published as an Artifact; multi-agent audited)
- [x] **Milestone 5 — Real deployment (Streamlit)** (extra, not graded) — 8 / 8 (2026-07-06: 5/5 self-audit PASS; app.py + test_app.py + M5.md; multi-agent audited, 1 BLOCKER + 2 HIGH fixed)
- [x] **Milestone 3.6 — Volatility Forecasting** (2026-07-07: notebook 04 + M3.6.md; 13/13 self-audit; 3 Model-QA audits) — **positive OOS result: realized volatility IS predictable (unlike return direction). Every model beats random-walk-vol on QLIKE OOS at both horizons; best h=5 GJR-GARCH/LightGBM, best h=20 LightGBM. Risk, not return alpha.**

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
- [x] **LSTM full training done** (FULL_TRAIN flag auto-on Colab GPU; 20-epoch attention + Optuna). Real GPU holdout DirAcc **0.491 (p=0.574, coin flip)**, RMSE worse than naive — the earlier forced-CPU 0.542 (p=0.005) did NOT reproduce, so the LSTM edge was run-dependent noise. Artifacts: lstm_attention_final.keras, lstm_best_params.json, lstm_val_metrics.json, y_pred_lstm filled. Guide: reports/COLAB_TRAINING_GUIDE.md
- [x] **Enhancement pass (2026-06-17):** +9 exogenous features (VIX/yields/dollar) in M1 (MODEL_FEATURES 30→39); LightGBM tabular; attention layer on LSTM; z-scored ensemble. M1 19/19, M3 19/19, multi-agent reviewed
- [x] **Enhancement finding (honest):** exogenous data IS used (8/15 LightGBM top importances) + lifts val DirAcc to 0.539 (p=0.015), but LightGBM OVERFITS holdout (RMSE 0.02165 > naive 0.02109, DM p=0.003; DirAcc 0.499). Confirms EMH ceiling — no OOS lift
- [x] **GARCH re-tested OOS (agent fix):** GJR-GARCH-mean holdout DirAcc 0.543 binomial p=0.002 (weak but significant) — the one surviving directional signal; ties naive on RMSE (DM p=0.59) + uneconomic after costs. holdout_predictions gains y_pred_garch
- [x] **Naming fix (agent):** "EGARCH" mislabel corrected to GJR-GARCH (o=1) everywhere; VIX release-timing wording softened; LightGBM noted deliberately un-tuned

## Milestone 4 — Evaluation & Presentation  *(status: x — done, multi-agent audited 2026-07-06)*

**File:** `milestone4_evaluation_presentation.md`  **Reads:** holdout predictions + all earlier reports.

- [x] Notebook `04_evaluation_presentation.py` (+ `.ipynb`) runs top-to-bottom with seed=42; 12/12 self-audit PASS
- [x] Final performance table: all 6 models vs naive + perfect-foresight gap-closure (no model beats naive; gap-closure ~0)
- [x] Top-10 worst forecasts table with narrative (DeepSeek shock, April tariff crash/rebound, earnings) → `models/m4_top10_worst.csv`
- [x] Error-distribution histogram + error-by-month plot + directional confusion matrix (M4_fig_err_dist/err_by_month/confusion)
- [x] 7 named finance-specific limitations (EMH, regime, black swans, calm holdout, survivorship, daily granularity, multiple comparisons)
- [x] Misuse audit: "not investment advice" disclaimer in `M4.md` header + section 5b + deck slide 1
- [x] Cumulative-return backtest with cost sensitivity + disclaimer (promoted model = buy-and-hold; ARIMA timer dies at 10 bps) → M4_fig_backtest_cum
- [x] Data-protection note (Yahoo public vs broker order flow: GDPR/CCPA, FINRA, MiFID II) — M4.md section 5d
- [x] Reproducibility documented: fresh venv → run 01-04 `.py`; Colab GPU step for the LSTM noted
- [x] Pinned `requirements.txt` + updated `README.md` reproduce section (all 4 notebooks)
- [x] Presentation deck `reports/M4_presentation.html` (12 slides + speaker notes, light clean-finance theme) + published claude.ai Artifact
- [x] Journey narrative (453 words) at top of `reports/milestones/M4.md`

**M4 key finding (honest):** no model beats naive on RMSE. The two "significant" directional edges (GJR-GARCH 0.543, tuned-LightGBM 0.542, p=0.002) both predict up 99-100% of days = the up-day rate 0.542 = market drift, not timing. The confusion matrix shows the promoted model never predicts a down day. The backtest of the promoted model is literally buy-and-hold. EMH ceiling holds, measured carefully.

---

## Decisions log — M4 (2026-07-06)

- `2026-07-06 | M4 | Built notebook 04 (final performance + error analysis + backtest) + M4.md (journey/limits/ethics) + a 12-slide light clean-finance HTML deck (frontend-slides) published as a claude.ai Artifact | Rubric wants the journey question->data->model->insight->limitations and an ethics audit. All copy run through the humanizer skill (no em-dashes/emojis). Deck theme user-chosen. 12/12 self-audit PASS`
- `2026-07-06 | M4 finding | The surviving directional edge is market drift, not skill | Both p=0.002 "edges" (GARCH, tuned-LGB) predict up 99-100% of days; DirAcc 0.542 == holdout up-day frequency. Confusion matrix: promoted model never predicts down. Backtest of promoted model == buy-and-hold. No tradeable timing alpha; ARIMA timer (174 switches) edges B&H at 0 cost but dies at 10 bps. Strengthens the EMH conclusion`
- `2026-07-06 | M4 | Refreshed PROJECT_DECISIONS_AND_ARCHITECTURE.md holdout table + README to canonical M3.md §8 numbers | The architecture doc was one revision stale (missing tuned-LightGBM + real-GPU LSTM). Now consistent across M3.md, M4.md, README, and the deck`
- `2026-07-06 | M4-audit | 3 adversarial subagents (Model QA + Investment Researcher + Data Engineer) reviewed M4; fixes applied | Verdict: honest, would pass buy-side review, no blockers. Fixes: (1) DE - self-audit crashed on a CPU-only reproduction (LSTM/ens NaN -> 5 models); gated the count on LSTM_READY. (2) IR - "lag-1 microstructure" contradicted the drift finding (lag-1 is negative/mean-reversion; survivors are always-up drift) -> corrected to unconditional drift in M4.md + CLAUDE.md. (3) QA/DE - stale MAE column in M4.md (4 cells) -> fixed to CSV values. (4) IR - added the honest directional test vs the up-day base rate: GARCH p=0.978, tuned-LGB p=1.000 (no skill beyond drift). (5) added t-test (t=1.38 p=0.168) + Sharpe (rf=0) to notebook 04. (6) deck - green "win" rows recolored to a neutral "up-only, drift" tag; disclaimer made verbatim. (7) doc consistency - cleared stale "M4 not started"/LSTM-0.542 text in arch doc, checklist, CLAUDE.md, README tree. 12/12 self-audit still PASS`

---

## Milestone 5 — Real deployment (Streamlit)  *(status: ☑ complete — extra, not graded)*

**Report:** [`reports/milestones/M5.md`](../reports/milestones/M5.md) | **Run:** `streamlit run app.py` | **Test:** `python test_app.py`

- [x] `app.py` — Streamlit app: ticker + model + date pickers over the sealed holdout; forecast vs actual with HIT/MISS verdict; full-holdout Altair chart with selected date marked
- [x] Not-investment-advice disclaimer front and center (red error box under the title, before any data; verbatim M4 §5b text; asserted by the smoke test)
- [x] Honest accuracy table (all 7 models, `Days scored` + `Naive RMSE (same days)` + DM p vs naive + drift debunk bullets) — consistent with M4.md, agent-verified
- [x] Live-model proof: tuned-LightGBM booster re-run on the selected date's features, asserted equal to the stored prediction (red error on mismatch)
- [x] Naive zero rendered as "no direction call" (no fabricated DirAcc); per-ticker captions compare each model to the always-up base rate
- [x] `test_app.py` smoke test — AppTest full render (0 exceptions) + booster reproduces all 1,380 stored predictions
- [x] Fresh-clone deployable: 4 small artifacts (~660 KB) un-gitignored via carve-outs; friendly `st.error`+`st.stop` if missing; `streamlit`+`altair` pinned in requirements.txt
- [x] Multi-agent audited (Model QA + Investment Researcher + code reviewer): 1 BLOCKER (gitignored artifacts) + 2 HIGH (table contradicted no-model-beats-naive bullet) + 3 MED + 5 LOW — all fixed; date semantics verified no off-by-one

## Milestone 3.6 — Volatility forecasting  *(status: ☑ complete — multi-agent audited)*

**Spec:** [`Plans/milestone_volatility.md`](milestone_volatility.md). Pivot from return direction (EMH coin flip) to
**5-day forward realized volatility** (val R² ≈ 0.50–0.56 in the feasibility probe — a real positive result).

- [x] Feasibility probe `scripts/vol_feasibility.py` (HAR-RV / LightGBM / random-walk, self-checked): 5-day RV R²(log) 0.52–0.56 vs random-walk floor 0.12
- [x] Return-horizon negative-result probe `scripts/horizon_experiment.py` (5/20-day return direction = drift + overlap artifact, no usable edge) — motivates the pivot
- [x] **Phase 1**: `fwd_rv_5` / `fwd_rv_20` targets wired into notebook 01 (per-split `shift(-h)`, leakage-safe) + exposed in `feature_roles.json` `vol_targets`; parquets regenerated offline from snapshots; ipynb regenerated
- [x] Phase 1 leakage guard: asserts exactly `h` tail rows/ticker NaN per split (catches sub-h vacuity + interior-NaN void)
- [x] Phase 1 adversarial audit (Model QA Specialist subagent): verdict **leakage-safe**; window provably `[t+1,t+h]`, embargo=horizon exactly sufficient. Fixes applied: tightened guard; corrected the plan's false "probe implements embargo" claim into a hard leak-trap warning (do NOT reuse `03`'s `embargo=1` for a multi-day target)
- [x] Phase 2: `notebooks/04_volatility_modeling.py` — RandomWalk + HAR-RV + GJR-GARCH + LightGBM(Optuna), **horizon-aware** purged CV (embargo=h), QLIKE + R² + Mincer-Zarnowitz + block-aware DM. **Result (val): 5-day RV LightGBM R²log 0.57, QLIKE -5.64, beats RandomWalk (DM p=2.5e-8) AND HAR (p=0.002) — real, robust vol skill.** 11/11 self-audit PASS
- [x] Phase 2 reporting fixes baked in: block-aware DM (Newey-West lag=h-1); Duan smearing on the `exp` back-transform; one common dropna mask across all models per horizon; QLIKE declared decisive
- [x] Phase 2 adversarial audit (Model QA Specialist): **leakage-safe + 5-day skill honest/reproducible** (DM survives 4× HAC widening). Fixes applied: (1) made the embargo assert load-bearing (tied embargo to horizon so a 1-day embargo on a multi-day target is impossible); (2) surfaced DM-vs-HAR in the summary so the h=20 HAR **tie** (p=0.165) isn't buried; (3) tightened R²_log comment. **Honest caveat: h=20 ties HAR and its RW edge is marginal/tuning-sensitive — Phase 4 must report h=20 as "skilled but not reliably above persistence", h=5 as the clean win**
- [~] Phase 3: ablation for new vol candidate features — **SKIPPED (user decision, YAGNI)**: existing features already give strong significant OOS skill; candidate vol features are speculative marginal gain
- [x] Phase 4: **holdout opened ONCE** → `reports/milestones/M3.6.md` + 6 figures. **OOS result: every model beats random-walk-vol on QLIKE (p≤1e-9 h=5, p≤0.033 h=20); best h=5 GJR-GARCH/LightGBM near-tie (QLIKE -5.47), best h=20 LightGBM (R²log 0.59, beats HAR p=0.005 — val tie resolved OOS). Volatility IS predictable OOS.** Models refit on train+val, LightGBM reused val-tuned params. 13/13 self-audit PASS
- [x] Phase 5: final adversarial audit (Model QA Specialist) — **VERDICT: leakage-free, honestly reported, reproducible to machine epsilon** (CSVs regenerate to ~1e-13; GARCH `last_obs` proven to exclude all holdout; holdout is a representative regime, not cherry-picked; no re-tuning on holdout). Fixes: bound 2 self-audit checks to computed results (no hardcoded True); corrected report's "regime change" over-characterization to honest "mild distribution shift". Graph refreshed (code-only: 481 nodes / 529 edges)

**M3.6 headline:** volatility is genuinely predictable out-of-sample where return direction was not. Every model beats random-walk-vol on QLIKE OOS (p≤1e-9 h=5, p≤0.033 h=20); best h=5 GJR-GARCH/LightGBM near-tie, best h=20 LightGBM (R²log 0.59, beats HAR p=0.005). This is about **risk (2nd moment)**, not tradeable return alpha — consistent with the M3/M4 EMH conclusion on returns.

## Status

> M1 + M2 + M3 + M3.5 + M4 + M5 complete, all multi-agent-audited. Full pipeline reproducible from committed snapshots (LSTM on Colab GPU); the M5 Streamlit app runs from a fresh clone with no pipeline run. Honest headline: no economically usable out-of-sample skill, the EMH ceiling holds. No blockers.

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
- `2026-07-06 | M5 | Streamlit app (app.py) replays the sealed holdout: stored predictions for all 7 models + a live tuned-LGB integrity re-prediction; Altair chart; disclaimer front and center | Deployment = replay-only by design (no "tomorrow" surface = structural misuse guard). One smoke test (AppTest render + 1380-row booster reproduction). streamlit+altair pinned`
- `2026-07-06 | M5-audit | 3 parallel agents (Model QA + Inv Researcher + code reviewer); all findings fixed | BLOCKER: app's 4 required artifacts were gitignored -> fresh clone crashed raw; carved out ~660 KB via dir/* + ! rules + st.error/st.stop guard. HIGH (both agents): accuracy table showed LSTM/ens RMSE below naive without the 1,140-row caveat (matched naive 0.02004) -> added Days scored + Naive RMSE (same days) + DM p columns + caption. MED: naive-zero fabricated "calls DOWN" + 45.8% DirAcc -> "no direction call"; static always-up caption false for below-drift models -> dynamic; "holdout never used for decisions" overclaim -> promote/revert wording. LOWs: mtime cache keys, warning/error on live-check gaps, delta colors off, na_rep, 54.2/54.3 precision`

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
`2026-07-07 | M3.6 | Pivot to volatility forecasting; wired fwd_rv_5/fwd_rv_20 targets into M1 (Phase 1) | Return direction is an EMH coin flip and a 5/20-day return-direction probe (scripts/horizon_experiment.py) confirmed the longer-horizon edge is drift + overlapping-window pseudo-replication, not skill. Volatility IS predictable (M2 vol clustering; feasibility probe scripts/vol_feasibility.py: 5-day forward RV val R2(log) 0.52-0.56 vs random-walk floor 0.12). Targets built per-split via shift(-h), leakage-safe (last h rows/ticker NaN), exposed in feature_roles.json vol_targets`
`2026-07-07 | M3.6-audit | Phase 1 adversarially audited (Model QA Specialist subagent): leakage-safe, verified | Target window provably [t+1,t+h] (day t excluded), per-split NaN boundary correct, embargo=horizon exactly sufficient, no return-metric smuggled in. Fixes: (1) tightened the leakage guard to assert exactly h NaN tail rows/ticker. (2) Corrected the plan's false claim that vol_feasibility.py implements horizon-aware embargo into a hard LEAK-TRAP warning: Phase 2 must NOT reuse 03's embargo=1 date_folds for a 5/20-day RV target. Deferred to Phase 2: overlap-aware DM significance, Jensen/Duan smearing on exp back-transform, common dropna mask`
`2026-07-07 | M3.6 | Phase 2 built notebook 04 (RandomWalk/HAR-RV/GJR-GARCH/LightGBM-Optuna) + adversarially audited | Forecasts h-day forward realized vol. Reused 03's metrics/date_folds/GARCH/Optuna; added block-aware Diebold-Mariano (Newey-West lag=h-1 for overlapping windows), Duan smearing on the exp back-transform, one common dropna mask per horizon, QLIKE decisive. Result (val): 5-day RV LightGBM R2log 0.57 QLIKE -5.64, beats RandomWalk (DM p=2.5e-8) AND HAR (p=0.002) - a real, robust positive result (survives 4x HAC widening). Model QA audit: leakage-safe (target [t+1,t+h], holdout sealed, GARCH val-excluded, embargo=h purge verified). Fixes: made embargo assert load-bearing by tying embargo to horizon; surfaced DM-vs-HAR so the h=20 HAR tie (p=0.165) is not buried. HONEST CAVEAT: h=20 ties HAR and its RW edge is marginal/tuning-sensitive; h=5 is the clean headline`
`2026-07-07 | M3.6 | Phases 4-5: opened holdout ONCE, wrote M3.6.md, final adversarial audit -> milestone COMPLETE | Skipped Phase 3 ablation (user decision, YAGNI - existing features already give strong significant OOS skill). Models refit on train+val (LightGBM reused val-tuned params, NOT re-tuned on holdout); holdout scored once. OOS result: every model beats random-walk-vol on QLIKE (h=5 p<=1e-9, h=20 p<=0.033); best h=5 GJR-GARCH/LightGBM near-tie (QLIKE -5.47), best h=20 LightGBM (R2log 0.59, beats HAR p=0.005 - the val-set HAR tie resolved OOS). Final Model QA audit verdict: leakage-free, honestly reported, reproducible to machine epsilon (CSVs regenerate to 1e-13; GARCH last_obs proven to exclude all holdout; representative regime not cherry-picked). Fixes: bound 2 self-audit checks to computed results; corrected report 'regime change' wording to honest 'mild distribution shift'. Headline: volatility (2nd moment/risk) is predictable OOS where return direction (1st moment) was an EMH coin flip - consistent with, not contradicting, M3/M4`
