# Milestone 3.6 — Volatility Forecasting (spec)

**Status:** Phase 1 landed (targets wired into M1). Phases 2–5 pending.
**Owner pattern:** same as M1–M4 — AI-executable spec, leakage-safe, adversarially audited before "done".

## Why this milestone exists

M3's honest headline: next-day **return direction** is an efficient-market coin flip (every model ties
naive on RMSE; the only significant directional edges were cost-erased lag-1 microstructure). The
`scripts/horizon_experiment.py` probe confirmed 5-/20-day return direction adds no usable edge either —
the apparent significance was upward **drift** plus overlapping-window pseudo-replication.

**Volatility is the opposite story.** M2 already documented volatility clustering, ACF of squared returns,
and the leverage effect; M3's GJR-GARCH standardised residuals came out white (Ljung-Box p > 0.29). A
feasibility probe (`scripts/vol_feasibility.py`, leakage-safe, self-checked) measured **out-of-sample R²
on val**:

| horizon | RandomWalk R²(log) | HAR-RV R²(log) | LightGBM R²(log) | best R²(level) |
|---|---|---|---|---|
| 1-day | −0.10 | 0.13 | 0.11 | 0.05 |
| **5-day** | 0.12 | 0.52 | **0.56** | **0.50** |
| 20-day | 0.57 | 0.68 | 0.62 | 0.60 |

**Primary target = 5-day forward realized volatility** (R² ≈ 0.50–0.56, models crush the random-walk floor
0.12 and beat HAR on QLIKE — a genuine, defensible positive result). 20-day is secondary (still skilled,
but random-walk already scores 0.57 there, so the model's *value-add over trivial persistence* is smaller).
1-day is a noisy RV estimate (single squared return) — reference only.

## Target definition (leakage-safe — Phase 1, DONE)

`fwd_rv_h` = sqrt( Σ log_return² over days t+1 … t+h ), per ticker, built **within one split** via
`shift(-h)`. The last `h` rows of every split are NaN (their forward window would cross the split edge).
Wired into `notebooks/01_*.py` alongside `target_log_return`; a leakage guard asserts the last `h`
rows/ticker are NaN in each split. Exposed in `feature_roles.json` as `vol_targets = {fwd_rv_5:5, fwd_rv_20:20}`.
Modelled in **log space** (RV is right-skewed); back-transform for reporting.

## Features

- **HAR backbone** (already in MODEL_FEATURES): `realized_vol_5/21/63` (daily/weekly/monthly RV).
- **Implied vol**: `vix_level`, `vix_log_change`, `vix_z_60` — genuinely forward-looking (already merged).
- Already present: `parkinson_vol_21`, `rolling_std_20`, `realized_vol_*`, leverage/return-lag terms.
- **New candidates** (gate through the existing §1b ablation harness — promote only survivors, no
  speculative adds): realized **semivariance** (down-vs-up vol; `semi_vol_21/63` already candidates),
  a **jump** proxy (|r| > k·rolling_std), VIX-minus-realized spread.

## Models (reuse the M3 stack, retargeted to RV)

1. **HAR-RV (OLS)** — the standard benchmark; every model must beat it on QLIKE or it's cut.
2. **GJR-GARCH** — already in the repo and *this is the target it was built for* (M3 already scored it by
   QLIKE / Mincer–Zarnowitz). Promote from side-character to a primary vol forecaster.
3. **LightGBM on log-RV** — best probe performer; tune via Optuna on the **purged** CV.
4. **LSTM (optional, Colab)** — sequence model over the vol features; only if time permits.
5. **Random-walk vol** (`RV_today·√h`) — the trivial floor everything is measured against.

## Evaluation (vol-appropriate — NOT return metrics)

- **R²** (level + log), **QLIKE** (vol-forecast loss), **Mincer–Zarnowitz** regression (forecast
  unbiasedness — helper already exists in `03`).
- **Diebold–Mariano** on QLIKE loss: each model vs HAR **and** vs random-walk (significance, not eyeballing).
- Per-ticker + pooled. Optional interpretable extra: vol-regime up/down classifier.

## CV & leakage discipline

- Purged walk-forward on **train only**, **embargo = horizon** (5 or 20 trading days).
  **LEAK TRAP (audit-confirmed):** `03_model_building.py`'s `date_folds` hardcodes `embargo=1` — reusing it
  for a 5-/20-day RV target **leaks** (the last kept train row's forward label reaches into the OOS block).
  Phase 2 MUST use a horizon-aware `date_folds` (the `embargo=horizon` variant in
  `scripts/horizon_experiment.py` is the correct template) and add a self-check asserting
  `last_kept_train_date + h < first_OOS_date`. One-sided purge (train edge only) is correct here: with a
  forward target and backward-looking features, the sole cross-boundary vector is a train *label* reaching
  into OOS, which the train-edge purge covers.
- val tunes; **holdout opened exactly once** at the very end (same discipline as M3).
- Overlapping-window caveat stated explicitly: h-day RV windows overlap, so the effective sample < n;
  report block-aware significance, don't trust naive binomial/DM n.

## Deliverables

- `notebooks/04_volatility_modeling.py` (+ generated `.ipynb`) mirroring `03`'s structure.
- Targets in `01` + `feature_roles.json` (**DONE**).
- `reports/milestones/M3.6.md` — honest write-up: positive result + caveats + Corrections section.
- Figures: forecast-vs-actual RV, QLIKE-by-model bar, VIX-vs-realized scatter, per-ticker R².
- Refresh the graphify graph after (`/graphify . --update`).
- Update `Plans/progress_checklist.md` Decisions Log.

## Adversarial audit gate (repo requirement)

A model-QA subagent must verify, before "done":
no forward window crosses a split; embargo ≥ horizon **everywhere** (CV + any inner split); HAR beaten on
QLIKE **and** DM-significant; MZ slope ≈ 1 (no systematic bias); holdout touched exactly once; **no
return-scale metric (RMSE-on-return) smuggled in as a volatility claim**; log↔level back-transform correct.

## Phases

1. **Target + selfchecks** into `01` + roles — **DONE** (leakage guard passes; parquets regenerated).
2. `04` notebook: HAR + GARCH + LightGBM, purged CV (embargo=h), full vol metrics.
3. Ablation for new vol candidate features (train-only).
4. Holdout opened once → `M3.6.md` + figures.
5. Adversarial audit + graphify refresh + checklist update.

## Exploratory scaffolding (not the sealed pipeline)

- `scripts/vol_feasibility.py` — the R² probe above (HAR / LightGBM / random-walk, self-checked).
- `scripts/horizon_experiment.py` — the return-horizon negative result that motivated the pivot.
