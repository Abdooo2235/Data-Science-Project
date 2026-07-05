# CLAUDE.md — Stock Market Trend Analysis

Architecture reference for this repo. Read before editing. Keep this file in sync when structure changes.

## What this project is

Educational data-science capstone: **forecast the NEXT-day log return** for four tickers
(`^GSPC`, `AAPL`, `AMZN`, `NVDA`), 10 years daily, and compare classical vs deep vs tree models.
**Not investment advice.** Honest headline result: **no economically usable out-of-sample skill** — the
efficient-market ceiling holds (see `reports/milestones/M3.md`).

Milestone-driven (M1 data → M2 EDA → M3 models → M4 evaluation), each spec in `Plans/`, each report in
`reports/milestones/`. Every milestone is adversarially reviewed by subagents before it is considered done.

## Knowledge graph (RAG — query before grepping)

A graphify knowledge graph indexes this whole repo (code + docs + reports + figures) as
155 nodes / 213 edges / 14 communities. **When locating where something lives, how concepts
connect, or which files touch a topic, query the graph first — it is faster than blind grep.**

- Query engine (RAG): `graphify query "<question>"` — BFS traversal, cites `source_location`.
- Shortest path between two things: `graphify path "make_features" "target_log_return"`.
- Explain one node: `graphify explain "LightGBM"`.
- Rebuild after code/doc changes: `/graphify . --update` (re-extracts only changed files).

Artifacts (in the **parent** dir, `../graphify-out/` relative to this file):
- `../graphify-out/graph.json` — machine-queryable graph (what the query commands read).
- `../graphify-out/graph.html` — interactive human view, open in a browser.
- `../graphify-out/GRAPH_REPORT.md` — god nodes, communities, surprising connections.

## Repo layout

```
Stock-Market-Trend-Analysis/
├── CLAUDE.md                      # this file
├── README.md                     # human how-to-run
├── requirements.txt              # pinned; local uses tensorflow-cpu, Colab uses its own tensorflow
├── .gitignore                    # data/processed, models, figures gitignored; data/raw kept
├── Plans/
│   ├── milestone{1,2,3,4}_*.md   # AI-executable specs (one per milestone)
│   └── progress_checklist.md     # master tracker + Decisions Log (append-only, one line/decision)
├── notebooks/
│   ├── 0{1,2,3}_*.py             # SOURCE OF TRUTH (jupytext "percent" format)
│   └── 0{1,2,3}_*.ipynb          # generated from .py — for Google Colab
├── scripts/py_to_nb.py           # .py -> .ipynb converter (nbformat; no jupyter needed)
├── data/
│   ├── raw/                      # *_10y.csv snapshots (COMMITTED = reproducibility anchor) + snapshot_hashes.json
│   └── processed/                # train_fe/val_fe/holdout_fe.parquet, feature_roles.json,
│                                 #   feature_dictionary.md, feature_nan_report.md, holdout_predictions.parquet
├── models/                       # arima_*.pkl, garch_*.pkl, lgbm_global{,_final}.txt,
│                                 #   lstm_{global,attention}_smoke.keras, val_scores.csv
├── reports/
│   ├── figures/                  # M1_*.png, M2_fig01..18_*.png, M3_fig_*.png
│   ├── milestones/               # M1.md, M2.md, M3.md (each with a Corrections section per audit)
│   └── PROJECT_DECISIONS_AND_ARCHITECTURE.md
└── References/                   # course PDFs + reference notebook (not part of the pipeline)
```

## Pipeline / data flow

```
Yahoo Finance (yfinance + direct chart-API fallback)
  → data/raw/*_10y.csv snapshots (4 tickers + 4 exogenous: VIX, TNX, IRX, DX-Y.NYB)
  → 01: clean, merge exogenous by date, engineer features, chronological split, next-day target
  → data/processed/{train,val,holdout}_fe.parquet  (68 columns; 39 are MODEL_FEATURES)
  → 02: EDA on train_fe ONLY (val/holdout sealed) → reports/figures/M2_*, reports/milestones/M2.md
  → 03: baselines + ARIMA + GJR-GARCH + LightGBM + LSTM(attention, smoke) + ensemble
        train on train, tune on val, HOLDOUT OPENED ONCE → holdout_predictions.parquet, models/, M3.md
```

Notebooks run **locally** (verification) and in **Google Colab** (final, GPU for LSTM). First cell auto-detects
Colab; a bootstrap markdown covers clone/upload + `pip install -r requirements.txt`.

## Model stack (03_model_building.py)

- **Baselines**: naive_zero, persistence, moving_avg_20.
- **ARIMA** per ticker — order by **BIC** (parsimonious: (1,0,0)/(0,0,1)); one-step-ahead (append-then-forecast).
- **GJR-GARCH(1,1) Student-t** per ticker (`arch`, `o=1` asymmetry for M2's leverage effect); out-of-sample via
  `last_obs`. Evaluated as a variance model (QLIKE, Mincer-Zarnowitz), not point-RMSE. (NB: GJR, not EGARCH.)
- **LightGBM** — tabular, 39 features + `ticker` categorical; fully local; deliberately un-tuned (shows overfit).
- **Global LSTM + attention** — Keras functional, 60-day sequences, ticker one-hot. `FULL_TRAIN` flag
  auto-enables the real 20-epoch run on a Colab GPU (`lstm_attention_final.keras` + `lstm_val_metrics.json`;
  fills holdout `y_pred_lstm`); on local CPU it's a 2-epoch smoke-test. See `reports/COLAB_TRAINING_GUIDE.md`.
- **Ensemble** — z-scored directional blend of LightGBM + LSTM (RMSE n/a by design; directional only).

Metrics: RMSE, MAE, **directional accuracy** + significance (binomial vs 0.50, **Diebold-Mariano** vs naive).

## Leakage-prevention invariants (NEVER break — the whole project rests on these)

1. **Target = next-day**: `target_log_return = groupby(ticker).log_return.shift(-1)`, built **per split** so the
   last row of each split is NaN (no label leaks forward across a boundary). Same-day `log_return` is a FEATURE.
2. **Rolling/lag features** use `.shift(1)` before `.rolling(...)`; lags use `groupby(ticker).shift(k)`.
3. **Target encoding** = per-ticker **expanding** mean/std, `.shift(1)` (strictly past-only; never a global mean).
4. **Fit on train only**: StandardScaler, LightGBM, and ARIMA/GARCH parameters (GARCH uses `last_obs` to exclude
   val). Applied read-only to val/holdout.
5. **Holdout opened exactly once**, at M3 §8. M2 never opens val/holdout (a logged `read_parquet` self-audit
   proves it).
6. **Exogenous** (VIX/yields/dollar): market-wide, merged by date, ffill is `groupby(ticker)` (never crosses a
   ticker boundary or pulls a future value). Verified leakage-free (VIX same-day corr −0.70 vs next-day +0.05).

`data/processed/feature_roles.json` is the machine-readable contract: `model_features` (39), `target`,
`raw_price_level_excluded` (non-stationary price levels kept but NOT in X), `diagnostic_cols`, `exog_raw`.
**M3 trains on `MODEL_FEATURES` only and `dropna(subset=MODEL_FEATURES)`** (warm-up NaNs; see
`feature_nan_report.md`).

## Conventions

- **Edit the `.py`, then regenerate the `.ipynb`**: `python scripts/py_to_nb.py notebooks/NN_x.py notebooks/NN_x.ipynb`.
  Never hand-edit the `.ipynb`.
- **Pinned date window** (`START`/`END` constants in `01`) — never anchor to "today"; snapshots + pins = reproducible.
- **Seed 42** everywhere (`np.random`, `tf.random.set_seed`, LightGBM `random_state`).
- **Prints are ASCII-only** (Windows cp1252 console can't encode `✅`, `Δ`, `²`); markdown/reports may use Unicode.
- **Every self-audit check is bound to a computed result** — no hardcoded `True`.
- Yahoo returns `Adj Close` split/dividend-adjusted; all returns use it (raw `Close` jumps on splits, e.g. NVDA 2024-06-10).

## Adversarial audit workflow (core to this project)

After building each milestone, spawn review subagents **in parallel** (they only report; the main thread applies
fixes and re-runs once): **Investment Researcher** (finance/quant correctness), **Model QA Specialist** (leakage,
alignment, honesty), **Data Engineer** (M1 pipeline). They have caught real defects every milestone — e.g. a
next-day-target off-by-one BLOCKER, global-mean target-encoding leakage, a vacuous sealing check, pooled-test
invalidity, an EGARCH mislabel. Findings + fixes are logged in each `M*.md` Corrections section and the Decisions
Log. Keep this discipline for M4.

## Status

- M1 (data + 9 exogenous features), M2 (EDA), M3 (models + enhancement pass) — **done, multi-agent audited**.
- **Attention-LSTM trained** (20-epoch, CPU-verified; `FULL_TRAIN` auto-runs on Colab GPU — guide in
  `reports/COLAB_TRAINING_GUIDE.md`). Holdout DirAcc 0.542 (p=0.005), RMSE worse than naive — weak/uneconomic.
- **Outstanding**: M4 (Evaluation & Presentation) not started. See `Plans/progress_checklist.md` for the tracker.

## How to run (local)

```bash
py -3.10 -m venv .venv && .venv\Scripts\activate
python -m pip install -r requirements.txt
python notebooks/01_data_collection_preprocessing.py   # then 02, then 03
```
`fc.yahoo.com` (yfinance consent endpoint) may be blocked on some networks; the notebook auto-falls back to the
direct Yahoo chart API. In Colab, `yfinance` works directly and the committed `data/raw/*.csv` are read if present.
