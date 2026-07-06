# Stock Market Trend Analysis — Capstone

Forecast next-day price movement for the S&P 500 index plus three tech stocks (AAPL, AMZN, NVDA) using 10 years of daily data. Compare **ARIMA** (classical statistical baseline) against **LSTM** (deep-learning sequence model).

**Project pattern** adapted from [Mhmd-Ram/DS-Project](https://github.com/Mhmd-Ram/DS-Project) — a milestone-driven, AI-executable capstone structure.

**Start here:** [CLAUDE.md](CLAUDE.md) (architecture, data flow, leakage invariants, conventions) · [reports/PROJECT_DECISIONS_AND_ARCHITECTURE.md](reports/PROJECT_DECISIONS_AND_ARCHITECTURE.md) (full decisions & findings report).

## Repo layout

```
.
├── data/
│   ├── raw/                       *.csv snapshots fetched from yfinance (committed)
│   └── processed/                 train_fe.parquet, val_fe.parquet, holdout_fe.parquet, feature_dictionary.md
├── notebooks/
│   ├── 0{1,2,3,4}_*.py            # jupytext source of truth (M1 data, M2 EDA, M3 models, M4 evaluation)
│   └── 0{1,2,3,4}_*.ipynb         # generated for Colab
├── models/                        # arima/garch/lgbm/lstm artifacts, val_scores.csv, m4_holdout_scores.csv
├── reports/
│   ├── figures/                   M1_*, M2_fig01..18_*, M3_fig_*, M4_fig_* .png
│   ├── milestones/                M1.md, M2.md, M3.md, M4.md
│   ├── M4_presentation.html       # 12-slide capstone deck (also a claude.ai Artifact)
│   ├── COLAB_TRAINING_GUIDE.md
│   └── PROJECT_DECISIONS_AND_ARCHITECTURE.md
├── Plans/
│   ├── milestone1_data_collection_preprocessing.md
│   ├── milestone2_eda.md
│   ├── milestone3_model_building.md
│   ├── milestone4_evaluation_presentation.md
│   └── progress_checklist.md
├── References/                    course PDFs + RestaurantSmartWait.ipynb style reference
├── scripts/                       small helper scripts (notebook build, snapshots)
├── requirements.txt
└── README.md
```

## Reproduce

### Option A — Google Colab (recommended for this project)

1. Open `notebooks/01_data_collection_preprocessing.ipynb` in Colab.
2. Run all cells top to bottom (the first cell installs `yfinance` automatically).
3. Outputs land in the Colab session disk under `data/processed/`, `reports/figures/`, etc.

### Option B — Local Python 3.10+

```bash
# create + activate venv
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# install pinned deps
python -m pip install -r requirements.txt

# run the pipeline end to end (the .py files are the source of truth)
python notebooks/01_data_collection_preprocessing.py   # raw snapshots -> features -> splits
python notebooks/02_eda.py                             # EDA on train only
python notebooks/03_model_building.py                  # models, holdout opened once
python notebooks/04_evaluation_presentation.py         # M4: final tables, figures, backtest
```

Outputs land in `data/processed/`, `reports/figures/`, `reports/milestones/`. Seed is fixed at `42` throughout.
The classical models, LightGBM, and all M4 outputs run on a local CPU. The 20-epoch attention LSTM needs a GPU
and runs on Google Colab (see `reports/COLAB_TRAINING_GUIDE.md`); its predictions are already saved in
`data/processed/holdout_predictions.parquet`.

### M5 — run the web app (after the pipeline has produced its artifacts)

```bash
streamlit run app.py        # opens http://localhost:8501
python test_app.py          # smoke test: full render + live-model reproduction check
```

The app loads only committed/produced artifacts (`holdout_predictions.parquet`, `m4_holdout_scores.csv`,
the tuned LightGBM booster) — no training, no network calls. Pick a ticker, model, and holdout date to see
the forecast versus what actually happened, with the honest accuracy table and the not-investment-advice
disclaimer front and center.

## Tickers

| Symbol | Description |
|---|---|
| `^GSPC` | S&P 500 index (benchmark, no dividends) |
| `AAPL` | Apple Inc. |
| `AMZN` | Amazon.com Inc. |
| `NVDA` | NVIDIA Corp. |

Date range: 10 years ending today (rolling window in `notebooks/01_*.ipynb` `start`/`end` cell).

## Milestones

See [Plans/progress_checklist.md](Plans/progress_checklist.md) for the live tracker.

- **M1** ✅ — Data collection & preprocessing. Spec: [Plans/milestone1_data_collection_preprocessing.md](Plans/milestone1_data_collection_preprocessing.md). Report: [reports/milestones/M1.md](reports/milestones/M1.md).
- **M2** ✅ — Exploratory data analysis. Spec: [Plans/milestone2_eda.md](Plans/milestone2_eda.md). Report: [reports/milestones/M2.md](reports/milestones/M2.md).
- **M3** ✅ — Model building (ARIMA + GJR-GARCH + LightGBM + attention-LSTM + ensemble; exogenous features). Spec: [Plans/milestone3_model_building.md](Plans/milestone3_model_building.md). Report: [reports/milestones/M3.md](reports/milestones/M3.md). LSTM training on Colab: [reports/COLAB_TRAINING_GUIDE.md](reports/COLAB_TRAINING_GUIDE.md).
- **M4** ✅ — Evaluation & presentation. Spec: [Plans/milestone4_evaluation_presentation.md](Plans/milestone4_evaluation_presentation.md). Report: [reports/milestones/M4.md](reports/milestones/M4.md). Slide deck: [reports/M4_presentation.html](reports/M4_presentation.html) (open in a browser, or print to PDF).

**Honest headline:** no model trained on price history beats a naive next-day forecast out of sample. The one edge that survives significance testing is the market's upward drift, not timing skill, and it does not survive transaction costs. The efficient-market ceiling holds, and measuring it carefully is the result.

## Limitations (preview — full discussion in M4)

Financial markets are **not** like the classic supervised-learning datasets:

- **Efficient-market hypothesis** says publicly known information is already priced in — predictability beyond chance is theoretically bounded.
- **Regime shifts** (COVID 2020, rate hikes 2022, AI boom 2023) break stationarity assumptions ARIMA relies on.
- **Black swans** (events with no historical analog) are unpredictable by definition.
- **Survivorship bias** — we picked tickers that *currently* exist and look interesting; models tested on these flatter forecasters.

**This project is for learning, not for trading.** Treat the numbers as evidence about modeling tradeoffs, not as investment advice.
