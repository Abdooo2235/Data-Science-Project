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
│   ├── 01_data_collection_preprocessing.ipynb    # M1 — Colab-ready
│   └── 01_data_collection_preprocessing.py       # jupytext source for diffs
├── models/                        # M3 will populate
├── reports/
│   ├── figures/                   M1_*.png ...
│   └── milestones/                M1.md, M2.md ...
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

# run M1
python -m jupyter execute notebooks/01_data_collection_preprocessing.ipynb
```

Outputs land in `data/processed/`, `reports/figures/`, `reports/milestones/`. Seed is fixed at `42` throughout.

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
- **M4** ☐ — Evaluation & presentation. Spec: [Plans/milestone4_evaluation_presentation.md](Plans/milestone4_evaluation_presentation.md).

## Limitations (preview — full discussion in M4)

Financial markets are **not** like the classic supervised-learning datasets:

- **Efficient-market hypothesis** says publicly known information is already priced in — predictability beyond chance is theoretically bounded.
- **Regime shifts** (COVID 2020, rate hikes 2022, AI boom 2023) break stationarity assumptions ARIMA relies on.
- **Black swans** (events with no historical analog) are unpredictable by definition.
- **Survivorship bias** — we picked tickers that *currently* exist and look interesting; models tested on these flatter forecasters.

**This project is for learning, not for trading.** Treat the numbers as evidence about modeling tradeoffs, not as investment advice.
