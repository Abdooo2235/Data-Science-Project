# Graph Report - .  (2026-07-05)

## Corpus Check
- 52 files · ~410,281 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 155 nodes · 213 edges · 14 communities (13 shown, 1 thin omitted)
- Extraction: 83% EXTRACTED · 17% INFERRED · 0% AMBIGUOUS · INFERRED: 36 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Data Collection & Preprocessing|Data Collection & Preprocessing]]
- [[_COMMUNITY_Project Architecture & Docs|Project Architecture & Docs]]
- [[_COMMUNITY_Model Building & Evaluation Code|Model Building & Evaluation Code]]
- [[_COMMUNITY_Model Results & Backtest|Model Results & Backtest]]
- [[_COMMUNITY_Return Distribution & Tail Risk|Return Distribution & Tail Risk]]
- [[_COMMUNITY_Autocorrelation & Volatility|Autocorrelation & Volatility]]
- [[_COMMUNITY_Notebook Conversion Utility|Notebook Conversion Utility]]
- [[_COMMUNITY_Milestone Specs & References|Milestone Specs & References]]
- [[_COMMUNITY_Price Trends & Normalisation|Price Trends & Normalisation]]
- [[_COMMUNITY_Feature Selection Signal|Feature Selection Signal]]
- [[_COMMUNITY_EDA Utilities|EDA Utilities]]
- [[_COMMUNITY_Cross-Asset Correlation|Cross-Asset Correlation]]
- [[_COMMUNITY_Seasonal Decomposition|Seasonal Decomposition]]
- [[_COMMUNITY_Monthly Seasonality|Monthly Seasonality]]

## God Nodes (most connected - your core abstractions)
1. `M3 Report — Model Building` - 10 edges
2. `CLAUDE.md — Architecture Reference` - 8 edges
3. `M1 Report — Data Collection & Preprocessing` - 8 edges
4. `Project Decisions & Architecture Report` - 7 edges
5. `LightGBM (tabular, exogenous)` - 7 edges
6. `enforce_schema()` - 6 edges
7. `_fetch_yfinance()` - 6 edges
8. `_fetch_yahoo_direct()` - 6 edges
9. `fetch_with_snapshot()` - 6 edges
10. `make_features()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Restaurant Wait Time Predictor — M1 Reference PDF (References)` --semantically_similar_to--> `Restaurant Wait Time Predictor — M1 Reference PDF (root)`  [INFERRED] [semantically similar]
  Stock-Market-Trend-Analysis/References/Milestone 1_ Data Collection & Preprocessing · Restaurant Wait Time Predictor.pdf → Milestone 1_ Data Collection & Preprocessing · Restaurant Wait Time Predictor.pdf
- `Milestone 1 Spec — Data Collection & Preprocessing` --references--> `Restaurant Wait Time Predictor — M1 Reference PDF (References)`  [INFERRED]
  Stock-Market-Trend-Analysis/Plans/milestone1_data_collection_preprocessing.md → Stock-Market-Trend-Analysis/References/Milestone 1_ Data Collection & Preprocessing · Restaurant Wait Time Predictor.pdf
- `Milestone-Driven Project Pattern (Mhmd-Ram/DS-Project)` --conceptually_related_to--> `Restaurant Wait Time Predictor — M1 Reference PDF (References)`  [INFERRED]
  Stock-Market-Trend-Analysis/README.md → Stock-Market-Trend-Analysis/References/Milestone 1_ Data Collection & Preprocessing · Restaurant Wait Time Predictor.pdf
- `requirements.txt — Pinned Dependencies` --conceptually_related_to--> `GJR-GARCH(1,1) Student-t`  [INFERRED]
  Stock-Market-Trend-Analysis/requirements.txt → Stock-Market-Trend-Analysis/reports/milestones/M3.md
- `requirements.txt — Pinned Dependencies` --conceptually_related_to--> `LightGBM (tabular, exogenous)`  [INFERRED]
  Stock-Market-Trend-Analysis/requirements.txt → Stock-Market-Trend-Analysis/reports/milestones/M3.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Milestone-Driven Pipeline (M1 to M4)** — stock_market_trend_analysis_plans_milestone1_data_collection_preprocessing_spec, stock_market_trend_analysis_plans_milestone2_eda_spec, stock_market_trend_analysis_plans_milestone3_model_building_spec, stock_market_trend_analysis_plans_milestone4_evaluation_presentation_spec [EXTRACTED 1.00]
- **Leakage-Prevention Pattern** — stock_market_trend_analysis_claude_leakage_prevention_invariants, stock_market_trend_analysis_data_processed_feature_dictionary_target_log_return, stock_market_trend_analysis_reports_milestones_m1_chronological_split, stock_market_trend_analysis_data_processed_feature_dictionary_model_features [EXTRACTED 1.00]
- **Model Stack (classical vs tree vs deep)** — stock_market_trend_analysis_reports_milestones_m3_arima, stock_market_trend_analysis_reports_milestones_m3_gjr_garch, stock_market_trend_analysis_reports_milestones_m3_attention_lstm, stock_market_trend_analysis_reports_milestones_m3_lightgbm [EXTRACTED 1.00]

## Communities (14 total, 1 thin omitted)

### Community 0 - "Data Collection & Preprocessing"
Cohesion: 0.13
Nodes (27): DataFrame, add_next_day_target(), add_warmup_flag(), enforce_schema(), fetch_with_snapshot(), _fetch_yahoo_direct(), _fetch_yfinance(), make_datetime_features() (+19 more)

### Community 1 - "Project Architecture & Docs"
Cohesion: 0.14
Nodes (28): CLAUDE.md — Architecture Reference, Leakage-Prevention Invariants, Multi-Agent Adversarial Audit, Feature Dictionary, MODEL_FEATURES Contract (feature_roles.json), Next-Day Target (target_log_return), Feature NaN / Warm-up Report, Trained LightGBM Model — Final (train+val) (+20 more)

### Community 2 - "Model Building & Evaluation Code"
Cohesion: 0.11
Nodes (21): add_onehot(), build_lstm(), build_matrix(), clean_xy(), diebold_mariano(), diracc_pvalue(), directional_accuracy(), mae() (+13 more)

### Community 3 - "Model Results & Backtest"
Cohesion: 0.18
Nodes (13): ARIMA Residual Diagnostics, Backtest Performance (Sharpe, growth-of-$1), Directional Accuracy, Efficient-Market Ceiling (no usable OOS skill), Error Analysis (RMSE by segment), Leftover ARCH / Volatility Clustering, LightGBM Feature Importance, M3 figure: ^GSPC ARIMA residual ACF (raw resid; leftover ARCH expected) (+5 more)

### Community 4 - "Return Distribution & Tail Risk"
Cohesion: 0.21
Nodes (12): Tail-event flagging / outlier retention (MAD z>3.5), M1: Tail-event flagging (retain + flag, no rows deleted) — log_return scatter per ticker, Long-run price trend (Adj Close, non-stationary level), M1: 10-year Adj Close price trends (Yahoo Finance, daily) for ^GSPC/AAPL/AMZN/NVDA, Log Returns Distribution, M1: Daily log-return distribution per ticker (10 yr) overlaid histogram, Fat Tails / Excess Kurtosis (leptokurtosis), M2 Fig 1: Log-return distribution vs normal fit (fat tails, excess kurtosis per ticker) (+4 more)

### Community 5 - "Autocorrelation & Volatility"
Cohesion: 0.24
Nodes (11): Autocorrelation (ACF), Leverage Effect, Partial Autocorrelation (PACF), Rolling Volatility, Volatility Clustering, Weak-Form Market Efficiency / Weak Mean Predictability, Fig 7 — 21-day rolling annualised volatility, volatility clustering visible, Fig 11 — ACF of ^GSPC log returns (~0: weak mean predictability) (+3 more)

### Community 6 - "Notebook Conversion Utility"
Cohesion: 0.31
Nodes (8): code_source(), convert(), markdown_source(), parse_cells(), Path, Convert a jupytext "percent" format .py file into a clean .ipynb that opens in J, Split a percent-format .py into a list of (cell_type, lines) tuples., Strip leading `# ` from comment-only lines in a markdown cell.

### Community 7 - "Milestone Specs & References"
Cohesion: 0.29
Nodes (7): Restaurant Wait Time Predictor — M1 Reference PDF (root), Milestone 1 Spec — Data Collection & Preprocessing, Milestone 2 Spec — EDA, Milestone 3 Spec — Model Building, Milestone 4 Spec — Evaluation & Presentation, Milestone-Driven Project Pattern (Mhmd-Ram/DS-Project), Restaurant Wait Time Predictor — M1 Reference PDF (References)

### Community 8 - "Price Trends & Normalisation"
Cohesion: 0.60
Nodes (5): Adjusted Close Price, Price Normalisation (Growth of $1), Fig 5 — Adjusted close price facets, ^GSPC/AAPL/AMZN/NVDA (2016→2023, train only), Fig 6 — Growth of $1 invested 2016-05, normalised price by ticker (NVDA ~46x), Fig 8 — NVDA Adj Close, last 2yr of train (AI-cycle run-up, ~11→50)

### Community 9 - "Feature Selection Signal"
Cohesion: 0.50
Nodes (5): Feature Selection / Predictive Signal, Mutual Information, Pearson Correlation (feature-target), Fig 16 — MODEL_FEATURES vs next-day target Pearson r, Fig 17 — MODEL_FEATURES ranked by mutual information vs next-day target

### Community 10 - "EDA Utilities"
Cohesion: 0.40
Nodes (3): max_drawdown(), Series, Max peak-to-trough drawdown from a log-return series (as a negative fraction).

### Community 11 - "Cross-Asset Correlation"
Cohesion: 0.50
Nodes (4): Correlation Heatmap, Rolling Correlation, Fig 14 — Daily log-return correlation heatmap (train), Fig 15 — Rolling 63-day correlation with ^GSPC

### Community 12 - "Seasonal Decomposition"
Cohesion: 1.00
Nodes (3): Seasonal Decomposition (trend/seasonal/residual), Fig 9 — ^GSPC seasonal decomposition, period=5 (trading week): observed/trend/seasonal/resid, Fig 10 — ^GSPC seasonal decomposition, period=252 (trading year): observed/trend/seasonal/resid

## Knowledge Gaps
- **22 isolated node(s):** `Milestone 4 Spec — Evaluation & Presentation`, `Restaurant Wait Time Predictor — M1 Reference PDF (root)`, `Chronological Time-Based Split`, `M2 Fig 4: Log-return by calendar month (pooled across tickers) — seasonality check`, `Tail-event flagging / outlier retention (MAD z>3.5)` (+17 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `M3 Report — Model Building` connect `Project Architecture & Docs` to `Milestone Specs & References`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `M1 Report — Data Collection & Preprocessing` connect `Project Architecture & Docs` to `Milestone Specs & References`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **What connects ``^GSPC` has a caret that makes a messy filename; strip it.`, `Coerce a freshly-fetched frame to the canonical dtypes so all fetch paths are id`, `Primary path: yfinance. Works on Colab and most networks.` to the rest of the system?**
  _42 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Data Collection & Preprocessing` be split into smaller, more focused modules?**
  _Cohesion score 0.12962962962962962 - nodes in this community are weakly interconnected._
- **Should `Project Architecture & Docs` be split into smaller, more focused modules?**
  _Cohesion score 0.13756613756613756 - nodes in this community are weakly interconnected._
- **Should `Model Building & Evaluation Code` be split into smaller, more focused modules?**
  _Cohesion score 0.11067193675889328 - nodes in this community are weakly interconnected._