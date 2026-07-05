# Milestone 4 — Evaluation & Presentation
**Project:** Stock Market Trend Analysis | **Week:** 13 | **Weight:** 7.5% of course | **Seed:** 42

> **How to use this file:** You are completing the final milestone. M3 produced holdout predictions and identified a winner (or honestly declared baselines as the winner). Your job: assess limitations, audit for ethical risks (financial-advice misuse), translate model performance into a careful narrative, and deliver a presentation that tells the **journey**: question → data → model → insight → limitations. The course rubric explicitly demands the journey narrative.

---

## 0. Prerequisites

- M3 outputs: `models/`, `data/processed/holdout_predictions.parquet`, all M3 figures.
- All `reports/milestones/M1.md`, `M2.md`, `M3.md` complete.
- Libraries: `pandas matplotlib seaborn python-pptx` (or `python-docx` / plain markdown if pptx unavailable).
- Open `notebooks/04_evaluation_presentation.ipynb`.

## 1. Final Performance Report

Build a clean comparison table in `M4.md`:

| Model | Holdout RMSE | MAE | Directional Acc | Δ vs best baseline |
|---|---|---|---|---|
| naive_yesterday | … | … | ~ 0.50 | — |
| naive_zero | … | … | n/a | — |
| moving_avg_20 | … | … | … | — |
| ARIMA (mean over tickers) | … | … | … | **±X%** |
| LSTM global | … | … | … | **±Y%** |

Add a one-paragraph interpretation: how much (if any) of the achievable gap (best baseline → perfect-foresight upper bound) did the model close? Be honest if it's small or negative — that's the more interesting finding.

## 2. Error Analysis

Load `holdout_predictions.parquet`. Produce:
- Top-10 worst forecasts (highest `abs_err`): table with `date, ticker, y_true, y_pred, abs_err` + a narrative explanation for each (was it a regime shift? an earnings day? a Fed announcement?).
- Distribution of `abs_err` (histogram + 50/90/99 percentile annotations) → `M4_fig_err_dist.png`.
- Error by month (regime check): is the model worse during 2025's volatile periods? → `M4_fig_err_by_month.png`.
- Directional confusion matrix: predicted-up actual-up / predicted-down actual-down → `M4_fig_confusion.png`.

## 3. Limitations (≥ 5, named explicitly)

Finance-specific required limitations:
1. **Efficient-market hypothesis (EMH):** publicly available price history is already priced in — predictability beyond chance is theoretically bounded; this constrains *any* model trained on prices alone.
2. **Regime shifts:** the holdout period (2025+) includes monetary-policy and AI-cycle changes the training data (2016–2023) cannot capture.
3. **Black swans:** unprecedented events (e.g. unscheduled Fed action, geopolitical shock) are inherently unpredictable; the model has no error bars on them.
4. **No exogenous features:** we used only price history. Real signals (earnings, news sentiment, macro releases, options-implied vol) are absent.
5. **Survivorship bias in ticker selection:** AAPL/AMZN/NVDA are currently large-cap survivors; a fair evaluation would include delisted/struggling tickers from 2016.
6. **Look-ahead bias risk:** even with the leakage assertions, target-encoded features computed on training stats are subject to subtle leaks if the train window shifts.
7. **Daily granularity:** intraday signal (e.g. opening gaps) is invisible to a daily model.

## 4. Ethical & Misuse Audit

Mandatory section even though the dataset has no human subjects.

### 4a. Misuse surface
One paragraph: a retail investor who follows this model's predictions in the real world could lose real money. The directional accuracy of stock-return models is consistently in the 50–55% range — barely better than a coin flip, and below transaction costs for most retail brokerages. State this plainly.

### 4b. Required disclaimer
Include in `M4.md` and on the title slide:

> **This project is for educational purposes only. The forecasts produced by this model are not investment advice. Past performance does not predict future returns. The author does not recommend trading on these predictions.**

### 4c. Fairness / equity considerations
Less applicable than a human-subject dataset, but state:
- Models trained on US large-cap tech may not generalise to small-cap, emerging-market, or non-equity assets.
- A retail trader using this model would be competing against quant funds with vastly better data, latency, and compute — there is an asymmetry-of-information harm.

### 4d. Data protection note
Yahoo Finance data is public. State what would change with a real broker's data: customer order flow is highly sensitive; access to it would require regulatory compliance (GDPR/CCPA, FINRA, MiFID II depending on jurisdiction).

## 5. Business-Impact Narrative (with strong disclaimer)

Run the **simple long-only backtest** from M3:
- Buy when `y_pred > 0`, hold cash otherwise.
- Compare cumulative return vs buy-and-hold ^GSPC over the holdout period.
- Assume 0 transaction costs (acknowledge this is unrealistic; real costs would erode 1–3% annually for a daily-rebalanced strategy).

Translate to dollars only as illustration:
```
Hypothetical $10,000 invested at holdout start:
  Buy-and-hold ^GSPC:    $X
  Long-when-positive:    $Y
  Difference:            $Y - $X  (over [holdout duration] period)
```

State explicitly: this is a back-test in hindsight, not a tradeable strategy, and the **method** of translation is what the rubric grades, not the dollar number.

## 6. Reproducibility Audit

Run from a fresh shell:
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
jupyter execute notebooks/01_data_collection_preprocessing.ipynb
jupyter execute notebooks/02_eda.ipynb
jupyter execute notebooks/03_model_building.ipynb
jupyter execute notebooks/04_evaluation_presentation.ipynb
```
All notebooks must finish without manual intervention. Pin every library version. Update `README.md` "How to reproduce" section.

## 7. Final Presentation (10–12 slides)

Build `reports/M4_presentation.pptx` (use `python-pptx`; fall back to `reports/M4_presentation_outline.md` if pptx isn't available). One bullet = one idea. Every slide needs **speaker notes**.

| # | Slide | Content |
|---|---|---|
| 1 | Title + disclaimer | Project name, your name, date, course, **the not-investment-advice disclaimer** |
| 2 | The question | Primary question + stakeholders (from M1) |
| 3 | The data | 4 tickers, 10 years, source = Yahoo Finance (from M1) |
| 4 | EDA highlight A | Strongest figure: rolling volatility / clustering (from M2) |
| 5 | EDA highlight B | ACF of returns ≈ 0 (EMH); ACF of `r²` strong (clustering) (from M2) |
| 6 | Modeling approach | Baselines + ARIMA + LSTM, why (from M3) |
| 7 | Results | Holdout comparison table + delta vs baseline (from §1 above) |
| 8 | Error analysis | Top errors + regime breakdown (from §2) |
| 9 | Limitations & ethics | The 5+ listed in §3 + misuse disclaimer |
| 10 | Backtest narrative | Cumulative-return plot + heavy disclaimer (from §5) |
| 11 | What I learned | The journey: what didn't work, what I changed |
| 12 | Q&A | "What would you ask me?" — prepare 3 anticipated questions with answers in speaker notes |

## 8. Journey Narrative (`M4.md` lead section)

Write 300–500 words narrating: question → data → model → insight → limitations. This is **the** rubric-critical section. Show iteration: what didn't work, what you changed, what you learned. Reference specific decisions from M1–M3 (e.g., "deciding to model log returns rather than prices after the M1 ADF test was the single most consequential choice — prices fail stationarity and ARIMA cannot fit them sensibly").

## Expected Outputs

- `notebooks/04_evaluation_presentation.ipynb`
- `reports/figures/M4_fig_{err_dist, err_by_month, confusion, backtest_cum}.png`
- `reports/M4_presentation.pptx` (or `_outline.md`)
- `reports/milestones/M4.md` (Journey → Performance → Error analysis → Limitations → Ethics → Backtest → Reproducibility → Self-Audit)
- Updated root `README.md` with reproduce instructions
- Pinned `requirements.txt`

## Self-Audit Table

| Criterion | Status | Evidence |
|---|---|---|
| Journey narrative present (question → … → limitations) | ☐ | … |
| Holdout numbers vs baselines reported | ☐ | … |
| Error analysis with top-10 worst + by-month plot | ☐ | … |
| ≥ 5 named limitations (finance-specific) | ☐ | … |
| Misuse / not-investment-advice disclaimer present | ☐ | … |
| Backtest with explicit assumptions + disclaimer | ☐ | … |
| Reproducibility verified (fresh-env notebook run) | ☐ | … |
| Presentation deck 10–12 slides + speaker notes | ☐ | … |
| All four `reports/milestones/M*.md` files present | ☐ | … |

## Forbidden

- Reporting only the headline metric without the baseline comparison.
- Presenting backtest cumulative returns without the "not investment advice" disclaimer.
- Claiming the model "works for trading."
- Skipping the limitations section because "the results were good."

## Project Complete

> **Final state:** Full stock-trend-analysis pipeline reproducible from yfinance snapshot → cleaned features → EDA → ARIMA + LSTM → holdout evaluation → backtest → presentation. All artifacts under `data/`, `models/`, `reports/`. Submit the four `reports/milestones/M*.md` files + the presentation + a link to the repo.
