# Milestone 3 — Model Building (ARIMA vs LSTM)
**Project:** Stock Market Trend Analysis | **Week:** 11 | **Weight:** 7.5% of course | **Seed:** 42

> **How to use this file:** You are completing Milestone 3. M1 produced the feature dataset, M2 confirmed the patterns. Build baselines, train two real models (one classical, one deep), pick a winner, and document the journey honestly — including the very real possibility that **the baselines win**. Never use random K-fold. Never touch holdout until the very last step.

---

## 0. Prerequisites

- M1 outputs: `data/processed/{train_fe,val_fe,holdout_fe}.parquet`
- M2 conclusions: ARIMA orders informed by ACF/PACF; log returns are the target; volatility clusters.
- Libraries: `statsmodels tensorflow scikit-learn matplotlib`. Add to `requirements.txt`: `tensorflow==2.18.0`.
- Open `notebooks/03_model_building.ipynb`. Set `SEED=42`. Set `tf.random.set_seed(42)`.

## 1. Metrics

```python
def rmse(y_true, y_pred):
    import numpy as np
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    import numpy as np
    return float(np.mean(np.abs(y_true - y_pred)))

def directional_accuracy(y_true, y_pred):
    import numpy as np
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))
```

Primary = **RMSE** (on log returns). Secondary = MAE, directional accuracy. Tertiary = per-ticker breakdown.

**Reality check (cite in M3.md):** Even with great features, expect directional accuracy in the 50–55% range on log returns — this is normal for stock prediction and a 52% directional model can still be profitable. Beating naive RMSE by 5–10% is a meaningful win.

## 2. Baselines (must build all three)

| Baseline | Definition |
|---|---|
| `naive_yesterday` | `ŷ_t = r_{t-1}` per ticker (predict tomorrow = yesterday) |
| `naive_zero` | `ŷ_t = 0` (predict no change — close to optimal under EMH) |
| `moving_avg_20` | `ŷ_t = mean(r_{t-20..t-1})` |

Score each on `val_fe`. Tabulate RMSE / MAE / directional accuracy in `M3.md`. Your trained models must beat the best of these by ≥ 5% RMSE on holdout to claim success (a stretch goal — the **honest** outcome may be that baselines win, which is itself the point of the milestone).

## 3. Classical Model — ARIMA (per ticker, 4 models)

For each ticker:
```python
from statsmodels.tsa.arima.model import ARIMA
s = train_fe.query("ticker == @ticker").set_index("date")["log_return"].dropna()
# Order suggested by M2's ACF/PACF — typically small. Try (1,0,1), (2,0,2), (5,0,0).
model = ARIMA(s, order=(1, 0, 1)).fit()
forecast = model.forecast(steps=len(val_slice))
```
- Diagnose residuals: Ljung–Box p-value (residual should look like white noise), residual ACF.
- Forecast over the val period; score against the corresponding `val_fe` log returns.
- **Honest documentation in `M3.md`:** "ARIMA on log returns approaches the naive zero baseline because EMH compresses the predictable signal."

> **M2 findings that shape this (read before fitting):**
> - **Target is `target_log_return` (next-day)** — fit ARIMA on `log_return` to forecast the next step; align to the next-day target. Use `MODEL_FEATURES` only; `dropna(subset=MODEL_FEATURES)`.
> - **ARIMA order:** M2 found the only non-trivial mean structure is a **negative lag-1** ACF (−0.162, a mean-reversion/bid-ask-bounce microstructure effect), so try small orders **(1,0,1), (1,0,0), (0,0,1)**; `d=0` (returns already mean-stationary). Expect ARIMA-on-mean to ≈ the naive baseline net of costs — that's the honest, expected result.
> - **Variance model = asymmetric GARCH.** M2 confirmed strong volatility clustering (ARCH-LM p≈0 all tickers) AND a significant **leverage effect** (down days raise next-day vol more, p≤0.035 all tickers). Pair each ARIMA with an **EGARCH or GJR-GARCH(1,1)** (e.g. `arch` package: `arch_model(r, vol="EGARCH", p=1, o=1, q=1, dist="t")`), NOT a symmetric GARCH(1,1). Use Student-t errors (returns are fat-tailed, JB rejects normality).

## 4. Deep Model — LSTM Global (1 model, all tickers)

```python
import tensorflow as tf
from tensorflow import keras

SEQ_LEN = 60  # 60 trading days ≈ 3 months
def make_sequences(df, feature_cols, target_col, seq_len):
    # Build (N, seq_len, n_features) array, target = next-day log_return
    ...

# FEATS = roles["model_features"] (30 cols); target = "target_log_return" (next-day, already aligned in M1)
X_tr, y_tr = make_sequences(train_fe, FEATS, "target_log_return", SEQ_LEN)
X_va, y_va = make_sequences(val_fe,   FEATS, "target_log_return", SEQ_LEN)

model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=False, input_shape=(SEQ_LEN, len(FEATS))),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=20, batch_size=64, callbacks=[
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
])
```

Critical details:
- **Scale features first** (StandardScaler fit on train only).
- **No shuffling** in `model.fit` — set `shuffle=False`.
- **Ticker** must be one-hot encoded so a single model can learn per-ticker effects.
- **TimeSeriesSplit** for hyperparameter selection (don't K-fold).
- Try at least 2 architecture variants: 1-layer LSTM vs 2-layer LSTM. Pick by val RMSE.

> **Why the LSTM is worth it (M2 evidence):** Pearson correlation of every feature with the next-day target is
> ~0, but **mutual information** ranks the volatility features ~67× the noise floor (`parkinson_vol_21` MI=0.16
> vs Pearson ~0.01). The exploitable signal is **nonlinear and volatility-based** — exactly what the LSTM can
> capture and ARIMA-on-mean cannot. Lead the LSTM justification with the MI result, not Pearson. Consider
> adding the M2-suggested **downside-vol** and **vol-regime** features.

## 5. Model Comparison Table (in `M3.md`)

| Model | Val RMSE | Val MAE | Val Directional Acc | Notes |
|---|---|---|---|---|
| naive_yesterday | … | … | ~ 0.50 | baseline |
| naive_zero | … | … | n/a | EMH benchmark |
| moving_avg_20 | … | … | … | baseline |
| ARIMA (per ticker, mean) | … | … | … | classical |
| LSTM global (tuned) | … | … | … | **candidate winner** |

Report mean + per-ticker breakdown. State the winner and the reasoning. **If baselines win, that is the correct finding — document it; don't fabricate a winner.**

## 6. Winner Diagnostics

- Residual plot: `pred - actual` vs `pred` → `M3_fig_residuals.png`.
- Error by ticker: barplot of per-ticker RMSE → `M3_fig_error_by_ticker.png`.
- Error by month: barplot → `M3_fig_error_by_month.png` (does the model fail in volatile months?).
- LSTM training curves: train/val loss → `M3_fig_lstm_curves.png`.

## 7. Final Holdout Evaluation

Only now:
```python
# Refit best model on train+val
X_final = pd.concat([train_fe, val_fe])
# ... refit ARIMA per ticker and refit LSTM
# ... predict on holdout_fe
final_rmse = rmse(holdout_fe.log_return, preds)
final_da   = directional_accuracy(holdout_fe.log_return, preds)
```
Save:
- `models/arima_<ticker>.pkl` (4 files) and `models/lstm_global.keras`
- `data/processed/holdout_predictions.parquet` with columns `date, ticker, y_true, y_pred_arima, y_pred_lstm`
- A simple long-only backtest: cumulative return if you `long when y_pred > 0, flat otherwise` vs buy-and-hold ^GSPC → `M3_fig_backtest.png`. Disclaimer required.

## 8. Leakage Verification

Run and document in `M3.md`:
```python
assert (train_fe.date.max() < val_fe.date.min()) and (val_fe.date.max() < holdout_fe.date.min())
# Spot-check: no NaN lag features past their warm-up window
# Spot-check: scaler fit on train only, applied to val/holdout
```

## Expected Outputs

- `notebooks/03_model_building.ipynb`
- `models/{arima_GSPC, arima_AAPL, arima_AMZN, arima_NVDA}.pkl`
- `models/lstm_global.keras`
- `data/processed/holdout_predictions.parquet`
- `reports/figures/M3_fig_{residuals, error_by_ticker, error_by_month, lstm_curves, backtest}.png`
- `reports/milestones/M3.md` (Metrics → Baselines → ARIMA → LSTM → Comparison → Diagnostics → Holdout → Backtest → Leakage check → Self-Audit)

## Self-Audit Table

| Criterion | Status | Evidence |
|---|---|---|
| ≥ 3 baselines scored | ☐ | … |
| 4 per-ticker ARIMA models fit, residuals diagnosed | ☐ | … |
| 1 global LSTM trained with sequence input, no shuffle | ☐ | … |
| TimeSeriesSplit / no random shuffle used throughout | ☐ | … |
| Per-ticker error breakdown reported | ☐ | … |
| Honest discussion if baselines win | ☐ | … |
| Simple long-only backtest with disclaimer | ☐ | … |
| Leakage assertion passes | ☐ | … |
| Notebook reproducible (seed=42 + tf.random.set_seed(42)) | ☐ | … |

## Forbidden

- `KFold` / `train_test_split(shuffle=True)`.
- `model.fit(..., shuffle=True)` on time-series sequences.
- Scoring on holdout before final retraining step.
- Fitting the StandardScaler on the full dataset.
- Reporting cumulative-return backtest without a "**not investment advice**" disclaimer.
- Cherry-picking a ticker where the model happened to do well to claim overall success.

## Hand-off to Milestone 4

> **State of project after M3:** Winner = [model name] with holdout RMSE = [X.XXXX] on log returns, directional accuracy = [Y%]. Baselines on holdout: best baseline = [name] with RMSE = [Z.ZZZZ]. Δ vs baseline = [±N%]. Model artifacts at `models/`; per-row predictions at `data/processed/holdout_predictions.parquet`. Next: error analysis by regime, limitations specific to finance, simple backtest narrative, and final presentation.
