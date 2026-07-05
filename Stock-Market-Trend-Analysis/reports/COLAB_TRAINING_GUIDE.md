# Colab Training Guide — Full Attention-LSTM (M3)

Run `notebooks/03_model_building.ipynb` on a Google Colab **GPU** to train the attention-LSTM for real
(20 epochs) and regenerate the prediction artifacts. The notebook auto-detects the GPU — **you do not edit any
epoch/config value by hand**.

> The classical models (ARIMA, GJR-GARCH, LightGBM) train in seconds on CPU and their numbers don't change on
> Colab. Only the LSTM needs the GPU.

---

## 0. Set the runtime to GPU

Colab menu → **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.
(Free T4 trains this model in ~1–2 min; CPU takes ~15–20 min but still works.)

## 1. Get the data + notebook onto Colab

The notebook expects everything under `/content/Stock-Market-Trend-Analysis/`. Pick **one**:

**Option A — clone from GitHub** (if you pushed the repo). First Colab cell:
```python
!git clone https://github.com/<you>/Stock-Market-Trend-Analysis.git /content/Stock-Market-Trend-Analysis
```

**Option B — upload the files manually** (no GitHub needed). Upload the notebook via **File → Upload notebook**,
then in the first cell create the folders and upload the 4 required data files:
```python
import os
from google.colab import files
os.makedirs("/content/Stock-Market-Trend-Analysis/data/processed", exist_ok=True)
os.makedirs("/content/Stock-Market-Trend-Analysis/data/raw", exist_ok=True)
os.chdir("/content/Stock-Market-Trend-Analysis/data/processed")
files.upload()   # pick: train_fe.parquet, val_fe.parquet, holdout_fe.parquet, feature_roles.json
os.chdir("/content")
```

**Required files** (from your local `data/processed/`):
`train_fe.parquet`, `val_fe.parquet`, `holdout_fe.parquet`, `feature_roles.json`.
(The `data/raw/*.csv` snapshots are **not** needed for M3 — they're only for M1.)

**Path handling:** you don't set any paths yourself. The notebook's setup cell detects Colab and sets
`ROOT = /content/Stock-Market-Trend-Analysis`, then reads `ROOT/data/processed/*.parquet`. As long as the files
sit there (Option A or B), there are no path errors. It also `pip install`s `requirements.txt` if present, else
`arch`, `tensorflow`, `lightgbm`.

## 2. Full 20-epoch GPU training — automatic

There is **no manual epoch change**. The LSTM cell has:
```python
FULL_TRAIN = bool(len(tf.config.list_physical_devices("GPU"))) or os.environ.get("LSTM_FULL_TRAIN") == "1"
EPOCHS = 20 if FULL_TRAIN else 2
lstm = build_lstm(..., units=64 if FULL_TRAIN else 32, layers=2 if FULL_TRAIN else 1, attention=True)
```
On a GPU runtime `FULL_TRAIN` becomes **True** by itself → 20 epochs, 2-layer, 64-unit, early stopping.
Just **Runtime → Run all**. Confirm the print near the LSTM cell says `FULL_TRAIN=True`.

(If you're on a CPU runtime but still want the full run, add a cell `import os; os.environ["LSTM_FULL_TRAIN"]="1"`
**before** the LSTM cell.)

## 3. Verify attention + LightGBM + ensemble are running

Check these in the cell outputs, in order:

1. **GPU active** — run once: `import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))` → non-empty.
2. **Attention layer present** — after the LSTM builds, run `lstm.summary()`. You must see a
   `Softmax`, a `Multiply`, and a `Lambda`/reduce-sum layer (the additive attention pooling over the 60-day
   window). If those are missing, `attention=True` didn't take.
3. **Full training, not smoke** — the LSTM print shows `FULL_TRAIN=True  LSTM+Attention (FINAL)` and ~20 epoch
   lines (early stopping may end sooner). The model row appears in the §6 comparison table (smoke never does).
4. **LightGBM ran + used exogenous data** — §5b prints `exogenous features in LightGBM top-15: [...]` (expect
   VIX/yield/dollar names) and saves `reports/figures/M3_fig_lgb_importance.png`.
5. **Ensemble used the real LSTM** — §5c prints `Ensemble (REAL LSTM; directional only)` (not "illustrative,
   LSTM=smoke").
6. **Holdout LSTM filled** — §8 prints a `Pooled holdout LSTM+Attention: ...` line, and
   `holdout_predictions.parquet` has a non-null `y_pred_lstm` (1,140 of 1,380 rows — the first 60/ticker stay
   NaN by design). The self-audit prints `All 19 M3 self-audit checks passed`.

## 4. Extract + download the artifacts

Everything is written under `ROOT/models/`, `ROOT/data/processed/`, `ROOT/reports/figures/`. Zip and download:

```python
import shutil
from google.colab import files
root = "/content/Stock-Market-Trend-Analysis"
shutil.make_archive("/content/m3_trained", "zip", root, base_dir="models")           # models/
shutil.make_archive("/content/m3_preds", "zip", root, base_dir="data/processed")     # incl. holdout_predictions.parquet
shutil.make_archive("/content/m3_figs", "zip", root, base_dir="reports/figures")     # M3 figures
for z in ("m3_trained", "m3_preds", "m3_figs"):
    files.download(f"/content/{z}.zip")
```

**What you get** (the things that update your M3 report):

| File | What it is |
|---|---|
| `models/lstm_attention_final.keras` | the trained attention-LSTM weights |
| `models/lstm_val_metrics.json` | real LSTM **val** RMSE / MAE / DirAcc / DirAcc_p |
| `models/val_scores.csv` | the full val comparison table incl. the real LSTM + ensemble rows |
| `data/processed/holdout_predictions.parquet` | now has `y_pred_lstm` filled (+ `y_pred_arima`, `y_pred_garch`, `y_pred_lgb`) |
| console line `Pooled holdout LSTM+Attention: RMSE=… DirAcc=… (binomial p=…)` | the real LSTM **holdout** metrics |

**Updating the M3 report:** replace the placeholder LSTM row in `reports/milestones/M3.md` §6 (val) and §8
(holdout) with the numbers from `lstm_val_metrics.json` and the `Pooled holdout LSTM+Attention` print. RMSE and
directional accuracy are already computed and printed — copy them straight in.

> **Reproducibility note:** GPU training has minor run-to-run nondeterminism (seeds pin the graph, not every
> GPU op). Expect the LSTM numbers to move by a small amount between runs; that is normal and doesn't change the
> conclusion. A local CPU 20-epoch run of this exact path gives LSTM holdout DirAcc ≈ 0.542 (binomial p ≈ 0.005)
> with RMSE slightly worse than naive — a weak, statistically-significant-but-uneconomic directional edge, same
> pattern as GJR-GARCH. Don't expect the GPU run to beat the naive baseline economically; the EMH ceiling holds.
