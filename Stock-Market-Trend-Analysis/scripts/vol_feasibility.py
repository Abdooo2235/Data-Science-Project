"""Volatility-prediction feasibility probe (exploratory — NOT the sealed pipeline).

Question: is forward realized volatility predictable where the return sign was not?
Target = h-day forward realized vol RV_t = sqrt(sum of squared log returns t+1..t+h), per ticker,
built leakage-safe (shift(-h); last h rows per split = NaN). Two models:
  - HAR-RV: OLS on log(realized_vol_5/21/63) — the standard, dead-simple volatility benchmark.
  - LightGBM on the full MODEL_FEATURES set.
Baseline = random-walk vol (predict today's realized_vol_h-equivalent). If OOS R2 >> 0 we have
a real, honest positive result (unlike returns). QLIKE reported too (the vol-forecasting loss).

ponytail: cheapest thing that proves signal before writing the M3.6 plan. Zero sealed files touched.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 42
HORIZONS = [1, 5, 20]
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
EPS = 1e-8


def load():
    roles = json.loads((PROC / "feature_roles.json").read_text(encoding="utf-8"))
    feats = roles["model_features"]
    train = pd.read_parquet(PROC / "train_fe.parquet")
    val = pd.read_parquet(PROC / "val_fe.parquet")
    return feats, train, val


def add_fwd_vol(split_df, horizons):
    """h-day forward realized vol = sqrt(sum of next-h squared log returns), per ticker.
    Built within one split → last h rows NaN (future in next split), no leak."""
    d = split_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    sq = d.groupby("ticker")["log_return"].transform(lambda s: s.pow(2))
    d["_sq"] = sq
    for h in horizons:
        fwd = d.groupby("ticker")["_sq"].transform(lambda s: s.rolling(h).sum().shift(-h))
        d[f"fwd_rv_{h}"] = np.sqrt(fwd)
    return d.drop(columns="_sq")


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def qlike(y_var, p_var):
    """QLIKE on variances (lower better). y_var, p_var are variances (>0)."""
    y_var = np.clip(y_var, EPS, None)
    p_var = np.clip(p_var, EPS, None)
    return float(np.mean(np.log(p_var) + y_var / p_var))


def har_baseline(train, val, h):
    """OLS on [1, log rv5, log rv21, log rv63] → log(fwd_rv_h)."""
    cols = ["realized_vol_5", "realized_vol_21", "realized_vol_63"]
    tcol = f"fwd_rv_{h}"
    tr = train.dropna(subset=[tcol] + cols)
    va = val.dropna(subset=[tcol] + cols)
    Xtr = np.column_stack([np.ones(len(tr))] + [np.log(tr[c].values + EPS) for c in cols])
    Xva = np.column_stack([np.ones(len(va))] + [np.log(va[c].values + EPS) for c in cols])
    ytr = np.log(tr[tcol].values + EPS)
    beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    pred_log = Xva @ beta
    pred = np.exp(pred_log)
    y = va[tcol].values
    return dict(model="HAR-RV(OLS)", horizon=h,
                r2_level=r2(y, pred), r2_log=r2(np.log(y + EPS), pred_log),
                qlike=qlike(y ** 2, pred ** 2), n=len(va))


def rw_baseline(val, h):
    """Random-walk vol: predict fwd_rv_h with the current same-window realized vol.
    Uses realized_vol_5/21 as the closest already-computed trailing window."""
    trail = {1: "realized_vol_5", 5: "realized_vol_5", 20: "realized_vol_21"}[h]
    tcol = f"fwd_rv_{h}"
    va = val.dropna(subset=[tcol, trail])
    y = va[tcol].values
    # scale trailing (per-day vol) to an h-day realized-vol comparable magnitude
    pred = va[trail].values * np.sqrt(h)
    return dict(model=f"RandomWalk({trail}*sqrt{h})", horizon=h,
                r2_level=r2(y, pred), r2_log=r2(np.log(y + EPS), np.log(pred + EPS)),
                qlike=qlike(y ** 2, pred ** 2), n=len(va))


def lgbm_model(train, val, feats, h):
    tcol = f"fwd_rv_{h}"
    tr = train.dropna(subset=[tcol] + feats)
    va = val.dropna(subset=[tcol] + feats)
    ytr = np.log(tr[tcol].values + EPS)  # predict log vol (right-skewed target)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    m.fit(tr[feats].values, ytr)
    pred_log = m.predict(va[feats].values)
    pred = np.exp(pred_log)
    y = va[tcol].values
    return dict(model="LightGBM(logRV)", horizon=h,
                r2_level=r2(y, pred), r2_log=r2(np.log(y + EPS), pred_log),
                qlike=qlike(y ** 2, pred ** 2), n=len(va))


def _selfcheck():
    r = np.array([0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.1], dtype=float)
    df = pd.DataFrame({"ticker": "X", "date": pd.date_range("2020-01-01", periods=7), "log_return": r})
    out = add_fwd_vol(df, [2])
    # fwd_rv_2 at row0 = sqrt(r1^2 + r2^2) = sqrt(0.04+0.09)=sqrt(0.13)
    assert abs(out["fwd_rv_2"].iloc[0] - np.sqrt(0.13)) < 1e-12, out["fwd_rv_2"].iloc[0]
    assert out["fwd_rv_2"].iloc[-2:].isna().all()  # last h rows sealed
    print("selfcheck OK: forward RV = sqrt(sum next-h squared returns), last h rows NaN")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        _selfcheck(); return
    feats, train, val = load()
    train = add_fwd_vol(train, HORIZONS)
    val = add_fwd_vol(val, HORIZONS)
    print(f"features {len(feats)} | train {len(train)} | val {len(val)}\n")
    rows = []
    for h in HORIZONS:
        rows.append(rw_baseline(val, h))
        rows.append(har_baseline(train, val, h))
        rows.append(lgbm_model(train, val, feats, h))
    df = pd.DataFrame(rows)[["horizon", "model", "r2_level", "r2_log", "qlike", "n"]]
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print("\nReading: R2 (log) materially > 0 = vol IS predictable (vs ~0 for returns). "
          "\nHAR/LightGBM beating RandomWalk on R2 AND QLIKE = a real, honest positive result.")


if __name__ == "__main__":
    main()
