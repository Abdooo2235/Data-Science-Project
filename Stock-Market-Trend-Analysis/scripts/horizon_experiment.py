"""Multi-horizon predictability probe (exploratory — NOT part of the sealed M1/M3 pipeline).

Question: is a 5-day / 20-day forward return (or its direction) more predictable than the
next-day return that M3 showed is a coin flip? Reuses the EXISTING processed features; only
the *target* changes. LightGBM only — the one model that retargets to arbitrary horizons
without semantic contortions (ARIMA/GARCH model the return series itself, so they don't apply).

Leakage rules honored (same bar as the rest of the repo):
  - Forward return built per-split via shift(-h): the last h rows of each split get NaN
    (their future lives in the next split) and are dropped. No label crosses a boundary.
  - Purged walk-forward CV on TRAIN ONLY, embargo = horizon (a 20-day target needs a 20-day
    gap, not 1). val is scored once at the end, holdout never touched here.

ponytail: probe first. If a horizon shows a real edge, THEN wire it into M1/M3. Until then no
sealed file is modified.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

SEED = 42
HORIZONS = [1, 5, 20]  # 1 = the M3 baseline for reference
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"


def load():
    roles = json.loads((PROC / "feature_roles.json").read_text(encoding="utf-8"))
    feats = roles["model_features"]
    train = pd.read_parquet(PROC / "train_fe.parquet")
    val = pd.read_parquet(PROC / "val_fe.parquet")
    return feats, train, val


def add_forward_targets(split_df: pd.DataFrame, horizons) -> pd.DataFrame:
    """h-day forward log return per ticker = sum of the NEXT h daily log returns, shift(-h).

    Built within one split only, so the last h rows per ticker are NaN (future in next split).
    """
    d = split_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = d.groupby("ticker")["log_return"]
    for h in horizons:
        if h == 1:
            d[f"fwd_ret_{h}"] = g.shift(-1)
        else:
            # rolling h-day sum aligned to the FIRST day of the window, then shift so row t
            # holds returns t+1..t+h. rolling().sum() is right-aligned → shift(-h).
            d[f"fwd_ret_{h}"] = g.transform(lambda s: s.rolling(h).sum().shift(-h))
        d[f"fwd_dir_{h}"] = (d[f"fwd_ret_{h}"] > 0).astype("float")
        d.loc[d[f"fwd_ret_{h}"].isna(), f"fwd_dir_{h}"] = np.nan
    return d


def date_folds(df, horizon, oos_years=(2021, 2022, 2023), embargo=None):
    """Expanding purged folds split by DATE (all tickers share the boundary).
    embargo defaults to `horizon`: purge the last `horizon` train dates so the last used
    train row's h-day target cannot overlap the OOS block."""
    embargo = horizon if embargo is None else embargo
    dts = df["date"]
    for y in oos_years:
        oo_mask = (dts >= pd.Timestamp(f"{y}-01-01")) & (dts <= pd.Timestamp(f"{y}-12-31"))
        tr_mask = dts < pd.Timestamp(f"{y}-01-01")
        if oo_mask.sum() == 0 or tr_mask.sum() == 0:
            continue
        tr_dates = df.loc[tr_mask, "date"].drop_duplicates().sort_values()
        if len(tr_dates) <= embargo:
            continue
        cut = tr_dates.iloc[-embargo]
        tr_idx = df.index[tr_mask & (dts < cut)]
        oo_idx = df.index[oo_mask]
        yield tr_idx, oo_idx


def dir_acc(y_true_ret, y_pred):
    """Directional accuracy on the return sign (zeros excluded from the denominator)."""
    s_true = np.sign(y_true_ret)
    s_pred = np.sign(y_pred)
    m = s_true != 0
    if m.sum() == 0:
        return np.nan, 0
    return float((s_true[m] == s_pred[m]).mean()), int(m.sum())


def binom_p(acc, n):
    """Two-sided binomial test that accuracy != 0.50."""
    if n == 0 or np.isnan(acc):
        return np.nan
    k = round(acc * n)
    return float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def run_regression(train, val, feats, h):
    tcol = f"fwd_ret_{h}"
    tr = train.dropna(subset=[tcol] + feats)
    va = val.dropna(subset=[tcol] + feats)
    Xtr, ytr = tr[feats].values, tr[tcol].values
    Xva, yva = va[feats].values, va[tcol].values

    # honest purged CV DirAcc on train (embargo = h)
    cv_accs = []
    for tr_idx, oo_idx in date_folds(tr, h):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        m.fit(tr.loc[tr_idx, feats].values, tr.loc[tr_idx, tcol].values)
        pred = m.predict(tr.loc[oo_idx, feats].values)
        a, _ = dir_acc(tr.loc[oo_idx, tcol].values, pred)
        if not np.isnan(a):
            cv_accs.append(a)
    cv_da = float(np.mean(cv_accs)) if cv_accs else np.nan

    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)

    rmse = float(np.sqrt(np.mean((yva - pred) ** 2)))
    naive_rmse = float(np.sqrt(np.mean(yva ** 2)))  # predict 0
    da, n = dir_acc(yva, pred)
    return dict(horizon=h, val_rmse=rmse, naive_rmse=naive_rmse,
                rmse_vs_naive=rmse / naive_rmse, cv_dir_acc=cv_da,
                val_dir_acc=da, val_dir_n=n, val_dir_p=binom_p(da, n))


def run_direction(train, val, feats, h):
    tcol = f"fwd_dir_{h}"
    tr = train.dropna(subset=[tcol] + feats)
    va = val.dropna(subset=[tcol] + feats)
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    model.fit(tr[feats].values, tr[tcol].values)
    proba = model.predict_proba(va[feats].values)[:, 1]
    pred = (proba > 0.5).astype(int)
    ytrue = va[tcol].values.astype(int)
    acc = float((pred == ytrue).mean())
    n = len(ytrue)
    base = float(ytrue.mean())  # class balance (up-rate) — the honest baseline, not always 0.5
    return dict(horizon=h, val_acc=acc, up_rate=base, n=n,
                p_vs_50=binom_p(acc, n))


def _selfcheck():
    """Prove the forward target is leakage-safe: row t holds returns t+1..t+h, and the last h
    rows per ticker are NaN (their future lives in the next split). Run: python ... selfcheck"""
    r = np.arange(1, 11, dtype=float)  # 10 days, log_return = 1..10
    df = pd.DataFrame({"ticker": "X", "date": pd.date_range("2020-01-01", periods=10), "log_return": r})
    out = add_forward_targets(df, [1, 5])
    # h=1 at row 0 = return of day 1 (index1) = 2.0
    assert out["fwd_ret_1"].iloc[0] == 2.0, out["fwd_ret_1"].iloc[0]
    # h=5 at row 0 = sum of days 1..5 (indices1..5) = 2+3+4+5+6 = 20
    assert out["fwd_ret_5"].iloc[0] == 20.0, out["fwd_ret_5"].iloc[0]
    # last h rows NaN → no label reaches past the split edge
    assert out["fwd_ret_1"].iloc[-1:].isna().all()
    assert out["fwd_ret_5"].iloc[-5:].isna().all()
    # direction matches sign, NaN where return NaN
    assert out["fwd_dir_5"].iloc[0] == 1.0
    assert out["fwd_dir_5"].iloc[-5:].isna().all()
    print("selfcheck OK: forward target holds t+1..t+h, last h rows sealed NaN")


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selfcheck":
        _selfcheck()
        return
    feats, train, val = load()
    train = add_forward_targets(train, HORIZONS)
    val = add_forward_targets(val, HORIZONS)

    print(f"features: {len(feats)} | train rows {len(train)} | val rows {len(val)}\n")

    print("=== REGRESSION (forward log return) — RMSE ratio <1.0 = beats naive; DirAcc p<0.05 = real sign edge ===")
    reg = pd.DataFrame([run_regression(train, val, feats, h) for h in HORIZONS])
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(reg.to_string(index=False))

    print("\n=== DIRECTION (up/down classifier) — acc vs up_rate; p<0.05 vs coin flip ===")
    di = pd.DataFrame([run_direction(train, val, feats, h) for h in HORIZONS])
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(di.to_string(index=False))

    print("\nReading: 1-day is the M3 coin-flip reference. A 5/20-day row is only interesting if"
          "\n  RMSE ratio < ~0.99 AND DirAcc p < 0.05 (regression), or classifier acc materially"
          "\n  beats up_rate with p < 0.05. Otherwise the longer horizon carries no usable edge either.")


if __name__ == "__main__":
    main()
