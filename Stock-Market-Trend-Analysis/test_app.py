"""Smoke test for the M5 Streamlit app: executes the full script headlessly,
fails on any uncaught exception, and checks the load-bearing logic
(live LightGBM re-prediction matches the stored holdout column).

Run:  python -m pytest test_app.py -q     (or just: python test_app.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent


def test_app_renders_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, f"app raised: {at.exception}"
    # disclaimer is front and center (M5 spec requirement)
    assert any("not investment advice" in e.value for e in at.error)
    # sidebar has the three pickers
    assert len(at.sidebar.selectbox) == 3


def test_live_lgb_matches_stored_predictions():
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(ROOT / "models" / "lgbm_global_tuned_final.txt"))
    hold = pd.read_parquet(ROOT / "data" / "processed" / "holdout_predictions.parquet")
    feats = pd.read_parquet(ROOT / "data" / "processed" / "holdout_fe.parquet")
    hold["date"] = pd.to_datetime(hold["date"])
    feats["date"] = pd.to_datetime(feats["date"])
    m = feats.merge(hold[["date", "ticker", "y_pred_lgb_tuned"]], on=["date", "ticker"])
    m = m.dropna(subset=["y_pred_lgb_tuned"])
    X = m[booster.feature_name()].copy()
    X["ticker"] = pd.Categorical(X["ticker"], categories=["AAPL", "AMZN", "NVDA", "^GSPC"])
    live = booster.predict(X)
    # pyrefly: ignore  # booster.predict/.values have broad union stubs; both are float ndarrays at runtime
    assert np.isclose(live, m["y_pred_lgb_tuned"].values, atol=1e-9).all()


if __name__ == "__main__":
    test_app_renders_without_exception()
    test_live_lgb_matches_stored_predictions()
    print("M5 smoke tests PASS")
