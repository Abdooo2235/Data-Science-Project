"""M5 deployment: Streamlit demo of the capstone's holdout forecasts.

Run locally:
    streamlit run app.py

Loads only committed artifacts (no training, no network):
- data/processed/holdout_predictions.parquet  (all 7 model prediction columns, sealed holdout)
- models/m4_holdout_scores.csv                (final M4 metrics table)
- models/lgbm_global_tuned_final.txt          (live LightGBM demo, optional)
- data/processed/holdout_fe.parquet           (features for the live demo, optional)
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Display name -> prediction column in holdout_predictions.parquet.
MODEL_COLS = {
    "Naive zero (baseline)": "y_pred_naive_zero",
    "ARIMA": "y_pred_arima",
    "GJR-GARCH-t (mean)": "y_pred_garch",
    "LightGBM (baseline)": "y_pred_lgb",
    "LightGBM (tuned)": "y_pred_lgb_tuned",
    "LSTM + attention (GPU)": "y_pred_lstm",
    "Ensemble GARCH + LSTM": "y_pred_ens",
}
# Ticker category order must match training (pandas sorts categories alphabetically).
TICKER_CATS = ["AAPL", "AMZN", "NVDA", "^GSPC"]

DISCLAIMER = (
    "This project is for educational purposes only. The forecasts produced by this "
    "model are not investment advice. Past performance does not predict future "
    "returns. The author does not recommend trading on these predictions."
)


# Loaders take the file's mtime so the cache invalidates if artifacts are
# regenerated while the server is running (otherwise stale numbers persist).
def _mtime(p: Path):
    return p.stat().st_mtime if p.exists() else None


@st.cache_data
def load_predictions(mtime):
    df = pd.read_parquet(DATA / "holdout_predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_scores(mtime):
    return pd.read_csv(MODELS / "m4_holdout_scores.csv")


@st.cache_resource
def load_lgb_booster(mtime):
    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(MODELS / "lgbm_global_tuned_final.txt"))
    except Exception:
        return None


@st.cache_data
def load_features(mtime):
    p = DATA / "holdout_fe.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


st.set_page_config(page_title="Stock market trend analysis", page_icon=":chart_with_upwards_trend:", layout="wide")

st.title("Stock market trend analysis")
st.caption("Next-day return forecasts on the sealed 2025-2026 holdout. Capstone milestone 5 demo.")
st.error("**Disclaimer.** " + DISCLAIMER)

missing = [p for p in (DATA / "holdout_predictions.parquet", MODELS / "m4_holdout_scores.csv")
           if not p.exists()]
if missing:
    st.error(
        "Required artifacts not found: " + ", ".join(str(p) for p in missing)
        + ". Run the pipeline first (notebooks 01-04) or pull the committed artifacts."
    )
    st.stop()

hold = load_predictions(_mtime(DATA / "holdout_predictions.parquet"))
scores = load_scores(_mtime(MODELS / "m4_holdout_scores.csv"))

# ---- sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Pick a forecast")
    ticker = st.selectbox("Ticker", TICKER_CATS, index=3)
    model_name = st.selectbox("Model", list(MODEL_COLS), index=4)
    col = MODEL_COLS[model_name]

    sub = hold[(hold.ticker == ticker) & hold[col].notna()].sort_values("date")
    dates = sub["date"].dt.date.tolist()
    date = st.selectbox("Date (trading days only)", dates, index=len(dates) - 1)
    st.caption(
        "The forecast on a given date is the model's prediction of the NEXT "
        "trading day's log return, made using only information available that day."
    )

row = sub[sub["date"].dt.date == date].iloc[0]
y_pred, y_true = float(row[col]), float(row["y_true"])
no_call = y_pred == 0.0  # naive_zero predicts exactly 0: that is NOT a direction call
pred_up, true_up = y_pred > 0, y_true > 0
hit = pred_up == true_up

# ---- headline: forecast vs actual ------------------------------------------
st.subheader(f"{ticker} on {date}: forecast vs what actually happened")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted next-day log return", f"{y_pred * 100:+.3f} %",
          delta="no direction call" if no_call else ("calls UP" if pred_up else "calls DOWN"),
          delta_color="off")
c2.metric("Actual next-day log return", f"{y_true * 100:+.3f} %",
          delta="went up" if true_up else "went down",
          delta_color="normal" if true_up else "inverse")
if no_call:
    c3.metric("Direction call", "n/a", delta="this baseline makes no call", delta_color="off")
else:
    c3.metric("Direction call", "HIT" if hit else "MISS",
              delta="correct side" if hit else "wrong side", delta_color="off")

up_rate = (sub["y_true"] > 0).mean()
if no_call:
    st.caption(
        f"Naive zero predicts exactly 0% every day, so it never calls a direction. "
        f"{up_rate:.1%} of the {len(sub)} {ticker} holdout days were up days."
    )
else:
    hits = ((sub[col] > 0) == (sub["y_true"] > 0)).mean()
    vs = ("about the same as" if abs(up_rate - hits) < 0.01
          else ("better than" if up_rate > hits else "worse than"))
    st.caption(
        f"Over all {len(sub)} {ticker} holdout days, {model_name} called the direction "
        f"correctly {hits:.1%} of the time. Simply predicting up every day would score "
        f"{up_rate:.1%} (the up-day rate) — {vs} this model."
    )

# ---- live model demo (LightGBM only) ----------------------------------------
if col == "y_pred_lgb_tuned":
    booster = load_lgb_booster(_mtime(MODELS / "lgbm_global_tuned_final.txt"))
    feats = load_features(_mtime(DATA / "holdout_fe.parquet"))
    if booster is None or feats is None:
        st.warning("Live model check unavailable (booster or features file not found).")
    else:
        frow = feats[(feats.ticker == ticker) & (feats["date"].dt.date == date)]
        if len(frow) != 1:
            st.warning("Live model check unavailable for this date (features row missing).")
        else:
            X = frow[booster.feature_name()].copy()
            X["ticker"] = pd.Categorical(X["ticker"], categories=TICKER_CATS)
            live = float(booster.predict(X)[0])
            if np.isclose(live, y_pred, atol=1e-9):
                st.info(
                    f"**Live model check.** The saved LightGBM booster was loaded and re-run "
                    f"on this date's features just now: it predicts {live * 100:+.3f} %, "
                    "matching the stored prediction above. This is the same trained model "
                    "file, not a lookup."
                )
            else:
                st.error(
                    f"**Live model check FAILED.** The booster predicts {live * 100:+.3f} % "
                    f"but the stored prediction is {y_pred * 100:+.3f} % — the committed "
                    "artifacts are out of sync."
                )

# ---- chart: predictions vs reality ------------------------------------------
st.subheader("The whole holdout at a glance")
chart_df = pd.concat([
    sub[["date", "y_true"]].rename(columns={"y_true": "value"}).assign(series="Actual"),
    sub[["date", col]].rename(columns={col: "value"}).assign(series="Forecast"),
])
sel_rule = alt.Chart(pd.DataFrame({"date": [pd.Timestamp(date)]})).mark_rule(
    color="#c9a227", strokeWidth=2).encode(x="date:T")
lines = alt.Chart(chart_df).mark_line(strokeWidth=1.5).encode(
    x=alt.X("date:T", title=None),
    y=alt.Y("value:Q", title="daily log return", axis=alt.Axis(format="%")),
    color=alt.Color("series:N", title=None,
                    scale=alt.Scale(domain=["Actual", "Forecast"],
                                    range=["#9fb3c8", "#1b4f8a"])),
    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("series:N"),
             alt.Tooltip("value:Q", format="+.4%")],
)
st.altair_chart((lines + sel_rule).properties(height=320), width="stretch")
st.caption(
    "The grey line is what the market did. The blue line is what the model predicted. "
    "The forecast line hugging zero is the finding, not a bug: daily returns are close "
    "to unpredictable, so an honest model predicts values near zero."
)

# ---- honest accuracy ---------------------------------------------------------
st.subheader("How good are these models, honestly?")
tbl = scores.copy()
show = tbl[["model", "n", "RMSE", "naive_rmse_same", "DM_vs_naive_p",
            "DirAcc", "DirAcc_p", "frac_pred_up"]].rename(columns={
    "n": "Days scored", "naive_rmse_same": "Naive RMSE (same days)",
    "DM_vs_naive_p": "DM p vs naive", "DirAcc": "Direction accuracy",
    "DirAcc_p": "p vs coin flip", "frac_pred_up": "Fraction predicted up"})
st.dataframe(show.style.format({"RMSE": "{:.5f}", "Naive RMSE (same days)": "{:.5f}",
                                "DM p vs naive": "{:.3f}", "Direction accuracy": "{:.3f}",
                                "p vs coin flip": "{:.3f}", "Fraction predicted up": "{:.2f}"},
                               na_rep="—"),
             width="stretch", hide_index=True)
st.caption(
    "Compare each RMSE to the naive RMSE on the SAME days. The LSTM and ensemble rows cover "
    "only 1,140 of 1,380 days (the LSTM needs a 60-day warm-up window); on those same days "
    "the naive forecast scores 0.02004, so their lower absolute RMSE does not beat naive either."
)
st.markdown(
    """
- No model beats the matched naive zero forecast on RMSE by a significant margin;
  two are significantly WORSE (baseline LightGBM, DM p=0.003; LSTM, p=0.004).
- The two models that look "significant" against a coin flip (GJR-GARCH 54.3% and
  tuned LightGBM 54.2% direction accuracy) predict up on nearly 100% of days.
  54.2% of holdout days were up days, so their edge is the market's upward drift,
  not timing skill. Tested against that base rate, p = 0.98 and 1.00.
- After realistic transaction costs, no strategy built on these forecasts beats
  simply buying and holding. Full analysis: `reports/milestones/M4.md`.
"""
)

st.divider()
st.caption(
    "Data: Yahoo Finance daily bars, 10 years, 4 tickers. Models trained through 2024, "
    "holdout Jan 2025 to May 2026, opened once — for final scoring and the single "
    "documented promote-or-revert check. Source and full reports: see the project README."
)
