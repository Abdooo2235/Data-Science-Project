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


@st.cache_data
def load_predictions():
    df = pd.read_parquet(DATA / "holdout_predictions.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_scores():
    return pd.read_csv(MODELS / "m4_holdout_scores.csv")


@st.cache_resource
def load_lgb_booster():
    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(MODELS / "lgbm_global_tuned_final.txt"))
    except Exception:
        return None


@st.cache_data
def load_features():
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

hold = load_predictions()
scores = load_scores()

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
pred_up, true_up = y_pred > 0, y_true > 0
hit = pred_up == true_up

# ---- headline: forecast vs actual ------------------------------------------
st.subheader(f"{ticker} on {date}: forecast vs what actually happened")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted next-day log return", f"{y_pred * 100:+.3f} %",
          delta="calls UP" if pred_up else "calls DOWN", delta_color="off")
c2.metric("Actual next-day log return", f"{y_true * 100:+.3f} %",
          delta="went up" if true_up else "went down",
          delta_color="normal" if true_up else "inverse")
c3.metric("Direction call", "HIT" if hit else "MISS",
          delta="correct side" if hit else "wrong side",
          delta_color="normal" if hit else "inverse")

hits = ((sub[col] > 0) == (sub["y_true"] > 0)).mean()
st.caption(
    f"Over all {len(sub)} {ticker} holdout days, {model_name} called the direction "
    f"correctly {hits:.1%} of the time. {(sub['y_true'] > 0).mean():.1%} of days were "
    "up days, so always predicting up would score about the same."
)

# ---- live model demo (LightGBM only) ----------------------------------------
if col == "y_pred_lgb_tuned":
    booster, feats = load_lgb_booster(), load_features()
    if booster is not None and feats is not None:
        frow = feats[(feats.ticker == ticker) & (feats["date"].dt.date == date)]
        if len(frow) == 1:
            X = frow[booster.feature_name()].copy()
            X["ticker"] = pd.Categorical(X["ticker"], categories=TICKER_CATS)
            live = float(booster.predict(X)[0])
            match = np.isclose(live, y_pred, atol=1e-9)
            st.info(
                f"**Live model check.** The saved LightGBM booster was loaded and re-run "
                f"on this date's features just now: it predicts {live * 100:+.3f} %, which "
                f"{'matches' if match else 'does NOT match'} the stored prediction above. "
                "This is the same trained model file, not a lookup."
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
show = tbl[["model", "RMSE", "MAE", "DirAcc", "DirAcc_p", "frac_pred_up"]].rename(columns={
    "DirAcc": "Direction accuracy", "DirAcc_p": "p vs coin flip", "frac_pred_up": "Fraction predicted up"})
st.dataframe(show.style.format({"RMSE": "{:.5f}", "MAE": "{:.5f}", "Direction accuracy": "{:.3f}",
                                "p vs coin flip": "{:.3f}", "Fraction predicted up": "{:.2f}"}),
             width="stretch", hide_index=True)
st.markdown(
    """
- No model beats the naive zero forecast on RMSE (0.02109) by a significant margin.
- The two models that look "significant" against a coin flip (GJR-GARCH and tuned
  LightGBM, about 54.3% direction accuracy) predict up on nearly 100% of days.
  54.2% of holdout days were up days, so their edge is the market's upward drift,
  not timing skill. Tested against that base rate, p = 0.98 and 1.00.
- After realistic transaction costs, no strategy built on these forecasts beats
  simply buying and holding. Full analysis: `reports/milestones/M4.md`.
"""
)

st.divider()
st.caption(
    "Data: Yahoo Finance daily bars, 10 years, 4 tickers. Models trained through 2024, "
    "holdout Jan 2025 to May 2026, never used for any modeling decision. "
    "Source and full reports: see the project README."
)
