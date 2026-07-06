# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Milestone 4 - Evaluation and Presentation
# **Project:** Stock Market Trend Analysis | **Tickers:** ^GSPC, AAPL, AMZN, NVDA | **Seed:** 42
#
# This notebook turns the sealed-holdout predictions from M3 into the final evaluation: a performance table
# against baselines and a perfect-foresight bound, an error analysis of where the model fails, a costed
# long-only backtest, and the figures the presentation uses. It reads only what M3 already produced
# (`data/processed/holdout_predictions.parquet`); it trains nothing new.
#
# **Honest headline carried in from M3:** no model beats the naive zero forecast on RMSE. Two models keep a
# weak, reproducible directional edge (GJR-GARCH-mean 0.543 and the tuned LightGBM 0.542, both p=0.002), and
# both are erased by transaction costs. The attention-LSTM edge did not reproduce on the real GPU run. This is
# the efficient-market ceiling, and per the course rubric a small or negative result, honestly shown, is the
# point.
#
# **Not investment advice.** The forecasts here are for education only. Past performance does not predict
# future returns. Do not trade on these predictions.

# %% [markdown]
# ## 0. Setup

# %%
try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

if IN_COLAB:
    ROOT = Path("/content/Stock-Market-Trend-Analysis")
elif "__file__" in globals():
    ROOT = Path(__file__).resolve().parent.parent
else:
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

PROC = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
FIG.mkdir(parents=True, exist_ok=True)

TICKERS = ["^GSPC", "AAPL", "AMZN", "NVDA"]

# Light "clean finance" style for every figure: white surface, navy ink, gain-green / loss-red accents.
# Colours follow the data-visualisation convention (green = up, red = down); one y-axis per chart, thin marks.
INK = "#0f2740"       # navy text
PRIMARY = "#1b4f8a"   # deep blue
GAIN = "#2e9e6b"      # green (up)
LOSS = "#d1495b"      # red (down)
GOLD = "#c9a227"      # highlight
MUTED = "#6b7a8d"     # secondary ink
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#c9d3de", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": "#eef2f6",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})

# %% [markdown]
# ## 1. Metrics (same helpers M3 used, so numbers reconcile with M3.md section 8)

# %%
def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true, y_pred):
    """Fraction of days where the predicted sign matches the actual sign (zeros excluded).
    NaN if the prediction has no directional opinion (all preds 0)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.all(y_pred == 0):
        return float("nan")
    mask = y_true != 0
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def diracc_pvalue(y_true, y_pred):
    """Binomial test that directional accuracy differs from 0.50."""
    from scipy.stats import binomtest
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.all(y_pred == 0):
        return float("nan")
    mask = y_true != 0
    hits = int(np.sum(np.sign(y_pred[mask]) == np.sign(y_true[mask])))
    n = int(mask.sum())
    return float(binomtest(hits, n, 0.5).pvalue)


def diebold_mariano(y_true, pred_a, pred_b):
    """Diebold-Mariano test (squared-error loss) that A and B have equal accuracy. Returns (stat, p)."""
    from scipy.stats import norm
    y_true = np.asarray(y_true, float)
    d = (y_true - np.asarray(pred_a, float)) ** 2 - (y_true - np.asarray(pred_b, float)) ** 2
    n = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(var / n)
    return float(dm), float(2 * (1 - norm.cdf(abs(dm))))


assert abs(rmse([0, 0], [3, 4]) - np.sqrt(12.5)) < 1e-9
assert np.isnan(directional_accuracy([1, -1], [0, 0]))
print("metric helpers ready")

# %%
hold = pd.read_parquet(PROC / "holdout_predictions.parquet")
hold["date"] = pd.to_datetime(hold["date"])
print("holdout rows:", len(hold), "| span", hold.date.min().date(), "->", hold.date.max().date())
print("tickers:", sorted(hold.ticker.unique()))
LSTM_READY = int(hold["y_pred_lstm"].notna().sum()) > 0  # filled only by the Colab GPU run
up_freq = float((hold.y_true > 0).mean())                # holdout up-day base rate (the drift benchmark)
print("LSTM/ensemble columns filled (real GPU run):", LSTM_READY, "| up-day base rate:", round(up_freq, 3))

# %% [markdown]
# ## 2. Final performance table (holdout, opened once in M3)
#
# Every model against the naive_zero benchmark, plus a perfect-foresight upper bound so the "how much of the
# achievable gap did we close" question has an honest answer. A model that ties naive closes ~0% of the RMSE
# gap; a directional accuracy of 0.54 closes ~8% of the 0.50 -> 1.00 direction gap.

# %%
MODELS_EVAL = [
    ("naive_zero", "y_pred_naive_zero"),
    ("ARIMA", "y_pred_arima"),
    ("GJR-GARCH-t (mean)", "y_pred_garch"),
    ("LightGBM (baseline)", "y_pred_lgb"),
    ("LightGBM (tuned)", "y_pred_lgb_tuned"),
    ("LSTM+Attention (GPU)", "y_pred_lstm"),
    ("Ensemble GARCH+LSTM", "y_pred_ens"),
]

rows = []
for name, col in MODELS_EVAL:
    sub = hold.dropna(subset=[col])
    if len(sub) == 0:
        continue
    yt, yp = sub.y_true.values, sub[col].values
    naive_sub = np.zeros(len(sub))
    da = directional_accuracy(yt, yp)
    # fraction predicted "up" exposes drift-followers (a model that just predicts positive most days)
    nz = yp[yp != 0]
    frac_up = float(np.mean(nz > 0)) if len(nz) else float("nan")
    rows.append({
        "model": name, "n": len(sub),
        "RMSE": rmse(yt, yp), "MAE": mae(yt, yp),
        "naive_rmse_same": rmse(yt, naive_sub),   # naive on the SAME rows, for a fair comparison
        "DirAcc": da, "DirAcc_p": diracc_pvalue(yt, yp),
        "DM_vs_naive_p": diebold_mariano(yt, yp, naive_sub)[1] if name != "naive_zero" else np.nan,
        "frac_pred_up": frac_up,
    })

perf = pd.DataFrame(rows)
naive_rmse = float(perf.loc[perf.model == "naive_zero", "RMSE"].iloc[0])
# Gap closure toward a perfect-foresight oracle (RMSE oracle = 0, DirAcc oracle = 1.0), vs naive on the same rows.
# Direction is measured two ways: against a 0.50 coin flip, and against the up-day drift rate (the honest floor
# for this data, because a model that just predicts up scores the drift rate for free).
perf["rmse_gap_closed_%"] = (perf["naive_rmse_same"] - perf["RMSE"]) / perf["naive_rmse_same"] * 100.0
perf["diracc_gap_vs_coin_%"] = (perf["DirAcc"] - 0.5) / (1.0 - 0.5) * 100.0
perf["diracc_gap_vs_drift_%"] = (perf["DirAcc"] - up_freq) / (1.0 - up_freq) * 100.0
perf.loc[perf.model == "naive_zero", ["rmse_gap_closed_%", "diracc_gap_vs_coin_%", "diracc_gap_vs_drift_%"]] = np.nan

perf_show = perf.copy()
for c in ["RMSE", "MAE"]:
    perf_show[c] = perf_show[c].round(5)
for c in ["DirAcc", "DirAcc_p", "DM_vs_naive_p", "frac_pred_up"]:
    perf_show[c] = perf_show[c].round(3)
for c in ["rmse_gap_closed_%", "diracc_gap_vs_coin_%", "diracc_gap_vs_drift_%"]:
    perf_show[c] = perf_show[c].round(1)
print(perf_show.to_string(index=False))
perf.to_csv(MODELS / "m4_holdout_scores.csv", index=False)
print("\nsaved:", MODELS / "m4_holdout_scores.csv")

# Honest read, all computed (not hardcoded).
# "Beats naive" must be STATISTICALLY significant (DM p<0.05) AND lower RMSE on the same rows - a nominal
# 0.00002 difference is a tie, not a win. The "worse" models (LightGBM baseline, LSTM) are significantly WORSE.
from scipy.stats import binomtest
beats_naive = perf[(perf.model != "naive_zero") & (perf.RMSE < perf.naive_rmse_same) & (perf.DM_vs_naive_p < 0.05)]
worse_than_naive = perf[(perf.RMSE > perf.naive_rmse_same) & (perf.DM_vs_naive_p < 0.05)]
sig_dir = perf[(perf.DirAcc_p < 0.05) & (perf.DirAcc > 0.5)]
drift_followers = perf[(perf.model != "naive_zero") & (perf.frac_pred_up > 0.95)]
print(f"\nModels SIGNIFICANTLY beating naive on RMSE (DM p<0.05): {list(beats_naive.model) or 'NONE'}")
print(f"Models SIGNIFICANTLY worse than naive on RMSE:          {list(worse_than_naive.model) or 'NONE'}")
print(f"Models with a significant directional edge vs a 0.50 coin flip: {list(sig_dir.model) or 'NONE'}")
print(f"\nUp-day frequency in the holdout: {up_freq:.3f}")
print(f"Models that predict 'up' >95% of days (drift-followers): {list(drift_followers.model) or 'NONE'}")

# The honest directional test: the 0.50 coin-flip null only detects "predicts up". The right skill null for an
# always-up model is the UP-DAY BASE RATE. Against that, the drift-followers show p ~ 1.0 = no timing skill.
print("\nDirectional edge re-tested against the up-day base rate (the honest skill null):")
for name, col in [("GJR-GARCH-t (mean)", "y_pred_garch"), ("LightGBM (tuned)", "y_pred_lgb_tuned")]:
    s = hold.dropna(subset=[col]); mk = s.y_true != 0
    hits = int((np.sign(s.loc[mk, col]) == np.sign(s.loc[mk, "y_true"])).sum()); n = int(mk.sum())
    p_base = binomtest(hits, n, up_freq).pvalue
    print(f"  {name}: DirAcc {hits/n:.3f} vs base rate {up_freq:.3f} -> binomial p={p_base:.3f} "
          f"({'no skill beyond drift' if p_base > 0.05 else 'skill'})")
print("So the p=0.002 edge is only 'better than a coin flip'; against the drift benchmark there is no skill.")

# %% [markdown]
# **Takeaway.** No model significantly beats naive_zero on RMSE (gap-closure near zero); the LightGBM baseline
# and the LSTM are significantly worse. Two models keep a statistically significant directional edge
# (GJR-GARCH-mean 0.543, tuned LightGBM 0.542, both p=0.002), but both predict "up" on 99-100% of days, so that
# accuracy is essentially the holdout's up-day frequency (0.542): it is the market's upward drift in a calm
# period, not day-to-day timing. The attention-LSTM edge did not reproduce on the real GPU run. This is the
# efficient-market ceiling, honestly earned.

# %% [markdown]
# ## 3. Error analysis - where and why the model misses
#
# We analyse the tuned LightGBM (the model M3 promoted). Because it is a 2-tree model it predicts a value very
# close to zero every day, so its large errors are simply the largest real market moves, which are the ones a
# price-history model cannot see coming.

# %%
CHOSEN_NAME, CHOSEN_COL = "LightGBM (tuned)", "y_pred_lgb_tuned"
ev = hold.dropna(subset=[CHOSEN_COL]).copy()
ev["abs_err"] = (ev.y_true - ev[CHOSEN_COL]).abs()

# Known holdout-period events, for the per-row narrative (public, widely reported).
EVENTS = {
    "2025-01-24": "DeepSeek AI shock (next session): NVDA and AI names sold off hard",
    "2025-02-28": "Late-Feb AI/tech drawdown",
    "2025-04-02": "US tariff announcement: broad risk-off crash",
    "2025-04-03": "Tariff sell-off continues",
    "2025-04-08": "Tariff pause: sharp relief rally the next session",
    "2025-04-09": "Relief-rally follow-through",
    "2025-10-30": "Mega-cap earnings reaction",
}
top10 = ev.nlargest(10, "abs_err")[["date", "ticker", "y_true", CHOSEN_COL, "abs_err"]].copy()
top10["event"] = top10.date.dt.strftime("%Y-%m-%d").map(lambda d: EVENTS.get(d, "large idiosyncratic move"))
top10_show = top10.copy()
top10_show["date"] = top10_show.date.dt.strftime("%Y-%m-%d")
for c in ["y_true", CHOSEN_COL, "abs_err"]:
    top10_show[c] = top10_show[c].round(4)
print("Top-10 worst forecasts (tuned LightGBM):")
print(top10_show.to_string(index=False))
top10_show.to_csv(MODELS / "m4_top10_worst.csv", index=False)

# %%
# Figure: error distribution with percentile markers.
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(ev.abs_err, bins=60, color=PRIMARY, alpha=0.85)
for q, lab in [(0.50, "50th"), (0.90, "90th"), (0.99, "99th")]:
    v = ev.abs_err.quantile(q)
    ax.axvline(v, color=GOLD if q < 0.99 else LOSS, lw=1.6, ls="--")
    ax.text(v, ax.get_ylim()[1] * 0.9, f" {lab} pct = {v:.3f}", color=INK, fontsize=9)
ax.set_title("Holdout absolute error (tuned LightGBM): most days tiny, a fat right tail of missed shocks")
ax.set_xlabel("absolute error (log-return units)")
ax.set_ylabel("count of days")
fig.tight_layout()
fig.savefig(FIG / "M4_fig_err_dist.png", dpi=120)
plt.close(fig)
print("saved:", FIG / "M4_fig_err_dist.png")

# %%
# Figure: mean absolute error by month (regime check).
by_month = ev.assign(month=ev.date.dt.to_period("M").astype(str)).groupby("month")["abs_err"].mean()
fig, ax = plt.subplots(figsize=(10, 4.5))
colors = [LOSS if v >= by_month.mean() else PRIMARY for v in by_month.values]
ax.bar(by_month.index, by_month.values, color=colors)
ax.axhline(by_month.mean(), color=MUTED, lw=1, ls="--", label=f"average = {by_month.mean():.4f}")
ax.set_title("Mean absolute error by month: error spikes in the volatile tariff months (Apr 2025)")
ax.set_ylabel("mean abs error")
plt.xticks(rotation=45, ha="right")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "M4_fig_err_by_month.png", dpi=120)
plt.close(fig)
print("saved:", FIG / "M4_fig_err_by_month.png")

# %%
# Figure: directional confusion matrix (predicted sign vs actual sign, zeros excluded).
m = ev.y_true != 0
pred_up = ev.loc[m, CHOSEN_COL].values > 0
act_up = ev.loc[m, "y_true"].values > 0
cm = np.array([
    [int(np.sum(pred_up & act_up)),  int(np.sum(pred_up & ~act_up))],
    [int(np.sum(~pred_up & act_up)), int(np.sum(~pred_up & ~act_up))],
])
acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
ax.set_xticks([0, 1], labels=["actual up", "actual down"])
ax.set_yticks([0, 1], labels=["pred up", "pred down"])
for i in range(2):
    for j in range(2):
        good = i == j
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                color=GAIN if good else LOSS, fontweight="bold")
ax.set_title(f"Directional confusion, tuned LightGBM\nit predicts UP every day: accuracy {acc:.3f} = up-day rate", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "M4_fig_confusion.png", dpi=120)
plt.close(fig)
print("saved:", FIG / "M4_fig_confusion.png")
print(f"confusion matrix [[pred_up-act_up, pred_up-act_down],[pred_down-act_up, pred_down-act_down]] = {cm.tolist()}")
print(f"predicted up on {pred_up.mean()*100:.1f}% of days -> the bottom 'predict down' row is empty; the model")
print("never calls a down day, so its accuracy is just the market's up-day rate (drift), not timing.")

# %% [markdown]
# ## 4. Costed long-only backtest (illustration, not a strategy)
#
# Rule: go long ^GSPC tomorrow when the model predicts a positive return, otherwise hold cash. Two honest points
# fall out. First, the two models with a "significant" directional edge (tuned LightGBM, GJR-GARCH) predict up
# almost every day, so their signal never leaves the market and their backtest is just buy-and-hold. To show
# what happens to a model that actually times the market we use the ARIMA signal, which switches position 174
# times. Second, once realistic per-trade costs are applied, even that timing edge disappears. This is a
# hindsight backtest on one calm regime, not a tradeable system.

# %%
gspc = hold[hold.ticker == "^GSPC"].sort_values("date").copy()
mkt = gspc.y_true.values                                  # realised next-day log return
bh_curve = np.exp(np.cumsum(mkt))

sig_lgbt = (gspc[CHOSEN_COL].values > 0).astype(float)    # promoted model: 1 = long, 0 = cash
sig_arima = (gspc["y_pred_arima"].values > 0).astype(float)  # the timer that actually switches
sw_lgbt = int(np.sum(np.abs(np.diff(sig_lgbt))))
sw_arima = int(np.sum(np.abs(np.diff(sig_arima))))
print(f"^GSPC holdout: {len(gspc)} days")
print(f"  promoted tuned-LightGBM signal: {int(sig_lgbt.sum())} long-days, {sw_lgbt} switches "
      f"-> {'ALWAYS long = buy-and-hold' if sw_lgbt == 0 else 'switches'}")
print(f"  ARIMA timing signal:            {int(sig_arima.sum())} long-days, {sw_arima} switches")


def equity_curve(signal, mkt, cost_bps):
    cost = np.zeros_like(signal)
    cost[1:] = np.abs(np.diff(signal)) * (cost_bps / 1e4)  # pay cost only when the position changes
    strat_ret = signal * mkt - cost
    return np.exp(np.cumsum(strat_ret)), strat_ret


print(f"\nARIMA timing strategy vs cost (Sharpe uses risk-free rate = 0; cash days earn 0):")
print(f"{'cost(bps)':>10} {'terminal x':>12} {'ann_ret%':>10} {'sharpe':>8}")
curves, sharpes = {}, {}
for c in (0, 5, 10):
    curve, sret = equity_curve(sig_arima, mkt, c)
    curves[c] = curve
    ann = sret.mean() * 252 * 100
    sharpe = (sret.mean() / sret.std() * np.sqrt(252)) if sret.std() > 0 else 0.0
    sharpes[c] = sharpe
    print(f"{c:>10} {curve[-1]:>12.3f} {ann:>10.1f} {sharpe:>8.2f}")
bh_sharpe = mkt.mean() / mkt.std() * np.sqrt(252)
print(f"{'buy&hold':>10} {bh_curve[-1]:>12.3f} {mkt.mean()*252*100:>10.1f} {bh_sharpe:>8.2f}")

# Is the timer's zero-cost daily edge even real? A one-sample t-test on its daily strategy returns.
from scipy.stats import ttest_1samp
_, sret0 = equity_curve(sig_arima, mkt, 0)
t_stat, t_p = ttest_1samp(sret0, 0.0)
print(f"ARIMA timer daily returns (0 cost): mean t-test t={t_stat:.2f}, p={t_p:.3f} "
      f"({'not' if t_p > 0.05 else ''} significantly positive)")
print(f"Sharpe: buy&hold {bh_sharpe:.2f}, timer 0 bps {sharpes[0]:.2f}, timer 10 bps {sharpes[10]:.2f} "
      f"(the edge at 0 cost collapses once trades are charged).")

# hypothetical $10,000 illustration (0 cost, the most generous case)
start = 10_000
print(f"\nHypothetical ${start:,} at holdout start (0 cost, unrealistic best case):")
print(f"  buy-and-hold ^GSPC:       ${start * bh_curve[-1]:,.0f}")
print(f"  ARIMA timing, 0 bps:      ${start * curves[0][-1]:,.0f}")
print(f"  ARIMA timing, 10 bps:     ${start * curves[10][-1]:,.0f}")
print("  The promoted model equals buy-and-hold (never trades). The timer's edge is erased by costs.")
print("  This is a hindsight backtest, not investment advice.")

# %%
# Figure: cumulative equity - buy-and-hold, the ARIMA timer at 0 and 10 bps.
fig, ax = plt.subplots(figsize=(10, 5))
d = gspc.date.values
ax.plot(d, bh_curve, color=INK, lw=2.2, label=f"buy & hold ^GSPC  (x{bh_curve[-1]:.2f})  = promoted model")
ax.plot(d, curves[0], color=GAIN, lw=1.8, label=f"ARIMA timer, 0 bps  (x{curves[0][-1]:.2f})")
ax.plot(d, curves[10], color=LOSS, lw=1.8, ls="--", label=f"ARIMA timer, 10 bps  (x{curves[10][-1]:.2f})")
ax.axhline(1.0, color=MUTED, lw=0.8)
ax.set_title("Long-only backtest on ^GSPC: the promoted model just holds the market;\n"
             "a real timer's edge is erased by transaction costs")
ax.set_ylabel("growth of $1")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "M4_fig_backtest_cum.png", dpi=120)
plt.close(fig)
print("saved:", FIG / "M4_fig_backtest_cum.png")

# %% [markdown]
# **Takeaway.** The promoted model's timing signal is "always long," so its backtest is identical to buy-and-hold:
# it captures the market's drift and adds no timing value. A model that does try to time (ARIMA) can edge
# buy-and-hold at zero cost, but it switches position 174 times and at a realistic 10 bps per trade the edge is
# gone. A backtest in hindsight is not a tradeable strategy, and the method, not the dollar figure, is the point.

# %% [markdown]
# ## 5. Self-audit (bound to computed results)

# %%
audit = {
    "holdout_loaded_once": len(hold) > 0 and "y_pred_lgb_tuned" in hold.columns,
    "perf_table_has_all_models": perf.model.nunique() >= (7 if LSTM_READY else 5),  # LSTM+ens absent on a CPU run
    "perf_scores_saved": (MODELS / "m4_holdout_scores.csv").exists(),
    "no_model_beats_naive_rmse": len(beats_naive) == 0,  # the honest EMH result
    "significant_dir_edge_only_garch_lgbtuned": set(sig_dir.model) <= {"GJR-GARCH-t (mean)", "LightGBM (tuned)"},
    "top10_worst_saved": (MODELS / "m4_top10_worst.csv").exists(),
    "err_dist_fig": (FIG / "M4_fig_err_dist.png").exists(),
    "err_by_month_fig": (FIG / "M4_fig_err_by_month.png").exists(),
    "confusion_fig": (FIG / "M4_fig_confusion.png").exists(),
    "backtest_cum_fig": (FIG / "M4_fig_backtest_cum.png").exists(),
    "backtest_has_cost_sensitivity": len(curves) == 3,
    "perfect_foresight_bound_reported": "rmse_gap_closed_%" in perf.columns,
}
for k, v in audit.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
assert all(audit.values()), "M4 self-audit failed!"
print(f"\nAll {len(audit)} M4 self-audit checks passed.")
if not LSTM_READY:
    print("NOTE: LSTM/ensemble rows absent - run the Colab GPU config (see COLAB_TRAINING_GUIDE.md) to fill them.")
