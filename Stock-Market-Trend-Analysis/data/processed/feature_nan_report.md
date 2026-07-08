# Feature NaN / warm-up report — M1

Per-split NaN counts for MODEL_FEATURES + target. Rows with any NaN feature are flagged `is_warmup=1`; M3 must drop them before fitting.

## train  (rows=7660, is_warmup=1: 800, target NaN: 4)

| column | NaN count |
|---|---|
| `price_to_sma_200` | 800 |
| `realized_vol_63` | 256 |
| `momentum_63` | 252 |
| `price_to_sma_50` | 200 |
| `log_return_lag_21` | 88 |
| `realized_vol_21` | 88 |
| `momentum_21` | 84 |
| `rolling_std_20` | 84 |
| `parkinson_vol_21` | 84 |
| `ticker_expanding_std` | 84 |
| `ticker_expanding_mean` | 84 |
| `volume_z_20` | 80 |
| `price_to_sma_20` | 80 |
| `vix_z_60` | 76 |
| `rsi_14` | 60 |
| `log_return_lag_5` | 24 |
| `realized_vol_5` | 24 |
| `momentum_5` | 20 |
| `log_return_lag_1` | 8 |
| `log_return` | 4 |
| `daily_return` | 4 |
| `vix_log_change` | 4 |
| `tnx_change` | 4 |
| `term_spread_change` | 4 |
| `dxy_log_return` | 4 |

## val  (rows=1008, is_warmup=1: 0, target NaN: 4)

## holdout  (rows=1384, is_warmup=1: 0, target NaN: 4)
