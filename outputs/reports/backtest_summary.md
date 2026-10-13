# Backtest — EP Postseason Series Predictor

106 completed postseason series, seasons [np.int64(2015), np.int64(2016), np.int64(2017), np.int64(2018), np.int64(2019), np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024), np.int64(2025), np.int64(2026)].
Weights frozen from `config.py`; predictions from 0-0 with regular-season data only.

## Scores (lower Brier / log loss = better; coin flip Brier = 0.250)

| Model | n | Brier | Log loss | Favorite won |
|---|---|---|---|---|
| component | 106 | 0.2426 | 0.6784 | 59.4% |
| pythag | 106 | 0.2424 | 0.6791 | 60.4% |
| record | 106 | 0.2463 | 0.6867 | 58.5% |
| coin | 106 | 0.2500 | 0.6931 | — |

## Is the component model better? (paired bootstrap, 10,000 resamples)

- Component vs Pythagorean: Brier difference +0.0002, 95% CI [-0.0120, +0.0122]
- Component vs coin flip: Brier difference -0.0074, 95% CI [-0.0266, +0.0125]

Negative = component model better. A CI that crosses 0 means no detectable difference.

## By round (component model)

| Round | n | Brier | Favorite won |
|---|---|---|---|
| Division Series | 44 | 0.2553 | 52.3% |
| LCS | 20 | 0.2279 | 65.0% |
| Wild Card | 32 | 0.2358 | 62.5% |
| World Series | 10 | 0.2380 | 70.0% |

## Calibration (component model)

| Favorite prob. bin | n | Predicted | Actual |
|---|---|---|---|
| 0.50-0.55 | 37 | 52.5% | 56.8% |
| 0.55-0.60 | 28 | 57.8% | 60.7% |
| 0.60-0.65 | 20 | 62.6% | 55.0% |
| 0.65-0.70 | 17 | 67.1% | 64.7% |
| 0.70-1.00 | 4 | 72.1% | 75.0% |
