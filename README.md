# EP Postseason Series Predictor — 2026 LCS

**Emerson Jimenez · Emerson Performance (EP)**

Rule-based, fully transparent series predictor for the four teams still alive in the 2026 MLB postseason:
**NLCS:** Milwaukee Brewers (home field) vs. Los Angeles Dodgers · **ALCS:** Tampa Bay Rays (home field) vs. Cleveland Guardians.

*Predictor de series basado en reglas y totalmente transparente para los cuatro equipos vivos en la postemporada 2026.*

---

## Why this exists / Por qué existe

Postseason series are close to coin flips. The question here is not "who wins?" but **how much does splitting a team into offense, rotation and bullpen change the odds compared with a simple run-differential baseline?** If the answer is "very little," that is a valid, publishable finding.

*Las series de playoffs son casi un volado. La pregunta no es "quién gana", sino cuánto cambian las probabilidades al separar bateo, rotación y bullpen frente a un modelo simple de diferencial de carreras. Si la respuesta es "muy poco", también es un hallazgo válido.*

## Method / Método

Two models run side by side on the same data, with the same home-field rule.

| Step | Component model (A) | Pythagorean baseline (B) |
|---|---|---|
| Offense | Team wRC+ (approximated from wOBA, see below) | Runs scored |
| Run prevention | 55% starter FIP + 45% reliever FIP (IP-weighted), relative to league FIP | Runs allowed |
| Expected runs | RS = lgR/G × (wRC+/100) × opponent pitching index | — |
| Game win % (neutral) | Pythagenpat on expected RS/RA | Pythagenpat season win %, then log5 |
| Home field | Odds ratio from a 0.54 home win rate | Same |
| Series | Exact best-of-7 recursion from the current series state, 2-3-2 format | Same |
| World Series | 100,000-run Monte Carlo; home field to the better regular-season record | Same |

**Data source:** the official MLB Stats API (`statsapi.mlb.com`). FanGraphs was the first choice, but it blocks automated `pybaseball` requests (HTTP error on `leaders-legacy.aspx`), so wOBA, wRC+ and FIP are computed in `src/data_loader.py` from raw counting stats:

- wOBA uses standard linear weights; wRC+ ≈ 100 × ((wOBA − lgwOBA)/1.24 + lgR/PA) / (lgR/PA).
- FIP = (13·HR + 3·(BB+HBP) − 2·K) / IP + C, with C chosen so league FIP equals league ERA.
- Pitchers are pulled per team, so a pitcher traded mid-season counts only for the innings he threw for each club.

*Fuente: API oficial de MLB. FanGraphs bloquea las descargas automáticas, así que wOBA, wRC+ y FIP se calculan aquí a partir de estadísticas básicas.*

Optional hook: the share of each team's relievers flagged **RED** by [`ep-bullpen-fatigue-risk`](https://github.com/ejimenezperformance/ep-bullpen-fatigue-risk) can inflate that bullpen's FIP (`BULLPEN_RED_SHARE` in `src/config.py`). It is **off by default** and the penalty size is an untested assumption.

## Validation (backtest) / Validación

`src/backtest.py` runs the **same model with the same frozen weights** on every completed postseason series from 2015 to 2026 (2020 excluded: 60-game season, different format), using only that season's regular-season stats and predicting each series from 0-0. Weights are **not** tuned on these results, so the backtest measures the model instead of fitting it.

It compares four forecasts on the same series:

| Model | What it uses |
|---|---|
| component | wRC+ / starter FIP / reliever FIP (this repo) |
| pythag | Runs scored/allowed (Pythagenpat + log5) |
| record | Regular-season win % + log5 |
| coin | 50% every series (no-skill benchmark) |

Scores: Brier score, log loss, share of series won by the model's favorite, calibration by probability bin, and a paired bootstrap (10,000 resamples) for whether the component model beats the Pythagorean baseline. Each round uses its real format: single-game Wild Card (2015-2021), best-of-3 at the higher seed (2022+), 2-2-1 Division Series, 2-3-2 LCS and World Series.

*El backtest corre el mismo modelo, con los pesos congelados, sobre cada serie de playoffs terminada 2015-2026 (sin 2020). Los pesos no se ajustan con estos resultados.*

## Labels / Etiquetas

- **CONFIRMED** — arithmetic and probability math, verified by 52 automated checks (`tests/test_pipeline.py` and `tests/test_backtest.py`), including a closed-form check of the series recursion and agreement between the Monte Carlo and the exact series odds.
- **INTERPRETATION** — every weight in `config.py` (55/45 starter/reliever split, 0.54 home win rate, fatigue penalty) and the wOBA linear weights / 1.24 scale. They are reasoned defaults, **not fitted** to past postseasons or to 2026 run environment.

## Limitations / Limitaciones

1. Regular-season stats; postseason rosters and usage differ (shorter starts, top relievers used more).
2. No game-by-game starting-pitcher matchups yet (planned Phase 2).
3. wRC+ here is an approximation with **no park adjustment** and generic linear weights, so it will not match FanGraphs exactly. Position players who pitched are included in the bullpen line (tiny innings, negligible effect).
4. Weights are not calibrated. The backtest checks them against past series, but with roughly 100 series the confidence intervals are wide: a small edge over the baselines may not be distinguishable from luck.
5. Even well-built playoff models pick series winners only modestly better than chance. Treat every number here with that in mind.

## How to run / Cómo correrlo

```
pip install -r requirements.txt
python tests/test_pipeline.py          # debe decir 28/28 checks passed
python src/data_loader.py              # pull real de la API de MLB -> "OK: los 4 equipos vivos aparecen"
python src/predict.py                  # probabilidades reales -> outputs/
python src/predict.py --synthetic      # demo con datos inventados (marcado SYNTHETIC)
python tests/test_backtest.py          # debe decir 24/24 checks passed
python src/backtest.py                 # validacion 2015-2026 (varios minutos; guarda cache)
```

**After every game:** update `SERIES` in `src/config.py` (`wins_home` / `wins_away`) and rerun `python src/predict.py`. The odds update from the new series state.

*Después de cada juego: actualiza `SERIES` en `src/config.py` y vuelve a correr `predict.py`.*

## Outputs

- `outputs/reports/series_predictions.csv` — game and series odds per model
- `outputs/reports/title_odds.csv` — pennant and World Series odds
- `outputs/reports/summary.md` — readable summary, including how much the two models disagree
- `outputs/figures/title_odds.png` — title odds chart (EP style)
- `outputs/reports/backtest_series.csv` — every past series with each model's pre-series odds and the result
- `outputs/reports/backtest_summary.md` — scores, bootstrap test, results by round, calibration
- `outputs/figures/backtest_calibration.png` — predicted vs. actual favorite win rate

## Results / Resultados

*Pending the real data pull. / Pendiente del pull real.*
