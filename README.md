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
| Offense | Team wRC+ | Runs scored |
| Run prevention | 55% starter FIP + 45% reliever FIP (IP-weighted), relative to league FIP | Runs allowed |
| Expected runs | RS = lgR/G × (wRC+/100) × opponent pitching index | — |
| Game win % (neutral) | Pythagenpat on expected RS/RA | Pythagenpat season win %, then log5 |
| Home field | Odds ratio from a 0.54 home win rate | Same |
| Series | Exact best-of-7 recursion from the current series state, 2-3-2 format | Same |
| World Series | 100,000-run Monte Carlo; home field to the better regular-season record | Same |

Optional hook: the share of each team's relievers flagged **RED** by [`ep-bullpen-fatigue-risk`](https://github.com/ejimenezperformance/ep-bullpen-fatigue-risk) can inflate that bullpen's FIP (`BULLPEN_RED_SHARE` in `src/config.py`). It is **off by default** and the penalty size is an untested assumption.

## Labels / Etiquetas

- **CONFIRMED** — arithmetic and probability math, verified by 23 automated checks (`tests/test_pipeline.py`), including a closed-form check of the series recursion and agreement between the Monte Carlo and the exact series odds.
- **INTERPRETATION** — every weight in `config.py` (55/45 starter/reliever split, 0.54 home win rate, fatigue penalty). They are reasoned defaults, **not fitted** to past postseasons.

## Limitations / Limitaciones

1. Regular-season stats; postseason rosters and usage differ (shorter starts, top relievers used more).
2. No game-by-game starting-pitcher matchups yet (planned Phase 2).
3. Pitchers traded mid-season appear on FanGraphs as a combined line and are left out of the team SP/RP split.
4. Weights are not calibrated; there is no backtest in this version, so the odds should be read as a structured estimate, not a validated forecast.
5. Even well-built playoff models pick series winners only modestly better than chance. Treat every number here with that in mind.

## How to run / Cómo correrlo

```
pip install -r requirements.txt
python tests/test_pipeline.py          # debe decir 23/23 checks passed
python src/data_loader.py              # pull real de FanGraphs -> "OK: los 4 equipos vivos aparecen"
python src/predict.py                  # probabilidades reales -> outputs/
python src/predict.py --synthetic      # demo con datos inventados (marcado SYNTHETIC)
```

**After every game:** update `SERIES` in `src/config.py` (`wins_home` / `wins_away`) and rerun `python src/predict.py`. The odds update from the new series state.

*Después de cada juego: actualiza `SERIES` en `src/config.py` y vuelve a correr `predict.py`.*

## Outputs

- `outputs/reports/series_predictions.csv` — game and series odds per model
- `outputs/reports/title_odds.csv` — pennant and World Series odds
- `outputs/reports/summary.md` — readable summary, including how much the two models disagree
- `outputs/figures/title_odds.png` — title odds chart (EP style)

## Results / Resultados

*Pending the real data pull. / Pendiente del pull real.*
