"""
test_backtest.py — checks for the validation code (no network).
Correr: python tests/test_backtest.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import backtest as bt  # noqa: E402
import model  # noqa: E402
from series import series_win_prob  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def game(pk, gtype, desc, n_series, num, home, away, home_won, state="Final", final=True):
    g = {"gamePk": pk, "gameType": gtype, "seriesDescription": desc, "gamesInSeries": n_series,
         "seriesGameNumber": num, "gameDate": f"2019-10-{num:02d}",
         "status": {"abstractGameState": state},
         "teams": {"home": {"team": {"id": home, "name": f"T{home}"}},
                   "away": {"team": {"id": away, "name": f"T{away}"}}}}
    if final:
        g["teams"]["home"]["isWinner"] = home_won
        g["teams"]["away"]["isWinner"] = not home_won
    return g


# 2-2-1 Division Series: team 1 (home field) wins 3-1 (games 3-4 at team 2's park)
ds = [game(1, "D", "ALDS", 5, 1, 1, 2, True), game(2, "D", "ALDS", 5, 2, 1, 2, False),
      game(3, "D", "ALDS", 5, 3, 2, 1, False), game(4, "D", "ALDS", 5, 4, 2, 1, False)]
# Rain-postponed duplicate entry of game 4 (no isWinner) — must be ignored
ds_dup = game(4, "D", "ALDS", 5, 4, 2, 1, False, state="Preview", final=False)
# Single-game wild card, road team wins
wc = [game(10, "F", "AL Wild Card Game", 1, 1, 5, 6, False)]
# Incomplete LCS (1-0) — must be flagged incomplete
lcs = [game(20, "L", "ALCS", 7, 1, 1, 7, True)]
schedule = {"dates": [{"games": ds + [ds_dup] + wc + lcs}]}
series = {s["series"]: s for s in bt.parse_series(schedule, 2019)}

check("Parses 3 series", len(series) == 3)
check("ALDS: home field = game-1 home team", series["ALDS"]["home_id"] == 1)
check("ALDS: 3-1 result counted across both parks",
      (series["ALDS"]["wins_home"], series["ALDS"]["wins_away"]) == (3, 1))
check("ALDS: complete and home-field team won", series["ALDS"]["complete"] and series["ALDS"]["home_won"])
check("Postponed duplicate ignored", series["ALDS"]["wins_home"] + series["ALDS"]["wins_away"] == 4)
check("Wild card game: road team won", series["AL Wild Card Game"]["home_won"] is False)
check("Unfinished series flagged incomplete", series["ALCS"]["complete"] is False)

# Patterns
check("Series patterns have the right lengths", all(len(p) == n for n, p in bt.PATTERNS.items()))
check("Best-of-1 series prob = single home-game prob",
      abs(series_win_prob(0.57, 0.40, best_of=1, pattern=bt.PATTERNS[1]) - 0.57) < 1e-12)
check("Best-of-3 all at home = 3-game binomial",
      abs(series_win_prob(0.6, 0.1, best_of=3, pattern=bt.PATTERNS[3]) - (0.6**2 + 2 * 0.6**2 * 0.4)) < 1e-12)

# Metrics
y = np.array([1, 0, 1, 1, 0, 1, 0, 0])
check("Coin flip Brier = 0.25", abs(bt.brier(np.full(8, 0.5), y) - 0.25) < 1e-12)
check("Perfect forecast Brier = 0", bt.brier(y, y) == 0)
check("Coin flip log loss = ln 2", abs(bt.log_loss(np.full(8, 0.5), y) - np.log(2)) < 1e-12)
check("Coin flip excluded from favorite accuracy", np.isnan(bt.favorite_accuracy(np.full(8, 0.5), y)))
p = np.array([0.7, 0.4, 0.6, 0.3, 0.45, 0.55, 0.65, 0.2])
check("Favorite accuracy counts correct sides",
      abs(bt.favorite_accuracy(p, y) - np.mean((p > 0.5) == (y == 1))) < 1e-12)
d, lo, hi = bt.paired_bootstrap(p, p, y)
check("Bootstrap of identical models = 0 difference", d == 0 and lo == 0 and hi == 0)
d, lo, hi = bt.paired_bootstrap(np.where(y == 1, 0.9, 0.1), np.full(8, 0.5), y)
check("Bootstrap: clearly better model has CI entirely below 0", hi < 0)
cal = bt.calibration_table(np.array([0.3, 0.7]), np.array([0, 1]))
check("Calibration folds underdog-side probs to favorite side",
      len(cal) == 1 and cal["n"].iloc[0] == 2 and cal["actual"].iloc[0] == 1.0)

# predict_series with records model is symmetric in home-field
check("Record model: equal records -> series odds > 0.5 only from home field",
      0.5 < series_win_prob(model.apply_home_field(0.5, True), model.apply_home_field(0.5, False)) < 0.55)

# End-to-end: one fake season through season_predictions
def hit_split(tid, runs):
    return {"team": {"id": tid, "name": f"T{tid}"}, "stat": {
        "gamesPlayed": 162, "runs": runs, "hits": 1350 + (runs - 700), "doubles": 280, "triples": 20,
        "homeRuns": 190, "baseOnBalls": 520, "intentionalWalks": 20, "hitByPitch": 60,
        "atBats": 5500, "sacFlies": 40, "plateAppearances": 6160}}


def pit_split(tid, ra, w):
    return {"team": {"id": tid, "name": f"T{tid}"}, "stat": {
        "gamesPlayed": 162, "runs": ra, "earnedRuns": ra - 50, "inningsPitched": "1440.0",
        "homeRuns": 180, "baseOnBalls": 500, "hitByPitch": 55, "strikeOuts": 1400,
        "wins": w, "losses": 162 - w}}


def pitcher_split(gs, ip, k):
    return {"player": {"fullName": "x"}, "stat": {
        "gamesPitched": 30, "gamesStarted": gs, "inningsPitched": ip,
        "homeRuns": 20, "baseOnBalls": 50, "hitByPitch": 5, "strikeOuts": k}}


tids = [1, 2, 5, 6, 7, 9]
raw = {"hit": [hit_split(t, 760 if t == 1 else 700) for t in tids],
       "pit": [pit_split(t, 620 if t == 1 else 700, 98 if t == 1 else 81) for t in tids],
       "pitchers": {str(t): [pitcher_split(30, "170.0", 190 if t == 1 else 160),
                             pitcher_split(0, "60.0", 70)] for t in tids}}
rows = bt.season_predictions(raw, schedule, 2019)
check("E2E: only the 2 completed series are scored", len(rows) == 2)
alds = next(r for r in rows if r["series"] == "ALDS")
check("E2E: better team (1) favored by all skill models",
      all(alds[f"p_{m}"] > 0.5 for m in ("component", "pythag", "record")))
check("E2E: coin model = 0.5", alds["p_coin"] == 0.5)
check("E2E: all probabilities in (0, 1)",
      all(0 < r[f"p_{m}"] < 1 for r in rows for m in bt.MODELS))
summ = bt.summarize(__import__("pandas").DataFrame(rows))
check("E2E: summary has one row per model", list(summ["model"]) == list(bt.MODELS))

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
