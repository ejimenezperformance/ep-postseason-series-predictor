"""
test_pipeline.py — sanity checks with synthetic data (no network).
Correr: python tests/test_pipeline.py
"""
import os
import sys
from math import comb

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config  # noqa: E402
import model  # noqa: E402
import predict  # noqa: E402
from data_loader import build_team_table, ip_to_float  # noqa: E402
from series import series_win_prob, length_distribution  # noqa: E402

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# --- series math ------------------------------------------------------------
check("Even teams -> 50% series", abs(series_win_prob(0.5, 0.5) - 0.5) < 1e-12)

p = 0.6
closed = sum(comb(3 + k, k) * p**4 * (1 - p)**k for k in range(4))
check("Constant p matches closed-form negative binomial",
      abs(series_win_prob(p, p) - closed) < 1e-12)

check("Series already won -> 1.0", series_win_prob(0.3, 0.3, 4, 2) == 1.0)
check("Series already lost -> 0.0", series_win_prob(0.9, 0.9, 1, 4) == 0.0)
check("Down 0-3 is hard but possible", 0 < series_win_prob(0.55, 0.45, 0, 3) < 0.15)

d = length_distribution(0.55, 0.45)
check("Length distribution sums to 1", abs(sum(d.values()) - 1) < 1e-12)
check("Length distribution agrees with series prob",
      abs(sum(v for (g, w), v in d.items() if w == "A") - series_win_prob(0.55, 0.45)) < 1e-12)
check("Series lengths only 4-7", all(4 <= g <= 7 for g, _ in d))

# --- model ------------------------------------------------------------------
check("Home-field at 0.5 neutral -> HOME_WIN_PCT",
      abs(model.apply_home_field(0.5, True) - config.HOME_WIN_PCT) < 1e-12)
check("Home + road for equal teams average to 0.5",
      abs((model.apply_home_field(0.5, True) + model.apply_home_field(0.5, False)) / 2 - 0.5) < 1e-12)
check("log5 symmetry", abs(model.log5(0.6, 0.55) + model.log5(0.55, 0.6) - 1) < 1e-12)
check("Pythag: equal RS/RA -> .500", abs(model.pythag_pct(700, 700, 162) - 0.5) < 1e-12)

df, lg_rpg, lg_fip = predict.synthetic_inputs()
rows = {r["team"]: r for _, r in df.iterrows()}
pab, _, _ = model.component_neutral(rows["LAD"], rows["CLE"], lg_rpg, lg_fip)
pba, _, _ = model.component_neutral(rows["CLE"], rows["LAD"], lg_rpg, lg_fip)
check("Component model symmetric (P(A>B) + P(B>A) = 1)", abs(pab + pba - 1) < 1e-9)
check("Better synthetic team favored (LAD over CLE)", pab > 0.5)

old = dict(config.BULLPEN_RED_SHARE)
config.BULLPEN_RED_SHARE["LAD"] = 0.5
p_tired, _, _ = model.component_neutral(rows["LAD"], rows["CLE"], lg_rpg, lg_fip)
config.BULLPEN_RED_SHARE.update(old)
check("Fatigued bullpen lowers win probability", p_tired < pab)

# --- simulation -------------------------------------------------------------
odds = predict.title_odds(df, lg_rpg, lg_fip, "component", n=20000, seed=1)
check("Title odds sum to 1", abs(odds["p_title"].sum() - 1) < 1e-9)
check("Pennant odds sum to 2", abs(odds["p_pennant"].sum() - 2) < 1e-9)

st = predict.series_table(df, lg_rpg, lg_fip)
for name, s in config.SERIES.items():
    r = st[(st["series"] == name) & (st["model"] == "component")].iloc[0]
    sim = odds.set_index("team").loc[s["home"], "p_pennant"]
    check(f"{name}: simulated pennant odds match exact series odds (+-1.5%)",
          abs(sim - r["p_series_home_field"]) < 0.015)

check("WS home field goes to better record (MIL over TB)", predict.ws_home_field("TB", "MIL")[0] == "MIL")

# --- loader -------------------------------------------------------------------
check("IP '45.2' = 45 2/3", abs(ip_to_float("45.2") - (45 + 2 / 3)) < 1e-9)
check("IP '18.0' = 18", ip_to_float("18.0") == 18)


def team_hit(tid, runs, hits, hr):
    # Same shape as stats[0].splits from /teams/stats?group=hitting
    return {"team": {"id": tid, "name": str(tid)}, "stat": {
        "gamesPlayed": 162, "runs": runs, "hits": hits, "doubles": 280, "triples": 20,
        "homeRuns": hr, "baseOnBalls": 520, "intentionalWalks": 20, "hitByPitch": 60,
        "atBats": 5500, "sacFlies": 40, "plateAppearances": 6160}}


def team_pit(tid, runs, er, hr, k):
    return {"team": {"id": tid, "name": str(tid)}, "stat": {
        "gamesPlayed": 162, "runs": runs, "earnedRuns": er, "inningsPitched": "1440.0",
        "homeRuns": hr, "baseOnBalls": 500, "hitByPitch": 55, "strikeOuts": k}}


def pitcher(name, g, gs, ip, hr, bb, hbp, k):
    return {"player": {"fullName": name}, "stat": {
        "gamesPitched": g, "gamesStarted": gs, "inningsPitched": ip,
        "homeRuns": hr, "baseOnBalls": bb, "hitByPitch": hbp, "strikeOuts": k}}


ids = [info["mlb_id"] for info in config.TEAMS.values()] + [147]
hit_s = [team_hit(t, 720, 1350, 190) for t in ids]
hit_s[0] = team_hit(ids[0], 800, 1420, 220)  # MIL better offense
pit_s = [team_pit(t, 680, 620, 180, 1400) for t in ids]
pl = {k: [pitcher("SP", 30, 30, "170.1", 20, 50, 5, 170),
          pitcher("RP", 66, 1, "64.0", 6, 22, 2, 75)] for k in config.TEAMS}
tbl = build_team_table(hit_s, pit_s, pl).set_index("team")
check("Loader finds all 4 teams by MLB id", set(tbl.index) == set(config.TEAMS))
flat = build_team_table([team_hit(t, 720, 1350, 190) for t in ids], pit_s, pl).set_index("team")
check("All-equal league -> every wRC+ = 100", (flat["wrc_plus"] - 100).abs().max() < 1e-6)
check("Average team in a league with one better team -> wRC+ < 100", tbl.loc["LAD", "wrc_plus"] < 100)
check("Better offense -> wRC+ > 100", tbl.loc["MIL", "wrc_plus"] > 100)
check("Reliever with 1 GS in 66 G stays RP", abs(tbl.loc["CLE", "ip_rp"] - 64) < 1e-9)
check("FIP constant makes league FIP = league ERA",
      abs(build_team_table(hit_s, pit_s, pl).attrs["lg_fip"] - 9 * 620 / 1440) < 1e-9)

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
