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
from data_loader import build_team_table, _ip_to_float  # noqa: E402
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
check("IP 45.2 = 45 2/3", abs(_ip_to_float(45.2) - (45 + 2 / 3)) < 1e-9)
bat = pd.DataFrame({"Team": ["MIL", "LAD", "TBR", "CLE", "NYY"], "G": [162] * 5,
                    "R": [790, 820, 720, 680, 800], "wRC+": [108, 116, 104, 96, 110]})
tp = pd.DataFrame({"Team": ["MIL", "LAD", "TBR", "CLE", "NYY"], "R": [610, 640, 600, 650, 660]})
pit = pd.DataFrame({"Team": ["MIL", "MIL", "LAD", "LAD", "TBR", "TBR", "CLE", "CLE", "- - -"],
                    "G": [30, 60, 31, 65, 30, 70, 28, 66, 40], "GS": [30, 0, 31, 0, 30, 0, 28, 1, 10],
                    "IP": [180.1, 65.0, 175.2, 66.1, 170.0, 70.2, 165.0, 64.0, 90.0],
                    "FIP": [3.6, 3.4, 3.5, 3.9, 3.4, 3.6, 3.9, 3.3, 4.2]})
tbl = build_team_table(bat, tp, pit)
check("Loader finds all 4 teams (incl. TBR alias)", set(tbl["team"]) == set(config.TEAMS))
check("Loader splits SP/RP (CLE reliever with 1 GS stays RP)",
      abs(tbl.set_index("team").loc["CLE", "fip_rp"] - 3.3) < 1e-9)

passed = sum(ok for _, ok in results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
