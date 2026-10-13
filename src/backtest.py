"""
backtest.py
Validation: run the SAME model, with the SAME frozen weights from config.py,
on every completed postseason series 2015-2026 (2020 excluded), using only
that season's regular-season stats, predicting from 0-0. Then compare with
what actually happened.

Validacion: el mismo modelo, con los mismos pesos congelados, sobre cada serie
de playoffs terminada 2015-2026 (sin 2020), con datos de esa temporada regular,
prediciendo desde 0-0. Los pesos NO se ajustan con estos resultados.

Models compared:
  component  — wRC+ / starter FIP / reliever FIP (this repo's model)
  pythag     — Pythagenpat from runs scored/allowed + log5
  record     — regular-season win% + log5 (naive benchmark)
  coin       — 50% every series (no-skill benchmark)

Correr: python src/backtest.py          (pull real, guarda cache en data/raw/backtest/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
import model  # noqa: E402
from data_loader import build_team_table, fetch_season_raw, make_session, API  # noqa: E402
from series import series_win_prob  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "raw", "backtest")
REPORTS = os.path.join(ROOT, "outputs", "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")

SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]
ROUND_NAMES = {"F": "Wild Card", "D": "Division Series", "L": "LCS", "W": "World Series"}
# True = game at the home-field team's park
PATTERNS = {
    1: [True],
    3: [True, True, True],                         # Wild Card Series (2022+)
    5: [True, True, False, False, True],           # 2-2-1
    7: [True, True, False, False, False, True, True],  # 2-3-2
}
MODELS = ("component", "pythag", "record", "coin")


# ---------------------------------------------------------------------------
# Series results from the schedule
# ---------------------------------------------------------------------------
def parse_series(schedule, season):
    """Group postseason games into series. Only Final games count."""
    games = {}
    for d in schedule.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            if "isWinner" not in g["teams"]["home"]:
                continue  # postponed / suspended entries
            games[g["gamePk"]] = g  # dedupes rescheduled duplicates

    groups = {}
    for g in games.values():
        h, a = g["teams"]["home"]["team"]["id"], g["teams"]["away"]["team"]["id"]
        key = (g["gameType"], g.get("seriesDescription", ""), frozenset((h, a)))
        groups.setdefault(key, []).append(g)

    out = []
    for (gtype, desc, _), gs in groups.items():
        gs.sort(key=lambda g: (g.get("seriesGameNumber", 0), g.get("gameDate", "")))
        first = gs[0]
        home = first["teams"]["home"]["team"]["id"]
        away = first["teams"]["away"]["team"]["id"]
        best_of = int(first.get("gamesInSeries", len(gs)))
        wins = {home: 0, away: 0}
        for g in gs:
            w = g["teams"]["home"] if g["teams"]["home"]["isWinner"] else g["teams"]["away"]
            wins[w["team"]["id"]] += 1
        need = best_of // 2 + 1
        complete = max(wins.values()) >= need
        out.append({
            "season": season, "round": ROUND_NAMES.get(gtype, gtype), "series": desc,
            "home_id": home, "away_id": away,
            "home_name": first["teams"]["home"]["team"]["name"],
            "away_name": first["teams"]["away"]["team"]["name"],
            "best_of": best_of, "wins_home": wins[home], "wins_away": wins[away],
            "complete": complete,
            "home_won": complete and wins[home] >= need,
        })
    return out


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def record_pct(pit_splits):
    out = {}
    for s in pit_splits:
        st = s["stat"]
        out[s["team"]["id"]] = st["wins"] / (st["wins"] + st["losses"])
    return out


def predict_series(table, lg_rpg, lg_fip, records, s, which):
    """P(home-field team wins the series) from 0-0."""
    if which == "coin":
        return 0.5
    h, a = table.loc[str(s["home_id"])], table.loc[str(s["away_id"])]
    if which == "component":
        neutral, _, _ = model.component_neutral(h, a, lg_rpg, lg_fip)
    elif which == "pythag":
        neutral = model.pythag_neutral(h, a)
    else:
        neutral = model.log5(records[s["home_id"]], records[s["away_id"]])
    p_home = model.apply_home_field(neutral, True)
    p_road = model.apply_home_field(neutral, False)
    return series_win_prob(p_home, p_road, best_of=s["best_of"], pattern=PATTERNS[s["best_of"]])


def season_predictions(raw, schedule, season):
    series = [s for s in parse_series(schedule, season) if s["complete"]]
    ids = sorted({s["home_id"] for s in series} | {s["away_id"] for s in series})
    built = build_team_table(raw["hit"], raw["pit"], raw["pitchers"],
                             teams={str(i): i for i in ids})
    lg_rpg, lg_fip = built.attrs["lg_rpg"], built.attrs["lg_fip"]
    table = built.set_index("team", drop=False)
    missing = {str(i) for i in ids} - set(table.index)
    if missing:
        raise RuntimeError(f"{season}: faltan datos para equipos {sorted(missing)}")
    records = record_pct(raw["pit"])
    rows = []
    for s in series:
        row = dict(s)
        for m in MODELS:
            row[f"p_{m}"] = predict_series(table, lg_rpg, lg_fip, records, s, m)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Metrics (CONFIRMED arithmetic)
# ---------------------------------------------------------------------------
def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def favorite_accuracy(p, y):
    """Share of series the model's favorite won (exact 50% calls excluded)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    mask = p != 0.5
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((p[mask] > 0.5) == (y[mask] == 1)))


def paired_bootstrap(p_a, p_b, y, n=10_000, seed=2026):
    """95% CI for Brier(A) - Brier(B). Negative = A is better."""
    rng = np.random.default_rng(seed)
    p_a, p_b, y = map(lambda v: np.asarray(v, float), (p_a, p_b, y))
    d = (p_a - y) ** 2 - (p_b - y) ** 2
    idx = rng.integers(0, len(y), size=(n, len(y)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def calibration_table(p, y, edges=(0.5, 0.55, 0.6, 0.65, 0.7, 1.0)):
    """Fold every series to the favorite's side, then bin by favorite probability."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    fav_p = np.where(p >= 0.5, p, 1 - p)
    fav_won = np.where(p >= 0.5, y, 1 - y)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fav_p >= lo) & (fav_p < hi)
        if m.sum():
            rows.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()),
                         "predicted": float(fav_p[m].mean()), "actual": float(fav_won[m].mean())})
    return pd.DataFrame(rows)


def summarize(df):
    y = df["home_won"].astype(float)
    rows = []
    for m in MODELS:
        p = df[f"p_{m}"]
        rows.append({"model": m, "n": len(df), "brier": brier(p, y), "log_loss": log_loss(p, y),
                     "favorite_won": favorite_accuracy(p, y)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def plot_calibration(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = df["home_won"].astype(float)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor=config.EP_OFFWHITE)
    ax.set_facecolor(config.EP_OFFWHITE)
    ax.plot([0.5, 0.8], [0.5, 0.8], ls="--", color="#888", lw=1, label="Perfect calibration")
    styles = {"component": (config.EP_NAVY, "o", "Component model"),
              "pythag": (config.EP_GOLD, "s", "Pythagorean baseline")}
    for m, (color, marker, label) in styles.items():
        cal = calibration_table(df[f"p_{m}"], y)
        ax.plot(cal["predicted"], cal["actual"], marker=marker, color=color, lw=1.5, label=label)
        for _, r in cal.iterrows():
            ax.annotate(f"n={r['n']}", (r["predicted"], r["actual"]), textcoords="offset points",
                        xytext=(6, -10), fontsize=7, color=color)
    ax.set_xlim(0.48, 0.82)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("Predicted P(favorite wins series)")
    ax.set_ylabel("Actual share of series won by favorite")
    ax.set_title(f"Backtest calibration — {len(df)} postseason series, "
                 f"{df['season'].min()}-{df['season'].max()} (no 2020)",
                 color=config.EP_NAVY, fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(df, summ, path):
    y = df["home_won"].astype(float)
    d, lo, hi = paired_bootstrap(df["p_component"], df["p_pythag"], y)
    d2, lo2, hi2 = paired_bootstrap(df["p_component"], df["p_coin"], y)
    lines = ["# Backtest — EP Postseason Series Predictor", "",
             f"{len(df)} completed postseason series, seasons {sorted(df['season'].unique())}.",
             "Weights frozen from `config.py`; predictions from 0-0 with regular-season data only.", "",
             "## Scores (lower Brier / log loss = better; coin flip Brier = 0.250)", "",
             "| Model | n | Brier | Log loss | Favorite won |", "|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        fw = "—" if np.isnan(r["favorite_won"]) else f"{r['favorite_won']:.1%}"
        lines.append(f"| {r['model']} | {r['n']} | {r['brier']:.4f} | {r['log_loss']:.4f} | {fw} |")
    lines += ["", "## Is the component model better? (paired bootstrap, 10,000 resamples)", "",
              f"- Component vs Pythagorean: Brier difference {d:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]",
              f"- Component vs coin flip: Brier difference {d2:+.4f}, 95% CI [{lo2:+.4f}, {hi2:+.4f}]",
              "", "Negative = component model better. A CI that crosses 0 means no detectable difference.",
              "", "## By round (component model)", "", "| Round | n | Brier | Favorite won |", "|---|---|---|---|"]
    for rnd, g in df.groupby("round"):
        yy = g["home_won"].astype(float)
        lines.append(f"| {rnd} | {len(g)} | {brier(g['p_component'], yy):.4f} | "
                     f"{favorite_accuracy(g['p_component'], yy):.1%} |")
    lines += ["", "## Calibration (component model)", "",
              "| Favorite prob. bin | n | Predicted | Actual |", "|---|---|---|---|"]
    for _, r in calibration_table(df["p_component"], y).iterrows():
        lines.append(f"| {r['bin']} | {r['n']} | {r['predicted']:.1%} | {r['actual']:.1%} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return d, lo, hi


def load_or_fetch(session, season):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{season}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    r = session.get(f"{API}/schedule", params={"sportId": 1, "season": season,
                                              "gameType": "F,D,L,W"}, timeout=60)
    r.raise_for_status()
    schedule = r.json()
    series = [s for s in parse_series(schedule, season) if s["complete"]]
    ids = sorted({s["home_id"] for s in series} | {s["away_id"] for s in series})
    raw = fetch_season_raw(session, season, ids)
    blob = {"schedule": schedule, "raw": raw}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh)
    return blob


def main():
    session = make_session()
    rows = []
    for season in SEASONS:
        print(f"Temporada {season}...", end=" ", flush=True)
        blob = load_or_fetch(session, season)
        season_rows = season_predictions(blob["raw"], blob["schedule"], season)
        rows += season_rows
        print(f"{len(season_rows)} series")

    df = pd.DataFrame(rows)
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    df.to_csv(os.path.join(REPORTS, "backtest_series.csv"), index=False)
    summ = summarize(df)
    summ.to_csv(os.path.join(REPORTS, "backtest_scores.csv"), index=False)
    d, lo, hi = write_summary(df, summ, os.path.join(REPORTS, "backtest_summary.md"))
    plot_calibration(df, os.path.join(FIGURES, "backtest_calibration.png"))

    print()
    print(summ.round(4).to_string(index=False))
    print(f"\nComponent - Pythag Brier: {d:+.4f}  (95% CI {lo:+.4f} a {hi:+.4f})")
    print("Listo. Ver outputs/reports/backtest_summary.md y outputs/figures/backtest_calibration.png")


if __name__ == "__main__":
    main()
