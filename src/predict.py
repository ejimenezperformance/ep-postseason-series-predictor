"""
predict.py
Runs both models on every remaining series, then simulates the World Series.

Outputs:
  outputs/reports/series_predictions.csv
  outputs/reports/title_odds.csv
  outputs/reports/summary.md
  outputs/figures/title_odds.png

Correr: python src/predict.py            (datos reales, despues de data_loader.py)
        python src/predict.py --synthetic (demo con datos inventados)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
import model  # noqa: E402
from series import series_win_prob, length_distribution  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "outputs", "reports")
FIGURES = os.path.join(ROOT, "outputs", "figures")
MODELS = ("component", "pythag")


def synthetic_inputs():
    """Clearly fake numbers used only for tests and the demo run."""
    df = pd.DataFrame([
        {"team": "MIL", "games": 162, "runs_scored": 790, "runs_allowed": 610,
         "wrc_plus": 108, "fip_sp": 3.70, "fip_rp": 3.55, "ip_sp": 880, "ip_rp": 570},
        {"team": "LAD", "games": 162, "runs_scored": 820, "runs_allowed": 640,
         "wrc_plus": 116, "fip_sp": 3.60, "fip_rp": 3.80, "ip_sp": 860, "ip_rp": 590},
        {"team": "TB", "games": 162, "runs_scored": 720, "runs_allowed": 600,
         "wrc_plus": 104, "fip_sp": 3.55, "fip_rp": 3.70, "ip_sp": 890, "ip_rp": 560},
        {"team": "CLE", "games": 162, "runs_scored": 680, "runs_allowed": 650,
         "wrc_plus": 96, "fip_sp": 3.85, "fip_rp": 3.50, "ip_sp": 850, "ip_rp": 600},
    ])
    return df, 4.45, 4.05


def load_inputs():
    path = os.path.join(PROCESSED, "team_inputs.csv")
    if not os.path.exists(path):
        sys.exit("No hay datos: corre primero  python src/data_loader.py")
    df = pd.read_csv(path)
    base = pd.read_csv(os.path.join(PROCESSED, "league_baseline.csv"), index_col=0)["value"]
    return df, float(base["lg_rpg"]), float(base["lg_fip"])


def game_probs(rows, home, away, lg_rpg, lg_fip, which):
    """P(home-field team wins) at its park and at the opponent's park."""
    a, b = rows[home], rows[away]
    if which == "component":
        neutral, _, _ = model.component_neutral(a, b, lg_rpg, lg_fip)
    else:
        neutral = model.pythag_neutral(a, b)
    return neutral, model.apply_home_field(neutral, True), model.apply_home_field(neutral, False)


def series_table(df, lg_rpg, lg_fip):
    rows = {r["team"]: r for _, r in df.iterrows()}
    out = []
    for name, s in config.SERIES.items():
        for which in MODELS:
            neutral, p_home, p_road = game_probs(rows, s["home"], s["away"], lg_rpg, lg_fip, which)
            p_series = series_win_prob(p_home, p_road, s["wins_home"], s["wins_away"])
            dist = length_distribution(p_home, p_road, s["wins_home"], s["wins_away"])
            likely = max(dist.items(), key=lambda kv: kv[1])
            winner = s["home"] if likely[0][1] == "A" else s["away"]
            out.append({
                "series": name, "model": which,
                "home_field": s["home"], "opponent": s["away"],
                "state": f"{s['wins_home']}-{s['wins_away']}",
                "p_game_neutral": neutral, "p_game_home": p_home, "p_game_road": p_road,
                "p_series_home_field": p_series, "p_series_opponent": 1 - p_series,
                "most_likely_result": f"{winner} in {likely[0][0]} ({likely[1]:.1%})",
            })
    return pd.DataFrame(out)


def ws_home_field(t1, t2):
    """World Series home field: better regular-season record."""
    w1 = config.TEAMS[t1]["wins"] / (config.TEAMS[t1]["wins"] + config.TEAMS[t1]["losses"])
    w2 = config.TEAMS[t2]["wins"] / (config.TEAMS[t2]["wins"] + config.TEAMS[t2]["losses"])
    return (t1, t2) if w1 >= w2 else (t2, t1)


def title_odds(df, lg_rpg, lg_fip, which, n=None, seed=None):
    """Monte Carlo: LCS winners drawn from exact series odds, then exact WS odds."""
    n = n or config.N_SIMULATIONS
    rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
    rows = {r["team"]: r for _, r in df.iterrows()}

    lcs = {}
    for name, s in config.SERIES.items():
        _, ph, pr = game_probs(rows, s["home"], s["away"], lg_rpg, lg_fip, which)
        lcs[name] = (s["home"], s["away"], series_win_prob(ph, pr, s["wins_home"], s["wins_away"]))

    ws_cache = {}

    def ws_prob(t1, t2):
        if (t1, t2) not in ws_cache:
            h, a = ws_home_field(t1, t2)
            _, ph, pr = game_probs(rows, h, a, lg_rpg, lg_fip, which)
            p = series_win_prob(ph, pr)
            ws_cache[(t1, t2)] = p if h == t1 else 1 - p
        return ws_cache[(t1, t2)]

    pennants = {t: 0 for t in config.TEAMS}
    titles = {t: 0 for t in config.TEAMS}
    nl = lcs["NLCS"]
    al = lcs["ALCS"]
    nl_draw = rng.random(n) < nl[2]
    al_draw = rng.random(n) < al[2]
    ws_draw = rng.random(n)
    for i in range(n):
        nl_w = nl[0] if nl_draw[i] else nl[1]
        al_w = al[0] if al_draw[i] else al[1]
        pennants[nl_w] += 1
        pennants[al_w] += 1
        champ = nl_w if ws_draw[i] < ws_prob(nl_w, al_w) else al_w
        titles[champ] += 1

    return pd.DataFrame([
        {"team": t, "model": which, "p_pennant": pennants[t] / n, "p_title": titles[t] / n}
        for t in config.TEAMS
    ])


def plot_title_odds(odds, path, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv = odds.pivot(index="team", columns="model", values="p_title")
    piv = piv.sort_values("component", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=config.EP_OFFWHITE)
    ax.set_facecolor(config.EP_OFFWHITE)
    y = np.arange(len(piv))
    ax.barh(y + 0.2, piv["component"], height=0.38, color=config.EP_NAVY, label="Component model")
    ax.barh(y - 0.2, piv["pythag"], height=0.38, color=config.EP_GOLD, label="Pythagorean baseline")
    for i, (c, p) in enumerate(zip(piv["component"], piv["pythag"])):
        ax.text(c + 0.005, i + 0.2, f"{c:.1%}", va="center", fontsize=9, color=config.EP_NAVY)
        ax.text(p + 0.005, i - 0.2, f"{p:.1%}", va="center", fontsize=9, color="#7a5c14")
    ax.axvline(0.25, color="#888", ls="--", lw=1)
    ax.text(0.252, len(piv) - 0.45, "coin-flip line (25%)", fontsize=8, color="#666")
    ax.set_yticks(y)
    ax.set_yticklabels([config.TEAMS[t]["name"] for t in piv.index])
    ax.set_xlabel("P(win 2026 World Series)")
    title = "2026 World Series title odds — EP Postseason Series Predictor"
    if synthetic:
        title += "\nSYNTHETIC DEMO DATA — NOT REAL"
    ax.set_title(title, color=config.EP_NAVY, fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_xlim(0, max(piv.max().max() * 1.25, 0.35))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(st, odds, lg_rpg, lg_fip, path, synthetic):
    lines = ["# EP Postseason Series Predictor — Results", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DEMO DATA — NOT REAL RESULTS**", ""]
    lines += [f"League baseline: {lg_rpg:.3f} R/G, FIP {lg_fip:.3f}", "",
              "## Series odds (team with home field listed first)", "",
              "| Series | State | Model | P(game, home) | P(game, road) | P(series) | Most likely |",
              "|---|---|---|---|---|---|---|"]
    for _, r in st.iterrows():
        lines.append(f"| {r['series']} {r['home_field']}-{r['opponent']} | {r['state']} | {r['model']} | "
                     f"{r['p_game_home']:.3f} | {r['p_game_road']:.3f} | "
                     f"{r['home_field']} {r['p_series_home_field']:.1%} | {r['most_likely_result']} |")
    lines += ["", "## Pennant and title odds", "",
              "| Team | Model | P(pennant) | P(title) |", "|---|---|---|---|"]
    for _, r in odds.sort_values(["model", "p_title"], ascending=[True, False]).iterrows():
        lines.append(f"| {r['team']} | {r['model']} | {r['p_pennant']:.1%} | {r['p_title']:.1%} |")
    gap = st.groupby("series")["p_series_home_field"].agg(lambda s: s.max() - s.min())
    lines += ["", "## Model disagreement (CONFIRMED arithmetic)", ""]
    for name, g in gap.items():
        lines.append(f"- {name}: the two models differ by {g:.1%} in series odds")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    df, lg_rpg, lg_fip = synthetic_inputs() if args.synthetic else load_inputs()
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    suffix = "_SYNTHETIC" if args.synthetic else ""

    st = series_table(df, lg_rpg, lg_fip)
    odds = pd.concat([title_odds(df, lg_rpg, lg_fip, m) for m in MODELS], ignore_index=True)

    st.to_csv(os.path.join(REPORTS, f"series_predictions{suffix}.csv"), index=False)
    odds.to_csv(os.path.join(REPORTS, f"title_odds{suffix}.csv"), index=False)
    write_summary(st, odds, lg_rpg, lg_fip, os.path.join(REPORTS, f"summary{suffix}.md"), args.synthetic)
    plot_title_odds(odds, os.path.join(FIGURES, f"title_odds{suffix}.png"), args.synthetic)

    pd.set_option("display.width", 160)
    print(st[["series", "model", "state", "p_game_home", "p_game_road",
              "p_series_home_field", "most_likely_result"]].round(3).to_string(index=False))
    print()
    print(odds.round(3).to_string(index=False))
    print(f"\nListo. Reportes en outputs/reports/, grafica en outputs/figures/{' (DEMO)' if args.synthetic else ''}")


if __name__ == "__main__":
    main()
