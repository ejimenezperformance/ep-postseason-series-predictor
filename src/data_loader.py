"""
data_loader.py
Pulls 2026 regular-season stats from the official MLB Stats API
(statsapi.mlb.com, free, no key) and builds one row per team still alive:

    team, games, runs_scored, runs_allowed, wrc_plus, fip_sp, fip_rp, ip_sp, ip_rp

FanGraphs was the original source, but it blocks automated pybaseball
requests (HTTP error on leaders-legacy.aspx), so offense and FIP are
computed here from raw counting stats:

  - wOBA from standard linear weights; wRC+ approximated from wOBA
    (NO park adjustment -> labeled INTERPRETATION in the README)
  - FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + C, with C set so league FIP = league ERA

Descarga datos de la API oficial de MLB. Correr: python src/data_loader.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
API = "https://statsapi.mlb.com/api/v1"

# Standard wOBA linear weights (recent-season typical values) and scale.
W = {"uBB": 0.69, "HBP": 0.72, "1B": 0.88, "2B": 1.25, "3B": 1.58, "HR": 2.03}
WOBA_SCALE = 1.24


def ip_to_float(ip):
    """MLB IP strings like '45.2' mean 45 2/3 innings."""
    whole, _, frac = str(ip).partition(".")
    return int(whole) + int(frac or 0) / 3


def _team_frame(splits):
    rows = []
    for s in splits:
        row = {"team_id": s["team"]["id"], "team_name": s["team"]["name"]}
        row.update(s["stat"])
        rows.append(row)
    return pd.DataFrame(rows)


def woba(df):
    singles = df["hits"] - df["doubles"] - df["triples"] - df["homeRuns"]
    ubb = df["baseOnBalls"] - df["intentionalWalks"]
    num = (W["uBB"] * ubb + W["HBP"] * df["hitByPitch"] + W["1B"] * singles
           + W["2B"] * df["doubles"] + W["3B"] * df["triples"] + W["HR"] * df["homeRuns"])
    den = df["atBats"] + ubb + df["sacFlies"] + df["hitByPitch"]
    return num, den


def fip_components(df):
    hbp = df["hitByPitch"] if "hitByPitch" in df else df["hitBatsmen"]
    return 13 * df["homeRuns"] + 3 * (df["baseOnBalls"] + hbp) - 2 * df["strikeOuts"]


def build_team_table(hit_splits, pit_splits, pitcher_splits_by_team, teams=None):
    """
    Pure function (testable without network).
    hit_splits / pit_splits: stats[0].splits from /teams/stats (all 30 teams)
    pitcher_splits_by_team: {our_key: stats[0].splits from /stats?teamId=...}
    teams: {our_key: mlb_id}; defaults to the 4 teams in config.TEAMS
    """
    hit = _team_frame(hit_splits)
    pit = _team_frame(pit_splits)
    pit["IP"] = pit["inningsPitched"].apply(ip_to_float)

    # League baselines
    lg_rpg = hit["runs"].sum() / hit["gamesPlayed"].sum()
    lg_pa = hit["plateAppearances"].sum() if "plateAppearances" in hit else (
        hit["atBats"] + hit["baseOnBalls"] + hit["hitByPitch"] + hit["sacFlies"]).sum()
    lg_r_per_pa = hit["runs"].sum() / lg_pa
    num, den = woba(hit)
    lg_woba = num.sum() / den.sum()
    lg_era = 9 * pit["earnedRuns"].sum() / pit["IP"].sum()
    fip_c = lg_era - fip_components(pit).sum() / pit["IP"].sum()
    hit["wOBA"] = num / den

    if teams is None:
        teams = {k: v["mlb_id"] for k, v in config.TEAMS.items()}

    rows = []
    for key, mlb_id in teams.items():
        h = hit[hit["team_id"] == mlb_id]
        p = pit[pit["team_id"] == mlb_id]
        if h.empty or p.empty or key not in pitcher_splits_by_team:
            continue
        h, p = h.iloc[0], p.iloc[0]
        wrc_plus = 100 * ((h["wOBA"] - lg_woba) / WOBA_SCALE + lg_r_per_pa) / lg_r_per_pa

        pl = pd.DataFrame([{**s["stat"], "player": s["player"]["fullName"]}
                           for s in pitcher_splits_by_team[key]])
        pl = pl[pl["gamesPitched"] > 0].copy()
        pl["IP"] = pl["inningsPitched"].apply(ip_to_float)
        pl["role"] = (pl["gamesStarted"] / pl["gamesPitched"] >= config.STARTER_GS_RATIO).map(
            {True: "SP", False: "RP"})

        row = {"team": key, "games": int(h["gamesPlayed"]),
               "runs_scored": float(h["runs"]), "runs_allowed": float(p["runs"]),
               "wrc_plus": float(wrc_plus), "woba": float(h["wOBA"])}
        for role in ("SP", "RP"):
            sub = pl[pl["role"] == role]
            ip = sub["IP"].sum()
            row[f"ip_{role.lower()}"] = ip
            row[f"fip_{role.lower()}"] = (fip_components(sub).sum() / ip + fip_c) if ip > 0 else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs.update({"lg_rpg": lg_rpg, "lg_fip": lg_era, "lg_woba": lg_woba, "fip_c": fip_c})
    return out


def _get(session, path, **params):
    r = session.get(f"{API}/{path}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()["stats"][0]["splits"]


def make_session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "ep-postseason-series-predictor (research)"
    return s


def fetch_season_raw(session, season, team_ids):
    """Team hitting/pitching (all 30) + per-team pitcher lines for team_ids."""
    common = {"season": season, "stats": "season", "sportIds": 1}
    hit = _get(session, "teams/stats", group="hitting", **common)
    pit = _get(session, "teams/stats", group="pitching", **common)
    pitchers = {}
    for tid in team_ids:
        pitchers[str(tid)] = _get(session, "stats", group="pitching", playerPool="ALL",
                                  limit=3000, teamId=tid, **common)
    return {"hit": hit, "pit": pit, "pitchers": pitchers}


def main():
    try:
        import requests
    except ImportError:
        sys.exit("Falta requests: pip install -r requirements.txt")

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "ep-postseason-series-predictor (research)"

    print(f"Descargando datos {config.SEASON} de la API oficial de MLB...")
    common = {"season": config.SEASON, "stats": "season", "sportIds": 1}
    hit = _get(s, "teams/stats", group="hitting", **common)
    pit = _get(s, "teams/stats", group="pitching", **common)
    pitchers = {}
    for key, info in config.TEAMS.items():
        pitchers[key] = _get(s, "stats", group="pitching", playerPool="ALL", limit=3000,
                             teamId=info["mlb_id"], **common)
        print(f"  {key}: {len(pitchers[key])} lanzadores")

    pd.json_normalize(hit).to_csv(os.path.join(RAW, "team_hitting.csv"), index=False)
    pd.json_normalize(pit).to_csv(os.path.join(RAW, "team_pitching.csv"), index=False)
    pd.concat([pd.json_normalize(v).assign(team_key=k) for k, v in pitchers.items()]).to_csv(
        os.path.join(RAW, "pitchers.csv"), index=False)

    table = build_team_table(hit, pit, pitchers)
    missing = set(config.TEAMS) - set(table["team"])
    if missing:
        sys.exit(f"FALTAN equipos: {sorted(missing)}")

    table.to_csv(os.path.join(PROCESSED, "team_inputs.csv"), index=False)
    pd.Series({"lg_rpg": table.attrs["lg_rpg"], "lg_fip": table.attrs["lg_fip"]}).to_csv(
        os.path.join(PROCESSED, "league_baseline.csv"), header=["value"])
    print()
    print(table.round(3).to_string(index=False))
    print(f"\nLiga: {table.attrs['lg_rpg']:.3f} R/G, ERA/FIP {table.attrs['lg_fip']:.3f}, "
          f"wOBA {table.attrs['lg_woba']:.3f}")
    print(f"OK: los {len(table)} equipos vivos aparecen en los datos")


if __name__ == "__main__":
    main()
