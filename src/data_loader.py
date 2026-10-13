"""
data_loader.py
Pulls 2026 regular-season team and pitcher stats from FanGraphs via pybaseball
and builds one row per team still alive:

    team, games, runs_scored, runs_allowed, wrc_plus, fip_sp, fip_rp, ip_sp, ip_rp

Descarga las estadisticas 2026 de FanGraphs (pybaseball) y arma una fila por
equipo vivo. Correr: python src/data_loader.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")


def _canonical(team_value):
    """Map a FanGraphs team label to our key (MIL, LAD, TB, CLE) or None."""
    for key, info in config.TEAMS.items():
        if str(team_value).strip() in info["aliases"]:
            return key
    return None


def _ip_to_float(ip):
    """FanGraphs IP like 45.2 means 45 2/3 innings."""
    whole = int(ip)
    outs = round((ip - whole) * 10)
    return whole + outs / 3


def build_team_table(bat, team_pit, pitchers):
    """Pure function (testable without network)."""
    bat = bat.copy()
    bat["key"] = bat["Team"].map(_canonical)
    team_pit = team_pit.copy()
    team_pit["key"] = team_pit["Team"].map(_canonical)

    p = pitchers.copy()
    p = p[p["G"] > 0]
    p["key"] = p["Team"].map(_canonical)
    p["IP_f"] = p["IP"].apply(_ip_to_float)
    p["role"] = (p["GS"] / p["G"] >= config.STARTER_GS_RATIO).map({True: "SP", False: "RP"})

    lg_rpg = bat["R"].sum() / bat["G"].sum()
    lg_fip = (p["FIP"] * p["IP_f"]).sum() / p["IP_f"].sum()

    rows = []
    for key in config.TEAMS:
        b = bat[bat["key"] == key]
        t = team_pit[team_pit["key"] == key]
        if b.empty or t.empty:
            continue
        row = {
            "team": key,
            "games": int(b["G"].iloc[0]),
            "runs_scored": float(b["R"].iloc[0]),
            "runs_allowed": float(t["R"].iloc[0]),
            "wrc_plus": float(b["wRC+"].iloc[0]),
        }
        for role in ("SP", "RP"):
            sub = p[(p["key"] == key) & (p["role"] == role)]
            ip = sub["IP_f"].sum()
            row[f"ip_{role.lower()}"] = ip
            row[f"fip_{role.lower()}"] = (sub["FIP"] * sub["IP_f"]).sum() / ip if ip > 0 else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.attrs["lg_rpg"] = lg_rpg
    out.attrs["lg_fip"] = lg_fip
    return out


def main():
    try:
        from pybaseball import team_batting, team_pitching, pitching_stats
    except ImportError:
        sys.exit("Falta pybaseball: pip install -r requirements.txt")

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(PROCESSED, exist_ok=True)

    print(f"Descargando datos {config.SEASON} de FanGraphs...")
    bat = team_batting(config.SEASON)
    team_pit = team_pitching(config.SEASON)
    pitchers = pitching_stats(config.SEASON, qual=0)

    bat.to_csv(os.path.join(RAW, "team_batting.csv"), index=False)
    team_pit.to_csv(os.path.join(RAW, "team_pitching.csv"), index=False)
    pitchers.to_csv(os.path.join(RAW, "pitchers.csv"), index=False)

    table = build_team_table(bat, team_pit, pitchers)
    missing = set(config.TEAMS) - set(table["team"])
    if missing:
        labels = sorted(bat["Team"].astype(str).unique())
        sys.exit(f"FALTAN equipos: {sorted(missing)}. Etiquetas en FanGraphs: {labels}")

    table.to_csv(os.path.join(PROCESSED, "team_inputs.csv"), index=False)
    pd.Series({"lg_rpg": table.attrs["lg_rpg"], "lg_fip": table.attrs["lg_fip"]}).to_csv(
        os.path.join(PROCESSED, "league_baseline.csv"), header=["value"])
    print(table.round(3).to_string(index=False))
    print(f"\nLiga: {table.attrs['lg_rpg']:.3f} R/G, FIP {table.attrs['lg_fip']:.3f}")
    print(f"OK: los {len(table)} equipos vivos aparecen en los datos")


if __name__ == "__main__":
    main()
