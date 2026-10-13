"""
config.py
EP Postseason Series Predictor — 2026 League Championship Series

Rule-based, transparent model: no machine learning, no fitted weights.
Every assumption below is written down so it can be audited and changed.

Modelo basado en reglas y transparente: sin machine learning ni pesos
ajustados. Cada supuesto esta escrito aqui para poder auditarlo y cambiarlo.
"""

SEASON = 2026

# ---------------------------------------------------------------------------
# Teams still alive (confirmed Oct 12, 2026) / Equipos vivos (confirmado 12-oct-2026)
# mlb_id = team id in the official MLB Stats API
# ---------------------------------------------------------------------------
TEAMS = {
    "MIL": {"name": "Milwaukee Brewers",   "league": "NL", "wins": 103, "losses": 59, "mlb_id": 158},
    "LAD": {"name": "Los Angeles Dodgers", "league": "NL", "wins": 100, "losses": 62, "mlb_id": 119},
    "TB":  {"name": "Tampa Bay Rays",      "league": "AL", "wins": 98,  "losses": 64, "mlb_id": 139},
    "CLE": {"name": "Cleveland Guardians", "league": "AL", "wins": 85,  "losses": 77, "mlb_id": 114},
}

# ---------------------------------------------------------------------------
# Series state — UPDATE AFTER EVERY GAME / ACTUALIZAR DESPUES DE CADA JUEGO
# home = team with home-field advantage (Games 1, 2, 6, 7 at home; 2-3-2)
# wins_home / wins_away = games already won in the series
# ---------------------------------------------------------------------------
SERIES = {
    "NLCS": {"home": "MIL", "away": "LAD", "wins_home": 2, "wins_away": 0},
    "ALCS": {"home": "TB",  "away": "CLE", "wins_home": 0, "wins_away": 0},
}

BEST_OF = 7
# 2-3-2 format: True = game played at the home-field team's park
HOME_PATTERN_7 = [True, True, False, False, False, True, True]

# ---------------------------------------------------------------------------
# Model parameters / Parametros del modelo
# ---------------------------------------------------------------------------
# Historical MLB home-team win rate is ~0.53-0.54. We use 0.54.
HOME_WIN_PCT = 0.54

# Share of postseason innings covered by starters vs. relievers.
# Postseason starters go shorter than in the regular season, so the bullpen
# carries more weight than its regular-season innings share would suggest.
STARTER_INNINGS_SHARE = 0.55
RELIEVER_INNINGS_SHARE = 1 - STARTER_INNINGS_SHARE

# A pitcher counts as a starter if at least this share of his appearances
# were starts (GS / G).
STARTER_GS_RATIO = 0.5

# Optional bullpen fatigue adjustment (from ep-bullpen-fatigue-risk).
# Fraction of a team's key relievers flagged RED. Each 10% of RED relievers
# inflates that bullpen's FIP by FATIGUE_FIP_PENALTY_PER_10PCT (as a ratio).
# Default: all zeros = adjustment OFF until real semaphore output is loaded.
BULLPEN_RED_SHARE = {"MIL": 0.0, "LAD": 0.0, "TB": 0.0, "CLE": 0.0}
FATIGUE_FIP_PENALTY_PER_10PCT = 0.02   # INTERPRETATION, not validated

# Monte Carlo
N_SIMULATIONS = 100_000
RANDOM_SEED = 2026

# ---------------------------------------------------------------------------
# EP brand / Marca EP
# ---------------------------------------------------------------------------
EP_NAVY = "#0B1B33"
EP_GOLD = "#D4A53A"
EP_OFFWHITE = "#F5F3EC"
