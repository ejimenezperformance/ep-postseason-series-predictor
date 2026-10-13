"""
model.py
Game-level win probability from two independent approaches:

A) Component model (bateo + rotacion + bullpen)
   Expected runs for team A vs team B:
     RS_A = lgR/G * (wRC+_A / 100) * (pitching_index_B)
     pitching_index = (SP_share * FIP_SP + RP_share * FIP_RP * fatigue) / lgFIP
   Then Pythagenpat converts RS/RA to a neutral-site win probability.

B) Pythagorean baseline
   Each team's season Pythagenpat win% from actual runs scored/allowed,
   combined with log5. This is the "naive" benchmark the component model
   has to beat to justify its extra complexity.

Both apply the same home-field adjustment (odds ratio).
"""
import config


def pythagenpat_exponent(rs_per_g, ra_per_g):
    return (rs_per_g + ra_per_g) ** 0.287


def pythag_pct(rs, ra, games=1.0):
    rpg, rapg = rs / games, ra / games
    x = pythagenpat_exponent(rpg, rapg)
    return rpg ** x / (rpg ** x + rapg ** x)


def log5(p_a, p_b):
    return (p_a - p_a * p_b) / (p_a + p_b - 2 * p_a * p_b)


def apply_home_field(p_neutral, home=True):
    """Shift a neutral-site probability by the home-field odds ratio."""
    k = config.HOME_WIN_PCT / (1 - config.HOME_WIN_PCT)
    if not home:
        k = 1 / k
    odds = p_neutral / (1 - p_neutral) * k
    return odds / (1 + odds)


def fatigue_multiplier(team):
    red = config.BULLPEN_RED_SHARE.get(team, 0.0)
    return 1 + config.FATIGUE_FIP_PENALTY_PER_10PCT * (red / 0.10)


def pitching_index(row, lg_fip):
    fip = (config.STARTER_INNINGS_SHARE * row["fip_sp"]
           + config.RELIEVER_INNINGS_SHARE * row["fip_rp"] * fatigue_multiplier(row["team"]))
    return fip / lg_fip


def component_neutral(row_a, row_b, lg_rpg, lg_fip):
    rs_a = lg_rpg * (row_a["wrc_plus"] / 100) * pitching_index(row_b, lg_fip)
    ra_a = lg_rpg * (row_b["wrc_plus"] / 100) * pitching_index(row_a, lg_fip)
    return pythag_pct(rs_a, ra_a), rs_a, ra_a


def pythag_neutral(row_a, row_b):
    pa = pythag_pct(row_a["runs_scored"], row_a["runs_allowed"], row_a["games"])
    pb = pythag_pct(row_b["runs_scored"], row_b["runs_allowed"], row_b["games"])
    return log5(pa, pb)
