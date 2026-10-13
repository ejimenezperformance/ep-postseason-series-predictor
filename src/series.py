"""
series.py
Exact best-of-N series probability from the current series state, with a
different win probability for home and road games (2-3-2 format).

Probabilidad exacta de ganar la serie desde el estado actual.
"""
from functools import lru_cache

import config


def series_win_prob(p_home_game, p_road_game, wins_a=0, wins_b=0,
                    best_of=None, pattern=None):
    """
    Probability that team A (the home-field team) wins the series.

    p_home_game : P(A wins) in a game at A's park
    p_road_game : P(A wins) in a game at B's park
    wins_a, wins_b : games already won
    """
    best_of = best_of or config.BEST_OF
    pattern = pattern or config.HOME_PATTERN_7
    need = best_of // 2 + 1

    @lru_cache(maxsize=None)
    def f(a, b):
        if a >= need:
            return 1.0
        if b >= need:
            return 0.0
        game_index = a + b
        p = p_home_game if pattern[game_index] else p_road_game
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)

    return f(wins_a, wins_b)


def length_distribution(p_home_game, p_road_game, wins_a=0, wins_b=0,
                        best_of=None, pattern=None):
    """P(series ends in exactly g games, winner) for reporting."""
    best_of = best_of or config.BEST_OF
    pattern = pattern or config.HOME_PATTERN_7
    need = best_of // 2 + 1
    out = {}

    def walk(a, b, prob):
        if a >= need or b >= need:
            key = (a + b, "A" if a >= need else "B")
            out[key] = out.get(key, 0.0) + prob
            return
        p = p_home_game if pattern[a + b] else p_road_game
        walk(a + 1, b, prob * p)
        walk(a, b + 1, prob * (1 - p))

    walk(wins_a, wins_b, 1.0)
    return out
