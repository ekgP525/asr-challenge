"""ITU-T P.56 active speech level (method B).

Returns the active speech level in dB relative to full scale (dBov, 1.0 = 0 dB)
and the activity factor. Implementation follows the recommendation's
envelope/hangover algorithm (as in the ITU-T STL `sv56` tool):

  envelope q[n] = g*q[n-1] + (1-g)*|x[n]|,  g = exp(-1/(fs*0.03))   (30 ms smoothing)
  for thresholds c_j = 2^-j * full scale (lowest first), count samples with
  q >= c_j plus a 200 ms hangover; active level A_j = 10 log10(sum(x^2) / a_j);
  the level is the A_j where A_j - C_j drops through 15.9 dB (interpolated),
  C_j = 20 log10(c_j).

Reference: ITU-T Recommendation P.56 (12/2011), "Objective measurement of active speech level".
"""
from __future__ import annotations

import math

import numpy as np

MARGIN_DB = 15.9
HANGOVER_S = 0.2
SMOOTH_S = 0.03


def active_speech_level(x: np.ndarray, fs: int, n_thresholds: int = 15):
    """-> (level_dBov, activity_factor). level is 20*log10(rms) of the active part."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return -np.inf, 0.0
    sq = float(np.sum(x * x))
    if sq <= 0:
        return -np.inf, 0.0
    g = math.exp(-1.0 / (fs * SMOOTH_S))
    # two-stage exponential envelope (P.56 uses p then q)
    from scipy.signal import lfilter
    p = lfilter([1 - g], [1, -g], np.abs(x))
    q = lfilter([1 - g], [1, -g], p)
    hang = int(round(HANGOVER_S * fs))
    long_rms_db = 10 * math.log10(sq / x.size)
    # thresholds from the lowest (almost everything active, A - C large) upwards;
    # the active level is where A_j - C_j drops through the 15.9 dB margin
    thresholds = [2.0 ** -(n_thresholds - 1 - j) for j in range(n_thresholds)]
    prev = None
    for c in thresholds:
        above = q >= c
        if not above.any():
            break
        act = above
        if hang > 0:
            idx = np.flatnonzero(above)
            act = np.zeros(x.size, dtype=bool)
            starts = idx[np.r_[True, np.diff(idx) > 1]]
            stops = idx[np.r_[np.diff(idx) > 1, True]]
            for s0, e0 in zip(starts, stops):
                act[s0: min(x.size, e0 + 1 + hang)] = True
        a = int(act.sum())
        A = 10 * math.log10(sq / a)
        C = 20 * math.log10(c)
        d = A - C
        if d <= MARGIN_DB:
            if prev is None:
                return A, a / x.size
            A_prev, d_prev, a_prev = prev
            if d_prev == d:
                return A, a / x.size
            t = (d_prev - MARGIN_DB) / (d_prev - d)
            return A_prev + t * (A - A_prev), (a_prev + t * (a - a_prev)) / x.size
        prev = (A, d, a)
    # never crossed: signal is essentially always active (e.g. noise) -> long-term rms
    return long_rms_db, 1.0


def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.mean(x * x)) if x.size else 0.0
    return 10 * math.log10(m) if m > 0 else -np.inf
