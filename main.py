#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).

For each case in the input directory, forecast every market's Yes price at
each of the next `horizon_minutes` minutes as five quantiles.

Method, per market: the honest distribution of the price change over h
minutes is estimated from the market's own history in the case. For a grid
of horizons H, take every overlapping h-minute change p[i] - p[i-H] in the
72 hours up to the checkpoint and read off its empirical quantiles at the
case's levels; interpolate each level's offset between grid horizons
linearly in sqrt(h) (price moves accumulate roughly like a random walk, but
the empirical quantiles keep the step-and-jump shape of these markets: a
large atom at 0 and fat tails, which a normal spread misprices). Markets
with little history are blended toward the house baseline (flat price,
sigma 0.0025 per sqrt-minute) in proportion to how much history they have.

The raw historical offsets are then shrunk: deviations from the median by
SPREAD_SCALE and the median's own drift by CENTRE_SHRINK. A 72-hour window
holds older, rougher regimes than the next 4 hours usually repeat, and the
house baseline's own published practice coverage (0.9496 for a nominal
90% interval) shows how much an unshrunk spread overpays at every quiet
minute; the shrink factors sit on the flat part of the walk-forward
backtest curve over the practice cases. No interval is allowed to be
narrower than FLOOR_FRACTION of the baseline's spread around the median,
as insurance against a market that slept through its history waking up.
Everything is deterministic: no randomness, no network, no clock.
"""

import json
import math
import os
import sys
from bisect import bisect_left
from statistics import NormalDist

BASE_SIGMA = 0.0025          # house baseline spread, price units per sqrt(minute)
SPREAD_SCALE = 0.50          # multiplier on deviations from the median offset
CENTRE_SHRINK = 0.30         # multiplier on the median offset (the drift)
FLOOR_FRACTION = 0.05        # min spread around the median, as a share of baseline
FULL_HISTORY_RETURNS = 1440  # returns at/above which the empirical forecast stands alone
MIN_SAMPLES = 40             # overlapping changes needed to trust a horizon's quantiles
MAX_SAMPLES = 3000           # cap on changes sampled per horizon (stride beyond this)
GRID = [1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 29, 38, 50, 66, 86, 113, 148, 194, 240]
TAUS_DEFAULT = [0.05, 0.25, 0.5, 0.75, 0.95]


def quantile(sorted_vals, tau):
    """Linear-interpolated quantile of an already-sorted list."""
    m = len(sorted_vals)
    if m == 1:
        return sorted_vals[0]
    pos = tau * (m - 1)
    i = int(pos)
    frac = pos - i
    if i + 1 < m:
        return sorted_vals[i] + frac * (sorted_vals[i + 1] - sorted_vals[i])
    return sorted_vals[i]


def price6(v):
    """Clamp to [0, 1] and round to 6 decimals (never -0.0)."""
    r = round(min(1.0, max(0.0, v)), 6)
    return 0.0 if r == 0 else r


def empirical_offsets(prices, taus, horizon):
    """Quantiles of the h-minute price change at each grid horizon.

    Returns (grid_hs, grid_offsets) where grid_offsets[k][j] is the tau_j
    quantile of the grid_hs[k]-minute change, or None when the history is
    too short to trust at any horizon.
    """
    n = len(prices)
    hs = []
    offsets = []
    for h in GRID:
        if h > horizon:
            break
        count = n - h
        if count < MIN_SAMPLES:
            continue
        stride = 1
        if count > MAX_SAMPLES:
            stride = -(-count // MAX_SAMPLES)
        deltas = sorted(prices[i] - prices[i - h] for i in range(h, n, stride))
        hs.append(h)
        offsets.append([quantile(deltas, t) for t in taus])
    if not hs:
        return None
    return hs, offsets


def interp_offsets(hs, offsets, h):
    """Each level's offset at horizon h, interpolated linearly in sqrt(h)."""
    if h <= hs[0]:
        # Below the first grid horizon: scale its offsets like a random walk.
        s = math.sqrt(h / hs[0])
        return [o * s for o in offsets[0]]
    if h >= hs[-1]:
        s = math.sqrt(h / hs[-1])
        return [o * s for o in offsets[-1]]
    k = bisect_left(hs, h)
    if hs[k] == h:
        return list(offsets[k])
    h0, h1 = hs[k - 1], hs[k]
    s0, s1 = math.sqrt(h0), math.sqrt(h1)
    w = (math.sqrt(h) - s0) / (s1 - s0)
    return [a + w * (b - a) for a, b in zip(offsets[k - 1], offsets[k])]


def forecast_market(prices, taus, zs, horizon):
    """horizon rows of len(taus) quantiles for one market."""
    x = prices[-1] if prices else 0.0
    n = len(prices)
    emp = empirical_offsets(prices, taus, horizon) if n >= 2 else None
    # Share of the forecast taken from the market's own history; the rest is
    # the house baseline. Full weight once a day of returns is in hand.
    w_emp = min(1.0, (n - 1) / FULL_HISTORY_RETURNS) if emp else 0.0
    med = min(range(len(taus)), key=lambda j: abs(taus[j] - 0.5))
    rows = []
    if emp:
        hs, offsets = emp
    for h in range(1, horizon + 1):
        sq = math.sqrt(h)
        if emp:
            e = interp_offsets(hs, offsets, h)
            offs = [w_emp * e[j] + (1.0 - w_emp) * zs[j] * BASE_SIGMA * sq
                    for j in range(len(taus))]
        else:
            offs = [zs[j] * BASE_SIGMA * sq for j in range(len(taus))]
        raw_centre = offs[med]
        centre = raw_centre * CENTRE_SHRINK
        row = []
        prev = 0.0
        for j in range(len(taus)):
            dev = (offs[j] - raw_centre) * SPREAD_SCALE
            floor = FLOOR_FRACTION * abs(zs[j]) * BASE_SIGMA * sq
            if abs(dev) < floor:
                dev = floor if zs[j] > 0 else (-floor if zs[j] < 0 else 0.0)
            v = price6(x + centre + dev)
            if v < prev:
                v = prev
            prev = v
            row.append(v)
        rows.append(row)
    return rows


def run_case(path, out_dir):
    with open(path, "r", encoding="utf-8") as f:
        case = json.load(f)
    horizon = int(case.get("horizon_minutes", 240))
    taus = [float(t) for t in case.get("quantiles", TAUS_DEFAULT)]
    nd = NormalDist()
    zs = [nd.inv_cdf(t) for t in taus]
    forecasts = []
    for m in case["markets"]:
        prices = [float(p) for p in m["prices"]]
        forecasts.append({"id": m["id"],
                          "q": forecast_market(prices, taus, zs, horizon)})
    out_path = os.path.join(out_dir, case["case"] + ".out")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(fn for fn in os.listdir(in_dir) if fn.endswith(".in"))
    for fn in names:
        run_case(os.path.join(in_dir, fn), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
