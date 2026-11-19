#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

Reads every ``*.in`` case in $MAKEFUN_INPUT_DIR (else /input) and writes one
``<id>.out`` JSON forecast per case to $MAKEFUN_OUTPUT_DIR (else /output).

Model, per trophy, on the logit scale
--------------------------------------
Two forces move an observed earn rate:

* dilution -- new players flood in, so the tracked-player count grows and
  every rate is pushed down;
* progression -- players already tracked keep earning story/activity
  trophies, pushing those rates up toward a ceiling below 100%.

The observed rate is the net of both.  A ridge regression of
logit(rate) on ln(players_tracked) and ln(1 + hours since release) is fit
over the case's whole history (not just the last 24 h), shrunk toward the
last observed rate when the history has few distinct snapshots.  The
tracked-player count at the target is forecast by fitting ln(players)
against ln(1 + hours) over the distinct snapshots and extrapolating with a
hard cap on growth, then the regression is read at that player count and
at the target hour.

Spread: the house random-walk baseline (sigma 0.5*sqrt(days)) lost ~1.51
on the practice because one Exophase refresh shifts every rate at once.
We therefore use a much wider logit spread (base ~1.25, larger for rare
trophies, growing slowly with horizon) with stretched outer quantiles, so a
refresh jump stays inside the 90% band most of the time.

Everything is deterministic (no wall clock, no randomness), stdlib only,
and runs in well under a second per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Stretched normal quantiles: heavier tails for refresh-jump risk.
Z = (-1.9, -0.75, 0.0, 0.75, 1.9)

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def parse_time(s):
    """ISO-8601 UTC timestamp -> datetime (accepts trailing Z)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logit(p):
    """p in percent -> logit of p/100, clamped to [0.01, 99.99]."""
    p = min(99.99, max(0.01, p))
    f = p / 100.0
    return math.log(f / (1.0 - f))


def inv_logit(x):
    """logit -> percent in [0, 100]."""
    if x >= 0:
        f = 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        f = e / (1.0 + e)
    return 100.0 * f


def hours_since_release(t):
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def solve_3x3(a, b):
    """Solve a 3x3 linear system by Gaussian elimination; None if singular."""
    n = 3
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                factor = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= factor * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def ridge_fit(xs, ys, lam):
    """Ridge least squares for y ~ b0 + b1*x1 + b2*x2; xs are (x1, x2) pairs.

    The intercept is not penalised; the slopes are.  Returns (b0, b1, b2)
    or None when the design is degenerate.
    """
    n = len(xs)
    if n == 0:
        return None
    mx1 = sum(x[0] for x in xs) / n
    mx2 = sum(x[1] for x in xs) / n
    my = sum(ys) / n
    s11 = s12 = s22 = s1y = s2y = 0.0
    for (x1, x2), y in zip(xs, ys):
        d1, d2, dy = x1 - mx1, x2 - mx2, y - my
        s11 += d1 * d1
        s12 += d1 * d2
        s22 += d2 * d2
        s1y += d1 * dy
        s2y += d2 * dy
    a = [[1.0, 0.0, 0.0],
         [0.0, s11 + lam, s12],
         [0.0, s12, s22 + lam]]
    b = [my, s1y, s2y]
    sol = solve_3x3(a, b)
    if sol is None:
        return None
    b0 = sol[0] - sol[1] * mx1 - sol[2] * mx2
    return (b0, sol[1], sol[2])


def forecast_players(history, target_hours):
    """Forecast players_tracked at the target hour.

    Fits ln(players) ~ a + b*ln(1+hours) over distinct snapshots and
    extrapolates, capped at 4x the last count (growth saturates).
    Returns the last count when there is nothing to fit.
    """
    seen = {}
    for h in history:
        p = h.get("players_tracked")
        if p and p > 0:
            seen[h["at"]] = p
    pts = sorted((parse_time(t), p) for t, p in seen.items())
    if not pts:
        return None
    last_p = pts[-1][1]
    if len(pts) < 2:
        return float(last_p)
    xs, ys = [], []
    for t, p in pts:
        xs.append(math.log1p(hours_since_release(t)))
        ys.append(math.log(p))
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return float(last_p)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    slope = min(1.5, max(0.0, slope))  # player count never shrinks here
    pred = my + slope * (math.log1p(target_hours) - mx)
    pred_p = math.exp(pred)
    return min(max(pred_p, float(last_p)), 4.0 * float(last_p))


def trophy_forecast(trophy, history, target_dt, horizon_minutes):
    """Return the five quantiles (percent) for one trophy."""
    idx = trophy["id"] - 1
    last_rate = None
    for h in reversed(history):
        rates = h.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            last_rate = float(rates[idx])
            break
    if last_rate is None:
        last_rate = 1.0
    if last_rate >= 99.99:
        # Everybody tracked already has it; it can only stay at 100%.
        return [100.0] * 5
    if last_rate <= 0.0:
        last_rate = 0.01

    last_logit = logit(last_rate)
    target_hours = hours_since_release(target_dt)
    horizon_days = horizon_minutes / 1440.0

    # Collect (log-players, log-hours, logit-rate) points, one per read.
    pts = []
    for h in history:
        rates = h.get("rates") or []
        if idx >= len(rates) or rates[idx] is None:
            continue
        p = h.get("players_tracked")
        if not p or p <= 0:
            continue
        t = parse_time(h["at"])
        pts.append((math.log(p), math.log1p(hours_since_release(t)),
                    logit(float(rates[idx]))))
    # Keep only distinct snapshots (the page sits unchanged between
    # Exophase refreshes); identical points carry no extra information.
    uniq = []
    seen = set()
    for pt in pts:
        key = (round(pt[0], 6), round(pt[1], 6), round(pt[2], 6))
        if key not in seen:
            seen.add(key)
            uniq.append(pt)

    median_logit = last_logit
    if len(uniq) >= 4:
        xs = [(p, h) for p, h, _ in uniq]
        ys = [y for _, _, y in uniq]
        fit = ridge_fit(xs, ys, lam=0.5)
        if fit is not None:
            p_target = forecast_players(history, target_hours)
            b0, b1, b2 = fit
            pred = b0 + b1 * math.log(p_target) + b2 * math.log1p(target_hours)
            # Shrink toward the last observed rate: the more distinct
            # snapshots, the more we trust the fit.
            w = len(uniq) / (len(uniq) + 4.0)
            median_logit = (1.0 - w) * last_logit + w * pred
            # Never let the model run away from the last observation.
            median_logit = min(max(median_logit, last_logit - 1.5),
                               last_logit + 1.5)

    # Spread: wide, wider for rare trophies, growing slowly with horizon.
    # The practice refresh moved every rate at once, outside the house
    # baseline's narrow band, so we pay for width rather than miss it.
    rarity = max(0.0, -last_logit)  # ~0 for common, ~4.6 for a 1% trophy
    sigma = 1.15 + 0.06 * rarity
    sigma *= math.sqrt(1.0 + horizon_days / 7.0)
    sigma = min(sigma, 1.7)

    qs = []
    for z in Z:
        q = inv_logit(median_logit + z * sigma)
        q = min(99.99, max(0.01, q))
        qs.append(round(q, 4))
    # Enforce monotonicity after rounding.
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    history = case.get("history") or []
    target_dt = parse_time(case["target"])
    horizon = float(case.get("horizon_minutes") or 0)
    rows = []
    for trophy in case["trophies"]:
        rows.append({"id": int(trophy["id"]),
                     "q": trophy_forecast(trophy, history, target_dt,
                                          horizon)})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
