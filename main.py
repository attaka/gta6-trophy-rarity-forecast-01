"""gta6-trophy-rarity-forecast-01: forecast GTA VI trophy earn rates one week out.

Reads one case (makefun/gta6-trophies-case/v1): the house's reads of Exophase's
GTA VI PS5 trophy page before a checkpoint, and writes five quantiles
(5/25/50/75/95%) of every trophy's earn rate at the case's target, scored by a
pinball loss on the logit scale.

Model, per trophy -- decompose the rate, extrapolate the parts, recombine:

    rate(t) = 100 * earners(t) / tracked(t)

A rate on a tracker page is a ratio, and its two parts move for different
reasons. `tracked` (players_tracked) surges after launch as the tracker's
coverage catches up with the player base, then decelerates; that surge alone
crushed most launch-night rates by ~50x in two days (the pre-release cohort of
638 tracked players had story trophies near 1%, the mass audience put them at
0.02%). `earners` never decreases and grows as players progress, fast for
early-game trophies, slowly for endgame and grind trophies. The house
baselines miss this split: persistence ignores both parts, and a trend fit on
the rate itself extrapolates the launch dilution as if it were the trophy's
own trajectory. So:

  * tracked: log tracked is extrapolated to the target by one of four curve
    families -- saturating exponential (a launch surge levelling off), linear
    in log time, damped linear trend, flat -- picked per case by an internal
    holdout (fit on the first 70% of the observed span, scored on the rest).
    Never shrinking, growth capped at 25x.
  * earners: the same family choice on log(earners + 1), the predicted growth
    then shrunk toward the cross-trophy median growth (weight
    earners/(earners+50), because a rate printed to 2 decimals makes small
    earner counts mostly rounding noise). Never shrinking, capped by tracked.
  * median rate = 100 * earners(target) / tracked(target).

Spread, on the logit scale: a base jitter (Exophase refresh timing and the
page's 2-decimal rounding), a random-walk term growing with the horizon, a
term proportional to the predicted logit move (a big predicted move is an
uncertain one), the tracked-growth uncertainty shared by all trophies, and
half the trophy's own observed refresh-step volatility (capped). A history
with no refresh in it at all (the page never changed while the house watched)
gets its spread widened: nothing observed is not nothing happening. Quantiles
are inv-logit(median logit + z * sigma), so they stay inside 0..100 and keep
their order by construction.

Two pointwise truths are enforced afterwards, quantile by quantile (set
inclusion holds in every state of the world, so it holds for every quantile):
the platinum's earners are a subset of every other trophy's earners, and
"Complete Chapter k+1" earners are a subset of "Complete Chapter k" earners,
with "Complete the final mission" below the last chapter.

Runs offline, standard library only, deterministic: the same case always
writes the same bytes. Any internal failure falls back to the house
persistence forecast for that case, so a valid file is always written.

Local run:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first (as the starter does).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

TAU_TRACKED = 48.0      # hours: damping constant of the tracked fallback family
TAU_EARNERS = 72.0      # hours: damping constant of the earners fallback family
BASE_JITTER = 0.15      # logit: refresh timing + 2-decimal rounding
RW_PER_SQRT_DAY = 0.45  # logit: random-walk spread growth
MOVE_UNCERTAINTY = 0.35 # share of the predicted logit move added to the spread
TRACKED_UNCERTAINTY = 0.25  # share of the predicted log tracked growth
FLAT_WIDEN = 1.6        # spread multiplier when the history shows no refresh
FLAT_ADD = 0.25         # ... plus this much logit


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as the starter (and JavaScript) does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def z_for(tau: float) -> float:
    if tau in Z_TABLE:
        return Z_TABLE[tau]
    # Acklam's rational approximation of the standard normal quantile.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if tau < plow:
        q = math.sqrt(-2 * math.log(tau))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if tau > phigh:
        q = math.sqrt(-2 * math.log(1 - tau))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = tau - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(text):
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def damped(slope_per_hour: float, hours: float, tau: float) -> float:
    """Integral of a slope that decays with time constant tau, over `hours`."""
    if hours <= 0:
        return 0.0
    return slope_per_hour * tau * (1.0 - math.exp(-hours / tau))


def lsq_line(points):
    """Least-squares (intercept, slope) of y vs t; (mean_y, 0.0) when undefined."""
    n = len(points)
    if n == 0:
        return 0.0, 0.0
    mt = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    if n < 2:
        return my, 0.0
    sxx = sum((p[0] - mt) ** 2 for p in points)
    if sxx <= 0:
        return my, 0.0
    slope = sum((p[0] - mt) * (p[1] - my) for p in points) / sxx
    return my - slope * mt, slope


SAT_TAUS = (12.0, 24.0, 48.0, 96.0, 192.0, 384.0)


def _fit_saturating(points):
    """Best (by SSE) fit of y = A + B * exp(-t/tau) over the tau grid."""
    best = None
    for tau in SAT_TAUS:
        xs = [(math.exp(-t / tau), y) for t, y in points]
        inter, slope = lsq_line(xs)
        sse = sum((y - (inter + slope * x)) ** 2 for x, y in xs)
        # Prefer the larger tau on near-ties: it extrapolates more gently.
        if best is None or sse < best[0] * 0.98:
            best = (sse, inter, slope, tau)
    _, inter, slope, tau = best
    return lambda t: inter + slope * math.exp(-t / tau)


def _fit_damped(points, tau):
    inter, slope = lsq_line(points)
    t_end, y_end = points[-1]
    return lambda t: y_end + damped(slope, max(0.0, t - t_end), tau)


def _fit_logtime(points):
    xs = [(math.log1p(t), y) for t, y in points]
    inter, slope = lsq_line(xs)
    return lambda t: inter + slope * math.log1p(t)


def extrapolate(points, t_target, tau_damped):
    """Predict y at t_target from step points [(t_hours, y)], choosing the
    curve family by an internal holdout: fit each family on the first ~70% of
    the observed span and score it on the rest. With too little data, prefer
    the saturating family (launch-week growth decelerates)."""
    if len(points) < 2 or t_target <= points[-1][0]:
        return points[-1][1]
    families = [
        lambda pts: _fit_saturating(pts),
        lambda pts: _fit_logtime(pts),
        lambda pts: _fit_damped(pts, tau_damped),
        lambda pts: (lambda t: pts[-1][1]),
    ]
    t_first, t_last = points[0][0], points[-1][0]
    cut = t_first + 0.7 * (t_last - t_first)
    train = [p for p in points if p[0] <= cut]
    hold = [p for p in points if p[0] > cut]
    chosen = families[0]
    if len(train) >= 3 and len(hold) >= 2:
        best_err = None
        for fam in families:
            try:
                f = fam(train)
                err = sum((f(t) - y) ** 2 for t, y in hold) / len(hold)
            except Exception:
                continue
            if best_err is None or err < best_err:
                best_err, chosen = err, fam
    return chosen(points)(t_target)


def forecast_tracked(reads, t_last, horizon_h):
    """reads: [(t_hours, players_tracked)]. Returns (n_target, d_log_n)."""
    n_last = reads[-1][1]
    steps = []
    for t, n in reads:
        if n is None or n <= 0:
            continue
        if not steps or n != steps[-1][1]:
            steps.append((t, float(n)))
    if not steps:
        return float(n_last or 0), 0.0
    if steps[-1][0] < t_last:
        steps.append((t_last, float(n_last)))
    pts = [(t, math.log(n)) for t, n in steps]
    pred = extrapolate(pts, t_last + horizon_h, TAU_TRACKED)
    d_log_n = min(max(pred - pts[-1][1], 0.0), math.log(25.0))
    return n_last * math.exp(d_log_n), d_log_n


def trophy_steps(reads, idx):
    """Step points (t, rate, tracked) for one trophy: reads where its rate or
    the tracked count changed, plus the last read that names it."""
    steps = []
    for t, rates, n in reads:
        r = rates[idx] if idx < len(rates) else None
        if r is None:
            continue
        if not steps or r != steps[-1][1] or n != steps[-1][2]:
            steps.append((t, float(r), n))
    return steps


def earners_log_growth(steps, t_last, horizon_h):
    """Predicted log growth of (earners + 1) from t_last to the target."""
    pts = [(t, math.log(r * n / 100.0 + 1.0)) for t, r, n in steps if n]
    if len(pts) < 2 or horizon_h <= 0:
        return 0.0
    pred = extrapolate(pts, t_last + horizon_h, TAU_EARNERS)
    return pred - pts[-1][1]


def step_volatility(steps):
    """RMS |delta logit| between consecutive step points of the rate."""
    if len(steps) < 2:
        return 0.0
    diffs = [abs(logit(steps[i + 1][1]) - logit(steps[i][1])) for i in range(len(steps) - 1)]
    return math.sqrt(sum(x * x for x in diffs) / len(diffs))


def forecast_case(case):
    trophies = case["trophies"]
    levels = case.get("levels") or LEVELS
    history = case.get("history") or []
    reads = []
    for h in history:
        ts = parse_time(h.get("slot") or h.get("at"))
        if ts is None:
            continue
        reads.append((ts, h.get("rates") or [], h.get("players_tracked")))
    if not reads:
        raise ValueError("no usable history")
    reads.sort(key=lambda x: x[0])
    t0 = reads[0][0]
    reads = [(ts - t0, rates, n) for ts, rates, n in reads]
    t_last = reads[-1][0] / 3600.0
    reads = [(ts / 3600.0, rates, n) for ts, rates, n in reads]

    target_ts = parse_time(case.get("target"))
    if target_ts is not None:
        horizon_h = max(0.0, (target_ts - t0) / 3600.0 - t_last)
    else:
        horizon_h = max(0.0, float(case.get("horizon_minutes") or 0) / 60.0)
    horizon_days = horizon_h / 24.0

    n_target, d_log_n = forecast_tracked([(t, n) for t, _, n in reads], t_last, horizon_h)
    n_last = reads[-1][2] or 0

    # First pass: each trophy's predicted earners log-growth, and their median
    # (small-count trophies shrink toward it: their printed rates are rounded
    # to 2 decimals, so their own growth reads are mostly rounding noise).
    per = []
    for i in range(len(trophies)):
        steps = trophy_steps(reads, i)
        per.append((steps, earners_log_growth(steps, t_last, horizon_h) if steps else 0.0))
    all_growth = sorted(g for _, g in per)
    median_growth = all_growth[len(all_growth) // 2] if all_growth else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        steps, raw_growth = per[i]
        if not steps:
            # Never seen in any read: a near-floor rate with a wide spread.
            rows.append([round4(min(100.0, max(0.01, inv_logit(logit(0.02) + z_for(tau) * 1.2))))
                         for tau in levels])
            continue
        t_s, r_last, n_at = steps[-1]
        e_last = r_last * (n_at or n_last) / 100.0
        w = e_last / (e_last + 50.0)
        d_log_e = w * raw_growth + (1.0 - w) * median_growth
        d_log_e = min(max(d_log_e, 0.0), math.log(60.0))
        e_target = (e_last + 1.0) * math.exp(d_log_e) - 1.0
        if e_last <= 0.0:
            e_target = max(e_target, 1.0)
        if e_last >= (n_at or 0):
            e_target = n_target  # earned by every tracked player: stays that way
        e_target = min(e_target, n_target)
        r_med = 100.0 * e_target / n_target if n_target > 0 else r_last
        l_med = logit(r_med)
        l_last = logit(r_last)

        vol = min(0.6, 0.5 * step_volatility(steps))
        var = (BASE_JITTER ** 2
               + (RW_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
               + (MOVE_UNCERTAINTY * abs(l_med - l_last)) ** 2
               + (TRACKED_UNCERTAINTY * d_log_n) ** 2
               + vol ** 2)
        sigma = math.sqrt(var)
        if len(steps) == 1:
            sigma = sigma * FLAT_WIDEN + FLAT_ADD
        row = []
        for tau in levels:
            q = inv_logit(l_med + z_for(tau) * sigma)
            row.append(round4(min(100.0, max(0.01, q))))
        # Rounding can only break monotonicity at the 0.01 floor; repair cheaply.
        for j in range(1, len(row)):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)

    apply_inclusion_caps(case, rows)
    return {"trophies": [{"id": t["id"], "q": rows[i]} for i, t in enumerate(trophies)]}


def apply_inclusion_caps(case, rows):
    """Cap quantiles where one trophy's earners are provably a subset of another's."""
    trophies = case["trophies"]

    def cap_at(sub_idx, super_rows):
        for j in range(len(rows[sub_idx])):
            cap = min(r[j] for r in super_rows)
            if rows[sub_idx][j] > cap:
                rows[sub_idx][j] = cap

    for i, t in enumerate(trophies):
        if t.get("tier") == "platinum":
            cap_at(i, [rows[k] for k in range(len(rows)) if k != i])
    chapters = []
    final_mission = None
    for i, t in enumerate(trophies):
        desc = (t.get("description") or "").strip()
        if desc.startswith("Complete Chapter ") and desc.endswith("."):
            num = desc[len("Complete Chapter "):-1]
            if num.isdigit():
                chapters.append((int(num), i))
        elif desc == "Complete the final mission.":
            final_mission = i
    chapters.sort()
    chain = [i for _, i in chapters]
    if final_mission is not None:
        chain.append(final_mission)
    for a, b in zip(chain, chain[1:]):
        cap_at(b, [rows[a]])


def persistence_case(case):
    """The house baseline, as a fallback: last rate, random-walk spread."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(float(case.get("horizon_minutes") or 0) / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if i < len(last["rates"]) else None
        lg = logit(rate if rate is not None else 0)
        rows.append([round4(min(100.0, max(0.01, inv_logit(lg + z_for(tau) * spread))))
                     for tau in (case.get("levels") or LEVELS)])
    return {"trophies": [{"id": t["id"], "q": rows[i]} for i, t in enumerate(case["trophies"])]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = persistence_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
