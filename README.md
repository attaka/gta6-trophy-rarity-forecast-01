# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/attaka/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/attaka/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts the share of GTA VI players who will have earned each of its 55 PS5 trophies by 2026-11-26, as five quantiles per trophy, by extrapolating earners and tracked players separately from the recorded Exophase reads.

## How it works

A trophy's earn rate is a ratio, `rate = 100 × earners / tracked`, and the two
parts move for different reasons: the tracked-player count surges after launch
as the tracker's coverage catches up (that surge alone cut most launch-night
rates by ~50× in two days), while earners only ever grow, fast for early-game
trophies and slowly for endgame and grind trophies. So `main.py` extrapolates
the two parts separately — each by a curve family (saturating exponential,
linear in log time, damped trend, or flat) chosen per series by an internal
holdout over the case's own reads — shrinks each trophy's predicted earners
growth toward the cross-trophy median, and recombines them into a median rate.
The spread around it, on the logit scale, grows with the horizon, with the
size of the predicted move, and with the trophy's own observed refresh-step
volatility. Two pointwise truths are enforced on the result: the platinum's
earners are a subset of every other trophy's, and each story chapter's earners
are a subset of the previous chapter's.

The program is Python 3.13, standard library only, offline and deterministic.
It reads every case file (`*.in`, schema `makefun/gta6-trophies-case/v1`) in
the input directory and writes one `<id>.out` per case to the output
directory, in the contest's format.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=tezwxqky&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
