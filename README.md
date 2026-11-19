# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Falken21/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Falken21/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty **"Forecast how many GTA VI players earn each trophy in week one"**
(`gta6-trophy-rarity-forecast-01`, $250): a Python 3.13, stdlib-only program
(`main.py`) that forecasts, for every one of the 55 GTA VI PlayStation 5
trophies, five quantiles (5/25/50/75/95%) of its earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z, from the house's recorded reads of that page.

## How it works

Per trophy, on the logit scale. Two forces move an observed earn rate:

- **Dilution** — new players flood in, the tracked-player count grows, and
  every rate is pushed down.
- **Progression** — players already tracked keep earning story/activity
  trophies, pushing those rates up toward a ceiling below 100%.

A ridge regression of `logit(rate)` on `ln(players_tracked)` and
`ln(1 + hours since release)` is fit over the case's **whole** history (not
just the last 24 h), shrunk toward the last observed rate when the history
has few distinct snapshots. The tracked-player count at the target is
forecast by fitting `ln(players)` against `ln(1 + hours)` and extrapolating
with a hard cap on growth; the regression is then read at that player count
and at the target hour, with the median clamped to ±1.5 logit of the last
observed rate.

**Spread:** the house random-walk baseline (σ = 0.5·√days) scored ~1.51 on
the practice because a single Exophase refresh shifts every rate at once,
outside its narrow band. This entry therefore uses a much wider logit
spread (base ≈ 1.15–1.4, larger for rarer trophies, growing slowly with
horizon) with stretched outer quantiles, so a refresh jump stays inside
the 90% band most of the time.

The program is deterministic (no wall clock, no randomness), reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`), writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`), and runs in
~0.1 s per case — far inside the 8 s limit.

## Local check

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Run twice and the outputs are byte-identical; each `<id>.out` holds exactly
55 rows, one per trophy, with non-decreasing quantiles in [0, 100].
