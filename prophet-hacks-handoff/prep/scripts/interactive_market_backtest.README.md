# Interactive Market Backtest

`interactive_market_backtest.py` is a point-in-time practice backtest for
forecasting against Kalshi market prices. It randomly samples a historical
market state, shows only information available at that timestamp, asks for your
prediction, then reveals the outcome and compares your Brier score against the
market-implied probability at that exact point.

## What It Supports

- Binary markets from `Kalshitopvolmarkets`
- Grouped sibling-outcome markets from `NonBinaryMarkets`
- Mixed sessions that randomly alternate between binary and nonbinary prompts
- Interactive prediction entry
- Non-interactive smoke modes using market or uniform predictions

## Data Expected

Binary mode expects:

```text
Kalshitopvolmarkets/
├── weekly_top_markets.csv
└── ohlcv/period_1m/week=YYYY-MM-DD/*.csv.gz
```

Nonbinary mode expects:

```text
NonBinaryMarkets/
├── weekly_top_groups.csv
├── markets/*_component_markets.jsonl
└── ohlcv/period_1m/week=YYYY-MM-DD/*.csv.gz
```

If your nonbinary pull is still using 60-minute candles, pass
`--nonbinary-period 60`.

## Basic Usage

Run from the repo root:

```bash
python3 prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py --mode mixed --rounds 10
```

Binary only:

```bash
python3 prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py --mode binary --rounds 5
```

Nonbinary only:

```bash
python3 prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py --mode nonbinary --rounds 5
```

Use a fixed seed for reproducible prompts:

```bash
python3 prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py --mode mixed --rounds 10 --seed 42
```

## How Predictions Are Entered

For binary prompts, enter `P(YES)` as either a decimal or a percent:

```text
0.62
62
62%
```

For nonbinary prompts, enter probabilities in the displayed component order:

```text
0.15, 0.55, 0.30
```

Percent entries are also accepted:

```text
15, 55, 30
```

The script normalizes nonbinary distributions before scoring, so the values do
not need to sum exactly to 1. Inside a nonbinary prompt, you can also type:

```text
market
uniform
```

## Scoring

Binary scoring uses standard binary Brier:

```text
(p_yes - outcome_yes)^2
```

Nonbinary scoring uses multiclass Brier over the displayed sibling set:

```text
sum((p_i - y_i)^2)
```

The market baseline is computed from the quote shown at the sampled timestamp.
For binary markets, market probability is the midpoint of YES bid and YES ask.
For nonbinary groups, each component midpoint is normalized across the displayed
components.

After every round, the script prints:

- Actual resolved outcome
- Your Brier score
- Market Brier score
- Delta vs market
- Whether you beat, tied, or lost to market

At the end, it prints average Brier and the number of rounds where you beat the
market.

## Useful Smoke Tests

Use the market as the automatic prediction. This should tie market every round:

```bash
python3 prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py --mode binary --rounds 1 --seed 1 --auto-market
python3 prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py --mode nonbinary --rounds 1 --seed 1 --auto-market
```

Use uniform predictions to sanity-check scoring:

```bash
python3 prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py --mode mixed --rounds 5 --seed 4 --auto-uniform
```

Compile check:

```bash
python3 -m py_compile prophet-hacks-handoff/prep/scripts/interactive_market_backtest.py
```

## Important Flags

```text
--mode binary|nonbinary|mixed
    Which type of prompt to sample.

--rounds N
    Number of prompts in the session.

--seed N
    Makes random sampling reproducible.

--history N
    Number of recent binary candles to show before the sampled timestamp.

--min-time-to-close-minutes N
    Excludes samples too close to market close.

--binary-period N
    Candle period for binary data. Default is 1.

--nonbinary-period N
    Candle period for nonbinary data. Default is 1. Use 60 for old hourly pulls.

--nonbinary-min-components N
    Minimum sibling components required for nonbinary prompts. Default is 3.

--allow-truncated-nonbinary
    Allows groups where selected components are fewer than total known components.
    By default, truncated groups are excluded.

--auto-market
    Non-interactive mode that predicts exactly the market baseline.

--auto-uniform
    Non-interactive mode that predicts 0.5 for binary or uniform for nonbinary.
```

## Caveats

- The script is for human practice and quick evaluation, not full model
  training.
- It avoids showing resolved result fields before prediction, but the market
  title/rules may still contain ordinary historical context.
- For nonbinary groups, the script only samples clean groups with exactly one
  YES among displayed components. By default it excludes truncated groups.
- Nonbinary market probabilities are normalized across displayed components.
  This is appropriate for complete mutually exclusive sibling sets, but less
  appropriate for ordinal ladders or groups where multiple contracts can resolve
  YES.

