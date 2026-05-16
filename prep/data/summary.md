# Local eval pack summary

_Generated: 2026-05-16T17:58:54.417625+00:00_

- **Resolved markets in pack: 79263**
- Snapshot dirs scanned: 22
- Outcomes on disk: 79380
- Class balance — YES: 22956, NO: 56307  (YES rate 0.290)

## Category breakdown

| Category | Count |
|---|---|
| Crypto | 46857 |
| Sports | 19889 |
| Financials | 4887 |
| Climate and Weather | 2013 |
| Entertainment | 1353 |
| Mentions | 1122 |
| Economics | 929 |
| Commodities | 802 |
| Elections | 745 |
| Politics | 510 |
| Companies | 105 |
| Science and Technology | 45 |
| Social | 5 |
| World | 1 |

## Baselines — the numbers your agent must beat

Brier scoring rewards calibration over confidence. Random = 0.25, perfect = 0.
ECE measures whether predicted probabilities match actual frequencies. Lower is better.

| Category | N | always_half Brier | **market Brier** | market ECE |
|---|---|---|---|---|
| **all** | 79263 | 0.2500 | **0.1207** | 0.0932 |
| Crypto | 46857 | 0.2500 | **0.1281** | 0.1354 |
| Sports | 19889 | 0.2500 | **0.1100** | 0.0480 |
| Financials | 4887 | 0.2500 | **0.1301** | 0.0162 |
| Climate and Weather | 2013 | 0.2500 | **0.0975** | 0.0419 |
| Entertainment | 1353 | 0.2500 | **0.1123** | 0.0997 |
| Mentions | 1122 | 0.2500 | **0.0420** | 0.0285 |
| Economics | 929 | 0.2500 | **0.1897** | 0.1579 |
| Commodities | 802 | 0.2500 | **0.0465** | 0.0463 |
| Elections | 745 | 0.2500 | **0.0838** | 0.0658 |
| Politics | 510 | 0.2500 | **0.1089** | 0.0759 |
| Companies | 105 | 0.2500 | **0.0981** | 0.2033 |
| Science and Technology | 45 | 0.2500 | **0.0613** | 0.0277 |
| Social | 5 | 0.2500 | **0.1001** | 0.2060 |
| World | 1 | 0.2500 | **0.0020** | 0.0450 |

**Reading this table (forecasting):** if your agent can't beat `market Brier`
on a category, you'd be better off just returning the market price for those events.
The aggregate number is dominated by Crypto (which is near-deterministic).
**Sports is the meaningful regression suite** — that's where agent skill differentiates.

## Trading P&L of baseline strategies

Each starts with $10,000 and buys-and-holds whichever side the strategy chooses,
realizing P&L at market resolution. Bid-ask spread costs are real.

| Strategy | Trades | P&L | Return | Win rate |
|---|---|---|---|---|
| never_trade | 0 | $+0.00 | +0.00% | n/a |
| market_anchor + default | 1,151 | $+221.00 | +2.21% | 19.3% |
| noisy_market + default | 6,780 | $+1,369.07 | +13.69% | 43.5% |

**IMPORTANT — read before trusting these numbers:**

- Our dataset is **71% NO outcomes**. Any strategy that bets NO often will look
  profitable on aggregate, regardless of real skill. The `noisy_market +12.20%`
  number above is largely this artifact.
- The sanity check: a strategy betting *against* the market (which should lose)
  instead "makes" +180% on this dataset — proving the aggregate is broken.
- **Use Sports-only numbers as the realistic benchmark** (run
  `python scripts/strategy_comparison.py --category Sports`). On Sports, every
  baseline LOSES money. The bid-ask spread is the real adversary.

Headline number to beat for Sports: **a real agent should lose less than `−$243`
with the `noisy + tight_band` baseline on multi-snapshot data.**

## Snapshots per market (trajectory length)

| #snapshots | markets |
|---|---|
| 1 | 66098 |
| 2 | 4589 |
| 3 | 2132 |
| 4 | 1333 |
| 5 | 2162 |
| 6 | 928 |
| 7 | 135 |
| 8 | 289 |
| 9 | 66 |
| 10 | 64 |
| 11 | 29 |
| 12 | 30 |
| 14 | 951 |
| 15 | 2 |
| 16 | 88 |
| 17 | 28 |
| 18 | 319 |
| 19 | 20 |

## Files

- `data/eval_pack.jsonl` — one row per market with full price trajectory and outcome
- `data/eval_pack_latest.csv` — flat CSV, latest snapshot per market
- `data/outcomes.jsonl` — raw outcomes log

Load via `prep.data.load_local_snapshots()` or just read `eval_pack.jsonl` directly.