# Local eval pack summary

_Generated: 2026-05-16T13:13:44.078241+00:00_

- **Resolved markets in pack: 42194**
- Snapshot dirs scanned: 18
- Outcomes on disk: 42194
- Class balance — YES: 12562, NO: 29632  (YES rate 0.298)

## Category breakdown

| Category | Count |
|---|---|
| Crypto | 21577 |
| Sports | 14335 |
| Other | 5562 |
| Weather | 410 |
| Economics | 310 |

## Baselines — the numbers your agent must beat

Brier scoring rewards calibration over confidence. Random = 0.25, perfect = 0.
ECE measures whether predicted probabilities match actual frequencies. Lower is better.

| Category | N | always_half Brier | **market Brier** | market ECE |
|---|---|---|---|---|
| **all** | 42194 | 0.2500 | **0.0645** | 0.0288 |
| Crypto | 21577 | 0.2500 | **0.0169** | 0.0245 |
| Sports | 14335 | 0.2500 | **0.1385** | 0.0461 |
| Other | 5562 | 0.2500 | **0.0516** | 0.0413 |
| Weather | 410 | 0.2500 | **0.1565** | 0.1940 |
| Economics | 310 | 0.2500 | **0.0627** | 0.0594 |

**Reading this table:** if your agent can't beat `market Brier` on a category,
you'd be better off just returning the market price for those events. The aggregate
number is dominated by Crypto (which is near-deterministic). **Sports is the meaningful
regression suite** — that's where agent skill differentiates.

## Snapshots per market (trajectory length)

| #snapshots | markets |
|---|---|
| 1 | 29530 |
| 2 | 4582 |
| 3 | 2090 |
| 4 | 1315 |
| 5 | 2162 |
| 6 | 928 |
| 7 | 75 |
| 8 | 288 |
| 9 | 40 |
| 10 | 64 |
| 11 | 27 |
| 12 | 28 |
| 14 | 950 |
| 16 | 87 |
| 17 | 28 |

## Files

- `data/eval_pack.jsonl` — one row per market with full price trajectory and outcome
- `data/eval_pack_latest.csv` — flat CSV, latest snapshot per market
- `data/outcomes.jsonl` — raw outcomes log

Load via `prep.data.load_local_snapshots()` or just read `eval_pack.jsonl` directly.