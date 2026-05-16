# Data sources — what to use, for what, and why

The repo accumulated several overlapping data files as people pulled from
HuggingFace, ran the Kalshi poller, or fetched the official benchmark.
This doc fixes which one to use for which task, so the team doesn't
silently disagree on what "the data" means.

---

## TL;DR — pick by task

| Task | Use this | Why |
|---|---|---|
| **Trading-track strategy backtest** | `data/external/subset_1200.csv` | Official 1,200-submission hand-curated hackathon benchmark. THE bar to beat (−$51 / −$2 per `STRATEGY_FINDINGS.md`). |
| **Forecasting-track ensemble backtest** | `data/eval_pack_live_clean.jsonl` | 13,165 markets, validated bid/ask, ≥2 snapshots per market, properly categorized. Our cleanest self-polled set. |
| **Cross-validate forecaster output** | `data/external/kalshi_markets.parquet` | 10,016 Kalshi markets with final outcomes (mid-2025), independent of our polling. |
| **Eyeball recent live market structure** | `data/external/athetus_live/kalshi_snapshot_2026-05-15.parquet` | Daily Kalshi snapshots Apr 23 → May 16, 2026. Newest = most recent. |
| **Category lookup for any ticker** | `data/kalshi_series_categories.json` | Authoritative `series_ticker → category` for 10,168 Kalshi series. Pulled from `/series` endpoint. **Use this, not prefix matching.** |
| **Live forecasting submission** | (nothing — the eval server pushes events to us) | Eval server POSTs `Event` JSON to our `/predict`. We don't poll. |
| **Live trading submission** | (nothing — the runner pulls market data per tick) | `prophet trade eval run` calls `claim_tick → load_candidates`. We don't poll. |

---

## Forbidden / known-bad

These look useful but are misleading. Listed so nobody re-discovers them.

| Don't use | Why |
|---|---|
| `data/eval_pack.jsonl` (full 43k version) | Contains backfill rows where prices were captured **after** market settlement. Treats post-settlement state as a pre-resolve bid/ask. Inflates inverse-market returns artificially. The `_live_clean` version filters these out. |
| `thomaswmitch/kalshi-prediction-markets-betting` (HF dataset) | Single-trade prices, not live bid/ask. `inverse_market` strategy falsely scored +435% on this. Excluded from the repo. |
| Crypto markets in any dataset | Markets are calibrated to ~0.017 Brier (essentially deterministic). No LLM edge. `tight_band_skip_crypto` strategy drops them. |

---

## Detailed file reference

### Authoritative (hackathon-published or ground-truth)

#### `data/external/subset_1200.csv` (4.4 MB, 1,201 rows incl. header)
**THE trading-track benchmark.** From `prophetarena/Prophet-Arena-Subset-1200` on HuggingFace — hand-curated by the same team running the hackathon.
- 1,200 submissions across **897 unique events**, June–November 2025
- Columns: `submission_id`, `event_ticker`, `title`, `snapshot_time`, `close_time`, `market_data` (JSON with bid/ask/liquidity per outcome), `market_outcome` (JSON with resolved 0/1 per outcome), `category`, `markets`, `augmented_title`, `rules`, `sources`
- Category distribution: Sports 894, Entertainment 93, Politics 91, Other 37, Companies 27, Mentions 26, Economics 19, Climate/Weather 13. **No Crypto** in this subset.
- Used by: `scripts/backtest_strategies.py` (default source)

#### `data/kalshi_series_categories.json` (330 KB, 10,168 entries)
Authoritative `series_ticker → category` lookup. Generated from Kalshi's public `/series` endpoint.
- Top categories: Entertainment 2389, Sports 1967, Politics 1916, Elections 1303, Economics 531, Companies 373, Mentions 352, Climate/Weather 266, Sci/Tech 245, **Crypto 231**
- Used by: `src/prep/trading/strategies.py:get_market_category()`, `scripts/consolidate.py`
- **Replaces** the fragile prefix-matching that was in `data.py` before.

### Self-polled (live Kalshi data via our `snapshot.py` + `resolve.py`)

#### `data/eval_pack_live_clean.jsonl` (13.3 MB, 13,165 markets) — **use this for forecasting backtest**
Cleanest version of our self-polled data:
- Markets captured during May 11–16 polling window
- ≥2 snapshots per market (so we have real bid/ask trajectories)
- Prices validated in `[0, 1]`
- Categorized via `kalshi_series_categories.json`
- Distribution: Sports 6,192 (47%), Crypto 3,079 (23%), Climate/Weather 1,149, Commodities 775, Economics 715, Entertainment 612, Financials 570, Politics 31, Elections 4
- Regenerate via: `python scripts/build_clean_eval_pack.py` (script lives in another teammate's local checkout — push if needed)

#### `data/eval_pack.jsonl` (31 MB, ~44k markets) — **don't backtest against**
Raw consolidated pack. Includes single-snapshot backfill rows where the price was captured **after** settlement. Distorts strategy P&L. The `_live_clean` version drops these.

#### `data/eval_pack_latest.csv` (7 MB)
Flat CSV mirror of `eval_pack.jsonl`. Same caveats — use `_live_clean` for backtest.

#### `data/outcomes.jsonl` (6.7 MB)
Resolution log keyed by `market_ticker`. Used by `consolidate.py` to attach outcomes.

#### `data/resolve_state.json` (223 B)
Checkpoint for `resolve.py`. Latest values as of this commit:
```
last_max_close_ts: 1778952017
last_run_completed_at: 2026-05-16T17:26:16+00:00
last_newly_resolved: 976
last_total_scanned: 129619
```

#### `data/snapshots/` (~375 MB, gitignored)
Raw snapshot directories, one per polling time. Re-generatable via `snapshot.py`. Not committed (size).

### External reference (HF datasets)

#### `data/external/kalshi_markets.parquet` (1.4 MB, 10,016 markets)
From `thomaswmitch/kalshi-prediction-markets-markets`. Full metadata + final outcomes, mid-2025. Use for cross-validation against `_live_clean`, not as primary backtest source (the *trades* dataset from the same author is misleading; the *markets* metadata is fine).

#### `data/external/athetus_live/` (3.9 MB, 23 daily snapshots)
From `athetus/predmarkets-kalshi-live`. One Kalshi snapshot per day Apr 23 → May 16, 2026. Useful for sanity-checking what current market structure looks like (categories, spreads, liquidity) without spinning up the poller.

---

## Polling — how to keep data fresh

Three scripts compose into the pipeline:

```
snapshot.py        # poll open Kalshi markets → data/snapshots/<ts>/
   ↓
resolve.py         # attach outcomes to settled markets → data/outcomes.jsonl
   ↓
consolidate.py     # roll snapshots + outcomes → data/eval_pack*.jsonl + summary
```

### Manual run
```bash
cd prep
python scripts/snapshot.py           # ~16 MB per run, ~5 min
python scripts/resolve.py            # writes new resolutions to outcomes.jsonl
python scripts/consolidate.py        # refreshes eval_pack.jsonl + summary.md
```

### Wrapped: resolve + consolidate
```bash
./scripts/cron_resolve.sh            # runs the latter two together; logs to data/resolve.log
```

### Suggested cron (during the hackathon)

Add this to your crontab if you want fresh data through Sunday — **only on a machine that stays awake**.

```cron
# Every 6 hours: full snapshot of open markets (adds ~16 MB to data/snapshots/)
0 */6 * * * cd /path/to/prep && /opt/homebrew/bin/python3 scripts/snapshot.py >> data/snapshot.log 2>&1

# Every hour: resolve newly-settled markets + refresh eval pack (cheap)
15 * * * * /path/to/prep/scripts/cron_resolve.sh >> data/resolve.log 2>&1
```

⚠️  As of this commit, **`crontab -l` shows no scheduled entries on Victor's machine**. The recent polls (latest snapshot at 16:36 UTC, latest resolve at 17:26 UTC) appear to be manual or driven by another teammate. **Decide as a team who owns continuous polling for the next ~24 h** — otherwise the local data goes stale and the live-Kalshi reference data drifts from market state.

### What polling does NOT do

It does **not** affect either submission track:
- Forecasting track: judges send us events via HTTP, we respond — no Kalshi data needed at inference time
- Trading track: the trading server provides per-tick market snapshots — we don't pull from Kalshi directly

Polling is purely for backtest validation and offline analysis. If polling drifts during the hackathon, our backtest numbers go stale, but live submissions are unaffected.

---

## Quick verifications

```bash
# Confirm subset_1200 loads + correct row count
python -c "import csv; print(sum(1 for _ in csv.DictReader(open('data/external/subset_1200.csv'))))"
# → 1200

# Confirm clean eval pack
wc -l data/eval_pack_live_clean.jsonl
# → 13165

# Run the canonical trading backtest (~30 sec, no LLM calls)
python scripts/backtest_strategies.py --strategy tight_band_skip_crypto --n-seeds 10
# → expect aggregate > -$51, Sports varies wildly (-$200 to +$60 single-seed)

# Confirm category lookup works
python -c "import json; m = json.load(open('data/kalshi_series_categories.json')); print('KXBTC ->', m.get('KXBTC'), '| KXNBAGAME ->', m.get('KXNBAGAME'))"
# → KXBTC -> Crypto | KXNBAGAME -> Sports
```
