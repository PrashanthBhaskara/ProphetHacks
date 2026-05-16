# prep

Shared backtest harness for our Prophet Hacks forecasting-track work. Direction-neutral — no architectural commitments. Any agent that exposes `predict(event) -> {"p_yes", "rationale"}` (the production contract) plugs in.

## What's here

```
prep/
├── reference/                      # vendored from HF: prophetarena/Prophet-Arena-Subset-100
│   ├── subset_data_100.csv         # 100 events, 1,061 binary markets, with ground truth + Kalshi snapshots
│   ├── standalone_predictor.py     # their reference predictor
│   ├── standalone_evaluator.py     # their reference evaluator (uses pm_rank)
│   └── README.md
├── src/prep/
│   ├── data.py                     # load_subset_100() → list[Sample]
│   ├── score.py                    # brier(), ece()
│   ├── eval.py                     # evaluate(predict_fn, samples) → metrics
│   └── baselines/
│       ├── always_half.py          # returns 0.5 (sanity check; Brier = 0.25)
│       ├── market.py               # returns Kalshi market price
│       └── claude_zero_shot.py     # zero-shot Sonnet 4 (matches example_agent.py)
└── scripts/run.py                  # CLI runner
```

## Setup

```bash
cd prep
pip install -r requirements.txt
```

## Quick start

```bash
# sanity check
python scripts/run.py always_half       # Brier 0.2500, ~instant

# market price as p_yes (the paper's reference baseline)
python scripts/run.py market            # Brier 0.0654 on this subset, ~instant

# zero-shot Claude (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run.py claude --workers 8

# filter by category
python scripts/run.py market --category Sports
python scripts/run.py market --category Politics
```

## Adding your own agent

Drop a module under `src/prep/baselines/` that exports `predict(event) -> dict`:

```python
def predict(event: dict) -> dict:
    # event has: event_ticker, market_ticker, title, subtitle,
    #            description, category, rules, close_time
    return {"p_yes": 0.42, "rationale": "..."}
```

Wire it into `BASELINES` in `scripts/run.py` and you can run it the same way.

The harness also supports `predict(event, market_info)` if your agent wants the Kalshi snapshot (yes_ask, no_ask, last_price, volume, etc.) — see `baselines/market.py`. **The production agent never sees market_info directly**, but it can fetch it via `KalshiForecastClient.get_market(ticker)` (no auth required — see `ai-prophet/packages/core/ai_prophet_core/forecast/kalshi_client.py`).

## Ensemble trading infrastructure

The repo now has a first-pass ensemble/trading stack under `src/prep/`:

```
src/prep/
├── schemas.py              # MarketPacket, ModelForecast, SupervisorForecast
├── packets.py              # build canonical model input packets
├── forecasters/
│   ├── gemini.py           # direct Gemini API adapter
│   ├── openrouter.py       # OpenRouter adapter for teammate lanes
│   └── mock.py             # deterministic no-key adapter for tests
├── ensemble.py             # supervisor weighted-logit aggregation
├── calibration.py          # shrink model edge toward market by category/horizon
└── trading/
    ├── risk.py             # final deterministic trade/no-trade decision
    ├── simulator.py        # conservative taker-fill simulator
    └── metrics.py          # PnL/ROI/trade-rate summaries
```

Run the no-key smoke backtest:

```bash
python scripts/backtest_ensemble.py --limit 200
```

The default config is `config/ensemble.example.json`. It enables two mock
Gemini lanes and leaves real Gemini/OpenRouter lanes disabled. To turn on real
models, copy `.env.example` to `.env` locally or export the variables, then set
the relevant model's `"enabled": true` in the config:

```bash
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY_CLAUDE_LANE=...
python scripts/backtest_ensemble.py --config config/ensemble.example.json --limit 20
```

Each model returns the same contract: forecast values, auditable reasoning
track, and diagnostics. The supervisor aggregates model probabilities and
reasoning, but the final trade decision stays deterministic in `trading/risk.py`.

## What to know before reading the numbers

**The 100-event subset is easier than the live evaluation.** Market prices are snapshotted at first-submission time, and 75% of the events are sports markets that often resolve cleanly. The paper reports market baseline Brier 0.187 over their full 1,367-event eval; we get 0.0654 here. The ECE matches the paper almost exactly (0.0707 vs 0.069), so the scorer is right — it's the data distribution that's softer.

Treat this subset as a **regression suite**: if a change makes one of the baselines worse here, that's a red flag. Don't treat it as a leaderboard.

**Brier scoring rewards calibration, not confidence.** A well-calibrated 0.65 beats an overconfident 0.95 when the event resolves NO. Per the paper, ECE is where models differ most (5× more variance than Brier), so this is the metric to optimize toward.

**The metrics:**

| Metric | What it measures | Direction |
|---|---|---|
| Brier | mean squared error vs binary outcome | lower better, perfect=0, random=0.25 |
| ECE | gap between predicted probability and observed frequency, binned | lower better, 0 = perfectly calibrated |

## Collecting fresh, contamination-free eval data (Kalshi polling)

The HF subset is one fixed snapshot. We can do better by polling Kalshi
ourselves before the hackathon — markets that resolve between today and
May 16 give us:

- **Zero training-data contamination** (events that haven't happened yet)
- **Multi-timestamp price tracks** (snapshot at T-5 days, T-2 days, T-1 hour → see how the market converges)
- **Way more sample size** than the 100-event subset

Pipeline:

```bash
# Snapshot all open Kalshi markets closing in next 7 days. Cheap. Run this
# 2–3 times a day for the days leading into the hackathon.
python scripts/snapshot.py --window-days 7

# After markets resolve, pull outcomes for every market we've snapshotted.
# Idempotent — re-running won't re-query already-resolved markets.
python scripts/resolve.py

# Then any agent can be scored on our fresh local data:
python scripts/run.py market --source eval_pack
```

Schedule it however — `cron`, `launchd`, or just running it before/after
breakfast and dinner. With a 0.3s pause between paginated pages it takes
~3 minutes per snapshot. A first snapshot is already in
`data/snapshots/<timestamp>/` for reference.

**What gets captured**: every non-combo binary Kalshi market with at
least one of `yes_ask_dollars` / `no_ask_dollars` populated. MVE
multi-leg combos are filtered out (use `--keep-mve` to override).

## Backfilling Kalshi training data

For larger training sets, use the historical collector. It writes raw JSONL
under `data/kalshi_training/` with source tags so downstream feature builders
can choose between official Kalshi, Oddpool, OddsPipe, and optional pmxt data.

```bash
# Official Kalshi only: market metadata, trades, L1 candles.
python scripts/collect_kalshi_training_data.py \
  --series-ticker KXFEDDECISION --max-markets 25

# Add current official L2 books.
python scripts/collect_kalshi_training_data.py \
  --series-ticker KXFEDDECISION --max-markets 25 --l2

# Add Oddpool historical L1/L2/trades when ODDPOOL_API_KEY is set.
export ODDPOOL_API_KEY=...
python scripts/collect_kalshi_training_data.py \
  --series-ticker KXFEDDECISION --max-markets 0 --oddpool --l2

# Optional normalized candle sources.
export ODDSPIPE_API_KEY=...
python scripts/collect_kalshi_training_data.py \
  --tickers KXFEDDECISION-26JUN-H0 --oddspipe --pmxt
```

Source coverage notes:

| Source | Use | Practical history |
|---|---|---|
| Kalshi official | Market metadata, trades, 1m/1h/1d L1 candles, current full book | Live tier plus `/historical/*`; target live window is about 3 months |
| Oddpool | Historical Kalshi top-of-book, trades, full-depth orderbook snapshots | Docs say Kalshi history from 2026-03-19 onward |
| OddsPipe | Normalized cross-venue candles/search | Free tier advertises 30 days; paid archive may be longer |
| pmxt | Unified SDK for current books/OHLCV/trades | Useful abstraction, but not a historical L2 archive |
| Dome API | Legacy historical orderbook/trades | Acquired/EOL; use pmxt or Oddpool unless an old key still works |

Official Kalshi does not provide historical full-depth L2 books. To build our
own archive from now forward, run:

```bash
python scripts/record_kalshi_orderbooks.py \
  --series-ticker KXFEDDECISION --seconds 30
```

## Open questions for the team

- Does an ensemble of zero-shot Sonnet + market-price (e.g. 0.5·LLM + 0.5·market) beat either alone on ECE?
- How much does the market-baseline edge decay if we filter to events where `volume < 1000` (illiquid markets, where reasoning should matter more)?
- Does category-conditional shrinkage toward the market price help? E.g. for Sports, trust market 80%; for Politics, trust LLM more.
- What does the per-bin ECE look like for `claude_zero_shot`? If it's biased high or low on extremes, that's a known fix (clamp or shrink).

## Paper for context

[`LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena`](https://arxiv.org/abs/2510.17638) — same authors as the hackathon hosts. See the project root for the PDF.
