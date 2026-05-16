# Prophet Arena Trading — Constraints & Quick Reference

Working notes for the trading track. Combines the official quick-start with what's actually in the `ai-prophet-core` SDK ([packages/core/](ai-prophet/packages/core)) so we don't get surprised by the wire model.

---

## Quick start

```bash
pip install ai-prophet-core   # Python ≥3.11
export PA_SERVER_URL=https://api.aiprophet.dev
export PA_SERVER_API_KEY=prophet_...
```

Run a first experiment (24 ticks ≈ 6 hours of market windows):

```bash
prophet trade eval run \
  -m openai:gpt-4o \
  --slug my_first_run \
  --max-ticks 24
```

Each tick the agent reviews ~256 live markets, searches the web for context, forecasts probabilities, and submits trades. `--slug` is the experiment name — rerun the same command to resume after a crash/stop.

Compare multiple models (4 participants = 2 models × 2 replicates, all see the same data, each gets its own $10k portfolio):

```bash
prophet trade eval run \
  -m openai:gpt-4o \
  -m anthropic:claude-sonnet-4 \
  --replicates 2 \
  --slug model_comparison \
  --max-ticks 96     # 96 ticks = 24h
```

Inspect results:

```bash
prophet trade progress <experiment_id>   # tick-by-tick
prophet trade dashboard                  # browser dashboard
```

The experiment ID is printed when the run starts.

---

## CLI flags

| Flag | What it does | Default |
|---|---|---|
| `-m`, `--models` | Model in `provider:model` format. Repeat for multiple. | required |
| `-s`, `--slug` | Experiment name. Reuse it to resume a stopped run. | required |
| `-r`, `--replicates` | Independent runs per model. | 1 |
| `--max-ticks` | How many 15-minute ticks to run. | 96 |
| `--starting-cash` | Simulated starting cash per agent. | 10000 |
| `-v`, `--verbose` | Print full LLM prompts and responses. | off |

## Supported models

| Provider | Examples |
|---|---|
| OpenAI | `openai:gpt-4o`, `openai:gpt-5.2` |
| Anthropic | `anthropic:claude-sonnet-4` |
| Google | `gemini:gemini-2.5-flash` |
| xAI | `xai:grok-3` |

## API credits

- **Build phase:** $50 OpenRouter credits per team for iteration.
- **Eval phase:** teams self-fund their own keys — plan model selection and tick budget accordingly.

---

## Trading rules (every agent, same rules)

| Rule | Value |
|---|---|
| Tick interval | 15 min (UTC `:00/:15/:30/:45`) |
| Starting cash | $10,000 |
| Max open positions | 30 |
| Max per-market exposure | $1,000 |
| Max total exposure | $10,000 |
| Max trades per tick | 10 |
| Fees | none |
| Slippage / partial fills | none — all-or-nothing at the snapshot's best bid/ask |
| Resolution payout | $1 winning side, $0 losing side |
| Submission deadline | within **9 minutes** of `tick_ts` or HTTP 409 |
| YES/NO on same market | not allowed simultaneously |

Buy YES fills at `best_ask`. Buy NO fills at `1 - best_bid`.

---

## Quick peek at live markets (no Python needed)

`/candidates/asof` with no params returns the latest tick's market universe:

```bash
curl -H "X-API-Key: $PA_SERVER_API_KEY" https://api.aiprophet.dev/candidates/asof
```

Useful for sanity-checking what's tradeable right now, eyeballing spreads, or grepping for a specific market_id without spinning up the SDK. Pipe through `jq '.markets[] | {market_id, question, quote}'` for a readable listing.

---

## Market data you actually get

This is the single most important constraint for strategy design. Per-market quote returned by `get_candidates()` / `get_market_snapshot()` ([client_models.py:139](ai-prophet/packages/core/ai_prophet_core/client_models.py:139)):

```python
class MarketQuote:
    best_bid: str       # top-of-book price only
    best_ask: str       # top-of-book price only
    volume_24h: float
    ts: datetime
```

**This is thinner than L1.** No `bid_size`, no `ask_size`, no second-best level, no depth. The server's internal `Quote` ([models.py:74](ai-prophet/packages/core/ai_prophet_core/models.py:74)) carries top-of-book sizes but **strips them before sending to clients**.

`volume_24h` is the only liquidity proxy. Reasonable rule of thumb: gate per-trade size by some fraction of 24h volume.

Each market also comes with: `market_id`, `question`, `short_label`, `description`, `resolution_time`, `source`, `source_url`, `topic`, `family`.

---

## Trade submission & fill model

Submit a batch of intents per tick via `POST /trade_intents`. Each intent maps to **exactly one** `FillData` or `RejectionData` — there is no partial-fill field ([client_models.py:208](ai-prophet/packages/core/ai_prophet_core/client_models.py:208)):

```python
class TradeIntentRequest:
    market_id: str
    action: str          # "BUY"
    side: str            # "YES" | "NO"
    shares: str
    idempotency_key: str # SDK auto-fills as {exp}:{participant}:{tick}:{i}

class FillData:
    shares, price, notional, filled_at, ...   # singular "shares" — no partial

class RejectionData:
    reason: str

class TradeSubmissionResult:
    accepted: int
    rejected: int
    fills: list[FillData]
    rejections: list[RejectionData]
```

Working model: full requested shares fill at the quoted price, or the intent is rejected. Probe rejection behavior empirically on test markets — `RejectionData.reason` is the only signal.

---

## Tick lifecycle (ordered sequence per tick)

1. `claim_tick()` — reserve the next interval (server enforces wall-clock; can't claim more than one ahead)
2. `load_candidates(lease)` — fetch market universe + prices
3. `get_portfolio()` — optional position review
4. `put_plan()` — optional audit log
5. `submit_trade_intents()` — execute trades
6. `finalize_participant()` — mark participant tick complete
7. `complete_tick()` — advance experiment

Run as a long-lived process that wakes on 15-minute boundaries.

---

## Full server endpoint surface

| Area | Endpoints |
|---|---|
| Health | `GET /health` |
| Market data | `GET /candidates`, `GET /candidates/asof` |
| Experiments | `POST /experiments`, `/experiments/{id}/{participants,progress,reasoning,ticks,ticks/{tick_id}:complete}` |
| Trading | `POST /trade_intents`, `GET /portfolio` |
| Forecast | `/forecast/{events,submit,teams/register,endpoints/register,endpoints/{team},scores}` |

There is **no** `/orderbook`, `/depth`, or `/quotes/l2`. Both market endpoints return the same skinny quote shape.

---

## Gotchas

- **`(owner, slug)` must be unique.** Two processes on the same slug fight over the tick lease — pick distinct slugs for distinct runs.
- **9-minute submission window.** Slow LLM calls will get HTTP 409. Budget for a ~5-min headroom; cache/parallelize web searches.
- **Idempotency keys are auto-generated**, so retrying a failed `submit_trade_intents` is safe.
- **Can't predict liquidity from the quote alone.** Without bid/ask sizes, your only pre-trade liquidity signal is `volume_24h`.
- **Resume = reuse the slug.** Crashes are non-fatal as long as you rerun with the same `--slug`.
