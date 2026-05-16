# PLAN — iterating on `caf3bba`

**Hackathon:** Prophet Hacks, May 16–17, 2026 · submission May 17, 5 PM
**Tracks:** Forecasting + Trading · **Eval:** `data/eval_pack.jsonl` (42,194 resolved markets)
**Working repo:** [PrashanthBhaskara/ProphetHacks](https://github.com/PrashanthBhaskara/ProphetHacks) · current HEAD `caf3bba`

---

## What's already built (don't redo)

Prashanth's `caf3bba` lands a working ensemble + trading harness:

- `forecasters/` — `ForecasterConfig` + `mock`, `gemini`, `openrouter` adapters
- `schemas.py` — `MarketPacket`, `ModelForecast`, `ReasoningTrack`, `ForecastDiagnostics`, `SupervisorForecast`
- `ensemble.py` — logit-pool with diagnostic-aware weights + market anchor (1.5)
- `calibration.py` — shrink-to-market by category × horizon
- `trading/` — risk gate + Kelly sizing + simulator + ROI metrics
- `scripts/backtest_ensemble.py` — end-to-end runner
- `data/eval_pack.jsonl` — 42k labeled markets with snapshots
- `config/ensemble.example.json` — per-lane config (mock + gemini/claude/openai/deepseek lanes)

**Eval-pack market baselines:** all 0.065 · Sports **0.139** · Crypto 0.017 · Weather 0.157 · Politics N/A in pack.
**Sports is the bar to beat.** Crypto is essentially solved by the market.

---

## Critical gaps in the harness (do these first)

These block the rest of the plan. Hour estimates are realistic with focused work.

| # | Gap | Owner | h | $ |
|---|---|---|---|---|
| **H1** | `backtest_ensemble.py` is single-threaded. 5 lanes × 1000 markets × 3 s = 4 h serial. Need `ThreadPoolExecutor` over markets (and over lanes within a market). Add `--workers N`. | Prashanth | 0.5 | 0 |
| **H2** | No response cache. A crash at row 700/1000 forces re-paying for the first 700. Cache `ModelForecast` to disk by `stable_prompt_hash`. Add `--cache-dir data/cache`. | Prashanth | 1 | 0 |
| **H3** | Summary outputs trading metrics only. We need Brier + ECE per-lane AND for the ensemble in the same pass. Hook `prep.score` into `backtest_ensemble.summarize_trades`. | Anyone | 0.5 | 0 |
| **H4** | OpenRouter + Gemini adapters have no retry. A single 5xx kills the run. Hand-rolled exponential backoff: 3 retries on 429/5xx/timeout. | Anyone | 0.5 | 0 |
| **H5** | `config/ensemble.example.json` ships with mock lanes `enabled:true`. Easy footgun — naive run looks fine but isn't real. Flip to `enabled:false` by default and add a `--mock` flag instead. | Anyone | 0.1 | 0 |

**Total: ~2.6 h of plumbing before any model lane is worth running at scale.** Most of these are 1-file edits.

---

## Phase A — Setup (mostly done)

- [x] Cloned repo, `.env` populated with shared OpenRouter + xAI keys
- [x] `grok_lane` added to `config/ensemble.example.json` (still `enabled:false`)
- [x] Mock backtest verified end-to-end on 5 samples (`python scripts/backtest_ensemble.py --limit 5`)
- [ ] **All four teammates** clone the repo, copy `.env` (see top of `.env`), confirm `python scripts/backtest_ensemble.py --limit 5` runs locally
- [ ] **Rotate** the shared OpenRouter + xAI keys (both currently leaked); update `.env` with new ones

---

## Phase B — Harness fixes (H1–H5 above, ~3 h, $0)

Do these in parallel branches. None require API calls.

---

## Phase C — Per-lane smoke (each teammate, ~15 min, ~$1 each)

Once H3 lands (per-lane Brier in the summary), each teammate runs **their** lane alone on a 100-market stratified slice:

```bash
# enable ONLY your lane in config, others enabled:false
python scripts/backtest_ensemble.py --source eval_pack --limit 100 \
       --out data/smoke_<your_model>.jsonl
```

Report back: Brier overall, Brier on Sports, ECE, n_trades, sample p_yes for one well-known market (smell test).

**Decision gate:** if your lane's Brier on Sports > 0.155, debug your prompt before joining the ensemble. The market baseline is 0.139 — being worse than that means we'd literally do better dropping you.

**Lane ownership:**
- Victor → `grok_lane` (x-ai/grok-4.3 via OpenRouter; optional direct xAI in Phase F)
- Teammate A → `claude_lane` (anthropic/claude-sonnet-4.6 or 4.7)
- Teammate B → `openai_lane` (openai/gpt-5.1)
- Teammate C → `gemini_lane` (google/gemini-3-pro) — via `forecasters/gemini.py` direct or OpenRouter

---

## Phase D — Full ensemble run (~1.5 h wall-clock with H1, ~$15)

After everyone clears Phase C:

```bash
# all four lanes enabled, 1000 stratified markets
python scripts/backtest_ensemble.py --source eval_pack --limit 1000 \
       --workers 8 --cache-dir data/cache --out data/ensemble_v1.jsonl
```

Inspect:
- Ensemble Brier vs best single lane (does the ensemble actually help?)
- Per-category Brier (where is the ensemble winning vs market?)
- Trading metrics (ROI, win rate, n_trades) at default risk settings

---

## Phase E — Calibration sweep (Victor, ~1 h, $0)

`scripts/calibration_sweep.py` (new) — reads cached forecasts from Phase D, sweeps `category_weights` × `horizon_weights` × `base_weight` over a 3-value grid each. Free because no LLM calls. Save best as `config/ensemble.tuned.json`.

Current defaults are a guess (`Sports: 0.20`, `Crypto: 0.08`). With real per-lane numbers we can pick rationally.

---

## Phase F — Direct xAI adapter (Victor — **done in this PR**, $0)

Lands `src/prep/forecasters/xai.py` mirroring `openrouter.py` at `https://api.x.ai/v1`. Dispatch wired in `base.py:forecast_from_config`. Config entry `grok_lane_xai_direct` added (`enabled:false`).

Use either lane:
- `grok_lane` → via OpenRouter (current default, simpler)
- `grok_lane_xai_direct` → direct xAI (skip the ~5 % markup at volume; flip `enabled:true` in config and set `XAI_API_KEY`)

---

## Phase G — Live submission scripts (~2 h)

- `scripts/run_live.py`: `pmxt.Kalshi().fetch_markets(...)` → build packets → call ensemble → write `data/live_run.jsonl`
- `scripts/submit_forecasting.py`: reshape into `{market_ticker, p_yes, rationale}` for the forecasting submission
- `scripts/submit_trading.py`: reshape `TradeDecision` for the trading submission

`pmxt` setup already verified — see `~/Prophet Hacks/prep/pmxt_kalshi_test.py` (will port to this repo).

---

## Phase H — Writeup + final submission (~1 h, day 2 afternoon)

- Reliability diagram per lane and for ensemble (matplotlib → PNG in `data/`)
- Bootstrap 95 % CI for ensemble Brier (1000 resamples on cached results)
- 1-page method summary for judges: "logit-pool 4-model ensemble, diagnostic-weighted, market-anchored at 1.5, category-shrunk calibration, Kelly-sized trades"

---

## Coordination

**Daily syncs** (Slack/Discord, async OK):
- **Day 1, noon (now):** confirm everyone has env set up, H1–H5 split across owners
- **Day 1, 6 PM:** per-lane smoke results, prune weak lanes
- **Day 1, 11 PM:** Phase D ensemble run kicked off overnight (~$15 spend)
- **Day 2, 8 AM:** review results, Phase E sweep, decide final config
- **Day 2, 2 PM:** Phase G live run + Phase H writeup
- **Day 2, 4:30 PM:** final submission, 30 min before deadline

**Branching:**
- `main` is shared, but **PR before merge** to avoid stomping each other
- Branch names: `<initials>/<short-desc>` (e.g. `vk/grok-lane`, `pb/parallelism`)
- Keep PRs small (1–2 files each ideally)

**Key spend tracking:**
- One shared OpenRouter org account → one shared spend cap
- Each phase tagged with a $ cap; if a run looks like it'll blow it, kill and ask
- Total expected spend: ~$50 across all 4 teammates

---

## Open questions (please edit answers here)

1. **Is `src/prep/baselines/` dead** or do we keep `claude_zero_shot.py` around for sanity? Answer: __________
2. **Sample size for Phase D**: 1k stratified (~$15) or full 42k (~$600, do not do this)? Answer: __________
3. **Trading-track risk limits**: current `max_stake=1.0, kelly_fraction=0.10`. Aggressive or conservative for the live demo? Answer: __________
4. **Submission day 2: who owns `scripts/submit_*.py` and the actual click-submit?** Answer: __________
