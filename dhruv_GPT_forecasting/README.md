# Dhruv GPT Forecasting Lane

This package implements an OpenRouter-only GPT lane for Prophet Arena and Kalshi-style prediction markets. The default package-level `predict(event)` is now the Prophet Arena forecasting path: it always returns a normalized probability distribution over exact outcome labels and optimizes Brier score, not trade execution.

The older Kalshi trading lane remains available through `dhruv_gpt_forecasting.forecaster.forecast_event`.

## Layout

- `configs/default.json`: model IDs, gates, risk settings, and cost assumptions.
- `prompts/`: static prompt prefixes for the cheap GPT lane and supervisor lane.
- `src/dhruv_gpt_forecasting/`: package code for features, stats, gating, OpenRouter calls, validation, and backtests.
- `src/dhruv_gpt_forecasting/arena_agent.py`: Prophet Arena local-module entrypoint for `prophet forecast predict`.
- `src/dhruv_gpt_forecasting/arena_priors.py`: no-market deterministic priors from resolved data, entity rates, and historical analogs.
- `src/dhruv_gpt_forecasting/arena_live_data.py`: optional cached live evidence retrieval.
- `src/dhruv_gpt_forecasting/grounded_research.py`: Gemini 3 Flash native-search source reading that turns contract-specific web research into an auditable evidence digest.
- `src/dhruv_gpt_forecasting/pit_evidence.py`: point-in-time Reddit/GDELT/ESPN/vendor/local external evidence filtering.
- `src/dhruv_gpt_forecasting/sentiment.py`: deterministic local sentiment features for timestamped evidence rows and digests.
- `src/dhruv_gpt_forecasting/evidence_sources.py`: source taxonomy used to tell Gemini 3 Flash how to weight ESPN, WRDS, LSEG, official series, market data, social evidence, prediction-market context, and native search grounding.
- `src/dhruv_gpt_forecasting/market_linker.py`: secondary linked-market model that turns sibling prediction markets into an inferred probability distribution.
- `src/dhruv_gpt_forecasting/arena_eval.py`: local Brier evaluator for Arena-shaped predictions.
- `src/dhruv_gpt_forecasting/stat_router.py`: shared statistical model registry and category/default routing for market-backed binary forecasts.
- `tests/`: non-network regression tests.
- `DATA_FINDINGS.md`: baseline results and interpretation for `prep/data` and `Kalshitopvolmarkets`.
- `context.py`: OOS-safe related-market context from `NonBinaryMarkets`, `Kalshitopvolmarkets`, and `prep/data/kalshi_polymarket`.

Default OpenRouter model IDs are now `google/gemini-3-flash-preview` for both the forecasting lane and supervisor lane. Live/current forecasts enable OpenRouter native web search grounding (`openrouter:web_search` with `engine="native"`) so Gemini can check current evidence inside the 5-minute budget. Historical backtests keep this off by default to preserve PIT validity.

## Quick Checks

```bash
cd dhruv_GPT_forecasting
python -m pytest
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli preflight --offline
ARENA_OFFLINE=1 PYTHONPATH=src python -m dhruv_gpt_forecasting.cli predict-arena-json event.json
prophet forecast predict --events events.json --local dhruv_gpt_forecasting.arena_agent
prophet forecast predict --events events.json --agent-url http://localhost:8000/predict
PYTHONPATH=src python -m dhruv_gpt_forecasting.arena_eval --submission predictions.json --actuals actuals.json --events events.json
prophet forecast retrieve --dataset sample-resolved --include-resolved -o logs/sample_resolved_events.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli arena-batch actuals --events logs/sample_resolved_events.json -o logs/sample_resolved_actuals.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli arena-batch predict --events logs/sample_resolved_events.json --limit 10 -o logs/sample_resolved_predictions.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli arena-batch benchmark --events logs/sample_resolved_events.json --limit 50 --seed 17
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli runbook --events logs/sample_resolved_events.json --limit 50
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli vendor-evidence normalize --source lseg --input exports/lseg_news.csv --output data/external_evidence/vendor/lseg_news.jsonl
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli vendor-evidence normalize --source wrds --input exports/wrds_news.csv --output data/external_evidence/vendor/wrds_news.jsonl
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli arena-eval --submission logs/sample_resolved_predictions.json --actuals logs/sample_resolved_actuals.json --events logs/sample_resolved_events.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli credentials
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli prophet-events --status open -o logs/prophet_events.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli kalshi-events --deadline 2026-05-19T00:00:00Z --max-items 100 -o logs/kalshi_events.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.backtest --source live_clean --mode market
PYTHONPATH=src python -m dhruv_gpt_forecasting.backtest --source live_clean --mode stat --limit 5000
PYTHONPATH=src python -m dhruv_gpt_forecasting.backtest --source topvol --mode market --limit 1000
PYTHONPATH=src python -m dhruv_gpt_forecasting.backtest --source nonbinary --mode market --limit 1000
PYTHONPATH=src python -m dhruv_gpt_forecasting.backtest --source unified --mode market --limit 1000
PYTHONPATH=src python -m dhruv_gpt_forecasting.backtest --source topvol --mode dryrun --limit 100
PYTHONPATH=src python -m dhruv_gpt_forecasting.experiments --source topvol --horizon-hours 24 --top 8
PYTHONPATH=src python -m dhruv_gpt_forecasting.oos_eval --source topvol --horizon-hours 24 --candle-stride-minutes 1 --context --out logs/oos_topvol_24h_1m_context.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.oos_eval --source nonbinary --candle-stride-minutes 1 --random-as-of --limit 2500 --out logs/oos_nonbinary_1m_random_asof.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.oos_eval --source unified --candle-stride-minutes 1 --random-as-of --limit 2500 --context --out logs/oos_unified_1m_random_asof.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.oos_eval --source topvol --horizon-hours 0.25 --candle-stride-minutes 1 --since-close 2026-02-16T00:00:00Z --context --out logs/oos_topvol_15m_1m_last3m_context.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.llm_tester --source topvol --horizon-hours 24 --candle-stride-minutes 1 --chronological-oos --limit 10
PYTHONPATH=src OPENROUTER_TIMEOUT_SECONDS=90 python -m dhruv_gpt_forecasting.llm_tester --source topvol --horizon-hours 0.25 --candle-stride-minutes 1 --since-close 2026-02-16T00:00:00Z --chronological-oos --limit 30 --model google/gemini-3-flash-preview
PYTHONPATH=src OPENROUTER_TIMEOUT_SECONDS=90 python -m dhruv_gpt_forecasting.llm_tester --source topvol --horizon-hours 0.25 --candle-stride-minutes 1 --since-close 2026-02-16T00:00:00Z --chronological-oos --force-cheap --pit-external-evidence --limit 30 --model google/gemini-3-flash-preview
```

## Prophet Arena Live Contract

The current Prophet Arena forecasting docs call either a Python module with `predict(event) -> dict` or an HTTP endpoint at `POST /predict`.
Our documented live entrypoints are:

```bash
# Local module path used by the current Prophet Arena docs.
PYTHONPATH=src prophet forecast predict \
  --events events.json \
  --local dhruv_gpt_forecasting.arena_agent \
  --timeout 300 \
  -o predictions.json

# HTTP endpoint path used for deployment.
python -m pip install -e ".[server]"
uvicorn dhruv_gpt_forecasting.server:app --host 0.0.0.0 --port 8000
prophet forecast predict \
  --events events.json \
  --agent-url http://localhost:8000/predict \
  --timeout 300 \
  -o predictions.json
```

Both entrypoints return exactly:

```json
{
  "probabilities": [
    {"market": "Pittsburgh", "probability": 0.68},
    {"market": "Atlanta", "probability": 0.32}
  ]
}
```

Every `market` value is copied from `event.outcomes`, every listed outcome appears once, and probabilities are normalized before return. Binary named outcomes, YES/NO markets, and non-binary/multi-outcome contracts all use the same response shape.

For a no-credit contract smoke, force deterministic mode:

```bash
ARENA_OFFLINE=1 ARENA_DISABLE_LIVE_DATA=1 PYTHONPATH=src python - <<'PY'
from dhruv_gpt_forecasting.arena_agent import predict
event = {
    "event_ticker": "task-001",
    "market_ticker": "task-001",
    "title": "Who will win: Pittsburgh or Atlanta?",
    "description": "Predict the winner.",
    "category": "Sports",
    "rules": "Resolves to the official winner.",
    "close_time": "2026-12-31T23:59:59Z",
    "outcomes": ["Pittsburgh", "Atlanta"],
}
print(predict(event))
PY
```

Note: the currently installed local `prophet` CLI in this workspace is older than the developer docs and still expects a binary `p_yes` return. It skips modern `probabilities` responses and cannot validate non-binary tasks. Use the raw `arena-batch`/`arena-eval` commands for local resolved-data testing until the CLI is upgraded, or use the HTTP endpoint path above when running against the current Prophet Arena contract.

Use `ARENA_OFFLINE=1` for deterministic no-network runs. By default, the Arena path will attempt Gemini 3 Flash if an OpenRouter key is available and will fall back to deterministic priors if the call fails. Native search grounding is enabled only for live/current `as_of` forecasts unless `OPENROUTER_NATIVE_SEARCH_GROUNDING` overrides it. Live evidence pulls are opt-in via `ARENA_ENABLE_LIVE_DATA=1`; results are cached under `logs/live_cache`.

When live data and GPT are enabled, the system now runs a grounded source-reading pass before the final probability prompt. It sends the exact contract parameters, rules, outcomes, close time, extracted Kalshi multi-leg structure, existing live evidence, and category-specific targeted questions into Gemini 3 Flash with OpenRouter native search grounding. The returned `gemini_native_search_grounded_research` evidence item summarizes macro drivers, breaking news, qualitative sentiment, contract-specific factors, source notes, information gaps, and retrieval confidence.

This pass is live-only by default. For exploratory historical backtests, enable `ARENA_ENABLE_BACKTEST_INTERNET=1` or pass `arena-batch ... --backtest-internet`; the source-reading prompt and runtime verifier then require every internet source to expose a clear source-specific `published_at` or update timestamp at or before the forecast `as_of`. Sources with missing, ambiguous, or post-`as_of` timestamps are discarded before the final probability model sees them. Current-only sources such as live Kalshi quotes, current Polymarket search, and current Polygon previous-close context are excluded from historical internet mode.

`--source topvol` reads `Kalshitopvolmarkets/`: weekly top-volume selected markets plus 1-minute OHLCV candles. `--source nonbinary` reads resolved component markets from `NonBinaryMarkets/`, which gives much broader category coverage for sports props, crypto, entertainment, and related context markets. `--source unified` combines both binary and nonbinary component samples so priors and OOS model routing see the whole local market universe. Backtests can downsample with `--candle-stride-minutes`; chronological OOS runs should use `1` when validating the full real-time signal path. The loader filters out candles after `close_time` and uses the requested point-in-time candle as the executable quote. This mirrors the kind of real-time packets the live system should feed into the lane.

The OOS evaluator now scores every registered statistical candidate, fits market-blend and Platt variants on the training split, and trains a category router using train-only Brier. The route report is written under `stat_model_routing`, so we can see whether crypto, sports, and other categories prefer different statistical families before handing the diagnostics to Gemini 3 Flash.

The linked-market model is emitted as `source="linked_market_model"` whenever same-event or sibling markets are available before `as_of`. It includes:

- `probabilities`: the target market probability lane.
- `component_distribution`: normalized sibling probabilities with labels.
- `inferred_structure`: whether the group looks like a coherent mutually exclusive distribution or a softer linked quote set.
- `quality`, `confidence`, and diagnostics such as component count, sum of mids, rank, entropy, and favorite gap.

Gemini 3 Flash sees this as a secondary model lane, not raw news. The stat router also uses it through context-normalized candidates when `--context` or live linked evidence is enabled.

The default stat config now promotes the 15-minute OOS winner for near-close market-backed forecasts: a 10% momentum-follow adjustment capped at 4pp, followed by Platt calibration (`a=0.09`, `b=1.21`) for horizons up to 0.5 hours. In Arena mode, Gemini 3 Flash is the final probability model for every prompted event. The deterministic/stat model is sent as calibrated context; runtime code only validates exact labels, clamps numeric errors, and normalizes.

Dry-run forecasts automatically attach related context when available:

- `NonBinaryMarkets/indexes/target_to_context_links.jsonl` gives exact same-event component groups.
- `NonBinaryMarkets/ohlcv` supplies component-market candle summaries at or before the target `as_of`.
- `Kalshitopvolmarkets` is a fallback for same-event sibling markets when no non-binary context link exists.
- `prep/data/kalshi_polymarket/map.csv` supplies cross-venue question mappings. It does not supply Polymarket prices yet; real-time Polymarket quotes should arrive as structured external evidence later.

The context builder strips post-settlement fields such as `result`, `status`, `settlement_ts`, and final metadata quotes before anything enters an LLM prompt.

## Point-in-Time External Evidence

The `pit_evidence` layer lets Gemini 3 Flash see Reddit, GDELT, ESPN, and normalized vendor context without reading ahead:

- Local archives live under `dhruv_GPT_forecasting/data/external_evidence/**/*.jsonl` by default.
- Each row should include `source`, `published_at`, `collected_at`, and at least one of `title`, `text`, `summary`, or `claim`.
- Optional `event_ticker` or `market_ticker` fields create exact joins; otherwise records are matched by title/outcome query overlap.
- Historical OOS tests default to strict mode: `published_at <= as_of` and `collected_at <= as_of` within a small clock tolerance.
- Live Reddit/GDELT/ESPN network pulls are only allowed when the packet `as_of` is near the current clock, except explicit GDELT timestamp-bounded archive pulls. This prevents a backtest for an old market from querying today's search index.
- Live fetches are appended back into `data/external_evidence/live_fetches/<source>/YYYY-MM-DD.jsonl`, creating the archive needed for later PIT replay.
- Normalized WRDS/LSEG rows can be used by placing JSONL records in `data/external_evidence/` with `source="wrds"` or `source="lseg"`, `published_at`, `collected_at`, and text fields. Use `vendor-evidence normalize` for CSV/JSON/JSONL exports.
- Optional live HTTP connectors use `WRDS_NEWS_API_URL`/`WRDS_API_URL` and `LSEG_NEWS_API_URL`/`LSEG_API_URL`. A custom endpoint should accept `q`, `as_of`, and `limit`, and return one of `results`, `articles`, `data`, `records`, `news`, `stories`, or `items`.
- Native vendor backends are supported when there is no custom HTTP bridge:
  - `LSEG_NEWS_API_URL=lseg-data-library://news` uses the LSEG Data Library / Workspace or Eikon app-key flow and retrieves timestamp-bounded headlines.
  - `WRDS_NEWS_API_URL=wrds-postgres://news` uses the WRDS Python/PostgreSQL path. WRDS is dataset-specific, so set `WRDS_NEWS_SQL` or `WRDS_NEWS_SQL_FILE` to a PIT-safe query template with placeholders `%(query)s`, `%(start)s`, `%(as_of)s`, and `%(limit)s`.
- Live LSEG queries are planned by GPT before headline retrieval when `ARENA_LSEG_LLM_QUERY_ENABLED=1`. The query planner follows LSEG Data Library guidance: `news.get_headlines` receives a query string while the runtime supplies `count/start/end`; RIC syntax such as `R:MSFT.O` is preferred only for high-confidence instruments; safe filters include `Language:LEN`, `Source:RTRS`, and `Topic:SIGNWS`; category logic differs for economics, crypto, sports, politics, weather, and reality TV/entertainment. If the query-planning call fails or the deadline is tight, the system falls back to a deterministic category-aware LSEG query.
- WRDS live auth supports `WRDS_API_KEY`/`WRDS_ACCESS_TOKEN` or `WRDS_USERNAME` + `WRDS_PASSWORD`. LSEG live auth supports `LSEG_API_KEY` or the app-key envs; `LSEG_APP_KEY*` defaults to an `App-Key` header unless `LSEG_API_KEY_HEADER` overrides it.
- All archived rows are enriched with `sentiment_score`, `sentiment_label`, and `sentiment_model="lexicon_v1"` before they reach GPT. This is a cheap deterministic tone feature, not a replacement for source reliability or market/stat priors.

Useful toggles:

```bash
FORECAST_ENABLE_PIT_EXTERNAL=1
PIT_EXTERNAL_ALLOW_NETWORK=1
REDDIT_USER_AGENT=ProphetHacksGPTForecasting/0.1
WRDS_NEWS_API_URL=wrds-postgres://news
WRDS_NEWS_SQL_FILE=dhruv_GPT_forecasting/configs/wrds_news.sql
LSEG_NEWS_API_URL=lseg-data-library://news
ARENA_ENABLE_LIVE_DATA=1
ARENA_ENABLE_BACKTEST_INTERNET=1
FORECAST_ENABLE_PIT_EXTERNAL=1
ARENA_LSEG_LLM_QUERY_ENABLED=1
ARENA_LSEG_QUERY_TIMEOUT_SECONDS=12
```

Check vendor readiness without spending GPT credits:

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli vendor-evidence status
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli vendor-evidence fetch --source lseg --query "Federal Reserve rate decision" --as-of "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --output logs/lseg_live_news_smoke.jsonl
```

For a 30-call Gemini 3 Flash sample, `llm_tester` now calls the configured LLM for every selected market by default. Attach PIT external evidence with `--pit-external-evidence`. Add `--pit-allow-network` only for live/current timestamps. For historical backtests, prefer archived records collected before each simulated `as_of`; `--pit-nonstrict-collected-at` is available for exploratory timestamp-only tests, but those results should not be treated as clean OOS. Use `--respect-gates` only when intentionally comparing against the older gated policy.

Archive the 30 forecast packets before running GPT:

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.evidence_archiver --source topvol --horizon-hours 0.25 --candle-stride-minutes 1 --since-close 2026-02-16T00:00:00Z --chronological-oos --force-cheap --limit 30 --sources reddit,gdelt,espn --allow-historical-backfill --sleep-seconds 1
```

`--allow-historical-backfill` enables GDELT timestamp-bounded article pulls. Reddit historical backfills are off by default because public Reddit search cannot prove the old search-index state; use `--reddit-historical-backfill` only for exploratory collection. Replay of those Reddit backfills is also blocked by default unless `PIT_EXTERNAL_ALLOW_REDDIT_BACKFILL_REPLAY=1` is set.

For clean PIT collection on events we are about to forecast, archive before predicting:

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.evidence_archiver --events-json events.json --sources reddit,gdelt,espn --run-name live_forecast_capture_$(date -u +%Y%m%dT%H%M%SZ)
```

Those live captures get `pit_mode=live_capture` and can be used in future strict backtests because `published_at` and `collected_at` are both at or before the forecast packet timestamp within the configured clock tolerance.

For the competition-style backtest where the forecast request time is random and the agent has five minutes to answer, use the random-as-of flags. These choose one valid request timestamp per resolved market, truncate the 1-minute market history exactly there, and store `decision_deadline_time` as metadata only.

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.oos_eval \
  --source topvol \
  --candle-stride-minutes 1 \
  --since-close 2026-02-16T00:00:00Z \
  --random-as-of \
  --random-seed 20260517 \
  --min-horizon-minutes 5 \
  --min-history-snapshots 5 \
  --decision-budget-minutes 5 \
  --limit 250 \
  --train-fraction 0.70
```

Archived evidence can be replayed into this OOS evaluator in two auditable modes:

- `strict_pit`: requires `published_at <= forecast_as_of` and `collected_at <= forecast_as_of` within the configured clock tolerance.
- `relaxed_published_at`: requires only `published_at <= forecast_as_of`, useful for exploratory historical backfills such as Reddit or GDELT archive searches.

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.oos_eval \
  --source unified \
  --candle-stride-minutes 1 \
  --random-as-of \
  --random-seed 20260517 \
  --min-horizon-minutes 5 \
  --min-history-snapshots 5 \
  --decision-budget-minutes 5 \
  --limit 300 \
  --train-fraction 0.70 \
  --context \
  --evidence-mode strict_pit \
  --evidence-manifest dhruv_GPT_forecasting/data/external_evidence/backfills/prophet_subset_1200_curated_sources_sentiment_YYYYMMDD/manifest.json
```

To archive timestamp-bounded GDELT news for the first 50 random-as-of events and write compact local digests for GPT prompts:

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.evidence_archiver \
  --source topvol \
  --candle-stride-minutes 1 \
  --since-close 2026-02-16T00:00:00Z \
  --random-as-of \
  --random-seed 20260517 \
  --min-horizon-minutes 5 \
  --min-history-snapshots 5 \
  --decision-budget-minutes 5 \
  --limit 50 \
  --max-candidates 500 \
  --sources gdelt \
  --allow-historical-backfill \
  --synthesize-news \
  --sleep-seconds 6 \
  --resume \
  --max-fetch-errors 8 \
  --run-name topvol_random_asof_50_gdelt_digest_20260517
```

GDELT has aggressive rate limiting. Use `--resume`, a multi-second `--sleep-seconds`, and a low `--max-fetch-errors` to build the archive in batches instead of wasting requests during a cooldown.

The teammate trading backtester uses the official-style `subset_1200.csv` row universe. To archive its bundled curated news/source snippets as PIT evidence plus compact sentiment digests without any network calls:

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.evidence_archiver \
  --source prophet_subset_1200 \
  --sources prophet_sources \
  --limit 1200 \
  --max-candidates 1200 \
  --synthesize-news \
  --run-name prophet_subset_1200_curated_sources_sentiment_YYYYMMDD
```

This treats the row `snapshot_time` as `collected_at` when the source does not include a publication timestamp. Those records are useful for Brier prompt context and backtest replay; the manifest records the timestamp basis so GPT can discount sources whose exact article publication time is unknown.

For live readiness, use `live-readiness` against an event file. This measures per-event elapsed time, deadline compliance, fallback errors, and final probabilities. Use `--with-gpt` only for small smoke tests because it spends OpenRouter credits.

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli live-readiness \
  --events-json dhruv_GPT_forecasting/logs/kalshi_events_deadline_YYYYMMDD.json \
  --limit 1 \
  --deadline-seconds 300 \
  --with-gpt \
  --live-data
```

## API Key Policy

The default provider is OpenRouter. This lane uses `OPENROUTER_API_KEY_1` by default so it does not consume teammates' keys. Lane-specific names remain optional aliases if we later split budgets.

```bash
OPENROUTER_API_KEY_1=...
OPENROUTER_API_KEY_GPT_CHEAP=...
OPENROUTER_API_KEY_GPT_SUPERVISOR=...
OPENROUTER_API_KEY_GPT_AUDIT=...
```

Never commit real `.env` values. The code can load a local `.env`, but examples contain placeholders only.
