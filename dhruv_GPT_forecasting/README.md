# Dhruv GPT Prophet Arena Agent

This package is scoped to the Prophet Arena forecasting submission path:

```python
from dhruv_gpt_forecasting.arena_agent import predict

response = predict(event)
```

`predict(event)` returns one normalized probability for each exact label in `event["outcomes"]`. It optimizes Brier score only; it does not emit trades, sizing, or no-trade recommendations.

## Layout

- `src/dhruv_gpt_forecasting/arena_agent.py`: primary `predict(event)` and `forecast_arena_event(...)` implementation.
- `src/dhruv_gpt_forecasting/server.py`: optional FastAPI wrapper exposing `POST /predict`.
- `src/dhruv_gpt_forecasting/arena_priors.py`: deterministic fallback priors from event text, category/entity history, local resolved data, and linked-market context.
- `src/dhruv_gpt_forecasting/arena_live_data.py`: optional cached live evidence retrieval for current forecasts.
- `src/dhruv_gpt_forecasting/grounded_research.py`: Gemini native-search source reading for compact, timestamp-audited evidence digests.
- `src/dhruv_gpt_forecasting/lseg_query.py`: optional LSEG query planning for licensed news evidence.
- `src/dhruv_gpt_forecasting/pit_evidence.py`: point-in-time filtering for timestamped local, Reddit, GDELT, ESPN, WRDS, and LSEG evidence.
- `src/dhruv_gpt_forecasting/market_linker.py`: related-market context model used as evidence for the final forecaster.
- `prompts/forecasting_brier_v1_system.txt`: final probability model prompt.
- `prompts/grounded_research_v1_system.txt`: optional Gemini source-reading prompt.
- `prompts/lseg_news_query_v1_system.txt`: optional LSEG query-planning prompt.
- `configs/default.json`: Arena model, live evidence, deadline, cache, and budget settings.

## Quick Checks

```bash
cd dhruv_GPT_forecasting
python -m pytest
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli preflight --offline
ARENA_OFFLINE=1 PYTHONPATH=src python -m dhruv_gpt_forecasting.cli predict-json event.json
```

For local resolved-data runs:

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli prophet-events --status open -o logs/prophet_events.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli arena-batch predict --events logs/prophet_events.json --limit 10 -o logs/predictions.json
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli arena-eval --submission logs/predictions.json --actuals logs/actuals.json --events logs/resolved_events.json
```

For the Prophet Arena local-module contract:

```bash
PYTHONPATH=src prophet forecast predict \
  --events events.json \
  --local dhruv_gpt_forecasting.arena_agent \
  --timeout 300 \
  -o predictions.json
```

For the HTTP contract:

```bash
python -m pip install -e ".[server]"
uvicorn dhruv_gpt_forecasting.server:app --host 0.0.0.0 --port 8000
prophet forecast predict \
  --events events.json \
  --agent-url http://localhost:8000/predict \
  --timeout 300 \
  -o predictions.json
```

## Response Shape

```json
{
  "probabilities": [
    {"market": "Pittsburgh", "probability": 0.68},
    {"market": "Atlanta", "probability": 0.32}
  ]
}
```

Every `market` value is copied from `event.outcomes`, every listed outcome appears once, and probabilities are normalized before return. Binary YES/NO markets, named binary outcomes, and multi-outcome contracts all use the same response shape.

## Runtime Modes

- Default config uses direct Gemini `gemini-3-flash-preview`.
- `ARENA_OFFLINE=1` disables GPT and live data for deterministic local tests.
- Live mode enables current evidence retrieval by default; set `ARENA_DISABLE_LIVE_DATA=1` to turn it off.
- Gemini native search grounding is used in the final probability call for live forecasts. The prompt includes targeted contract-specific search questions, deadline context, source status, and deterministic priors.
- `ARENA_ENABLE_PRE_GROUNDED_RESEARCH=1` enables the older separate Gemini source-reading pass when you explicitly want it; the fast live path normally uses one final grounded Gemini call.
- `ARENA_LIVE_ACCELERATE_AFTER_SECONDS=360` skips optional repair/audit calls after six minutes, and `ARENA_FINAL_FALLBACK_RESERVE_SECONDS=20` preserves time for a market/deterministic fallback before the eight-minute deadline.
- `FORECAST_ENABLE_PIT_EXTERNAL=1` enables timestamp-filtered local/live evidence rows.

If GPT or live evidence fails, `forecast_arena_event(...)` falls back to deterministic Arena priors and still returns a valid probability distribution.
