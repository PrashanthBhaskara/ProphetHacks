# API Key Setup

This file documents the keys needed by `dhruv_GPT_forecasting`. Keep real values in a local `.env` only. Do not commit secrets.

## Current Local Audit

Detected as present under the currently checked local env files:

- `FRED_API_KEY`
- `POLYGON_API_KEY`
- `ODDSPIPE_API_KEY`
- `EIA_API_KEY`
- `BEA_API_KEY`
- Dhruv OpenRouter key: `OPENROUTER_API_KEY_1`
- Prophet Arena key: `PA_SERVER_API_KEY`
- Kalshi access key aliases: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-KEY-DEMO`

Not needed right now:

- `OPENROUTER_API_KEY_GPT_CHEAP`
- `OPENROUTER_API_KEY_GPT_SUPERVISOR`
- `OPENROUTER_API_KEY_GPT_AUDIT`

Not detected and out of scope for now:

- `KALSHI_PRIVATE_KEY_B64`
- `REDDIT_USER_AGENT`
- `WRDS_API_URL` / `WRDS_API_KEY`
- `LSEG_NEWS_API_URL` / `LSEG_API_KEY`

## Required For GPT Forecasting

1. OpenRouter
   - Go to `https://openrouter.ai/settings/keys`.
   - Use only `OPENROUTER_API_KEY_1` for this lane. Keys 2-4 belong to teammates and should not be used by this package.

```bash
OPENROUTER_API_KEY_1=...
```

The default model is `google/gemini-3-flash-preview`. Live/current forecasts use OpenRouter native search grounding by default, while historical backtests keep direct final-prompt search grounding disabled to preserve PIT validity. The package intentionally does not fall back to `OPENROUTER_API_KEY_2`, `OPENROUTER_API_KEY_3`, or `OPENROUTER_API_KEY_4`.

The live Arena path also uses the same key for a grounded source-reading pass when `ARENA_ENABLE_LIVE_DATA=1` and `ARENA_ENABLE_GROUNDED_RESEARCH=1`. That pass asks Gemini 3 Flash to search for contract-specific macro drivers, breaking news, and qualitative sentiment, then stores the compact digest as evidence for the final probability model.

For exploratory historical backtests, set `ARENA_ENABLE_BACKTEST_INTERNET=1` or use `arena-batch --backtest-internet`. The source-reading pass may then search the internet, but the prompt and runtime verifier require every used source to have a clear publication/update timestamp at or before the simulated `as_of`; undated, ambiguous-date, and post-`as_of` sources are dropped.

## Required For Live Kalshi Retrieval

1. Create a Kalshi API key in the Kalshi account/developer settings.
2. Save the API key ID.
3. Save the RSA private key immediately. Kalshi returns the private key once.
4. Base64-encode the private key PEM without printing it in logs, then add:

```bash
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_B64=...
```

The loader also maps `KALSHI-ACCESS-KEY` to `KALSHI_API_KEY_ID`, and `KALSHI-ACCESS-KEY-DEMO` to `KALSHI_DEMO_API_KEY_ID`. Those access keys are present in `prophet-hacks-handoff/prep/.env`.

The private key is still required for signed Kalshi requests and for the installed `prophet forecast retrieve` path. The local `dhruv_gpt_forecasting kalshi-events` command can use public Kalshi market endpoints without it.

## Required For Prophet Arena Server API

Ask the Prophet Arena organizers/dashboard for the team API key, then add:

```bash
PA_SERVER_API_KEY=...
```

The local benchmark path does not need this key. Server endpoints and leaderboard queries do.
`PA_SERVER_API_KEY` is present locally.

## Reddit

For our current public-search usage, Reddit does not need a secret key. The code uses a default user agent if none is set. You can override it with:

```bash
REDDIT_USER_AGENT=python:prophet-hacks-forecasting:v0.1 (by /u/<your_reddit_username>)
```

Historical Reddit search is not strict PIT unless we captured it live before the forecast cutoff. Use Reddit mostly for live pre-prediction capture.

## ESPN Sports News

No key is required for the current ESPN news adapter. It is live-capture only for strict PIT backtests: run the archive command before prediction and replay those archived rows later. ESPN is categorized as high-quality sports availability/context evidence, not as an odds source.

## WRDS And LSEG

These are licensed sources. The code supports two paths:

- Preferred for backtests: export CSV/JSON/JSONL rows and run `vendor-evidence normalize` into `data/external_evidence/` with `source` set to `wrds` or `lseg`, plus `published_at` and `collected_at`.
- Optional live connector: configure `WRDS_NEWS_API_URL`/`WRDS_API_URL` or `LSEG_NEWS_API_URL`/`LSEG_API_URL`; the runtime sends `q`, `as_of`, and `limit` query params and normalizes common `results`/`articles`/`data`/`records`/`stories` rows.
- WRDS live auth supports `WRDS_API_KEY`/`WRDS_ACCESS_TOKEN` or `WRDS_USERNAME` + `WRDS_PASSWORD`. LSEG live auth supports `LSEG_API_KEY` or app-key envs such as `LSEG_APP_KEY`.

Example normalization commands:

```bash
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli vendor-evidence normalize --source lseg --input exports/lseg_news.csv --output data/external_evidence/vendor/lseg_news.jsonl
PYTHONPATH=src python -m dhruv_gpt_forecasting.cli vendor-evidence normalize --source wrds --input exports/wrds_news.csv --output data/external_evidence/vendor/wrds_news.jsonl
```

For clean historical tests, use source-native release/vintage timestamps. Do not replay rows collected after a simulated forecast time as strict PIT evidence.

## No Key Needed

- GDELT DOC API: timestamp-bounded news/article search, useful as a replacement for X.
- ESPN news adapter for live sports context.
- Local Kalshi/Polymarket context already downloaded under the repo data folders.

## Optional Data Keys Already Detected

These are present locally and should remain in `.env`:

```bash
FRED_API_KEY=...
POLYGON_API_KEY=...
ODDSPIPE_API_KEY=...
EIA_API_KEY=...
BEA_API_KEY=...
```

## Recommended `.env` Location

Use one of:

- `dhruv_GPT_forecasting/.env`
- repo-root `.env`
- `prophet-hacks-handoff/prep/.env`

The loader checks all three, but `dhruv_GPT_forecasting/.env` is the cleanest for this package.
