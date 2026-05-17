# API Key Setup

Keep real values in a local `.env` only. Do not commit secrets.

## Required For GPT Forecasting

The Arena agent uses direct Gemini by default:

```bash
GEMINI_API_KEY=...
```

`GOOGLE_API_KEY` is accepted as a fallback alias. The current model is `gemini-3-flash-preview`, configured in `configs/default.json`.

When `ARENA_ENABLE_LIVE_DATA=1` and `ARENA_ENABLE_GROUNDED_RESEARCH=1`, the same Gemini key is used for the optional native-search source-reading pass. That pass creates timestamp-audited evidence for the final Brier-score probability model; it does not make the final forecast.

## Optional Prophet Arena API

Use this only for organizer/server calls such as event retrieval, health, and scores:

```bash
PA_SERVER_API_KEY=...
```

The local `predict(event)` path does not need this key.

## Optional Live Evidence Keys

Kalshi public event lookups can run without a private key. Signed Kalshi requests need:

```bash
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_B64=...
```

The loader also maps `KALSHI-ACCESS-KEY`, `KALSHI_ACCESS_KEY`, and related aliases into the internal names above.

Optional vendor/source keys:

```bash
FRED_API_KEY=...
POLYGON_API_KEY=...
ODDSPIPE_API_KEY=...
EIA_API_KEY=...
BEA_API_KEY=...
REDDIT_USER_AGENT=ProphetHacksGPTForecasting/0.1
WRDS_NEWS_API_URL=wrds-postgres://news
WRDS_API_KEY=...
LSEG_NEWS_API_URL=lseg-data-library://news
LSEG_APP_KEY=...
```

GDELT and ESPN currently do not need secret keys.

## Recommended `.env` Location

Use one of:

- `dhruv_GPT_forecasting/.env`
- repo-root `.env`
- `prophet-hacks-handoff/prep/.env`

The loader checks all three, but `dhruv_GPT_forecasting/.env` is the cleanest for this package.
