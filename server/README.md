# Prophet Hacks forecast API

FastAPI server that returns `p_yes` per Kalshi binary market for the
2-week eval window after the May 17, 2026 submission.

## Endpoints

```
GET  /                         service info
GET  /healthz                  liveness probe (Render hits this)
GET  /predict?ticker=KX...     single prediction
POST /predict                  body: {"tickers": ["KX...", "KX..."]}
POST /predict_event            body: {"market_ticker": "KX...", "title": "..."}
```

Response shape:

```json
{
  "market_ticker": "KXATPMATCH-25JUL02SHEHIJ",
  "p_yes": 0.421,
  "raw_market_p": 0.421,
  "source": "mid",
  "calibration": "passthrough",
  "status": "active",
  "close_time": "2026-05-20T19:00:00Z",
  "ts": 1747526400,
  "cached": false
}
```

## Model

Market mid-price from `kalshi /markets/{ticker}`, with optional
global Platt shrinkage if `server/calibration.json` exists.

To enable Platt: drop `{"a": <slope>, "b": <intercept>}` into
`server/calibration.json` and redeploy. With no file, predictions
pass through unchanged — that's our validated floor.

## Local run

```bash
cd server
pip install -r requirements.txt
uvicorn server.main:app --reload --port 8000  # run from repo root
```

Test it:

```bash
curl http://localhost:8000/healthz
curl 'http://localhost:8000/predict?ticker=KXATPMATCH-25JUL02SHEHIJ'
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"tickers": ["KXATPMATCH-25JUL02SHEHIJ"]}'
```

## Deploy to Render

See `../DEPLOY.md`.
