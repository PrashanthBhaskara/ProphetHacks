# ProphetHacks Forecast Agent

Production forecasting service for the Prophet Arena `/predict` contract.

The default runtime is `config/FINAL.json`. It runs four lanes in parallel:

- `gemini_pro`: direct Gemini API, `gemini-3-pro-preview`, Google Search grounding required.
- `dhruv_gemini_lane`: Dhruv lane adapter using direct Gemini API, `gemini-3-flash-preview`, with live grounding enabled.
- `claude_lane`: Claude through OpenRouter.
- `grok_lane`: Grok through OpenRouter.

The final judge is OpenAI through OpenRouter: `openai/gpt-5.4`. It receives the lane JSON outputs, deterministic ensemble, market context, justifications, and probabilities, then blends its judgment into the final distribution.

## Timing

- Lane budget: 450 seconds each.
- Judge budget: 120 seconds.
- Full `/predict` budget: 585 seconds.
- If a lane times out, it returns a market-mirror placeholder.
- If the whole ensemble times out or no lane produces a usable prediction, the service returns the current Kalshi market-implied distribution when available, otherwise a normalized uniform fallback.

## Local Setup

```bash
cd prophet-hacks-handoff/prep
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill `prophet-hacks-handoff/prep/.env` from `.env.example`. The agent server loads it automatically before reading `PROPHET_CONFIG`. Required keys for the full production config:

- `PA_SERVER_API_KEY` for Prophet Arena/server authentication.
- `GEMINI_API_KEY1` for the Gemini Pro lane.
- `GEMINI_API_KEY2` for the Dhruv Gemini Flash lane. Use a different Gemini key from `GEMINI_API_KEY1`.
- `OPENROUTER_API_KEY` for the Claude lane, Grok lane, and OpenAI judge.

Optional OpenRouter fallback:

- `OPENROUTER_API_KEY2` shared by Claude, Grok, and the OpenAI judge.

Run locally:

```bash
uvicorn scripts.agent_server:app --host 0.0.0.0 --port 8000
```

Smoke-check the wire contract:

```bash
python scripts/smoke_predict.py
```

## Output Contract

`POST /predict` returns:

```json
{
  "probabilities": [
    {"market": "Pittsburgh", "probability": 0.68},
    {"market": "Atlanta", "probability": 0.32}
  ]
}
```

The response preserves the exact outcome labels from the inbound event.
