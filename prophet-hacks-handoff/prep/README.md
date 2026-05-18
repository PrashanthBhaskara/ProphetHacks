# ProphetHacks Forecast Agent

Production forecasting service for the Prophet Arena `/predict` contract.

The default runtime is `config/FINAL.json`. It runs four lanes in parallel:

- `gemini_pro`: direct Gemini API, `gemini-3-pro-preview`, Google Search grounding required.
- `dhruv_gemini_lane`: Dhruv lane adapter using `google/gemini-3-flash-preview` through OpenRouter with live grounding enabled.
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

- `GEMINI_PRO_API_KEY` for the Gemini Pro lane.
- `DHRUV_OPENROUTER_API_KEY` for the Dhruv Gemini Flash lane.
- `CLAUDE_OPENROUTER_API_KEY` for the Claude lane.
- `GROK_OPENROUTER_API_KEY` for the Grok lane.
- `OPENAI_JUDGE_OPENROUTER_API_KEY` for the OpenAI judge.

Optional fallbacks:

- `GEMINI_FALLBACK_API_KEY` for Gemini Pro.
- `OPENROUTER_FALLBACK_API_KEY` shared by Dhruv, Claude, Grok, and the OpenAI judge.

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
