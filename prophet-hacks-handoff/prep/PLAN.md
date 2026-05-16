# PLAN — how we win Prophet Hacks (revised after dev-docs discovery)

**Submission:** May 17, 5 PM (≈24 h from Sat 1 PM). **Tracks:** Forecasting + Trading.
**Working repo:** `PrashanthBhaskara/ProphetHacks` (private; we're all collaborators).

---

## 0 · What we know now (confirmed, not assumed)

From **https://prophetarena.co/developer** (confirmed live), plus org Q&A:

### Wire contract
- **Endpoint:** `POST /predict`
- **Input body:** `Event` JSON with `outcomes: list[str]` and `resolved_outcome: null`. Other fields: `event_ticker`, `market_ticker`, `title`, `subtitle?`, `description?`, `category`, `rules?`, `close_time`. Dataset newer-format uses `task_id`, `predict_by`, `context` aliases.
- **Output:** `{"probabilities": [{"market": <outcome_label>, "probability": <float>}, ...]}`
- Each `market` MUST be one of the event's outcomes. Probabilities in `[0,1]`. Don't need to sum to 1 — scorer normalizes.
- **One event per request. 10 minutes per event.** Parallelism is *within* a request (run all lanes in parallel), not across.

### Scoring
- Multi-class Brier per event: `sum((p_i - outcome_i)^2)` across submitted outcome probabilities.
- Perfect = 0.0. Random baseline depends on number of outcomes (2-way uniform → 0.5; 4-way uniform → 0.75).

### Dataset
- `github.com/ai-prophet/ai-prophet-datasets` — currently published: `sample-sports`, `sample-resolved`, `sample-entertainment`, `sample-economics`. `hackathon-day` (org default) not yet published as of Sat 12:50 PM CT.
- Sample tasks are mostly binary (2 outcomes) but the schema supports multi-outcome and judges will test that.

---

## 1 · What we've already built (this branch)

`vk/multi-outcome-pivot` lands the contract:

| Component | Status |
|---|---|
| `schemas.py` — `MarketPacket.outcomes`, `ForecastValues.probabilities`, `SupervisorForecast.calibrated_probabilities` | ✅ done |
| `forecasters/base.py` — prompt asks per-outcome, parses `probabilities` dict (with binary fallback) | ✅ done |
| `forecasters/mock.py` — returns distributions (binary + multi) | ✅ done |
| `ensemble.py` — logit-pool per outcome, market anchor when binary Kalshi, uniform anchor otherwise | ✅ done |
| `calibration.py` — binary YES/NO shrinkage when Kalshi quote present, uniform-anchor shrinkage for multi | ✅ done |
| `packets.py` — `packet_from_arena_event()` for the new Event shape | ✅ done |
| `scripts/agent_server.py` — FastAPI app: `POST /predict`, `GET /health`, plus `predict(event: dict)` for `prophet forecast predict --local` | ✅ done |
| Mock backtest still passes (3 markets, eval_pack) | ✅ verified |
| `predict_endpoint` returns valid distributions for 2-outcome and 4-outcome events | ✅ verified |

**Back-compat:** `p_yes` stays as a derived property everywhere it was a field. Trading-track code that reads `.calibrated_p_yes` keeps working for binary Kalshi markets.

---

## 2 · The 24-hour critical path

| When | Phase | Who | What | $ |
|---|---|---|---|---|
| **NOW (Sat 1 PM)** | A | All 4 | Land this PR. Land Prashanth's H1–H4 harness fixes (parallelism, cache, retries) on top. Distribute new `.env`. | 0 |
| Sat 2–4 PM | B (per-lane smoke) | Each teammate | Enable ONLY your lane, run `prophet forecast predict --local scripts.agent_server` against `ai-prophet-datasets/sample-resolved`, report multi-class Brier per event. | ~$1 each |
| Sat 4 PM | sync 1 | All 4 | Prune any lane that's worse than uniform-random baseline. Tune temperature / reasoning_effort. | 0 |
| Sat 4–6 PM | C (full ensemble dry-run) | Whoever has spend | All lanes enabled, 50 events from sample-resolved (mix of categories). Confirm ensemble Brier < best single lane. | ~$5 |
| Sat 6 PM | sync 2 | All 4 | Decide final ensemble weights. Lock config. | 0 |
| **Sat 7–10 PM** | D (live retrieval) | Victor + Dhruv | Wire Grok 4.3 native search + GPT-5.1 web tool into their respective adapters. Re-run sample-resolved to measure lift. | ~$5 |
| Sat 11 PM | sleep | All | 8 hours. Submission isn't won at 4 AM. | 0 |
| **Sun 7–10 AM** | E (deploy) | Victor | Deploy `agent_server.py` on a public URL (Render / Fly.io / ngrok tunnel). Verify with `curl POST /predict`. Register endpoint with `prophet forecast register --team-name "<us>" --endpoint-url <url>`. | $0 |
| Sun 10 AM–2 PM | F (final QA) | All 4 | Hammer the live endpoint with sample-resolved events. Verify response time < 10 min consistently. Watch latency tail. | ~$10 |
| Sun 2–4 PM | G (trading track) | Prashanth | Re-purpose ensemble output for trading-track submission. The risk gate already exists. | ~$5 |
| **Sun 4:30 PM** | H (submit) | Victor | Final endpoint registration + trading submission. 30-min buffer. | 0 |
| **Sun 5 PM** | DEADLINE | | | |

**Total spend: ~$30.** Plenty of budget remains for re-runs.

---

## 3 · Five things that win it

1. **The endpoint actually responds.** Hosted, reachable, under 10 min per event, returns valid JSON. Most teams will fumble deployment. We deploy Saturday night, hammer it Sunday morning.
2. **Multi-outcome distributions are coherent.** Models should return probabilities that roughly sum to 1 for mutually-exclusive outcomes. Our logit-pool + normalize handles this; verify on 4-outcome test cases.
3. **Retrieval on Grok 4.3.** Live search gives us current-events edge. Multi-class Brier rewards both calibration AND directional accuracy — knowing "Sweden definitely won't win Eurovision" is worth a lot.
4. **Calibration shrinkage when uncertain.** Multi-outcome scoring punishes overconfidence quadratically. Shrink toward uniform when evidence is weak. Our `calibration.py` does this.
5. **Lane diversity in the ensemble.** 4 different labs (Anthropic / OpenAI / Google / xAI) — different priors, different training data. Each contributes signal the others miss.

---

## 4 · Five things that will lose it

1. **Endpoint not deployed by Sunday noon.** Local-only `--local` testing isn't enough; judges hit a URL. Deploy by Sun 10 AM at latest, ideally Sat night.
2. **Model returns invalid JSON for one event → 502 on that event.** Need robust JSON parsing (we have it) + retry on parse failure (need to add).
3. **All 4 lanes return the same answer.** Ensemble adds no signal. Mitigate with prompt diversity, different reasoning_effort, possibly different prompt templates per lane.
4. **Outcome label mismatch.** Model returns `{"Yes": 0.7}` but event has `outcomes=["YES","NO"]`. Case mismatch → outcome dropped → score destroyed. Add label normalization (case-insensitive match against `event.outcomes`).
5. **Deadline panic at 4:55 PM.** Phase F deadline = Sun 4 PM, not 5 PM. 1-hour buffer minimum.

---

## 5 · Open items / what I'm flagging

- **`prophet forecast retrieve` CLI** needs the ai-prophet pip package installed. We should add it to requirements.txt and verify the CLI works locally.
- **Endpoint hosting choice:** Render free tier, Fly.io free tier, or ngrok-tunneled local? Need 10-min request timeout, so most free PaaS will work but check.
- **Outcome-label normalization** — TODO before live submission. Add a step in `agent_server.predict_endpoint` that case-fold-matches model output to `event.outcomes`.
- **Retry on adapter failure** — PLAN's old H4. Still needed.
- **Hackathon-day dataset** — not published yet; org may push it close to eval window. Watch the Discord.

---

## 6 · Per-teammate ownership

| Teammate | Lane | Other |
|---|---|---|
| **Victor** | Grok 4.3 (xAI + OpenRouter) | `agent_server.py`, deployment, calibration sweep |
| **Prashanth** | Gemini 3.1 Pro (OpenRouter) | Harness fixes H1–H4, trading-track wrap-up |
| **Franklin** | Claude Sonnet 4.6 (OpenRouter) | Reliability diagrams + writeup |
| **Dhruv** | GPT-5.1 (OpenRouter) | GPT live-search wiring |
