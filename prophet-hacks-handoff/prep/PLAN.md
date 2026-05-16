# PLAN — how we win Prophet Hacks (updated Sat ~1 PM)

**Submission:** May 17, 5 PM (≈28 h from Sat 1 PM). **Tracks:** Forecasting + Trading.
**Working repo:** `PrashanthBhaskara/ProphetHacks` (private; we're all collaborators).

---

## 0 · State of the world right now

### Open PRs (notveiker)
| PR | Branch | Status | What |
|---|---|---|---|
| **#3** | `vk/multi-outcome-pivot` | ⏳ open, current | Schema pivot + FastAPI `/predict` server (this PR's PLAN.md update lives here) |
| #2 | `vk/bidirection-fallback-fixes` (fork) | 🪦 stale, **close it** | Bi-direction was for binary; obsolete after PR #3 |
| #1 | `vk/grok-lane-and-plan` (fork) | 🪦 stale, **close + re-port** | Grok lane needs to ride on top of PR #3 as a new PR |

### What landed on main since I started

| SHA | Author | What |
|---|---|---|
| `9d07871` | Victor (other Claude session) | `CONSTRAINTS.md` — trading-track quick reference. **Read it before doing trading work.** |
| `99dd16e` | Victor (other Claude session) | `/candidates/asof` curl one-liner addition |
| `a678e94` | Dhruv | ~22 MB of personal coursework PDFs + `.DS_Store` files at repo root. **Repo is now 71 MB tree size.** Worth asking Dhruv to move to their own repo or a `notes/` branch. |
| `ed1d706` | Dhruv | Leaked FRED + WRDS (incl. password) + LSEG keys to `.env.example`. Repo was already private — exposure contained to collaborators, but **Dhruv should change his WRDS password tonight.** |
| `d24e875` | Dhruv | Leaked 4 OR keys to `.env.example`. Same scope. Keys are sponsored grants — can't rotate; live with the residual risk. |

### What I'm holding open as TODO (not started)
- **Grok lane port** — needs to land *after* PR #3 merges (or stacked on top)
- **Outcome-label normalization** — case-fold matcher in `agent_server.predict_endpoint` against `event.outcomes`
- **Adapter retry/backoff** — H4 from earlier; still needed for overnight runs
- **Endpoint deployment** — Render / Fly.io / ngrok decision pending
- **Cleanup PR**: gitignore root `.DS_Store`; ideally move Dhruv's PDFs out

---

## 1 · The contract (recap, was unconfirmed until today)

From **https://prophetarena.co/developer** + org Q&A:

### Forecast track wire contract
- **Endpoint:** agent exposes `POST /predict`. Eval server hits it once per event.
- **Input:** `Event` with `outcomes: list[str]` + standard metadata. Newer dataset uses `task_id` / `predict_by` / `context` aliases — our `packet_from_arena_event()` handles both.
- **Output:** `{"probabilities": [{"market": <outcome>, "probability": <p>}, ...]}`. Probabilities in `[0,1]`; scorer normalizes.
- **Timeout:** 10 minutes per event. Parallelism *within* the request (run all lanes in parallel), not across requests.
- **Scoring:** multi-class Brier per event, `sum((p_i - outcome_i)^2)`. Perfect = 0.0. Random baseline depends on outcome count.

### Trading track rules (from [CONSTRAINTS.md](../CONSTRAINTS.md))
- 15-min ticks; agent has 9 min to submit each tick or HTTP 409
- $10,000 starting cash, max 30 positions, $1k per market, 10 trades per tick
- Top-of-book L1 only — `best_bid` / `best_ask` / `volume_24h`, no depth, no sizes
- All-or-nothing fills at the quoted price; no partial fills, no slippage
- YES and NO on the same market not allowed simultaneously
- Buy YES at `best_ask`; buy NO at `1 - best_bid`

### 🚨 Budget model — build vs eval
**$50 OpenRouter grant per teammate is BUILD-PHASE ONLY.** Eval phase = teams self-fund their own keys. So:
- Spend the $200 grant pool aggressively during prep (today + tomorrow morning)
- Before eval window opens, switch endpoint to a self-funded key
- Pick lane budget assuming eval-phase tokens cost real money

### Dataset
- `github.com/ai-prophet/ai-prophet-datasets` — `sample-sports`, `sample-resolved`, `sample-entertainment`, `sample-economics` published. **`hackathon-day` (org default) not yet published as of Sat 12:50 PM CT.** Watch Discord.

---

## 2 · What's already built (PR #3)

| Component | Status |
|---|---|
| `schemas.py` — `MarketPacket.outcomes`, `ForecastValues.probabilities`, `SupervisorForecast.calibrated_probabilities` | ✅ done |
| `forecasters/base.py` — prompt asks per-outcome; parses `probabilities` dict (with binary fallback) | ✅ done |
| `forecasters/mock.py` — returns distributions (binary, 2-outcome named, N-outcome) | ✅ done |
| `ensemble.py` — logit-pool per outcome, market anchor when binary Kalshi, uniform anchor otherwise | ✅ done |
| `calibration.py` — binary YES/NO shrinkage when Kalshi quote present, uniform-anchor shrinkage for multi | ✅ done |
| `packets.py` — `packet_from_arena_event()` for the new Event shape | ✅ done |
| `scripts/agent_server.py` — FastAPI `POST /predict` + `GET /health` + CLI-compatible `predict()` | ✅ done |
| Mock backtest still passes (3 markets, eval_pack) | ✅ verified |
| `predict()` returns valid distributions for 2-outcome named + 4-outcome events | ✅ verified |
| `p_yes` back-compat preserved for trading code | ✅ verified |

---

## 3 · The 28-hour critical path (revised)

| When | Phase | Who | What | $ |
|---|---|---|---|---|
| **NOW** | A0 | Victor | Open Grok-lane port PR (stacked on #3 or independent). Without it, ensemble has 3 lanes not 4 when #3 merges. | 0 |
| NOW | A1 | Victor | Outcome-label normalization PR (case-fold matcher) | 0 |
| NOW | A2 | Anyone | Cleanup PR: gitignore `.DS_Store`, propose moving Dhruv's PDFs out of repo | 0 |
| Sat 2–4 PM | B (per-lane smoke) | Each teammate | Enable ONLY your lane, run `prophet forecast predict --local scripts.agent_server` against `sample-resolved`, report per-event multi-class Brier | ~$1 each |
| Sat 4 PM | sync 1 | All 4 | Prune lanes that don't beat uniform. Tune temperature / reasoning_effort per lane. | 0 |
| Sat 4–6 PM | C (full ensemble dry-run) | Whoever has spend | All surviving lanes enabled, 50 events stratified across sample-* datasets. Verify ensemble Brier < best single lane. | ~$5 |
| Sat 6 PM | sync 2 | All 4 | Decide final ensemble weights. Lock config. | 0 |
| **Sat 7–10 PM** | D (live retrieval) | Victor + Dhruv | Wire Grok 4.3 native search (`live_search` param) + GPT-5.1 web tool. Re-run sample-resolved to measure lift. | ~$5 |
| Sat 10 PM | sleep prep | Victor | First-pass deployment of `agent_server.py` to Render/Fly + smoke test with curl | 0 |
| Sat 11 PM | sleep | All | 8 hours. Submission isn't won at 4 AM. | 0 |
| **Sun 7–9 AM** | E (deploy hardening) | Victor | Full deployment + register endpoint with `prophet forecast register --team-name "<us>" --endpoint-url <url>`. Switch from grant key to self-funded key. | 0 |
| Sun 9 AM–12 PM | F (live QA) | All 4 | Hammer the live endpoint with sample-resolved events. p50/p99 latency. Watch tail. | ~$10 |
| Sun 12–3 PM | G (trading track) | Prashanth | `prophet trade eval run -m <provider:model> --slug ...` — schedule 96-tick eval (~24h coverage). Use risk gate from `trading/risk.py`. | ~$10 |
| **Sun 4:30 PM** | H (submit) | Victor | Final endpoint registration confirmation + trading run summary. 30-min buffer. | 0 |
| **Sun 5 PM** | DEADLINE | | | |

**Total spend: ~$30–40** (within the $200 build-phase grant pool, before eval-phase self-funded costs kick in).

---

## 4 · Five things that win it

1. **Endpoint actually responds.** Hosted, reachable, ≤10 min per event, valid JSON. Most teams will fumble deployment. Deploy Sat night, hammer Sun morning.
2. **Multi-outcome distributions are coherent** — probabilities roughly sum to 1 for mutually-exclusive outcomes. Logit-pool + normalize handles this; verify on 4-outcome cases.
3. **Live retrieval on Grok 4.3** — paper held retrieval fixed across models; we don't have to. For current-events markets this is genuine new information none of the other lanes have.
4. **Calibration shrinkage when uncertain.** Multi-class Brier punishes overconfidence quadratically. Shrink toward uniform on weak evidence.
5. **Lane diversity in the ensemble.** 4 different labs → uncorrelated errors → ensemble Brier strictly better than mean-of-lanes by `Var(predictions)`. Grok specifically is the most diverse (different data, different reasoning).

---

## 5 · Five things that will lose it

1. **Endpoint not deployed by Sun noon.** Local-only `--local` testing isn't enough. Phase E deadline = Sun 9 AM.
2. **Outcome-label mismatch.** Model returns `{"Yes": 0.7}` but event has `outcomes=["YES","NO"]` → silently dropped → score destroyed. **Phase A1 (case-fold matcher) is non-negotiable.**
3. **Adapter failure cascades.** One 5xx on Grok = whole `/predict` returns 502 = that event scored as zero info. Need retry-with-backoff before live.
4. **All 4 lanes agree → ensemble adds no signal.** Mitigate with prompt diversity, different reasoning_effort, possibly different prompt templates per lane.
5. **Deadline panic at 4:55 PM.** Phase F deadline = Sun 4 PM, not 5 PM. Hard cutoff for last commit.

---

## 6 · Per-teammate ownership

| Teammate | Lane | Other |
|---|---|---|
| **Victor** | Grok 4.3 (OR + direct xAI) | `agent_server.py`, deployment, calibration sweep, label normalization, Grok lane port |
| **Prashanth** | Gemini 3.1 Pro (OR) | Harness retries (H4), Gemini smoke, trading-track wrap-up |
| **Franklin** | Claude Sonnet 4.6 (OR) | Claude smoke, reliability diagrams + writeup |
| **Dhruv** | GPT-5.1 (OR) | GPT smoke, GPT live-search wiring |

---

## 7 · Open questions for the team

1. **Endpoint hosting:** Render free tier (10-min request timeout OK), Fly.io, or ngrok-tunneled local box left running on someone's laptop?
2. **Self-funded eval key:** whose card eats the eval-phase model spend? (Estimated <$30 for the eval window if we're careful.)
3. **Dhruv's coursework PDFs** at repo root — move out, or leave (~22MB ongoing clone cost)?
4. **Trading track aggression:** default `kelly_fraction=0.10` is conservative; bump to 0.15 if calibration looks good in Phase C?
