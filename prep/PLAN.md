# Prophet Hacks — Victor's track-plan (Grok + Ensemble)

**Hackathon:** May 16–17, 2026. **Submission:** May 17, 5:00 PM (≈30h from now).
**Scoring:** Brier + ECE on resolved Kalshi binary markets, 10-day eval window after submission.
**My slot on the team:** (a) Grok 4 as the 4th frontier model for ensemble diversity, (b) build the aggregator that combines all four teammates' `p_yes` into one calibrated team submission.

---

## Guiding principles (from the Prophet Arena paper + our HF-subset results)

1. **ECE has 5× more cross-model variance than Brier.** Calibration is where most of the win is. Ensembling + isotonic recalibration is the high-leverage move.
2. **Reasoning-mode models top the rankings.** Always prefer the reasoning variant of every model when both are available.
3. **Markets beat LLMs in the last 3 hours before close, LLMs beat markets at longer horizons.** A time-aware shrinkage toward market price is worth ~0.01–0.02 Brier.
4. **Retrieval is uncontrolled in the paper.** Live web search is a real edge for the team — Grok 4 has it natively.
5. **Per-category Brier varies wildly** (Sports 0.13, Politics 0.18). The aggregator should be allowed per-category weights or per-category recalibration.

---

## Phase 0 — Setup (30 min, $0)

- [x] Scaffold `grok_zero_shot` baseline → `prep/src/prep/baselines/grok_zero_shot.py`
- [x] Register `grok` in `prep/scripts/run.py`
- [x] Create `.env`, `.env.example`, `.gitignore`
- [ ] **Rotate the leaked xAI + OpenRouter keys; paste new ones into `prep/.env`**
- [ ] Add OpenRouter baseline so we can ad-hoc test any model from one key (DeepSeek R1, Qwen, …) → `prep/src/prep/baselines/openrouter_zero_shot.py`
- [ ] Agree the **team interface** with the other three (see § Coordination below). This is the most important Phase-0 deliverable.

**Why:** the team can't ship an ensemble if every teammate's output has a different schema. Lock the interface tonight.

---

## Phase 1 — Grok zero-shot baseline (45 min, ~$1–3)

Run the existing scaffold against the 100-event HF subset:

```bash
source .venv-pmxt/bin/activate && set -a && source prep/.env && set +a
python prep/scripts/run.py grok --limit 5            # smoke
python prep/scripts/run.py grok --workers 8          # full subset
python prep/scripts/run.py grok --category Politics --workers 4
python prep/scripts/run.py grok --category Sports   --workers 4
```

**Compare against the anchors we already have:**
| baseline      | Brier | ECE   |
|---------------|-------|-------|
| always_half   | 0.250 | 0.113 |
| market price  | 0.065 | 0.071 |
| claude_zero_shot (TBD) | – | – |
| **grok_zero_shot** | **fill in** | **fill in** |

**Decision gate:** if Grok zero-shot is within 0.02 Brier of the market baseline, proceed to Phase 2. If it's way off (e.g. Brier > 0.15), debug prompt/parsing before moving on.

---

## Phase 2 — Grok variants worth trying (1.5h, ~$5)

Three small experiments, all cheap, all run on the same 100-event subset so they're directly comparable:

1. **Reasoning mode.** Set `GROK_MODEL=grok-4.3-reasoning` (or whichever the console actually exposes — confirm from <https://console.x.ai>). Reasoning models top the paper's rankings. Default base model in `grok_zero_shot.py` is now `grok-4.3`.
2. **Live web search.** xAI exposes `search_parameters={"mode": "on"}` in the chat-completions request. This is Grok's biggest differentiator vs. the other three labs.
3. **Self-ensemble.** Same model, `n=5` samples at `temperature=0.7`, average the `p_yes`. Costs 5× but often improves ECE more than a model upgrade.

Record Brier/ECE for each into a `prep/data/grok_variants.csv`. Pick the best as our Grok contribution to the ensemble.

---

## Phase 3 — The aggregator (3h, $0 — pure compute)

This is the highest-EV piece of the project and nobody else is doing it.

**Inputs:** four files in `prep/data/predictions/{sonnet,gpt5,gemini,grok}.jsonl`, each row:
```json
{"market_ticker": "KX...", "p_yes": 0.42, "rationale": "..."}
```

**Aggregator chain** (`prep/src/prep/aggregator.py`):
1. **Logit-pool average:** `p̄ = σ( mean( logit(p_i) ) )`. Simple, well-behaved at extremes, beats arithmetic mean.
2. **Optional model weights:** start uniform; tune on the HF subset if there's a clear weak model.
3. **Isotonic recalibration:** fit `IsotonicRegression` on a holdout split (50/50 of the HF subset). Saves a `calibration.pkl`.
4. **Market shrinkage:** `p_final = (1 - α) · p_ens + α · p_market`, with α scaled by time-to-close (small at +30d, ~0.5 at +3h). Use the snapshot's `last_price` when available.
5. **Per-category weights** (stretch goal): fit a separate isotonic per category — likely useful for Politics, Sports.

**Build it against stub teammate predictions first** so it's ready the instant they ship real ones. Stubs: write `always_half`, `market`, `claude_zero_shot` outputs into the four slots — that gives us a real run end-to-end before any teammate is ready.

---

## Phase 4 — Live Kalshi evaluation harness (2h, $0)

We already have `pmxt` working for live Kalshi reads. Add:

- `prep/scripts/snapshot.py` already exists — confirm it pulls current open markets.
- New script `prep/scripts/run_live.py`:
  1. Pull all currently-open binary markets closing in the eval window.
  2. For each, fetch market price snapshot.
  3. Call each model's `predict()` and write into the per-model jsonl files.
  4. Run the aggregator → produce `submission.jsonl`.

**This is also the harness that will run during the 10-day eval window**, so it has to be robust to API errors (retry + skip, never crash the run).

---

## Phase 5 — Calibration + sanity (2h, ~$2)

- Reliability diagram on the 100-event subset for each model + the ensemble. Save as PNG to `prep/data/`. Easy "wow" for any demo.
- Bootstrap CI on Brier (1000 resamples) so we can claim "ensemble Brier 0.06 ± 0.01".
- **Adversarial check:** look at the 10 events with worst per-prediction loss. Pattern? (e.g. ensemble systematically over-confident on sports moneylines).

---

## Phase 6 — Submission (1h)

- Final aggregator run on whatever set the judges hand us.
- Submit `submission.jsonl`.
- Write the 1-page method summary (judges love these): "logit-pool 4-model ensemble, isotonic-recalibrated, market-shrinkage near close, per-category weights for Sports/Politics."

---

## Coordination with teammates

**Send this to the team now.** All four of us should output predictions to a shared schema so the aggregator just consumes files:

```jsonl
{"market_ticker": "KX...-T", "p_yes": 0.42, "rationale": "...", "model": "claude-sonnet-4-6", "ts": "2026-05-17T..."}
```

Proposal:
- One folder per model: `prep/data/predictions/{sonnet,gpt5,gemini,grok}.jsonl`.
- Whoever runs predictions appends rows. Aggregator reads the latest row per `market_ticker` per model.
- **Each model must hit the same set of `market_ticker`s** — otherwise the ensemble has missing data. Easy fix: I publish the canonical list as `prep/data/markets_to_predict.jsonl` after Phase 4, everyone reads from that.
- **Settle calibration philosophy:** do we want each teammate to run their own self-ensemble (`n=5`), or should they all submit a single sample and we ensemble across models only? My vote: each model self-ensembles to reduce variance, then we average across models — but it 5×'s their cost. Decide tonight.

---

## Budget guesstimate

| Item                              | $          |
|-----------------------------------|------------|
| Grok zero-shot, full subset       | ~$2        |
| Grok variants (reasoning, search, self-ens) | ~$5  |
| Live eval, ~50 markets × 5 samples × 4 models (mine) | ~$15 |
| OpenRouter wildcard model spelunking | ~$3     |
| Buffer for re-runs / mistakes     | ~$10       |
| **Total my-side spend**           | **~$35**   |

OpenRouter sees the same prices as direct providers + ~5% markup; using direct xAI for Grok saves a couple of bucks.

---

## What I want from you before I burn the first credit

1. **Did you rotate both keys yet?** (yes/no)
2. Are we OK with the team interface above? (`predictions/{model}.jsonl`)
3. For Phase 2, do you want me to test all three Grok variants, or pick one? (default: all three — only ~$5)
4. Anything you want to add/remove/reprioritize?
