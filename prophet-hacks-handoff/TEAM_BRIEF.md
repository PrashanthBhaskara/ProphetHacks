# Prophet Hacks — team brief

A pre-hackathon summary of what we're building, how we're scored, and what's already set up. Before May 16.

---

## The hackathon

- **Event:** [Prophet Hacks](https://prophethacks.com) — AI Forecasting Hackathon
- **Dates:** May 16–17, 2026 (30-hour sprint, submission deadline Sun 5pm)
- **Locations:** UChicago JCL 390 / Fleet AI office SF / Remote
- **Hosts:** Prophet Arena Team (Sigma Lab / UChicago) + Fleet AI
- **Track we're entering:** **Forecasting** (not Trading)
- **Prize:** sponsored trip to Korea to present at an ICML workshop (winner) + $500 (runner-up)
- **Eval window:** the submitted agent runs autonomously for **10 days after** the deadline. Winners announced May 28.

## What we're actually building

An AI agent that estimates the probability `p_yes ∈ [0.01, 0.99]` that a given binary Kalshi prediction market resolves YES.

**Agent contract** (authoritative, from [prophetarena.co/developer](https://prophetarena.co/developer)):

```
POST {event_json}  →  {"p_yes": 0.72, "rationale": "..."}
```

Where `event_json` contains: `event_ticker`, `market_ticker`, `title`, `subtitle`, `description`, `category`, `rules`, `close_time`. **Note: the event JSON does NOT include the market price** — but we can fetch it from Kalshi ourselves with no auth (see "Hidden gems" below).

**Two submission modes** (pick either or both — server takes the latest):
- **Manual:** run `prophet forecast predict` locally, then `prophet forecast submit`
- **Endpoint:** register a URL, organizers POST to it daily during the 10-day eval window

**Scoring:** Brier score, `(1/N) · Σ(p_yes − actual)²`. Lower is better. Random = 0.25, perfect = 0.0.

## Setup before the hackathon

Each of us should do this **before May 16**:

1. `pip install ai-prophet` (the official CLI + SDK; both also on PyPI)
2. Get API keys (some take 24h to activate):
   - **Anthropic** (required for the example agent)
   - **OpenRouter** (lets us swap models freely)
   - **Perplexity** (default search backend) — or Brave / Exa / Tavily as alternates
   - **Kalshi** account (optional for now; free, useful to browse the markets in their UI)
3. Join the [Discord](https://discord.gg/aTsY7979zP) — official announcements drop here first
4. Read the [developer docs](https://prophetarena.co/developer) (10 min)
5. Try the example agent locally:
   ```bash
   prophet forecast retrieve --deadline "2026-05-25T23:59:59Z" --output events.json
   prophet forecast predict --events events.json --local ai_prophet.forecast.example_agent
   ```

## Useful context (not the rules, but informs strategy)

The hosts published a [research paper on the Prophet Arena benchmark](https://arxiv.org/abs/2510.17638) (Oct 2025, 23 LLMs evaluated on 1,367 events). It's *not* hackathon-specific but the eval methodology is the same. Key findings worth knowing:

- **Brier scores cluster tight (0.18–0.22) across all models.** Random is 0.25. The differences are small in absolute terms.
- **ECE (calibration error) varies 5× more.** Calibration is where the real headroom is — a well-calibrated 0.65 beats an overconfident 0.95.
- **Reasoning-mode models (GPT-5ᴿ, o3, Claude Sonnet 4ᴿ) consistently top the rankings.** Use the reasoning variants.
- **The Kalshi market price alone scored Brier 0.187 — beating 4 of 5 frontier LLMs.** Anchor on the market price; deviate only with evidence.
- **Markets beat LLMs in the last ~3 hours before resolution.** LLMs beat markets at longer horizons. Strategy should probably differ by time-to-close.
- **The paper held retrieval fixed across all models.** So they cannot say "better data wins" — but in the hackathon we control retrieval, so this is plausibly an edge axis they didn't measure.

The three named bottlenecks from the paper: (1) inaccurate event recall, (2) misunderstanding data sources, (3) slower info aggregation than markets near resolution. Each is a concrete failure mode to defend against.

## Hidden gems most teams won't find

1. **The market price.** Not in the event JSON, but `ai_prophet_core.forecast.kalshi_client.KalshiForecastClient.get_market(ticker)` fetches it with no auth. Embedding the market price as a feature in any agent is probably the single highest-leverage architectural decision.
2. **A public 100-event eval set:** [`prophetarena/Prophet-Arena-Subset-100`](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-100) on HuggingFace. 100 events / ~1,000 binary markets with ground truth AND Kalshi market snapshots at submission time.
3. **`mini-prophet` repo on GitHub** (`ai-prophet/mini-prophet`) — a more sophisticated baseline than the official `example_agent`. Has a planning phase that decomposes the question before searching. Worth deciding whether we fork it or build from scratch.

## Data collection running in the background (do this NOW)

I set up a Kalshi polling pipeline so we accumulate **fresh,
contamination-free** eval data before May 16. The HF 100-event subset is
useful but small and frozen; markets that resolve between today and
hackathon day are bigger, post-training-cutoff, and let us see price
evolution.

In `prep/`:

```bash
python scripts/snapshot.py --window-days 7   # ~3min, captures all open markets
python scripts/resolve.py                    # after markets close, pulls outcomes
```

I've already taken the first snapshot — captured **8,074 binary markets
across 672 events** closing in the next 7 days. If we run snapshot every
~8 hours from now through May 16, we'll have a few thousand resolved
markets with multi-timestamp price tracks to evaluate any agent against.

**Useful for one of us to set up:** a cron / launchd entry that runs
`python scripts/snapshot.py` every 6–8 hours through May 16. Optional but
high-leverage — more data = more confident architectural decisions on
Day 1.

## What's already set up in the shared repo

I built a small **direction-neutral sandbox** at `prep/` — no agent architecture committed, just shared infrastructure. Any predict function we write can plug in and get scored.

```
prep/
├── README.md                       # full setup + how to add a predictor
├── requirements.txt
├── reference/                      # vendored from HF: the 100-event subset + their scripts
├── src/prep/
│   ├── data.py                     # load_subset_100() → list[Sample]
│   ├── score.py                    # brier(), ece()
│   ├── eval.py                     # evaluate(predict_fn, samples) → metrics
│   └── baselines/
│       ├── always_half.py          # sanity check (Brier 0.25)
│       ├── market.py               # use Kalshi price as p_yes
│       └── claude_zero_shot.py     # what the official example_agent does
├── scripts/
│   ├── run.py                      # CLI: python scripts/run.py market --source hf|local
│   ├── snapshot.py                 # poll Kalshi for open markets
│   └── resolve.py                  # attach outcomes once markets close
└── data/snapshots/                 # accumulated snapshots (first one already here)
```

**To use it:**
```bash
cd prep
pip install -r requirements.txt
python scripts/run.py always_half        # sanity
python scripts/run.py market             # Kalshi price as p_yes
python scripts/run.py claude --workers 8 # needs ANTHROPIC_API_KEY
```

**Numbers already validated locally** (on the 100-event HF subset, 1,061 binary markets):

| Baseline | Brier | ECE | Note |
|---|---|---|---|
| always_half | 0.2500 | 0.1126 | exactly random — sanity check |
| market price | 0.0654 | 0.0707 | ECE matches paper's 0.069 → scorer is right |

**Caveat:** the 100-event subset is *easier* than the live evaluation (snapshots are at first-submission time when many sports markets are already near-resolved). The paper's full 1,367-event eval got market-baseline Brier 0.187, not 0.065. Treat the subset as a **regression suite**, not a leaderboard. If a change makes baselines worse on it, that's a real red flag.

**Per-category Brier varies a lot.** Market price baseline is much better on some categories than others:

| Category | N markets | Market Brier |
|---|---|---|
| Sports | 339 | 0.1257 |
| Politics | 41 | 0.1843 |

A category-aware router (e.g. trust market more for liquid sports, trust LLM more for politics) is plausibly worth more than another round of prompt tuning.

## Where I'd suggest the edges live (sorted by likely ROI)

These are *suggestions for Day 1 discussion*, not commitments:

1. **Market-price anchoring** — biggest free win, paper-validated baseline
2. **Reasoning models with high thinking budget** — paper-validated
3. **Calibration layer** (clamp tighter than [0.01, 0.99]; shrink toward market) — where ECE headroom lives
4. **Category-specific specialists** — e.g. for sports, hit an odds API; for weather, NOAA — likely high ROI given 76% sports weighting
5. **LLM ensembling on calibration** — unvalidated but plausible; ECE varies most across models
6. **Better retrieval pipeline** — paper held this fixed; probably matters but we don't have evidence

## Team split idea for the day

4 of us, 30 hours. Specialization > parallel cloning:

- **Research pipeline** — search backends (Perplexity/Exa/etc.), caching, source extraction
- **Reasoning & calibration** — prompting, model choice, calibration layer
- **Harness + ops** — eval loop, leaderboard polling, the zip + run command we submit
- **Category specialists** — deterministic shortcuts for sports/crypto/weather/etc.

## Open questions to think about

- Do we fork `mini-prophet` or build from scratch with `ai-prophet-core`?
- Single model or ensemble?
- How much do we lean on the market price vs. fight it?
- Do we run our agent once per event, or re-forecast as close_time approaches?
- During the 10-day eval window, who maintains the running agent if it breaks?

---

**TL;DR:** Read [the developer docs](https://prophetarena.co/developer) (10 min) and skim [the paper](https://arxiv.org/abs/2510.17638). Get API keys this week. The biggest non-obvious insight: **the market price alone beats most LLMs, and it's one Kalshi API call away.** I have a working scorer in `prep/` you can drop any predictor into — we figure out the actual agent together on Day 1.
