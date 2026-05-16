# Prophet Hacks — team brief (trading track)

What we're building, how we're scored, and what's already set up.

---

## The hackathon

- **Event:** [Prophet Hacks](https://prophethacks.com)
- **Dates:** May 16–17, 2026 — submission deadline Sun 5pm CDT
- **Track:** **Trading** (P&L, not forecast accuracy)
- **Prize:** Korea trip + ICML workshop slot (winner), $500 (runner-up)
- **Eval window:** submitted agent runs autonomously for **10 days after** the deadline

## What we're building

A trading agent that places real (paper-money) bets on binary Kalshi markets and is scored on **realized P&L** over the 10-day eval window.

**Critical insight:** a trading agent is **forecasting + betting strategy**. The forecasting work (estimate `p_yes`) is step 1, then a `BettingStrategy.evaluate(market_id, p_yes, yes_ask, no_ask) -> BetSignal` converts the probability into a trade. The hackathon ships two built-in strategies (`DefaultBettingStrategy` and `RebalancingStrategy`) — both in `ai_prophet_core.betting`. You can plug in your own.

**Three things change vs. forecasting track:**

1. **Score is P&L, not Brier.** A perfectly calibrated 0.50 makes $0 if you never trade.
2. **The market is the adversary, not the anchor.** You only profit when you *disagree* with the market price AND you're right.
3. **Bid-ask spread is a real cost.** If `yes_ask = 0.55`, your true probability must be **>0.55** to profit on a YES buy — not >0.50. The spread is the threshold.

## Agent surface in `ai-prophet`

```bash
prophet trade eval run \
  -m anthropic:claude-sonnet-4-20250514 \
  --slug our-team-v1 \
  --strategy default                # or 'rebalancing'
  --starting-cash 10000
  --max-ticks 96                    # 96 trading ticks over 10 days
```

The trade pipeline (in `ai-prophet/packages/cli/ai_prophet/trade/agent/stages/`) is four stages: **search → forecast → action → review**. Forecast and action are *separate* LLM calls — the team explicitly designed this to let you measure "forecasting skill" independently from "betting/risk skill."

## Setup before the hackathon

1. `pip install ai-prophet` (the official CLI + SDK)
2. API keys:
   - **Anthropic** + **OpenRouter** (let us swap models freely)
   - **Perplexity** (default search backend) or Brave/Exa/Tavily
3. Join the [Discord](https://discord.gg/aTsY7979zP)
4. Skim `prophetarena.co/developer` (the public docs are forecast-track focused; trade-specific runtime docs come via the Discord at kickoff)

## Useful context (informs strategy, not the rules)

The hosts published a [paper on the Prophet Arena benchmark](https://arxiv.org/abs/2510.17638). It's a forecasting-track analysis, but several findings transfer:

- **Reasoning models (GPT-5ᴿ, o3, Claude Sonnet 4ᴿ) consistently top the rankings.** Use the reasoning variants.
- **Markets beat LLMs in the last ~3 hours before resolution.** For trading: don't try to compete with market consensus near close — your edge lives at longer horizons.
- **The paper holds retrieval fixed.** "Better data sources" is an unvalidated bet by them but is plausibly an edge axis we control.

## Hidden gems most teams won't find

1. **The Kalshi market price IS the prediction-market consensus.** No auth needed to fetch it via `ai_prophet_core.forecast.kalshi_client.get_market(ticker)`. For trading, the *spread* and *volume* also matter — both come back in that same call.
2. **Public 100-event eval set:** [`prophetarena/Prophet-Arena-Subset-100`](https://huggingface.co/datasets/prophetarena/Prophet-Arena-Subset-100) on HuggingFace.
3. **`mini-prophet` repo** (`ai-prophet/mini-prophet`) has a more sophisticated forecasting agent with a planning phase. Worth deciding whether we fork it or build from scratch.

## What's already set up in `prep/`

I've been polling Kalshi for the past 5 days. We have **42,194 resolved binary markets** with bid/ask snapshots and outcomes — directly usable as a trading backtest dataset.

```
prep/
├── README.md
├── requirements.txt
├── reference/                  # vendored HF 100-event subset + reference scripts
├── src/prep/
│   ├── data.py                 # loaders (HF subset + our local snapshots)
│   ├── score.py                # Brier + ECE
│   ├── eval.py                 # forecasting harness
│   ├── trade.py                # trading backtest harness ← NEW
│   ├── kalshi.py               # anonymous Kalshi client
│   └── baselines/              # forecasting baselines (still useful for step-1)
├── scripts/
│   ├── run.py                  # forecasting scorer (Brier, ECE)
│   ├── run_trade.py            # trading scorer (P&L, win rate, sharpe) ← NEW
│   ├── snapshot.py             # Kalshi polling cron
│   ├── resolve.py              # outcome attachment
│   ├── consolidate.py          # builds eval_pack.jsonl + summary.md
│   └── cron_resolve.sh
└── data/
    ├── eval_pack.jsonl         # 42,194 markets with prices + trajectory + outcome
    ├── eval_pack_latest.csv    # spreadsheet form
    ├── summary.md              # AUTO-GENERATED counts + baselines (forecast + trade)
    └── outcomes.jsonl          # raw outcomes log
```

## Numbers you'll see in `data/summary.md`

**Forecasting baselines (still relevant as step-1 of the trading agent):**

| Category | N | market Brier (target to beat in forecasting) |
|---|---|---|
| Sports | 14,335 | 0.1385 |
| Weather | 410 | 0.1565 |
| Other | 5,562 | 0.0516 |
| Crypto | 21,577 | 0.0169 (near-deterministic, ignore) |

**Trading baselines (run on the same data):**

| Strategy | P&L | Notes |
|---|---|---|
| `never_trade` | $0 | sanity control |
| `market_anchor + default` | +$201 (2%) | finds crossed markets only |
| `noisy_market + default` (all categories) | +$1,220 (12%) | LOOKS great, but it's just exploiting Crypto's NO-bias |
| `noisy_market + default` (**Sports only**) | **-$313 (-3%)** | random forecasts LOSE on balanced markets |

**The honest interpretation:** the aggregate "12% return" from random noise is fake — it's structural NO-bias in crypto strikes. **On Sports (the only meaningful category for trading), a forecast with no real edge LOSES money to bid-ask spread.** Your agent must beat `noisy_market + default` on Sports to be worth submitting.

## Where the edges plausibly live, ranked

**(Re-ordered after deep dive on the paper's appendices — see `PAPER_NOTES.md`)**

1. **Use the right model for the trading objective.** The paper shows the **best forecaster is NOT the best trader**: GPT-5ᴿ wins Brier but only ranks 6th in market returns. **Claude Opus 4.1ᴿ (rank 1)** and **GPT-4o (rank 2)** dominate trading returns. (§C.1, Table 6)
2. **Skip Crypto AND prioritize non-Sports.** Both are surprises. Crypto is near-deterministic (no edge). But also: the **market is most efficient on Sports** — the LLM's edge over the market actually *grows* when sports is downweighted. **Politics, Economics, World, Entertainment** are where the inefficiencies live. (§B.8, Table 5)
3. **Optimize for DIRECTION, not calibration.** Paper proves with explicit example: a "worse-Brier" model that picks the right side of the market beats a "better-Brier" model on the wrong side. For trading, optimize for "are we on the right side of `q`", not for matching the truth probability. (§B.4)
4. **Bi-direction prompting (free win).** Ask P(YES) AND P(NO) separately, then average. Improves calibration on 4/5 models with zero extra cost. (§C.3, Table 8)
5. **Don't use self-consistency (10-rollout majority vote).** Strictly worse than direct probability elicitation on every tested model. Saves tokens too. (§C.3)
6. **Cross-family ensembling beats within-family.** Average Claude + GPT + Gemini, not GPT-5-High + GPT-5-Medium. Diversity is what makes ensembling useful. (§C.5)
7. **Don't trust LLM disagreement at extremes.** When market is at 0.95 and LLM says 0.7, that's likely LLM miscalibration (they're systematically conservative). Shrink toward market when `q ∈ [0, 0.10] ∪ [0.90, 1]`. (§4.2.3, Fig 6)
8. **Filter source quality, especially for crypto-like quant markets.** Aggregator blogs (cryptopredictions.com, priceforecastbot.com, ChatGPT-based forecasts) introduce noise that worsens predictions. Curate to primary sources. (§D.3)
9. **Avoid Kelly with noisy forecasts.** Our local backtest confirms: Kelly is *worse* than the default strategy when probabilities have error (Kelly over-sizes bad bets). Default strategy first; move to Kelly only after calibrating.
10. **Don't trade in the last 3 hours.** Market dominates LLMs there. Submit forecasts but skip bets. (§3.2.1)

## The single most sobering finding (and what it means)

**All 24 models in the paper had negative Sharpe ratios.** Best was o3ᴿ at −0.013; market baseline at −0.090. Trading these markets is a structurally losing proposition on a risk-adjusted basis. **Reframe the team goal: it's "lose less than other teams" not "make money".** The winning trading agent will likely have a *less negative* P&L than competitors, not a positive one.

## Team split (4 people, 30 hours)

- **Forecaster** — owns step 1: LLM + search → calibrated `p_yes`. Reuses any forecasting work.
- **Strategy / risk** — owns step 2: which strategy (default/rebalancing/Kelly/custom), how to size, spread thresholds, category filters. This is where trading-specific work lives.
- **Harness + ops** — owns the submission infrastructure, the live agent loop, the leaderboard polling, the trade history dashboard.
- **Category specialists** — deterministic helpers per category (sports odds APIs, weather forecasts, etc.).

## Open questions to bring to Day 1

- Default vs Rebalancing strategy — they have different risk profiles. Backtest both with our data.
- Single model or ensemble? (For forecasting; trading inherits from forecast quality.)
- Categories to **exclude** entirely vs trade selectively.
- Do we re-trade as new info arrives, or one trade per market and hold?
- Backup forecaster in case primary model API errors during the 10-day eval window.

---

**TL;DR:** A trading agent = forecasting agent + betting strategy. The forecasting prep work is *step 1* of the trading agent, not wasted. The new piece is the betting strategy + sizing, where the highest-leverage decisions are **category filtering** (skip crypto entirely) and **avoiding Kelly on noisy forecasts**. Everything's in `prep/` — `cat prep/data/summary.md` to see all baselines, run `python prep/scripts/run_trade.py <forecast> --strategy <strategy>` to backtest anything.
