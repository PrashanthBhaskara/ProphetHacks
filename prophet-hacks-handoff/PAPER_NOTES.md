# Paper notes — `LLM-as-a-Prophet` (arXiv 2510.17638v2, Dec 2025)

Distilled findings from the full 53-page version of the paper, ranked by direct relevance to the **trading track**. All page/section references are to the December 2025 v2.

The paper is from the same team running Prophet Hacks. Their analysis covers 23 LLMs on 1,367 resolved Kalshi events.

---

## §B.4 — Brier ≠ Market Returns. Direction beats calibration.

The paper formally proves with an explicit example:

- Ground truth `p* = 0.6`, market price `q = 0.5`
- Model **A** predicts 0.45 → **better Brier (0.74)**, but predicts BELOW market → buys NO → expected return **−$0.10**
- Model **B** predicts 0.9 → **worse Brier (0.67)**, but predicts ABOVE market → buys YES → expected return **+$0.10**

**Implication for our trading agent:** don't optimize calibration alone. What matters is being on the correct side of the market price. A noisier model that's directionally right beats a precise model that's on the wrong side.

---

## §C.1, Table 6 — Best forecaster ≠ best trader

Top 6 by **market return** (out of 23 LLMs evaluated):

| Rank | Model | Avg Return | Brier rank |
|---|---|---|---|
| 1 | **Claude Opus 4.1ᴿ** | 0.982 | 10 |
| 2 | GPT-4o | 0.970 | 16 |
| 3 | Kimi-K2 | 0.966 | 14 |
| 4 | DeepSeek-V3 | 0.963 | 17 |
| 5 | o3ᴿ | 0.959 | 5 |
| 6 | GPT-5ᴿ (High) | 0.943 | 1 (best Brier) |
| — | Market Baseline | 0.899 | — |

**Suggested model choice for trading: Claude Opus 4.1 (best returns) or GPT-4o (#2 + much cheaper).** GPT-5 reasoning is the best forecaster but underperforms as a trader.

---

## §C.2, Table 7 — Every model has NEGATIVE Sharpe ratio

| Rank | Model | Sharpe |
|---|---|---|
| 1 | o3ᴿ | **−0.0131** (least bad) |
| 2 | GPT-5ᴿ | −0.0212 |
| 3 | Gemini 2.5 Proᴿ | −0.0230 |
| 18 | Market Baseline | −0.0897 |
| 24 | Gemini 2.0 Flash (Lite) | −0.1842 |

**Reframe the team goal:** the trading track is about *losing less than other teams*, not making money. Beating Sharpe-0 is essentially impossible on these markets. The realistic bar is beating the market baseline's -0.090.

---

## §B.8, Table 5 — Market is WEAKEST on non-sports

When the eval set is rebalanced away from sports, the LLM's edge over the market grows:

| Subset | GPT-5ᴿ Brier | Market Brier | LLM edge over market |
|---|---|---|---|
| Original (75% Sports) | 0.179 | 0.188 | +0.009 |
| Moderate (50% Sports) | 0.146 | 0.164 | +0.018 |
| Balanced (25% Sports) | **0.128** | **0.149** | **+0.021** |

**Strategy: prioritize Politics, Economics, World, Entertainment for trading.** The sports market is efficient; non-sports has more LLM-exploitable inefficiency.

This *reverses* the earlier guidance ("focus on Sports") — that was forecasting-track thinking, where Sports has the most volume. For trading edge, Sports is the wrong place to look.

---

## §C.3, Table 8 — Probability elicitation tricks

**Use Bi-direction prompting** — ask P(YES) AND P(NO) separately, then combine as `½(p_yes + (1 − p_no))`. Improves calibration (lower ECE) on 4/5 frontier models. Zero extra inference cost vs default.

**Avoid self-consistency / majority-vote rollouts** — querying 10 times for YES/NO and averaging is *worse* for accuracy than direct probability elicitation. Don't burn tokens on this.

Per-model results (Brier / ECE):

| Method | Grok 4 | Gemini 2.5 Flash | Sonnet 4 | GPT-5 | Llama 4 Scout |
|---|---|---|---|---|---|
| Default | 0.186 / 0.117 | 0.166 / 0.036 | 0.173 / 0.046 | 0.165 / 0.020 | 0.196 / 0.153 |
| Variation C (rewrite) | 0.176 / 0.117 | 0.167 / 0.027 | 0.172 / 0.039 | 0.159 / 0.016 | 0.195 / 0.167 |
| **Bi-direction** | 0.180 / 0.101 | 0.164 / **0.031** | **0.165 / 0.028** | **0.158 / 0.023** | 0.203 / 0.140 |
| Self-consistency (10 rollouts) | 0.238 / 0.115 | 0.231 / 0.110 | 0.241 / 0.071 | 0.239 / 0.071 | 0.267 / 0.129 |

---

## §B.5 — Optimal bet sizing under different risk preferences

For a market with model probability `p` and YES contract price `q`, allocate `a_Y` to YES and `a_N = 1 − a_Y` to NO:

- **Risk-neutral (γ=0):** all-in on whichever side has `p > q` (or `p < q` → NO). Maximizes expected return.
- **Log-utility / proper Kelly (γ=1):** `a_Y = p, a_N = 1 − p`. Maximizes expected log-wealth.
- **In between:** closed form `a_Y = q^(1−1/γ) p^(1/γ) / [q^(1−1/γ) p^(1/γ) + (1−q)^(1−1/γ) (1−p)^(1/γ)]`

The `DefaultBettingStrategy` in `ai_prophet_core.betting` sits between these — it sizes by `|p − q|` (proportional to disagreement) and has spread-band filters.

**Suggested:** start with the built-in `DefaultBettingStrategy`. If your forecaster is well-calibrated, move toward proper Kelly (`RebalancingStrategy`). Risk-neutral all-in only if you trust your direction.

---

## §B.6, Theorem B.4 — A calibrated symmetric predictor has equal YES and NO returns

If your predictor is perfectly calibrated AND symmetric (over- and under-shoots market equally), then `E[YES return | bet YES] = E[NO return | bet NO]`. **No need to skew toward YES or NO trades** — the symmetry takes care of it. Practical implication: don't tune your strategy to favor one side.

---

## §4.2.1, Fig 4 — LLMs fail at recalling Politics and Weather events

Recall rate by category (testing on 100 events from before each model's training cutoff):

- Most reliable: **Entertainment** (~95% across all tested models)
- Worst: **Climate/Weather** and **Politics** (frequent mis-recall — high false positive recall rate)
- Economics and Financials: variable; GPT-5 high, smaller models often wrong

**Trading implication:** be especially skeptical of LLM forecasts on Politics and Weather markets. The model may "remember" the wrong outcome from training. Verify with external sources.

---

## §4.2.2, Fig 5 — Sources help more in some categories than others

Brier scores by available context:
- No context: 0.235
- Sources only: 0.191
- Market data only: 0.173
- Both: 0.169

**Market data alone is the biggest single gain** (0.235 → 0.173). Adding sources on top of market data is a marginal further win.

But by category: sources help A LOT in Politics, Economics, World; help LITTLE in Sports, Entertainment. So **route by category** — only burn search tokens where they matter.

---

## §4.2.3 + Fig 6 — LLMs are systematically conservative vs the market

When market price is near 0 or 1, LLMs refuse to go to extremes. Even when the market is at 0.95, models like Llama 4 Scout cluster around 0.5–0.7. GPT-5 and Grok 4 are less conservative; Claude Sonnet 4 is slightly more assertive.

**Trading implication:** when market is at 0.95 and your LLM says 0.7, that's *probably LLM miscalibration, not signal*. **Don't trust LLM disagreement at extremes blindly.** Consider clamping or shrinking LLM probabilities toward the market when the market is at <0.10 or >0.90.

---

## §C.5, Fig 8 — Cross-family ensembling > within-family

Average L2 distance between model predictions:
- GPT-5 (H/M/L variants): ~0.06 (tight cluster)
- GPT-5 vs Claude: 0.17
- Claude vs Llama 4 Maverick: 0.20
- Llama 4 Maverick vs Llama 4 Scout: **0.43** (highest in matrix — same family!)

**For ensembling:** prefer **Claude + GPT + Gemini** over **GPT-5-High + GPT-5-Medium + GPT-5-Low**. Cross-family gives more diversity to average over.

---

## §C.8 + §D.3 — Sources can HURT, especially for crypto strikes

Case study: same Bitcoin price prediction task, two different dates.
- **July 4**: with sources → Brier 0.20 (was 0.40 without). Sources helped.
- **July 9**: with sources → Brier 0.32 (was 0.30 without). Sources hurt.

The difference was **source quality**. The bad-result sources were generic crypto blogs (cryptopredictions.com, priceforecastbot.com, ChatGPT-based forecasts) with widely-ranging numbers. **Quality > quantity for sources.** Curate.

**Trading implication:** when building the search pipeline, **filter out aggregator/blog content** for crypto and other quantitative markets. Direct primary sources (exchanges, official data) beat 10 aggregator predictions.

---

## §E.1.2 — The exact production prediction prompt

The paper publishes the actual prompt the organizers use to elicit forecasts. Key features worth copying:

- System: "You are an AI assistant specialized in analyzing and predicting real-world events. You have deep expertise in predicting the outcome of the event: '{event_title}'"
- Explicit `IMPORTANT CONSTRAINTS` block (probabilities for exact outcomes listed, sum to 1, case-sensitive names)
- Output structured JSON: `{ "rationale": "...", "probabilities": { "outcome1": float, ... } }`
- Sources are passed as ranked list: "The smaller the ranking number, the more you should weight the source"
- Market snapshot passed at the end: "CURRENT ONLINE TRADING DATA: ... (last trading price of each outcome turned out to be yes) from a popular prediction market at the moment of your prediction"
- Explicit hedge: "you should not rely on market data alone to make your prediction"

The full prompt is reproduced in `ai-prophet/packages/cli/ai_prophet/forecast/example_agent.py` — it's the same one.

---

## Things the paper does NOT validate (so don't claim them)

- **Better search wins.** Retrieval is held fixed (GPT-4o searcher) across all 23 models. Whether better retrieval beats reasoning improvements is an open question.
- **Ensembling.** Not tested directly. Cross-model diversity (§C.5) suggests it would help, but the paper doesn't verify.
- **Kelly-fraction sizing under noisy forecasts.** Theory only — no empirical evaluation of how Kelly degrades when probabilities have error.
- **Trade frequency optimization.** The paper uses a single forecast per event per horizon; doesn't study "re-forecast as new info arrives" patterns.

These are all *plausible edges* but they're hypotheses, not findings.
