# Forecasting benchmarks — data-only floor and what to measure

A team-facing benchmark report. The goal: establish the **data-only Brier floor** any LLM-based forecaster has to beat, and document what the eval scale (N=200) actually lets us measure. Everything here is reproducible from this repo.

---

## ⭐ Headline — use this baseline

The recommended data-only predictor is **`RecommendedPredictor`** in `prep/src/prep/baselines/fair_price.py`. Single function call:

```python
from prep.baselines.fair_price import RecommendedPredictor
from prep.data import load_subset_1200

predictor = RecommendedPredictor.fit(load_subset_1200())     # once at startup

# In a tick handler with full candidate set:
preds = predictor.predict_batch(candidate_samples)
# preds[market_ticker] -> p_yes

# Or per-market with explicit context:
p_yes = predictor.predict(event, market_info, candidate_set=candidate_samples)

# Or per-market without context (uses prefix fallback):
p_yes = predictor.predict(event, market_info)
```

What it computes:
```
p_es     = sigmoid(0.61 + 1.17·logit(q) − 0.46·log(N_event))
p_final  = sigmoid(0.5·logit(p_es) + 0.5·logit(q))           # logit-space shrinkage
```

`q` is the bid/ask midpoint. `N_event` is the number of markets in the same event ticker. Shrinkage α=0.5 trades half the in-distribution mean alpha for tighter variance, which is the right tradeoff at the eval scale.

---

## Brier at the eval scale (N=200 calls / 2 weeks)

Bootstrap of 1,500 N=200 resamples from subset_1200 holdout (latest 30% by time). Three modes corresponding to how much sibling context the endpoint sees:

| Mode | What it represents | Brier mean | 95% CI | P(beats market) |
|---|---|---:|---:|---:|
| market (just q) | the do-nothing baseline | 0.185 | [0.165, 0.207] | — |
| **A — no sibling context (prefix fallback)** | endpoint sees one market per call, predictor falls back to typical event size for that ticker prefix | **0.181** | [0.159, 0.205] | **87.9%** |
| **B — sibling-group batch** | endpoint receives all markets of an event in one call | **0.174** | [0.112, 0.227] | **70.0%** |
| **C — full universe cached** | agent caches sibling counts across calls (best case) | **0.171** | [0.148, 0.194] | **99.9%** |

Mode B has the widest CI because event-level resampling clusters by event — one 60-market strike grid can dominate a 200-sample bootstrap.

**Key implementation point:** to get Mode C performance, the agent must **maintain a cache of `event_ticker → n_markets` updated from every candidate set it sees across the 14 days**, then pass it to `predict(...)` via the `n_event` argument.

---

## Noise floor at N=200

Empirical standard error of Brier (across the bootstrap resamples) is **σ ≈ 0.011**. 95% CI half-width is **~0.021**. So:

- **Minimum detectable Brier difference at N=200 ≈ 0.02.** Two methods within that of each other are statistically indistinguishable.
- A method that improves Brier by 0.005 looks great on subset_1200 (which has N=2090) but is invisible at N=200.
- **This means variance reduction matters more than chasing fractional mean improvements.** Shrinkage to a stable anchor is more valuable than the last bit of calibration.

---

## What we tested and learned

### Recalibration variants of `q` (all data-only)

All evaluated on the same time-split holdout (N=2,090, no filtering). Numbers come from `python scripts/verify_recommendations.py`.

| Variant | Full-test Brier | Verdict |
|---|---:|---|
| just `q` (market mid) | 0.185 | baseline |
| `mean_bias_market` (q + constant shift learned on train) | 0.177 | weak improvement |
| `platt_market` (logistic regression on logit(q)) | 0.175 | better, captures bias |
| `beta_calibration` | 0.175 | ties Platt |
| `decile_isotonic` | 0.175 | ties Platt; brittle in P&L (bin-boundary artifacts) |
| `multi_feature` (q + spread + liquidity + category one-hot) | 0.181 | overfits — spread/liquidity are noise |
| `hierarchical_platt` (per-cat Bayesian-shrunk) | 0.175 | ties Platt; worse P&L in walk-forward CV |
| `platt_max_pnl` (grid-search P&L on train) | 0.193 | **worse than market** — overfits the strategy |
| `category_platt` (raw per-cat fits) | 0.209 | overfits hard |
| `event_size_platt` (no shrinkage) | **0.173** | best of the fitted models |
| **`event_size_platt + logit-shrink α=0.5` ⭐** | **0.172** | **recommended — adds `log(N_event)`, lower variance** |

### Why shrinkage α=0.5

Pure `event_size_platt` has Brier 0.173, shrunk to α=0.5 has Brier 0.172 — essentially the same mean. But at N=200, the pure version has **P(beats market) ≈ 94%** while shrunk has **99.9%**. Half the alpha, but you almost can't lose. At N=200 that's the right operating point.

Logit-space shrinkage (averaging log-odds) is marginally better than probability-space averaging and theoretically cleaner.

### Things that didn't help (recorded for posterity)

- **Cross-market sum constraint** (rescale events to expected sum_yes): per-event-size sum estimates too noisy with 1–5 events per size bucket. Real signal (21+-outcome events overprice YES by ~3.7), but not exploitable via naive rescaling. Negative result.
- **Trajectory features from trade history** (VWAP, drift, volatility): looked huge (Brier 0.090) but was contaminated by near-settlement prices. At honest horizons (24h–6h before close), no edge beyond just using current `q`. The market has already absorbed the trajectory.
- **OOD generalization test** against `kalshi_trades_*.parquet` (10K markets, different time window): predictors LOSE to raw market price. Methodology issue: the parquet has trade-execution prices, not bid/ask midpoints (different signal). So this is a "doesn't transfer cleanly to different price types," not "doesn't generalize." Walk-forward CV within subset_1200 (which is apples-to-apples) holds up: positive in all 5 folds, mean +$110.

### Direct trading P&L (separate metric from Brier)

Walk-forward CV on subset_1200, default trading strategy:

| Forecaster | Mean P&L (per ~1k market fold) | Min fold | Notes |
|---|---:|---:|---|
| `event_size_platt` | +$199 | +$40 | best mean, positive every fold |
| `platt_market` | +$110 | +$43 | also positive every fold |
| `multi_feature` | −$19 | −$147 | overfits |

**P&L caveat at eval scale:** these were on ~1k-market folds. At N=200, scaling roughly linearly: mean ≈ +$26, **95% CI [−$40, +$96]** (includes zero). Realistic eval-window P&L is genuinely uncertain.

---

## What this means for the LLM lane(s)

1. **The bar to beat is Brier ≈ 0.17–0.18 depending on call mode.** Anything that doesn't clear this on subset_1200 holdout should *defer to the data baseline*, not get shipped.
2. **The LLM has to add information that isn't in the order book.** Spread, liquidity, category, trajectory — all tested, all add nothing. The only signal in `market_info` that survived is the implicit `n_event` from the candidate set. The LLM's edge has to come from news, schedule data, primary sources, domain reasoning.
3. **Use heavy shrinkage in any LLM ensemble.** At N=200, a forecaster that occasionally produces bad outliers will lose to this baseline even if its mean Brier is better. Ensemble weights should look like `0.5·data_baseline + 0.5·LLM_lane`, not `0.9·LLM + 0.1·anchor`.
4. **Maintain an event-size cache** across calls. Mode A (no context) gets to Brier 0.181; Mode C (full universe known) gets to 0.171. The 0.01 gap is half the noise floor — meaningful at scale.
5. **Don't tune to ≤0.02 Brier differences** in subset_1200 evaluation. Below the noise floor at N=200.

---

## Reproduce everything

```bash
cd prep

# Headline N=200 bootstrap (3 modes)
python scripts/n200_bootstrap.py
python scripts/n200_variance_reduction.py

# Walk-forward CV
python scripts/walk_forward_cv.py --folds 5

# Where the P&L came from (residuals by category, price band, side)
python scripts/residual_analysis.py --strategy default

# Full 17-metric suite × 12 baselines × 3 strategies
python scripts/forecast_benchmarks.py --no-bootstrap

# Time-split full grid (extra detail; note: this script filters bid/ask-invalid markets,
# producing slightly different Brier numbers — verify_recommendations is canonical)
python scripts/forecast_benchmarks.py --no-bootstrap --fair-price-split

# Single source of truth for all headline numbers
python scripts/verify_recommendations.py
```

Anything in this doc that doesn't reproduce from these scripts is a bug — please flag.

---

## File map

```
prep/src/prep/
├── baselines/
│   ├── fair_price.py            ← THIS IS THE ENTRY POINT for the agent
│   └── data_fair_price.py       ← lower-level fitters (Platt, beta, multi-feature, etc.)
├── score.py                     ← 17-metric suite (Brier, log_loss, ECE, BSS, direction-vs-market, …)
prep/scripts/
├── verify_recommendations.py    ← ⭐ single source of truth — reproduces every number in this doc
├── n200_bootstrap.py            ← N=200 sampling-variance analysis
├── n200_variance_reduction.py   ← shrinkage / ensemble comparison at N=200
├── walk_forward_cv.py           ← 5-fold time-ordered CV
├── residual_analysis.py         ← per-category / per-price-band breakdown
├── forecast_benchmarks.py       ← full metric × baseline × strategy grid (filters bid/ask-invalid markets)
├── horizon_analysis.py          ← trajectory-feature horizon test (negative result)
├── sum_constrained.py           ← cross-market sum (negative result)
└── trajectory_features.py       ← trade-trajectory features (negative result at honest horizons)
```
