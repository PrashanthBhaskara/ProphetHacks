# Prophet Arena Competition Guide

This folder is for building, testing, and improving our GPT API based agent so it can become the strongest possible forecasting and trading system for Prophet Arena.

The goal is not just to call an LLM and submit outputs. The goal is to build a disciplined agent that can read market data, estimate probabilities, size trades, manage risk, log reasoning, and improve through backtesting and live results.

## Competition Overview

Prophet Arena evaluates AI systems on prediction markets. The platform has two related tracks:

1. Forecasting: submit calibrated probabilities for real-world events.
2. Trading: operate a paper-trading agent that trades prediction market snapshots under fixed rules.

Both tracks test whether an AI agent can reason under uncertainty, use current information, avoid overconfidence, and make decisions that are useful before outcomes are known.

## Forecasting Track

The forecasting benchmark asks the agent to assign probabilities to event outcomes.

The agent receives event objects with fields such as:

- `event_ticker`
- `market_ticker`
- `title`
- `description`
- `category`
- `rules`
- `close_time`
- `outcomes`
- `resolved_outcome`

The agent must return probabilities for the listed outcomes. Each probability must be between `0` and `1`, and the outcome labels must match the event's `outcomes` list.

Forecasting is scored with Brier score. Lower is better. A perfect score is `0.0`.

The main forecasting objective is calibration:

- Do not simply predict the most likely outcome.
- Produce probabilities that match real outcome frequencies over time.
- Avoid unsupported certainty.
- Update beliefs when new information changes the expected outcome.
- Keep rationales tied to the event rules and resolution criteria.

## Trading Track

The trading benchmark is a simulated prediction-market trading competition.

The server owns the experiment state, tick schedule, market snapshots, fills, portfolio, and PnL. Our agent is a client that reads available markets and submits trade intents.

The trading system works on deterministic 15-minute ticks anchored to UTC boundaries: `:00`, `:15`, `:30`, and `:45`.

Each tick is a decision window:

1. Claim the next available tick.
2. Load the candidate market snapshot for that tick.
3. Review current cash, equity, and positions.
4. Generate a trading plan.
5. Submit trade intents.
6. Finalize the participant tick.
7. Complete the tick.

Every participant sees the same pinned market snapshot for a given tick. Fills are deterministic against those snapshot prices.

The expected starting capital is `$10,000` in simulated cash.

## Trade Intent Shape

A trade intent describes what the agent wants to do in a market:

```python
TradeIntentRequest(
    market_id="kalshi:example-market-id",
    action="BUY",
    side="YES",
    shares="10",
    idempotency_key="",
)
```

Core fields:

- `market_id`: the market from the tick candidate set.
- `action`: `BUY` or `SELL`.
- `side`: `YES` or `NO`.
- `shares`: decimal quantity encoded as a string.
- `idempotency_key`: used to make retries safe; the SDK can generate this.

## Trading Rules And Limits

The rules are enforced server-side. The exact authoritative values should always be verified from the installed `ai_prophet_core.ruleset`, but the public developer docs describe the current benchmark shape as:

- 15-minute tick interval.
- 9-minute submission deadline after the tick timestamp.
- `$10,000` initial cash.
- Maximum 20 filled trades per tick.
- Maximum 100 filled trades per rolling 24-hour window.
- Maximum 30 distinct open positions.
- Maximum `$1,000` notional exposure per market.
- Maximum `$10,000` gross exposure.
- No trading fees in the documented benchmark rules.

The agent must treat these limits as hard constraints.

## Execution Semantics

Prediction market prices are interpreted as probabilities with `YES` and `NO` sides.

Documented fill behavior:

- `BUY YES` fills at the market `best_ask`.
- `BUY NO` fills at `1 - best_bid`.
- Positions are tracked by `(market_id, side)`.
- Repeated buys on the same side increase the position and update average entry price.
- Realized PnL comes from market resolution.
- Unrealized PnL is marked to market using the current snapshot.

The agent should avoid submitting conflicting or redundant orders. It should always inspect existing positions before buying or selling.

## Primary Objectives

Our agent should optimize for these objectives, in order:

1. Long-term risk-adjusted return in the trading benchmark.
2. Accurate and calibrated probability estimates in the forecasting benchmark.
3. Robustness under missing data, stale data, API errors, and delayed ticks.
4. Transparent reasoning and audit logs for every forecast and trade.
5. Consistent behavior across repeated runs.

The agent should not chase short-term leaderboard variance by taking reckless exposure.

## Model Objectives

The GPT API agent should be designed as a decision system with separate responsibilities:

- Information gathering: collect the market question, rules, prices, dates, and relevant external evidence.
- Forecasting: estimate a fair probability for each outcome.
- Market comparison: compare fair probability to market price.
- Edge calculation: identify positive expected value opportunities.
- Sizing: convert edge and confidence into position size under risk limits.
- Execution: submit valid trade intents only when the expected value clears a threshold.
- Logging: store forecast, rationale, price, edge, confidence, and final action.
- Review: evaluate mistakes after resolution and use them to improve future prompts and strategy logic.

## Forecasting Strategy Guidelines

For each event, the agent should:

- Read the exact resolution rules before estimating probability.
- Identify the event deadline and whether new information can arrive before close.
- Separate base rates from current evidence.
- Consider the market price as useful information, not as ground truth.
- Produce a fair probability before deciding whether a trade exists.
- Use calibrated language and avoid overconfident probabilities unless evidence is overwhelming.
- Prefer probability ranges internally, then choose a final point estimate.
- Save the reason for the final probability.

## Trading Strategy Guidelines

For each candidate market, the agent should:

- Estimate fair value for `YES` and `NO`.
- Compare fair value to executable prices.
- Trade only when the estimated edge is large enough to overcome model uncertainty.
- Use smaller sizes when uncertainty is high.
- Avoid exhausting daily trade limits on weak signals.
- Avoid concentration in one theme, event type, or correlated outcome.
- Respect per-market and gross exposure limits before every order.
- Prefer no trade over a low-confidence trade.

## Suggested Edge And Sizing Logic

The agent should use conservative sizing until performance is proven.

Example logic:

- No trade if expected edge is below 3 percentage points.
- Small trade if edge is 3 to 6 percentage points and confidence is moderate.
- Medium trade if edge is 6 to 12 percentage points and evidence is strong.
- Large trade only when edge is above 12 percentage points, evidence is strong, and portfolio exposure remains controlled.

This should be implemented as code, not left entirely to free-form GPT judgment.

## Folder Purpose

This `dhruv_GPT_forecasting` folder should contain everything needed to help our GPT API agent become the best possible Prophet Arena forecasting and trading model.

Recommended contents:

- Prompt templates for forecasting and trading.
- Strategy notes and model instructions.
- Backtest outputs.
- Experiment configs.
- Evaluation logs.
- Error analyses.
- Market feature engineering notes.
- Risk management rules.
- API integration notes.
- Leaderboard and performance snapshots.

The folder should act as the agent's development notebook and operating manual.

## Engineering Principles

The agent should be built with these principles:

- Deterministic structure around probabilistic reasoning.
- Explicit inputs and outputs for every decision step.
- Strict validation before any API submission.
- Clear separation between forecasting, sizing, and execution.
- Reproducible configs with versioned strategy names.
- Full logs for every decision.
- Backtests before live competition use.
- Conservative risk settings by default.

## What Success Looks Like

Success means the agent can:

- Produce well-calibrated forecasts.
- Identify mispriced prediction markets.
- Trade only when there is a clear expected edge.
- Stay inside all benchmark risk limits.
- Survive long-running execution without manual babysitting.
- Explain every decision after the fact.
- Improve through measured review rather than random prompt changes.

The best version of this agent should combine GPT's reasoning ability with hard-coded risk controls, structured market math, and continuous evaluation.
