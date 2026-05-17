# Data Findings

Generated during implementation on 2026-05-16.

## Existing `prep/data`

`prophet-hacks-handoff/prep/data/eval_pack_live_clean.jsonl` has 13,165 resolved markets with at least two snapshots. The market-price baseline remains hard to beat:

| Source | Mode | N | Brier | ECE |
|---|---:|---:|---:|---:|
| live_clean | market | 13,165 | 0.0968 | 0.0406 |
| live_clean | stat, first 5,000 | 5,000 | 0.0803 | 0.0565 |

The stat lane can improve Brier on some slices, but it can worsen calibration. Use it as a confidence/shrinkage lane, not as an unconstrained replacement for market price.

## `Kalshitopvolmarkets`

The top-volume folder is now first-class backtest data. The loader finds 5,688 finalized, candle-backed binary markets with at least two pre-close 15-minute snapshots.

| Category | N |
|---|---:|
| Sports | 5,440 |
| Other | 155 |
| Entertainment | 50 |
| Politics | 17 |
| Climate and Weather | 11 |
| Economics | 9 |
| Crypto | 6 |

Full-source baselines:

| Source | Mode | N | Brier | ECE |
|---|---:|---:|---:|---:|
| topvol | market | 5,688 | 0.0683 | 0.0157 |
| topvol | stat | 5,688 | 0.0597 | 0.1183 |

Interpretation:

- The top-volume candle data is useful because the logit/Kalman stat lane improves Brier, especially on sports-heavy markets.
- The stat lane is over-adjusted from a calibration perspective, so LLM/supervisor prompts should see its probability, uncertainty, and reason codes, but final output should remain market-anchored.
- The trading gate currently produces no trades on this source because edges fail buffers or spreads are too wide. That is intentional until calibration is improved.

## New Context Data

The new context pull is wired in as side-channel evidence for binary tradable markets, not as directly tradable labels.

| Source | Scope |
|---|---:|
| `NonBinaryMarkets` context groups | 475 |
| `NonBinaryMarkets` component markets | 1,316 |
| `NonBinaryMarkets` target-to-context links | 1,245 |
| `NonBinaryMarkets` 60-minute OHLCV files | 1,316 |
| `prep/data/kalshi_polymarket/map.csv` mapped tickers | 149 |
| `prep/data/kalshi_polymarket/rejected.csv` rejected candidates | 107 |

Prompt safety rules implemented:

- Context rows strip `result`, `status`, `settlement_ts`, and final metadata quote fields before reaching the LLM.
- Context candle summaries only use component-market candles with timestamps at or before the target market `as_of`.
- If no `NonBinaryMarkets` exact link exists, `Kalshitopvolmarkets` same-event markets are used as a fallback context source.
- The Polymarket file is currently a mapping layer, not a price feed. It can tell the LLM that a comparable cross-venue question exists; real-time Polymarket price/volume should be added later as structured external evidence.

Full top-volume dry-run with context:

| Source | Mode | N | Avg context records | Context sources |
|---|---:|---:|---:|---|
| topvol | dryrun | 5,688 | 1.0 | 1,245 `kalshi_nonbinary_context`, 4,443 `kalshi_topvol_same_event` |

The dry-run score matches the stat lane because no API calls are made. The value of this context should be measured on gated LLM subsets before allowing it to drive larger fair-value deviations.

## Point-in-Time Model Experiments

The first top-volume tests were too close to settlement because they used the last pre-close candle. The experiment harness now supports true point-in-time truncation with `--horizon-hours`.

Best Brier results from resolved PIT runs:

| Source | Horizon | N | Best variant | Brier | Market Brier |
|---|---:|---:|---|---:|---:|
| topvol | 24h | 4,518 | `momentum_revert_5pct` | 0.2158 | 0.2164 |
| topvol | 6h | 5,203 | `momentum_revert_5pct` | 0.2168 | 0.2174 |
| topvol | 1h | 5,670 | `stat_default` | 0.1648 | 0.1654 |
| live_clean | 24h | 9,964 | `stat_cap_6pp` | 0.1243 | 0.1264 |
| eval_pack | 24h | 17,883 | `stat_cap_6pp` | 0.1319 | 0.1328 |

Interpretation:

- The best general-purpose stat input for the GPT lane is the wider 6pp capped Kalman/AR lane. It wins on the broader live/eval PIT sets and is competitive on top-volume horizons.
- Small mean reversion helps top-volume sports at 24h/6h, but the gains are narrow and less stable than the capped stat lane across sources.
- Context-normalized deterministic probabilities did not improve Brier on top-volume resolved data. Related markets should be included in the LLM prompt as audit evidence and structure, not as a direct probability override yet.
- PIT histograms remain concentrated because many markets are already high-confidence binary outcomes. Use Brier/log loss for ranking model variants, and use ECE/PIT only to decide shrinkage and trade sizing.

Production config now uses `max_market_deviation=0.06` and `default_market_deviation=0.03`. Keep the market price as the supervisor's anchor; this stat lane should be evidence, not permission to trade.

## Chronological OOS On 1-Minute Top-Volume Data

The longer OOS harness uses `Kalshitopvolmarkets/ohlcv/period_1m`, truncates each resolved market to a fixed point-in-time horizon, sorts samples chronologically, trains/calibrates on the first 70%, and evaluates on the final 30%. Results are saved under `dhruv_GPT_forecasting/logs/`.

| Horizon | OOS N | Market Brier | Best variant | Best Brier | Improvement |
|---|---:|---:|---|---:|---:|
| 24h | 1,357 | 0.2186 | `stat_cap_6pp:platt_a=-0.03_b=0.70` | 0.2161 | +0.0025 |
| 6h | 1,564 | 0.2201 | `recent_revert_10pct:platt_a=-0.00_b=0.69` | 0.2170 | +0.0032 |
| 1h | 1,704 | 0.1677 | `stat_cap_6pp:platt_a=0.03_b=0.93` | 0.1665 | +0.0012 |
| 15m, last 3 months | 1,111 | 0.0663 | `momentum_follow_10pct:platt_a=0.09_b=1.21` | 0.0645 | +0.0018 |

Trading simulation with the conservative production risk gate:

| Horizon | Trades | PnL | ROI on stake | Notes |
|---|---:|---:|---:|---|
| 24h | 35 | +7.01 | +26.6% | Mostly BUY_NO; positive but small stake and sample. |
| 6h | 3 | -2.02 | -100.0% | Forecast Brier improves, but trade gate examples lose. |
| 1h | 0 | 0.00 | 0.0% | Market is tighter/near-settled; no executable edge after buffers. |

Contract segments that currently look most useful:

- Best broad slice is still Sports, but only because it dominates the dataset. OOS Brier improvement is small, so size should stay capped.
- 24h market-price buckets `10-25`, `25-40`, `60-75`, and `75-90` improve versus market; `75-90` is the most executable in this split because the model often prefers BUY_NO against overconfident favorites.
- 24h/6h large move buckets show signal: `down_15pp_plus` and `up_15pp_plus` have positive Brier improvement and positive raw unit-edge ROI. Treat this as late price correction/mean-reversion evidence.
- WTA match markets were the strongest series-level OOS segment in the 24h run: `KXWTAMATCH` had +0.0109 Brier improvement and positive simulated trade PnL. This needs more data before promotion.

Segments to avoid or shrink:

- ATP match raw trades were negative in the 24h run despite positive Brier improvement, so the trading gate should not treat "forecast improvement" as enough.
- 6h and 1h strict-gate trades are not proven. Forecast accuracy improves, but executable edge mostly disappears after spread/uncertainty buffers.
- Structural same-event context alone did not justify GPT spend. It is useful audit context, not enough to open a GPT-5.4 call by itself.

## GPT-5.4 Gated Smoke Test

The cheap lane now uses `openai/gpt-5.4` through OpenRouter. A 20-case chronological OOS, gated-only 24h batch was run after the deterministic tests:

| Batch | N | Market Brier | Stat Brier | GPT-5.4 Brier | Cost estimate |
|---|---:|---:|---:|---:|---:|
| topvol 24h OOS gated | 20 | 0.1935 | 0.1942 | 0.1937 | $0.4645 |

Operational read:

- GPT-5.4 valid JSON parsing worked and all 20 responses remained conservative `NO_TRADE`.
- Average absolute GPT move from market was only 5.2 bps. With only Kalshi structural context, GPT mostly copied the market/stat prior.
- Because this did not beat market on the smoke batch, the gate was tightened: `kalshi_topvol_same_event` context can be included in prompts but no longer triggers a GPT call by itself.
- Updated shadow call rates after tightening are much lower: 24h 40/1,357 = 2.9%, 6h 20/1,564 = 1.3%, 1h 245/1,704 = 14.4%. The 1h rate is higher because stat disagreement and near-close conditions interact more often.

Current recommendation:

- Use the chronological OOS best stat/calibration as the default forecast lane.
- Call GPT-5.4 only for high-spread/stat-disagreement/cross-venue/real-time evidence cases, not for ordinary sports markets.
- Supervisor GPT should stay off until a gated GPT subset beats market on at least a few hundred OOS examples or real-time external evidence is added.
- The next GPT-5.4 evaluation should use the new PIT external-evidence layer. Historical Reddit/X/search records must be archived with both `published_at` and `collected_at` before the simulated `as_of`; otherwise they should be treated as exploratory timestamp-only evidence, not clean OOS.
- Use `dhruv_gpt_forecasting.evidence_archiver` to build the evidence archive for the same 30 GPT-5.4 test packets. X full-archive pulls can be bounded to `as_of`; Reddit public-search historical backfills should remain off for strict OOS unless we have prior live captures.

## Last-Three-Month 15-Minute OOS Run

The latest resolved top-volume pull contains 3,701 binary markets closing on or after 2026-02-16 with a usable 1-minute candle at least 15 minutes before close. The chronological 70/30 split gives 2,590 train and 1,111 OOS test markets.

| Run | N test | Market Brier | Model Brier | Improvement |
|---|---:|---:|---:|---:|
| Full deterministic/stat OOS | 1,111 | 0.0663 | 0.0645 | +0.0018 |
| Runtime promoted near-close stat | 1,111 | 0.0663 | 0.0645 | +0.0018 |

Best model:

- `momentum_follow_10pct:platt_a=0.09_b=1.21`
- Formula: take the 15-minute market prior, add 10% of the point-in-time move from first observed candle to forecast time, cap the adjustment at 4pp, then apply Platt calibration.
- This is now enabled in `configs/default.json` as `stat.near_close_brier_enabled=true` for horizons up to 0.5 hours.

Gating and cost read:

| Gate mode | Cheap GPT calls on OOS test | Call rate |
|---|---:|---:|
| Without related context as trigger | 52 / 1,111 | 4.7% |
| With related context evidence | 169 / 1,111 | 15.2% |

GPT smoke tests:

| Batch | N | Market Brier | Stat Brier | GPT Brier | Cost estimate |
|---|---:|---:|---:|---:|---:|
| `openai/gpt-5-nano`, 15m gated | 5 | 0.0797 | 0.0996 | 0.0996 | $0.0000, all fell back |
| `openai/gpt-5.4`, 15m gated | 5 | 0.0797 | 0.0996 | 0.0838 | $0.1295 |
| `openai/gpt-5.4`, 15m gated | 30 | 0.1773 | 0.1817 | 0.1782 | $0.7924 |

Interpretation:

- GPT-5 Nano is not usable with the current OpenRouter JSON-mode prompt path; it returned empty message content and fell back to stat-only forecasts.
- GPT-5.4 is operational and improves over the old Kalman/AR stat lane on the 30-case smoke batch, but it did not beat the market there. It should remain gated and should consume the newly promoted near-close Brier stat prior.
- At 15 minutes before close, most Brier improvement comes from calibrated price trajectory behavior, not from LLM reasoning. Live GPT should be reserved for cases with fresh external evidence, stale/ambiguous market structure, or cross-market disagreement.

## Prophet Arena Brier-Only Refactor

The forecasting-track path is now separate from the Kalshi trading lane. `dhruv_gpt_forecasting.arena_agent.predict(event)` is the intended local module for:

```bash
prophet forecast predict --events events.json --local dhruv_gpt_forecasting.arena_agent
```

Important behavior:

- Every event returns exactly one probability for each provided outcome label.
- Probabilities are normalized before return and use a light `0.001-0.999` internal clamp.
- No trade recommendation or no-trade option appears in the Arena prompt or response parser.
- If GPT-5.4 or live data fails, the agent returns deterministic priors from category base rates, entity historical rates, nearest-neighbor analogs, and any matched live probability evidence.
- Active live pulls are opt-in with `ARENA_ENABLE_LIVE_DATA=1`; local tests should use `ARENA_OFFLINE=1`.
