# Kalshi Top Volume Markets

This folder is for the January 1 through May 9, 2026 Kalshi data pull.

The collector ranks markets independently for each 7-day window anchored at
January 1. The practical default ranks markets closing in that week by Kalshi's
final `volume_fp`. There is also an exact trade-volume mode, but the public
trades feed can require thousands of pages per active day, so it is not the
default for this large January-May pull.

It then filters out markets that are not usable for the Prophet Arena trading
track:

- non-Kalshi data sources
- non-binary markets
- multivariate/combo markets
- markets missing title, rules, open time, or close time
- markets with no weekly trade volume

The default candle granularity is 1 minute, because Kalshi only exposes
1-minute, 1-hour, and 1-day OHLCV candles and 1 minute is the closest match to a
15-minute trading tick backtest.

## Layout

- `collect_kalshi_top_volume.py`: resumable collector
- `rankings/`: weekly aggregate volume rankings and selected top-300 lists
- `markets/`: selected market metadata by week
- `ohlcv/`: compressed OHLCV files by period and week
- `logs/errors.jsonl`: per-market download errors
- `manifest.json`: run configuration and source notes

## Run

```bash
python3 collect_kalshi_top_volume.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-09 \
  --top-n 300 \
  --period-interval 1 \
  --ranking-source market_close_volume
```

The run is resumable. Existing candle files are skipped unless `--force` is set.

If the weekly ranking files already exist, download or resume only the candle
files with:

```bash
python3 collect_kalshi_top_volume.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-09 \
  --top-n 300 \
  --period-interval 1 \
  --candles-from-rankings
```

## Sparse LLM Test Points

To keep LLM calls sparse, sample `x` random market-time points per week:

```bash
python3 sample_llm_test_points.py --calls-per-week 1 --seed 42
```

With this dataset's 19 weekly windows, `--calls-per-week 1` creates 19 planned
LLM calls. Increase `--calls-per-week` to sweep larger budgets. The sampler
writes JSONL model packets and a CSV summary under `samples/`.
