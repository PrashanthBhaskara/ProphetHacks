# Non-Binary / Context Market Groups

This folder stores event-level context groups for the January 1 through
May 9, 2026 Kalshi pull.

Kalshi markets are primarily binary contracts. For LLM forecasting, the useful
"non-binary" context is usually the set of related component markets around a
single event: winner markets, spreads, totals, ladder markets, and multivariate
combo markets when available. These groups can inform the target binary market
without becoming direct trading targets.

The default collector is intentionally smaller than the 1-minute target-market
dataset:

- top 25 context groups per week
- up to 8 component markets per group
- 1-hour OHLCV candles
- same January 1 through May 9, 2026 window

## Layout

- `collect_kalshi_context_markets.py`: context group collector
- `rankings/`: weekly group rankings and selected top groups
- `markets/`: selected context group metadata and component market metadata
- `ohlcv/`: compressed OHLCV files for selected component markets
- `indexes/target_to_context_links.jsonl`: exact same-event links from the
  binary target dataset to context groups
- `manifest.json`: run configuration and source notes

## Run

```bash
python3 collect_kalshi_context_markets.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-09 \
  --top-groups-per-week 25 \
  --max-markets-per-group 8 \
  --period-interval 60
```

The default ranking source is the already downloaded
`Kalshitopvolmarkets/markets/historical_markets_2026-01-01_2026-05-09.jsonl`
cache. This keeps the run bounded. Use `--include-live-supplement` to query
`/markets` by close-window as well, but that is slower and unnecessary for this
completed historical window. For weeks not present in the historical cache, the
collector supplements from the already downloaded
`Kalshitopvolmarkets/markets/{week}_selected_markets.jsonl` files. Use
`--force-rebuild-cache` only if you specifically want to rebuild a fresh
historical cache without the original binary/MVE exclusions.

## OOS Prompt Use

At forecast time, only pass context rows with candle timestamps at or before the
target `as_of`. Do not pass result, settlement, final volume, final rank, or
post-`as_of` OHLCV values into the LLM prompt.
