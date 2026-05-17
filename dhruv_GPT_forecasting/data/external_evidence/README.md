# External Evidence Archive

Store point-in-time social/search/vendor evidence here as JSONL files.

Minimum useful row:

```json
{"source":"reddit","published_at":"2026-03-01T11:30:00Z","collected_at":"2026-03-01T11:45:00Z","market_ticker":"KXEXAMPLE-YES","title":"Example discussion","text":"Short source text"}
```

For clean historical OOS tests, both timestamps must be before the simulated forecast `as_of`. Rows without `market_ticker` or `event_ticker` are matched by title/outcome token overlap.

When live Reddit/GDELT/ESPN pulls are enabled, fetched rows are appended under `live_fetches/<source>/YYYY-MM-DD.jsonl` with `collected_at` set at retrieval time.

Use `python -m dhruv_gpt_forecasting.cli vendor-evidence normalize` to convert WRDS/LSEG exports into the shared JSONL shape. WRDS/LSEG exports should use source-native release timestamps when available.
