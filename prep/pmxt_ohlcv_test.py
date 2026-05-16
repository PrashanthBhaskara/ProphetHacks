"""Try OHLCV on a high-volume Kalshi market."""
import pmxt

api = pmxt.Kalshi()

# Try a few queries likely to have active trading
for query in ["bitcoin", "fed", "trump", "recession"]:
    markets = api.fetch_markets(query=query, limit=3)
    print(f"\n--- query={query!r}: {len(markets)} markets ---")
    for m in markets:
        title = getattr(m, "title", getattr(m, "name", "?"))
        yes = getattr(m, "yes", None) or (getattr(m, "outcomes", []) or [None])[0]
        if yes is None:
            print(f"  {title!r}: no outcome")
            continue
        try:
            candles = api.fetch_ohlcv(yes, resolution="1h", limit=5)
            print(f"  {title!r}: {len(candles)} candles")
            for c in candles[:2]:
                print(f"    {c}")
            if candles:
                break
        except Exception as exc:
            print(f"  {title!r}: OHLCV error: {type(exc).__name__}: {exc}")
    else:
        continue
    break
