"""Smoke-test pmxt against Kalshi: list markets, peek at an event, pull OHLCV."""
import pmxt

api = pmxt.Kalshi()

print("=== fetch_markets(query='election', limit=3) ===")
markets = api.fetch_markets(query="election", limit=3)
for m in markets:
    print(f"  id={getattr(m, 'id', '?')}  title={getattr(m, 'title', getattr(m, 'name', '?'))!r}")
    outcomes = getattr(m, "outcomes", None) or []
    for o in outcomes:
        price = getattr(o, "price", None)
        name = getattr(o, "name", getattr(o, "label", "?"))
        print(f"     outcome={name!r}  price={price}")

print("\n=== fetch_events(limit=3) ===")
try:
    events = api.fetch_events(limit=3)
    for e in events:
        print(f"  id={getattr(e, 'id', '?')}  title={getattr(e, 'title', getattr(e, 'name', '?'))!r}")
except Exception as exc:
    print(f"  fetch_events failed: {type(exc).__name__}: {exc}")

print("\n=== fetch_ohlcv on first market's YES outcome ===")
if markets:
    m = markets[0]
    yes = getattr(m, "yes", None)
    if yes is None:
        outcomes = getattr(m, "outcomes", None) or []
        yes = outcomes[0] if outcomes else None
    if yes is not None:
        try:
            candles = api.fetch_ohlcv(yes, resolution="1h", limit=5)
            print(f"  got {len(candles)} candles")
            for c in candles[:3]:
                print(f"   {c}")
        except Exception as exc:
            print(f"  fetch_ohlcv failed: {type(exc).__name__}: {exc}")
    else:
        print("  no outcome to test OHLCV with")
