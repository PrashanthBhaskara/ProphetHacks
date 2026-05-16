"""Market-price baseline.

The single highest-leverage signal per the Prophet Arena paper: just
return the Kalshi-implied probability. In Table 2 of the paper this
scored Brier 0.187, beating 4 of 5 frontier LLMs.

Production agents receive an event dict without prices and must fetch
them via KalshiForecastClient.get_market(ticker). Here we already have
the snapshot, so we accept market_info directly.

Price preference:
1. midpoint of yes_ask and (100 - no_ask) when both available
2. last_price
3. fallback 0.5

All Kalshi prices are in cents (0-100) — divide by 100 for probability.
"""


def _price_to_prob(market_info: dict) -> float:
    yes_ask = market_info.get("yes_ask")
    no_ask = market_info.get("no_ask")
    last_price = market_info.get("last_price")

    if yes_ask is not None and no_ask is not None and yes_ask + no_ask > 0:
        # Average the two implied probabilities to reduce spread bias.
        p = (yes_ask + (100 - no_ask)) / 200
        return p

    if last_price is not None:
        return last_price / 100

    return 0.5


def predict(event: dict, market_info: dict) -> dict:
    p = _price_to_prob(market_info)
    return {"p_yes": p, "rationale": f"market mid-price ({p:.3f})"}
