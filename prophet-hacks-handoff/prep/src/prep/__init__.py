"""Prep package for the Prophet Hacks forecasting track.

Direction-neutral utilities the team can share regardless of which agent
architecture we end up building:

    from prep.data import load_subset_100
    from prep.eval import evaluate
    from prep.baselines.market import predict as market_predict

    events, outcomes = load_subset_100()
    result = evaluate(market_predict, events, outcomes)
    print(result)  # {"brier": 0.187, "ece": 0.069, "n": 175, ...}
"""
