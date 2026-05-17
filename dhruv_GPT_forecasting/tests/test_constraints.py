from dhruv_gpt_forecasting.constraints import enforce_constraints


def test_binary_probabilities_normalize_to_outcomes():
    probs = enforce_constraints({"yes": 0.8, "NO": 0.4}, ["YES", "NO"], "binary")
    assert set(probs) == {"YES", "NO"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["YES"] > probs["NO"]


def test_threshold_ladder_is_monotone_and_normalized():
    probs = enforce_constraints(
        {"Above $450": 0.7, "Above $500": 0.8, "Above $600": 0.5},
        ["Above $450", "Above $500", "Above $600"],
        "threshold_ladder",
    )
    assert probs["Above $450"] >= probs["Above $500"] >= probs["Above $600"]
    assert abs(sum(probs.values()) - 1.0) < 1e-9
