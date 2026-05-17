from dhruv_gpt_forecasting.market_linker import linked_distribution_for_outcomes


def test_linked_distribution_maps_component_labels_to_named_outcomes():
    distribution = [
        {"label": "Oklahoma City", "probability": 0.68},
        {"label": "Los Angeles L", "probability": 0.32},
    ]

    probs = linked_distribution_for_outcomes(["Los Angeles L", "Oklahoma City"], distribution)

    assert probs is not None
    assert probs["Oklahoma City"] > probs["Los Angeles L"]
    assert abs(sum(probs.values()) - 1.0) < 1e-9
