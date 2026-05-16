"""Always-0.5 baseline. Expected Brier = 0.25 on any binary set."""


def predict(event: dict) -> dict:
    return {"p_yes": 0.5, "rationale": "uniform prior"}
