from dhruv_gpt_forecasting.data_loaders import load_topvol_samples
from dhruv_gpt_forecasting.experiments import point_in_time_samples, random_point_in_time_samples
from dhruv_gpt_forecasting.features import parse_dt


def test_point_in_time_samples_truncate_before_horizon():
    samples = load_topvol_samples(limit=20)
    pit = point_in_time_samples(samples, horizon_hours=24)
    assert pit
    for sample in pit:
        close_dt = parse_dt(sample.event["close_time"])
        last_dt = parse_dt(sample.snapshots[-1]["t"])
        assert close_dt is not None
        assert last_dt is not None
        assert (close_dt - last_dt).total_seconds() >= 24 * 3600


def test_random_point_in_time_samples_are_reproducible_and_truncated():
    samples = load_topvol_samples(limit=20, candle_stride_minutes=15, min_snapshots=5)
    first = random_point_in_time_samples(
        samples,
        n_events=5,
        seed=123,
        min_horizon_minutes=5,
        min_history_snapshots=5,
        decision_budget_minutes=5,
    )
    second = random_point_in_time_samples(
        samples,
        n_events=5,
        seed=123,
        min_horizon_minutes=5,
        min_history_snapshots=5,
        decision_budget_minutes=5,
    )
    assert [row.market_info["forecast_request_time"] for row in first] == [
        row.market_info["forecast_request_time"] for row in second
    ]
    assert first
    for sample in first:
        close_dt = parse_dt(sample.event["close_time"])
        request_dt = parse_dt(sample.market_info["forecast_request_time"])
        assert close_dt is not None
        assert request_dt is not None
        assert parse_dt(sample.snapshots[-1]["t"]) == request_dt
        assert len(sample.snapshots) >= 5
        assert (close_dt - request_dt).total_seconds() >= 5 * 60
        assert sample.market_info["decision_budget_minutes"] == 5
