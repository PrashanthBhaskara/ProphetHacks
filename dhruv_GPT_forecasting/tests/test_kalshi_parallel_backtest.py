from dhruv_gpt_forecasting.data_loaders import BacktestSample
from dhruv_gpt_forecasting.kalshi_parallel_backtest import run_parallel_kalshi_backtest


def _sample(ticker: str, *, outcome: int) -> BacktestSample:
    snapshots = [
        {
            "t": "2026-03-01T10:00:00Z",
            "yes_bid": 0.40,
            "yes_ask": 0.42,
            "no_ask": 0.60,
            "last_price": 0.41,
            "volume": 10,
            "open_interest": 20,
        },
        {
            "t": "2026-03-01T11:00:00Z",
            "yes_bid": 0.45,
            "yes_ask": 0.47,
            "no_ask": 0.55,
            "last_price": 0.46,
            "volume": 12,
            "open_interest": 22,
        },
        {
            "t": "2026-03-01T12:00:00Z",
            "yes_bid": 0.48,
            "yes_ask": 0.50,
            "no_ask": 0.52,
            "last_price": 0.49,
            "volume": 14,
            "open_interest": 24,
        },
    ]
    event = {
        "event_ticker": ticker.rsplit("-", 1)[0],
        "market_ticker": ticker,
        "title": f"Will {ticker} happen?",
        "category": "Sports",
        "rules": "Resolves Yes if it happens.",
        "close_time": "2026-03-01T13:00:00Z",
        "outcomes": ["YES", "NO"],
    }
    market_info = {"ticker": ticker, "event_ticker": event["event_ticker"], "close_time": event["close_time"]}
    market_info.update(snapshots[-1])
    market_info["snapshots"] = snapshots
    return BacktestSample(event=event, market_info=market_info, snapshots=snapshots, outcome=outcome)


def test_parallel_backtest_runs_balanced_random_pit_without_gpt(monkeypatch, tmp_path):
    topvol = [_sample(f"KXTOP-{idx}", outcome=idx % 2) for idx in range(4)]
    nonbinary = [_sample(f"KXNON-{idx}", outcome=(idx + 1) % 2) for idx in range(4)]
    monkeypatch.setattr("dhruv_gpt_forecasting.kalshi_parallel_backtest.load_topvol_samples", lambda **kwargs: topvol)
    monkeypatch.setattr(
        "dhruv_gpt_forecasting.kalshi_parallel_backtest.load_nonbinary_component_samples",
        lambda **kwargs: nonbinary,
    )

    result = run_parallel_kalshi_backtest(
        total=4,
        topvol_count=2,
        nonbinary_count=2,
        forecast_mode="stat",
        max_workers=2,
        min_history_snapshots=2,
        output_dir=tmp_path / "run",
        progress_every=0,
    )

    assert result["summary"]["n"] == 4
    assert result["summary"]["api_call_count"] == 0
    assert {row["source_dataset"] for row in result["rows"]} == {"topvol_binary", "nonbinary_component"}
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "rows.jsonl").exists()
