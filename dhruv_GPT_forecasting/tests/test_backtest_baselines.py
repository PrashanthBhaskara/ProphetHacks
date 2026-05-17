from dhruv_gpt_forecasting.backtest import run_backtest
from dhruv_gpt_forecasting.data_loaders import load_eval_pack, load_topvol_samples


def test_live_clean_market_baseline_matches_known_result():
    samples = load_eval_pack()
    result = run_backtest(samples, "market")
    assert result["n"] == 13165
    assert abs(result["brier"] - 0.0964) < 0.001


def test_topvol_market_baseline_runs_on_recent_top_volume_data():
    samples = load_topvol_samples(limit=200)
    result = run_backtest(samples, "market")
    assert result["n"] == 200
    assert 0.0 <= result["brier"] <= 0.25
    assert result["category_metrics"]

