from dhruv_gpt_forecasting.arena_batch import actuals_from_events, benchmark_events, predict_events
from dhruv_gpt_forecasting.arena_batch import _load_events
from dhruv_gpt_forecasting.arena_eval import actual_market_label, evaluate_predictions, event_brier, load_events


def test_event_brier_scores_multi_outcome_distribution():
    probs = {"A": 0.7, "B": 0.2, "C": 0.1}
    assert round(event_brier(probs, "A", ["A", "B", "C"]), 4) == 0.14


def test_evaluate_predictions_matches_by_market_ticker():
    predictions = [
        {
            "market_ticker": "task-001",
            "probabilities": [
                {"market": "A", "probability": 0.7},
                {"market": "B", "probability": 0.3},
            ],
        }
    ]
    actuals = {"task-001": "B"}
    events = [{"market_ticker": "task-001", "category": "Demo", "outcomes": ["A", "B"]}]
    result = evaluate_predictions(predictions, actuals, events=events)
    assert result["n"] == 1
    assert round(result["brier"], 4) == 0.98
    assert result["category_metrics"][0]["segment"] == "Demo"


def test_resolved_outcome_value_payload_matches_prophet_scorer():
    actual = {"value": ["B"], "resolved_at": "2026-05-13T17:02:27Z", "source": "task-001"}
    assert actual_market_label(actual) == "B"
    assert round(event_brier({"A": 0.7, "B": 0.3}, actual, ["A", "B"]), 4) == 0.98


def test_actuals_from_retrieved_resolved_events_uses_market_ticker():
    events = [
        {
            "event_ticker": "event-001",
            "market_ticker": "task-001",
            "outcomes": ["A", "B"],
            "resolved_outcome": {"value": ["B"], "resolved_at": "2026-05-13T17:02:27Z"},
        }
    ]
    assert actuals_from_events(events) == {"task-001": "B"}


def test_predict_events_writes_prophet_submission_shape(monkeypatch):
    class Forecast:
        probabilities = {"A": 0.25, "B": 0.75}
        audit = {}
        source = "test"

    monkeypatch.setattr("dhruv_gpt_forecasting.arena_batch.forecast_arena_event", lambda *args, **kwargs: Forecast())
    result = predict_events([{"market_ticker": "task-001", "outcomes": ["A", "B"]}], use_gpt=False)
    assert result["predictions"][0]["market_ticker"] == "task-001"
    assert result["predictions"][0]["probabilities"] == [
        {"market": "A", "probability": 0.25},
        {"market": "B", "probability": 0.75},
    ]


def test_batch_loader_accepts_raw_tasks_jsonl(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        (
            '{"task_id":"task-001","title":"Who wins?","outcomes":["A","B"],'
            '"predict_by":"2026-05-13T09:58:49+00:00","context":"Rules text",'
            '"metadata":{"category":"Sports","source":{"event_ticker":"event-001","rules":"Source rules"}}}\n'
        ),
        encoding="utf-8",
    )
    event = _load_events(path)[0]
    assert event["market_ticker"] == "task-001"
    assert event["event_ticker"] == "event-001"
    assert event["category"] == "Sports"
    assert event["close_time"] == "2026-05-13T09:58:49+00:00"
    assert event["rules"] == "Source rules"


def test_eval_loader_accepts_raw_tasks_jsonl_with_metadata_category(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        (
            '{"task_id":"task-001","title":"Who wins?","outcomes":["A","B"],'
            '"predict_by":"2026-05-13T09:58:49+00:00","context":"Rules text",'
            '"metadata":{"category":"Sports","source":{"event_ticker":"event-001"}}}\n'
        ),
        encoding="utf-8",
    )
    event = load_events(path)[0]
    assert event["market_ticker"] == "task-001"
    assert event["event_ticker"] == "event-001"
    assert event["category"] == "Sports"


def test_benchmark_events_writes_run_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret")
    events = [
        {
            "task_id": "task-001",
            "title": "Who wins?",
            "outcomes": ["A", "B"],
            "predict_by": "2026-05-13T09:58:49+00:00",
            "context": "Rules text",
            "metadata": {"category": "Sports", "source": {"event_ticker": "event-001"}},
            "resolved_outcome": {"value": ["A"]},
        }
    ]
    out_dir = tmp_path / "run"

    result = benchmark_events(
        events,
        output_dir=out_dir,
        dataset="sample",
        release="test",
        seed=7,
        evidence_mode="strict_pit",
        limit=1,
        with_gpt=False,
    )

    assert result["run_config"]["dataset"] == "sample"
    assert result["run_config"]["random_seed"] == 7
    assert result["run_config"]["api_key_fingerprint"]
    assert (out_dir / "run_config.json").exists()
    assert (out_dir / "predictions.json").exists()
    assert (out_dir / "actuals.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "report.md").exists()
    assert "deterministic" in result["metrics"]
