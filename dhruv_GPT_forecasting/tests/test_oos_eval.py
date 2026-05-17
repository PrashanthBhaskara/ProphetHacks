from dhruv_gpt_forecasting.oos_eval import run_oos_evaluation
import json


def test_oos_eval_runs_on_small_topvol_slice():
    result = run_oos_evaluation(
        source="topvol",
        horizon_hours=24,
        candle_stride_minutes=15,
        train_fraction=0.6,
        limit=80,
        min_segment_n=5,
        top_segments=3,
    )
    assert result["n_train"] > 0
    assert result["n_test"] > 0
    assert result["best_model"]["n"] == result["n_test"]
    assert result["top_models"]
    assert result["segment_report"]["category"]
    assert result["stat_model_routing"]["candidate_count"] > 0


def test_oos_eval_random_as_of_runs_on_small_topvol_slice():
    result = run_oos_evaluation(
        source="topvol",
        horizon_hours=24,
        candle_stride_minutes=15,
        train_fraction=0.6,
        limit=80,
        random_as_of=True,
        random_seed=123,
        min_horizon_minutes=5,
        min_history_snapshots=5,
        decision_budget_minutes=5,
        min_segment_n=5,
        top_segments=3,
    )
    assert result["random_as_of"] is True
    assert result["decision_budget_minutes"] == 5
    assert result["n_train"] > 0
    assert result["n_test"] > 0
    assert result["best_model"]["n"] == result["n_test"]
    assert result["stat_model_routing"]["routes"]


def test_oos_eval_accepts_archive_replay_manifest(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"archive_files": {}, "digest_file": None}), encoding="utf-8")
    result = run_oos_evaluation(
        source="topvol",
        horizon_hours=24,
        candle_stride_minutes=15,
        train_fraction=0.6,
        limit=50,
        random_as_of=True,
        random_seed=456,
        min_horizon_minutes=5,
        min_history_snapshots=5,
        decision_budget_minutes=5,
        min_segment_n=5,
        top_segments=3,
        evidence_mode="strict_pit",
        evidence_manifest_paths=[manifest_path],
    )

    assert result["evidence_replay"]["mode"] == "strict_pit"
    assert result["evidence_replay"]["loaded_records"] == 0
    assert result["evidence_replay"]["test"]["n_packets"] == result["n_test"]
