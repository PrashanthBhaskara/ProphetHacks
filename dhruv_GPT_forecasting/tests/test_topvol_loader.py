from dhruv_gpt_forecasting.data_loaders import (
    load_nonbinary_component_samples,
    load_prophet_subset_events,
    load_topvol_samples,
    load_unified_binary_samples,
)


def test_topvol_loader_reads_candle_trajectories():
    samples = load_topvol_samples(limit=3)
    assert len(samples) == 3
    assert all(sample.snapshots for sample in samples)
    assert all(sample.event["outcomes"] == ["YES", "NO"] for sample in samples)
    assert all(sample.outcome in {0, 1} for sample in samples)


def test_nonbinary_component_loader_reads_resolved_components():
    samples = load_nonbinary_component_samples(limit=3)
    assert len(samples) == 3
    assert all(sample.snapshots for sample in samples)
    assert all(sample.event["outcomes"] == ["YES", "NO"] for sample in samples)
    assert all(sample.outcome in {0, 1} for sample in samples)


def test_unified_loader_combines_binary_and_nonbinary_samples():
    samples = load_unified_binary_samples(limit=6)
    assert len(samples) == 6
    assert all(sample.snapshots for sample in samples)
    assert {sample.event["outcomes"][0] for sample in samples} == {"YES"}


def test_prophet_subset_loader_preserves_curated_sources():
    events = load_prophet_subset_events(limit=2)
    assert len(events) == 2
    assert events[0]["as_of"]
    assert events[0]["outcomes"]
    assert events[0]["features"]["source_dataset"] == "prophet_subset_1200"
    assert events[0]["features"]["curated_sources"]
    assert events[0]["features"]["curated_sources"][0]["collected_at"] == events[0]["as_of"]
