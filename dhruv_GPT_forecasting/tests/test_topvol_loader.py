import csv
import gzip
import json

from dhruv_gpt_forecasting.data_loaders import (
    GIT_LFS_POINTER_PREFIX,
    load_nonbinary_component_samples,
    load_prophet_subset_events,
    load_topvol_samples,
    load_unified_binary_samples,
)


def test_topvol_loader_reads_candle_trajectories(tmp_path):
    root = _write_market_fixture(tmp_path / "topvol", "selected_markets", count=3)
    samples = load_topvol_samples(root=root, limit=3)
    assert len(samples) == 3
    assert all(sample.snapshots for sample in samples)
    assert all(sample.event["outcomes"] == ["YES", "NO"] for sample in samples)
    assert all(sample.outcome in {0, 1} for sample in samples)


def test_nonbinary_component_loader_reads_resolved_components(tmp_path):
    root = _write_market_fixture(tmp_path / "nonbinary", "component_markets", count=3)
    samples = load_nonbinary_component_samples(root=root, limit=3)
    assert len(samples) == 3
    assert all(sample.snapshots for sample in samples)
    assert all(sample.event["outcomes"] == ["YES", "NO"] for sample in samples)
    assert all(sample.outcome in {0, 1} for sample in samples)


def test_unified_loader_combines_binary_and_nonbinary_samples(tmp_path, monkeypatch):
    import dhruv_gpt_forecasting.data_loaders as data_loaders

    topvol_root = _write_market_fixture(tmp_path / "topvol", "selected_markets", count=3)
    nonbinary_root = _write_market_fixture(tmp_path / "nonbinary", "component_markets", count=3)
    monkeypatch.setattr(data_loaders, "TOPVOL_ROOT", topvol_root)
    monkeypatch.setattr(data_loaders, "NONBINARY_ROOT", nonbinary_root)

    samples = load_unified_binary_samples(limit=6)
    assert len(samples) == 6
    assert all(sample.snapshots for sample in samples)
    assert {sample.event["outcomes"][0] for sample in samples} == {"YES"}


def test_lfs_pointer_market_files_are_skipped_cleanly(tmp_path):
    root = tmp_path / "topvol"
    market_dir = root / "markets"
    market_dir.mkdir(parents=True)
    (market_dir / "2026-01-01_selected_markets.jsonl").write_text(
        f"{GIT_LFS_POINTER_PREFIX}\noid sha256:abc\nsize 123\n",
        encoding="utf-8",
    )

    assert load_topvol_samples(root=root, limit=1) == []


def test_lfs_pointer_candle_files_are_skipped_cleanly(tmp_path):
    root = tmp_path / "topvol"
    ticker = "KXTEST-1"
    _write_jsonl(root / "markets" / "2026-01-01_selected_markets.jsonl", [_market_row(ticker, 1)])
    candle_path = root / "ohlcv" / "period_1m" / "week=2026-01-01" / f"{ticker}.csv.gz"
    candle_path.parent.mkdir(parents=True)
    candle_path.write_text(f"{GIT_LFS_POINTER_PREFIX}\noid sha256:abc\nsize 123\n", encoding="utf-8")

    assert load_topvol_samples(root=root, limit=1) == []


def test_prophet_subset_loader_preserves_curated_sources():
    events = load_prophet_subset_events(limit=2)
    assert len(events) == 2
    assert events[0]["as_of"]
    assert events[0]["outcomes"]
    assert events[0]["features"]["source_dataset"] == "prophet_subset_1200"
    assert events[0]["features"]["curated_sources"]
    assert events[0]["features"]["curated_sources"][0]["collected_at"] == events[0]["as_of"]


def _write_market_fixture(root, suffix: str, *, count: int):
    week = "2026-01-01"
    rows = [_market_row(f"KXTEST-{idx}", idx) for idx in range(1, count + 1)]
    _write_jsonl(root / "markets" / f"{week}_{suffix}.jsonl", rows)
    for idx, row in enumerate(rows, start=1):
        _write_candles(root, week, row["ticker"], base_price=40 + idx)
    return root


def _market_row(ticker: str, idx: int):
    return {
        "ticker": ticker,
        "event_ticker": f"KXTESTEVENT-{idx}",
        "title": f"Will fixture event {idx} happen?",
        "subtitle": "Fixture market",
        "yes_sub_title": "YES",
        "rules_primary": "Resolves Yes if the fixture condition occurs.",
        "close_time": "2026-01-02T00:00:00Z",
        "category": "Politics",
        "result": "yes" if idx % 2 else "no",
    }


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_candles(root, week: str, ticker: str, *, base_price: int):
    path = root / "ohlcv" / "period_1m" / f"week={week}" / f"{ticker}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "end_period_time",
                "end_period_ts",
                "yes_bid_close",
                "yes_ask_close",
                "price_close",
                "volume",
                "open_interest",
            ],
        )
        writer.writeheader()
        for minute in (0, 16, 32):
            writer.writerow({
                "end_period_time": f"2026-01-01T00:{minute:02d}:00Z",
                "end_period_ts": str(1767225600 + minute * 60),
                "yes_bid_close": str(base_price + minute),
                "yes_ask_close": str(base_price + minute + 4),
                "price_close": str(base_price + minute + 2),
                "volume": str(10 + minute),
                "open_interest": str(100 + minute),
            })
