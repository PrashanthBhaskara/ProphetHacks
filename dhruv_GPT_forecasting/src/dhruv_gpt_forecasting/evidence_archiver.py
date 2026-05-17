"""Archive point-in-time external evidence for forecast packets."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .config import load_config, load_local_env
from .data_loaders import (
    load_nonbinary_component_samples,
    load_prophet_subset_events,
    load_topvol_samples,
    load_unified_binary_samples,
)
from .backtest import load_samples
from .experiments import point_in_time_samples, random_point_in_time_samples
from .features import build_feature_packet, parse_dt
from .forecaster import forecast_event
from .arena_priors import build_arena_packet
from .news_synthesizer import synthesize_news_digest
from .pit_evidence import (
    _resolve_evidence_root,
    annotate_external_records,
    build_evidence_query,
    fetch_external_records_for_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events-json",
        type=Path,
        help="Archive evidence for Arena/Prophet events instead of resolved backtest samples.",
    )
    parser.add_argument(
        "--source",
        choices=["live_clean", "eval_pack", "topvol", "nonbinary", "unified", "prophet_subset_1200"],
        default="topvol",
    )
    parser.add_argument("--horizon-hours", type=float, default=0.25)
    parser.add_argument("--candle-stride-minutes", type=int, default=1)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--chronological-oos", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--gated-only", action="store_true")
    parser.add_argument("--force-cheap", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--since-close")
    parser.add_argument("--until-close")
    parser.add_argument(
        "--sources",
        default="reddit,gdelt",
        help="Comma-separated evidence sources. Add prophet_sources for subset_1200 curated snippets.",
    )
    parser.add_argument("--random-as-of", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260517)
    parser.add_argument("--min-horizon-minutes", type=float, default=5.0)
    parser.add_argument("--max-horizon-hours", type=float)
    parser.add_argument("--min-history-snapshots", type=int, default=5)
    parser.add_argument("--decision-budget-minutes", type=float, default=5.0)
    parser.add_argument("--synthesize-news", action="store_true")
    parser.add_argument("--digest-max-records", type=int, default=8)
    parser.add_argument("--digest-max-chars", type=int, default=1800)
    parser.add_argument(
        "--pit-lookback-hours",
        type=int,
        help="Override archive lookback window for timestamp-bounded evidence pulls.",
    )
    parser.add_argument(
        "--pit-max-records",
        type=int,
        help="Override max evidence records requested per source and forecast packet.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--max-fetch-errors", type=int, default=25)
    parser.add_argument(
        "--allow-historical-backfill",
        action="store_true",
        help="Allow timestamp-bounded historical pulls, currently GDELT.",
    )
    parser.add_argument(
        "--reddit-historical-backfill",
        action="store_true",
        help="Allow Reddit published_at-only historical backfills. Not strict PIT without prior archive.",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    load_local_env()
    cfg = load_config()
    if args.pit_lookback_hours is not None:
        cfg.arena.pit_external_live_lookback_hours = max(1, int(args.pit_lookback_hours))
    if args.pit_max_records is not None:
        cfg.arena.pit_external_max_records = max(1, int(args.pit_max_records))
    packets = _selected_packets(args, cfg)
    sources = {item.strip().lower() for item in args.sources.split(",") if item.strip()}
    run_id = args.run_name or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = args.out_dir or (_resolve_evidence_root(cfg) / "backfills" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_keys: set[str] = set()
    records_by_source: Counter[str] = Counter()
    errors_by_source: Counter[str] = Counter()
    strict_count = 0
    published_count = 0
    packet_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped_packets = 0

    source_paths = {source: out_dir / f"{source}.jsonl" for source in sorted(sources)}
    digest_path = out_dir / "news_digest.jsonl"
    skipped_packet_keys: set[str] = set()
    if args.resume:
        existing_keys, existing_counts, existing_strict, existing_published = _load_existing_archive_state(out_dir)
        written_keys.update(existing_keys)
        records_by_source.update(existing_counts)
        strict_count += existing_strict
        published_count += existing_published
        skipped_packet_keys = _load_attempted_packet_keys(out_dir / "manifest.partial.json")
    if args.manifest_only:
        if not args.resume:
            existing_keys, existing_counts, existing_strict, existing_published = _load_existing_archive_state(out_dir)
            written_keys.update(existing_keys)
            records_by_source.update(existing_counts)
            strict_count += existing_strict
            published_count += existing_published
        prior_manifest = _load_manifest(out_dir / "manifest.partial.json") or _load_manifest(out_dir / "manifest.json")
        packet_rows = list(prior_manifest.get("packet_rows") or []) if prior_manifest else []
        errors = list(prior_manifest.get("errors") or []) if prior_manifest else []
        errors_by_source.update(prior_manifest.get("errors_by_source") or {})
        manifest_path = out_dir / "manifest.json"
        manifest = _write_manifest(
            manifest_path,
            args=args,
            run_id=run_id,
            packets=packets,
            sources=sources,
            source_paths=source_paths,
            digest_path=digest_path if args.synthesize_news else None,
            records_by_source=records_by_source,
            errors_by_source=errors_by_source,
            strict_count=strict_count,
            published_count=published_count,
            packet_rows=packet_rows,
            errors=errors,
            skipped_packets=len(skipped_packet_keys),
            stopped_early_reason="manifest_only_refresh",
        )
        print(json.dumps({
            "archive_dir": str(out_dir),
            "manifest": str(manifest_path),
            "n_packets": manifest["n_packets"],
            "n_records": manifest["n_records"],
            "records_by_source": manifest["records_by_source"],
            "strict_pit_eligible_records": strict_count,
            "published_at_pit_eligible_records": published_count,
            "errors_by_source": manifest["errors_by_source"],
        }, indent=2, sort_keys=True))
        return 0
    handles = {source: path.open("a", encoding="utf-8") for source, path in source_paths.items()}
    digest_handle = digest_path.open("a", encoding="utf-8") if args.synthesize_news else None
    try:
        for idx, packet in enumerate(packets):
            packet_key = _packet_key(packet)
            if args.resume and packet_key in skipped_packet_keys:
                skipped_packets += 1
                continue
            query = build_evidence_query(packet)
            fetched: list[dict[str, Any]] = []
            fetch_errors: list[dict[str, Any]] = []
            if "prophet_sources" in sources:
                fetched.extend(_curated_records_for_packet(packet, cfg, query))
            network_sources = sources - {"prophet_sources"}
            if network_sources:
                network_fetched, fetch_errors = fetch_external_records_for_packet(
                    packet,
                    cfg,
                    sources=network_sources,
                    allow_historical_backfill=args.allow_historical_backfill,
                    allow_reddit_historical_backfill=args.reddit_historical_backfill,
                )
                fetched.extend(network_fetched)
            packet_rows.append({
                "idx": idx,
                "market_ticker": packet.market_ticker,
                "event_ticker": packet.event_ticker,
                "as_of": packet.as_of,
                "close_time": packet.close_time,
                "title": packet.title,
                "query": query,
                "n_records": len(fetched),
                "n_errors": len(fetch_errors),
            })
            for error in fetch_errors:
                source = str(error.get("source") or "unknown")
                errors_by_source[source] += 1
                errors.append({
                    "market_ticker": packet.market_ticker,
                    "as_of": packet.as_of,
                    **error,
                })
            if args.max_fetch_errors >= 0 and len(errors) >= args.max_fetch_errors:
                _write_manifest(
                    out_dir / "manifest.partial.json",
                    args=args,
                    run_id=run_id,
                    packets=packets,
                    sources=sources,
                    source_paths=source_paths,
                    digest_path=digest_path if args.synthesize_news else None,
                    records_by_source=records_by_source,
                    errors_by_source=errors_by_source,
                    strict_count=strict_count,
                    published_count=published_count,
                    packet_rows=packet_rows,
                    errors=errors,
                    skipped_packets=skipped_packets,
                    stopped_early_reason="max_fetch_errors_reached",
                )
                break
            for record in fetched:
                key = _record_key(record)
                if key in written_keys:
                    continue
                written_keys.add(key)
                record["archive_run_id"] = run_id
                source = str(record.get("source") or "unknown").lower()
                records_by_source[source] += 1
                if record.get("strict_pit_eligible"):
                    strict_count += 1
                if record.get("published_at_pit_eligible"):
                    published_count += 1
                handle = handles.get(source)
                if handle is None:
                    path = out_dir / f"{source}.jsonl"
                    handle = path.open("a", encoding="utf-8")
                    handles[source] = handle
                    source_paths[source] = path
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if digest_handle is not None:
                digest = synthesize_news_digest(
                    fetched,
                    packet,
                    query,
                    max_records=args.digest_max_records,
                    max_chars=args.digest_max_chars,
                )
                if digest is not None:
                    digest["archive_run_id"] = run_id
                    records_by_source[str(digest["source"])] += 1
                    published_count += 1 if digest.get("published_at_pit_eligible") else 0
                    digest_handle.write(json.dumps(digest, sort_keys=True) + "\n")
                    digest_handle.flush()
            for handle in handles.values():
                handle.flush()
            _write_manifest(
                out_dir / "manifest.partial.json",
                args=args,
                run_id=run_id,
                packets=packets,
                sources=sources,
                source_paths=source_paths,
                digest_path=digest_path if args.synthesize_news else None,
                records_by_source=records_by_source,
                errors_by_source=errors_by_source,
                strict_count=strict_count,
                published_count=published_count,
                packet_rows=packet_rows,
                errors=errors,
                skipped_packets=skipped_packets,
                stopped_early_reason=None,
            )
            if args.sleep_seconds > 0.0 and idx < len(packets) - 1:
                time.sleep(args.sleep_seconds)
    finally:
        for handle in handles.values():
            handle.close()
        if digest_handle is not None:
            digest_handle.close()

    manifest_path = out_dir / "manifest.json"
    manifest = _write_manifest(
        manifest_path,
        args=args,
        run_id=run_id,
        packets=packets,
        sources=sources,
        source_paths=source_paths,
        digest_path=digest_path if args.synthesize_news else None,
        records_by_source=records_by_source,
        errors_by_source=errors_by_source,
        strict_count=strict_count,
        published_count=published_count,
        packet_rows=packet_rows,
        errors=errors,
        skipped_packets=skipped_packets,
        stopped_early_reason="max_fetch_errors_reached" if args.max_fetch_errors >= 0 and len(errors) >= args.max_fetch_errors else None,
    )
    print(json.dumps({
        "archive_dir": str(out_dir),
        "manifest": str(manifest_path),
        "n_packets": manifest["n_packets"],
        "n_records": manifest["n_records"],
        "records_by_source": manifest["records_by_source"],
        "strict_pit_eligible_records": strict_count,
        "published_at_pit_eligible_records": published_count,
        "errors_by_source": manifest["errors_by_source"],
    }, indent=2, sort_keys=True))
    return 0


def _write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    run_id: str,
    packets: list[Any],
    sources: set[str],
    source_paths: dict[str, Path],
    digest_path: Path | None,
    records_by_source: Counter[str],
    errors_by_source: Counter[str],
    strict_count: int,
    published_count: int,
    packet_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    skipped_packets: int,
    stopped_early_reason: str | None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_id,
        "events_json": str(args.events_json) if args.events_json else None,
        "source": args.source if not args.events_json else "events_json",
        "horizon_hours": args.horizon_hours,
        "candle_stride_minutes": args.candle_stride_minutes if args.source in {"topvol", "nonbinary", "unified"} else None,
        "since_close": args.since_close,
        "until_close": args.until_close,
        "random_as_of": args.random_as_of,
        "random_seed": args.random_seed if args.random_as_of else None,
        "min_horizon_minutes": args.min_horizon_minutes if args.random_as_of else None,
        "max_horizon_hours": args.max_horizon_hours if args.random_as_of else None,
        "min_history_snapshots": args.min_history_snapshots if args.random_as_of else None,
        "decision_budget_minutes": args.decision_budget_minutes if args.random_as_of else None,
        "chronological_oos": args.chronological_oos,
        "train_fraction": args.train_fraction,
        "gated_only": args.gated_only,
        "force_cheap": args.force_cheap,
        "sources": sorted(sources),
        "sleep_seconds": args.sleep_seconds,
        "resume": args.resume,
        "max_fetch_errors": args.max_fetch_errors,
        "allow_historical_backfill": args.allow_historical_backfill,
        "reddit_historical_backfill": args.reddit_historical_backfill,
        "synthesize_news": args.synthesize_news,
        "digest_max_records": args.digest_max_records if args.synthesize_news else None,
        "digest_max_chars": args.digest_max_chars if args.synthesize_news else None,
        "pit_lookback_hours": args.pit_lookback_hours,
        "pit_max_records": args.pit_max_records,
        "n_packets": len(packets),
        "n_packets_attempted": len(packet_rows),
        "n_packets_skipped_resume": skipped_packets,
        "stopped_early_reason": stopped_early_reason,
        "n_records": sum(records_by_source.values()),
        "records_by_source": dict(records_by_source),
        "strict_pit_eligible_records": strict_count,
        "published_at_pit_eligible_records": published_count,
        "errors_by_source": dict(errors_by_source),
        "archive_files": {source: str(path) for source, path in source_paths.items() if path.exists()},
        "digest_file": str(digest_path) if digest_path and digest_path.exists() else None,
        "packet_rows": packet_rows,
        "errors": errors[:200],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _selected_packets(args: argparse.Namespace, cfg: Any):
    if args.events_json:
        events = json.loads(args.events_json.read_text(encoding="utf-8"))
        if isinstance(events, dict):
            events = events.get("events") or events.get("data") or [events]
        return [build_arena_packet(event, include_historical_analogs=False) for event in events]
    if args.source == "prophet_subset_1200":
        events = load_prophet_subset_events(limit=args.max_candidates)
        packets = [build_arena_packet(event, include_historical_analogs=False) for event in events]
        return packets[args.offset: args.offset + args.limit]
    return [
        build_feature_packet(sample.event, sample.market_info, price_trajectory=sample.snapshots)
        for sample in _selected_samples(args, cfg)
    ]


def _selected_samples(args: argparse.Namespace, cfg: Any):
    if args.source == "topvol":
        raw_samples = load_topvol_samples(candle_stride_minutes=args.candle_stride_minutes)
    elif args.source == "nonbinary":
        raw_samples = load_nonbinary_component_samples(candle_stride_minutes=args.candle_stride_minutes)
    elif args.source == "unified":
        raw_samples = load_unified_binary_samples(candle_stride_minutes=args.candle_stride_minutes)
    else:
        raw_samples = load_samples(args.source, None)
    samples = _filter_by_close_time(raw_samples, since_close=args.since_close, until_close=args.until_close)
    samples.sort(key=lambda sample: (
        parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time")).timestamp()
        if parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time")) else 0.0,
        sample.event.get("market_ticker") or sample.market_info.get("ticker") or "",
    ))
    if args.chronological_oos:
        split = max(1, min(len(samples) - 1, int(len(samples) * args.train_fraction)))
        samples = samples[split:]
    if args.random_as_of:
        samples = random_point_in_time_samples(
            samples,
            seed=args.random_seed,
            min_horizon_minutes=args.min_horizon_minutes,
            max_horizon_hours=args.max_horizon_hours,
            min_history_snapshots=args.min_history_snapshots,
            decision_budget_minutes=args.decision_budget_minutes,
        )
    else:
        samples = point_in_time_samples(samples, horizon_hours=args.horizon_hours)
        samples.sort(key=lambda sample: (
            _sample_as_of(sample),
            sample.event.get("market_ticker") or sample.market_info.get("ticker") or "",
        ))
    candidates = samples[args.offset: args.offset + args.max_candidates]
    selected = []
    for sample in candidates:
        if not args.gated_only:
            selected.append(sample)
        else:
            dry = forecast_event(
                sample.event,
                sample.market_info,
                price_trajectory=sample.snapshots,
                dry_run=True,
                config=cfg,
                force_cheap=args.force_cheap,
            )
            if dry.audit_summary.get("gates", {}).get("call_cheap"):
                selected.append(sample)
        if len(selected) >= args.limit:
            break
    return selected


def _curated_records_for_packet(packet: Any, cfg: Any, query: str) -> list[dict[str, Any]]:
    features = getattr(packet, "features", {}) or {}
    records = features.get("curated_sources") or []
    if not isinstance(records, list):
        return []
    return annotate_external_records(
        [record for record in records if isinstance(record, dict)],
        packet,
        cfg,
        query,
        mode="prophet_subset_curated_sources",
    )


def _record_key(record: dict[str, Any]) -> str:
    parts = [
        str(record.get("source") or ""),
        str(record.get("target_market_ticker") or record.get("market_ticker") or ""),
        str(record.get("published_at") or ""),
        str(record.get("url") or ""),
        str(record.get("text") or "")[:200],
    ]
    return "\x1f".join(parts)


def _packet_key(packet: Any) -> str:
    return "\x1f".join([
        str(getattr(packet, "market_ticker", "") or ""),
        str(getattr(packet, "as_of", "") or ""),
    ])


def _load_existing_archive_state(out_dir: Path) -> tuple[set[str], Counter[str], int, int]:
    keys: set[str] = set()
    counts: Counter[str] = Counter()
    strict_count = 0
    published_count = 0
    if not out_dir.exists():
        return keys, counts, strict_count, published_count
    for path in out_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                keys.add(_record_key(row))
                counts[str(row.get("source") or _infer_source_from_name(path.name))] += 1
                if row.get("strict_pit_eligible"):
                    strict_count += 1
                if row.get("published_at_pit_eligible"):
                    published_count += 1
    return keys, counts, strict_count, published_count


def _infer_source_from_name(name: str) -> str:
    lower = name.lower()
    if "digest" in lower:
        return "pit_news_digest"
    if "gdelt" in lower:
        return "gdelt"
    if "reddit" in lower:
        return "reddit"
    return "external_jsonl"


def _load_attempted_packet_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    keys: set[str] = set()
    for row in manifest.get("packet_rows") or []:
        if isinstance(row, dict):
            keys.add("\x1f".join([str(row.get("market_ticker") or ""), str(row.get("as_of") or "")]))
    return keys


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _sample_as_of(sample) -> float:
    if sample.snapshots:
        value = sample.snapshots[-1].get("t") or sample.snapshots[-1].get("snapshot_time")
    else:
        value = sample.market_info.get("snapshot_time")
    parsed = parse_dt(value)
    return parsed.timestamp() if parsed is not None else 0.0


def _filter_by_close_time(samples, *, since_close: str | None, until_close: str | None):
    if not since_close and not until_close:
        return samples
    since_dt = parse_dt(since_close) if since_close else None
    until_dt = parse_dt(until_close) if until_close else None
    out = []
    for sample in samples:
        close_dt = parse_dt(sample.event.get("close_time") or sample.market_info.get("close_time"))
        if close_dt is None:
            continue
        if since_dt is not None and close_dt < since_dt:
            continue
        if until_dt is not None and close_dt > until_dt:
            continue
        out.append(sample)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
