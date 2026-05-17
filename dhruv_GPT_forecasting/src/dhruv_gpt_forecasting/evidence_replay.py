"""Manifest-scoped PIT evidence replay for OOS evaluation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from .config import ForecastConfig
from .features import parse_dt
from .pit_evidence import (
    _is_relevant_record,
    _rank_records,
    _summarize_records,
    build_evidence_query,
)


EvidenceReplayMode = str


@dataclass(frozen=True)
class EvidenceReplayStats:
    mode: EvidenceReplayMode
    manifest_count: int
    loaded_records: int
    records_by_source: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "manifest_count": self.manifest_count,
            "loaded_records": self.loaded_records,
            "records_by_source": self.records_by_source,
        }


class EvidenceReplayIndex:
    """Load archive manifests once, then replay eligible evidence per packet."""

    def __init__(self, records: list[dict[str, Any]], *, manifest_paths: list[Path]) -> None:
        self.records = records
        self.manifest_paths = manifest_paths
        self._source_counts = Counter(str(row.get("source") or "unknown") for row in records)

    @classmethod
    def from_manifests(cls, manifest_paths: Iterable[Path]) -> "EvidenceReplayIndex":
        paths = [Path(path) for path in manifest_paths]
        records: list[dict[str, Any]] = []
        seen_files: set[Path] = set()
        for manifest_path in paths:
            manifest = _load_manifest(manifest_path)
            if not manifest:
                continue
            for archive_path in _manifest_jsonl_paths(manifest, manifest_path):
                resolved = archive_path.resolve()
                if resolved in seen_files or not resolved.exists():
                    continue
                seen_files.add(resolved)
                records.extend(_iter_jsonl(resolved))
        return cls(records, manifest_paths=paths)

    @property
    def stats(self) -> EvidenceReplayStats:
        return EvidenceReplayStats(
            mode="loaded",
            manifest_count=len(self.manifest_paths),
            loaded_records=len(self.records),
            records_by_source=dict(self._source_counts),
        )

    def evidence_for_packet(
        self,
        packet: Any,
        cfg: ForecastConfig,
        *,
        mode: EvidenceReplayMode,
        max_records: int | None = None,
    ) -> list[dict[str, Any]]:
        if mode not in {"strict_pit", "relaxed_published_at"}:
            return []
        as_of_dt = parse_dt(str(getattr(packet, "as_of", "") or ""))
        if as_of_dt is None:
            return []
        query = build_evidence_query(packet)
        eligible = [
            row for row in self.records
            if _eligible_for_mode(row, as_of_dt, cfg, mode=mode)
            and _is_relevant_record(row, packet, query)
        ]
        if not eligible:
            return []
        limit = max_records or cfg.arena.pit_external_max_records
        ranked = _rank_records(eligible, packet, query)[:limit]
        if not ranked:
            return []
        summary = _summarize_records(
            ranked,
            packet,
            query,
            as_of_dt,
            strict=(mode == "strict_pit"),
        )
        summary["source"] = "archive_replay_evidence"
        summary["archive_replay_mode"] = mode
        summary["archive_manifest_count"] = len(self.manifest_paths)
        summary["claim"] = (
            "Manifest-scoped archived evidence was replayed using strict collection-time PIT."
            if mode == "strict_pit"
            else "Manifest-scoped archived evidence was replayed using publication-time PIT."
        )
        return [summary]


def coverage_summary(rows: list[tuple[Any, Any, int]], *, mode: EvidenceReplayMode) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    total_records = 0
    with_evidence = 0
    for _sample, packet, _outcome in rows:
        count = int(packet.features.get("archive_replay_record_count") or 0)
        if count > 0:
            with_evidence += 1
            total_records += count
        for source, value in (packet.features.get("archive_replay_source_counts") or {}).items():
            source_counts[str(source)] += int(value)
    n = len(rows)
    return {
        "mode": mode,
        "n_packets": n,
        "packets_with_evidence": with_evidence,
        "coverage_rate": with_evidence / n if n else 0.0,
        "record_count": total_records,
        "records_by_source": dict(source_counts),
    }


def _eligible_for_mode(
    record: dict[str, Any],
    as_of_dt: Any,
    cfg: ForecastConfig,
    *,
    mode: EvidenceReplayMode,
) -> bool:
    published = parse_dt(str(record.get("published_at") or record.get("created_at") or record.get("timestamp") or ""))
    if published is None or published > as_of_dt:
        return False
    if mode == "relaxed_published_at":
        return True
    collected = parse_dt(str(record.get("collected_at") or record.get("retrieved_at") or record.get("ingested_at") or ""))
    if collected is None:
        return False
    tolerance = timedelta(seconds=cfg.arena.pit_external_clock_tolerance_seconds)
    return collected <= as_of_dt + tolerance


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _manifest_jsonl_paths(manifest: dict[str, Any], manifest_path: Path) -> list[Path]:
    raw_paths = list((manifest.get("archive_files") or {}).values())
    digest = manifest.get("digest_file")
    if digest:
        raw_paths.append(digest)
    out: list[Path] = []
    for raw in raw_paths:
        path = Path(str(raw))
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.suffix == ".jsonl":
            out.append(path)
    return out


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row.setdefault("record_path", str(path))
            rows.append(row)
    return rows
