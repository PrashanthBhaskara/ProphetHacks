"""WRDS/LSEG evidence connectors and normalizers.

The live connector intentionally expects a small normalized HTTP contract. For
licensed SDK workflows, export vendor rows to JSON/JSONL/CSV and normalize them
with this module so backtests and live runs use the same record shape.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .config import PACKAGE_ROOT, ForecastConfig, load_config, load_local_env


VENDOR_SOURCES = {"wrds", "lseg"}


def vendor_env_status(source: str) -> dict[str, Any]:
    """Return sanitized connector status for preflight output."""
    source = _canonical_vendor(source)
    url, url_env = _vendor_url(source)
    key, key_env, header = _vendor_key(source)
    username = os.environ.get("WRDS_USERNAME") if source == "wrds" else None
    password = os.environ.get("WRDS_PASSWORD") if source == "wrds" else None
    backend = _vendor_backend(source, url)
    missing: list[str] = []
    configured = False
    if backend == "http":
        configured = bool(url and (key or (username and password) or source == "lseg"))
        if not url:
            missing.append("url")
        if source == "wrds" and not (key or (username and password)):
            missing.append("WRDS_API_KEY/WRDS_ACCESS_TOKEN or WRDS_USERNAME+WRDS_PASSWORD")
    elif backend == "lseg_data_library":
        configured = bool(key)
        if not key:
            missing.append("LSEG_APP_KEY or LSEG_API_KEY")
        if not _module_available("lseg.data"):
            missing.append("optional package lseg-data")
    elif backend == "wrds_postgres":
        configured = bool(username and password and _wrds_sql())
        if not (username and password):
            missing.append("WRDS_USERNAME+WRDS_PASSWORD")
        if not _wrds_sql():
            missing.append("WRDS_NEWS_SQL or WRDS_NEWS_SQL_FILE")
        if not _module_available("wrds"):
            missing.append("optional package wrds")
    else:
        configured = False
        if not url:
            missing.append("url")
    auth_mode = None
    if key:
        auth_mode = f"header:{header}"
    elif username and password:
        auth_mode = "basic_auth"
    return {
        "source": source,
        "configured": configured,
        "backend": backend,
        "url_present": bool(url),
        "url_env": url_env,
        "key_present": bool(key),
        "key_env": key_env,
        "username_present": bool(username),
        "password_present": bool(password),
        "auth_mode": auth_mode,
        "missing": missing,
        "accepted_url_envs": _url_envs(source),
        "accepted_key_envs": _key_envs(source),
        "native_url_schemes": _native_url_schemes(source),
        "local_archive_supported": True,
    }


def fetch_vendor_records(
    source: str,
    query: str,
    as_of: Any,
    cfg: ForecastConfig,
    *,
    deadline_at: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch normalized vendor records from configured WRDS/LSEG endpoints."""
    source = _canonical_vendor(source)
    url, _ = _vendor_url(source)
    if not url:
        return [], [{"source": source, "error": "missing_vendor_url"}]
    backend = _vendor_backend(source, url)
    if backend == "lseg_data_library":
        return _run_native_fetch_with_timeout(
            source,
            lambda: _fetch_lseg_data_library_records(query, as_of, cfg),
            timeout_seconds=_source_timeout(cfg, deadline_at),
        )
    if backend == "wrds_postgres":
        return _run_native_fetch_with_timeout(
            source,
            lambda: _fetch_wrds_postgres_records(query, as_of, cfg),
            timeout_seconds=_source_timeout(cfg, deadline_at),
        )
    if backend != "http":
        return [], [{"source": source, "error": f"unsupported_vendor_url_scheme:{urlparse(url).scheme}"}]
    headers, auth = _vendor_headers_and_auth(source)
    params = {
        "q": query,
        "as_of": _as_iso(as_of),
        "limit": min(50, max(1, cfg.arena.pit_external_max_records)),
    }
    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            auth=auth,
            timeout=_source_timeout(cfg, deadline_at),
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - evidence failures should be reported, not fatal.
        return [], [{"source": source, "error": f"{type(exc).__name__}:{exc}"}]
    return normalize_vendor_payload(source, data, collected_at=_now()), []


def _fetch_lseg_data_library_records(
    query: str,
    as_of: Any,
    cfg: ForecastConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch LSEG Workspace/Data Library headlines when no HTTP bridge is used."""
    key, key_env, _ = _vendor_key("lseg")
    if not key:
        return [], [{"source": "lseg", "error": "missing_lseg_app_key"}]
    try:
        ld = importlib.import_module("lseg.data")
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        return [], [{"source": "lseg", "error": f"missing_lseg_data_package:{type(exc).__name__}:{exc}"}]
    try:
        ld.open_session(app_key=key)
        end_dt = _as_datetime(as_of) or datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(hours=cfg.arena.pit_external_live_lookback_hours)
        headlines = ld.news.get_headlines(
            query=query,
            count=min(100, max(1, cfg.arena.pit_external_max_records)),
            start=_lseg_dt(start_dt),
            end=_lseg_dt(end_dt),
        )
        rows = _dataframe_rows(headlines)
    except Exception as exc:  # noqa: BLE001 - evidence failures should not fail forecasts.
        return [], [{"source": "lseg", "error": f"lseg_data_library:{type(exc).__name__}:{exc}", "key_env": key_env}]
    return normalize_vendor_rows("lseg", rows, collected_at=_now()), []


def _fetch_wrds_postgres_records(
    query: str,
    as_of: Any,
    cfg: ForecastConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch WRDS rows through the official Python/PostgreSQL path.

    WRDS is a dataset platform, not a single generic news endpoint. The project
    therefore expects a licensed news SQL template in WRDS_NEWS_SQL or
    WRDS_NEWS_SQL_FILE. Use placeholders %(query)s, %(as_of)s, %(start)s, and
    %(limit)s so the query stays point-in-time.
    """
    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")
    if not (username and password):
        return [], [{"source": "wrds", "error": "missing_wrds_username_password"}]
    raw_sql = _wrds_sql()
    end_dt = _as_datetime(as_of) or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=cfg.arena.pit_external_live_lookback_hours)
    sql = _render_wrds_sql(raw_sql, end_dt) if raw_sql else None
    if not sql:
        return [], [{"source": "wrds", "error": "missing_wrds_news_sql"}]
    try:
        wrds = importlib.import_module("wrds")
    except Exception as exc:  # noqa: BLE001 - optional dependency.
        return [], [{"source": "wrds", "error": f"missing_wrds_package:{type(exc).__name__}:{exc}"}]
    params = {
        "query": query,
        "as_of": end_dt.isoformat(),
        "start": start_dt.isoformat(),
        "limit": min(100, max(1, cfg.arena.pit_external_max_records)),
    }
    connection = None
    try:
        connection = wrds.Connection(autoconnect=False, wrds_username=username, wrds_password=password)
        connection._Connection__make_sa_engine_conn(raise_err=True)
        rows_df = connection.raw_sql(sql, params=params)
        rows = _dataframe_rows(rows_df)
    except Exception as exc:  # noqa: BLE001 - evidence failures should not fail forecasts.
        return [], [{"source": "wrds", "error": f"wrds_postgres:{type(exc).__name__}:{exc}"}]
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:  # noqa: BLE001
                pass
    return normalize_vendor_rows("wrds", rows, collected_at=_now()), []


def normalize_vendor_payload(source: str, payload: Any, *, collected_at: str | None = None) -> list[dict[str, Any]]:
    """Normalize common WRDS/LSEG API response shapes to evidence records."""
    rows = _extract_rows(payload)
    return normalize_vendor_rows(source, rows, collected_at=collected_at)


def normalize_vendor_rows(
    source: str,
    rows: list[dict[str, Any]],
    *,
    collected_at: str | None = None,
) -> list[dict[str, Any]]:
    source = _canonical_vendor(source)
    collected = collected_at or _now()
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        record = {
            "source": source,
            "published_at": _jsonable(_first_value(
                row,
                (
                    "published_at",
                    "published",
                    "timestamp",
                    "date",
                    "datetime",
                    "story_date",
                    "firstCreated",
                    "versionCreated",
                    "dateTime",
                    "Date",
                    "Time",
                    "created_at",
                ),
            )),
            "collected_at": _jsonable(_first_value(row, ("collected_at", "retrieved_at", "ingested_at")) or collected),
            "title": _jsonable(_first_value(row, ("title", "headline", "Headline", "storyHeadline", "subject", "caption"))),
            "text": _jsonable(_first_value(row, ("text", "body", "story", "summary", "description", "content", "article"))),
            "url": _jsonable(_first_value(row, ("url", "link", "storyUrl", "source_url"))),
            "vendor_id": _jsonable(_first_value(
                row,
                ("id", "storyId", "StoryId", "story_id", "news_id", "ric", "perm_id", "accession_number"),
            )),
            "vendor_source": _jsonable(_first_value(
                row,
                ("provider", "source", "sourceCode", "Source", "source_name", "database"),
            )),
        }
        for key in ("market_ticker", "event_ticker", "target_market_ticker", "target_event_ticker"):
            if row.get(key) is not None:
                record[key] = _jsonable(row.get(key))
        extra = {
            key: _jsonable(row.get(key))
            for key in (
                "language",
                "company",
                "ticker",
                "ric",
                "country",
                "subjects",
                "topic",
                "category",
                "relevance",
            )
            if row.get(key) is not None
        }
        if extra:
            record["vendor_metadata"] = extra
        if record["title"] or record["text"]:
            out.append(record)
    return out


def records_to_live_evidence(
    source: str,
    records: list[dict[str, Any]],
    *,
    query: str,
    max_records: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    source = _canonical_vendor(source)
    compact = []
    for record in records[:max_records]:
        compact.append({
            "title": record.get("title"),
            "summary": record.get("text"),
            "published_at": record.get("published_at"),
            "collected_at": record.get("collected_at"),
            "url": record.get("url"),
            "vendor_id": record.get("vendor_id"),
            "vendor_source": record.get("vendor_source"),
            "vendor_metadata": record.get("vendor_metadata"),
        })
    return [{
        "source": source,
        "timestamp": _now(),
        "query": query,
        "claim": f"Licensed {source.upper()} records were retrieved and normalized for this forecast.",
        "records": compact,
    }]


def archive_vendor_records(records: list[dict[str, Any]], cfg: ForecastConfig) -> None:
    if not records or not cfg.arena.pit_external_archive_live_fetches:
        return
    root = Path(cfg.arena.pit_external_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    date = time.strftime("%Y-%m-%d", time.gmtime())
    for record in records:
        source = _canonical_vendor(str(record.get("source") or "vendor"))
        archive_dir = root / "live_fetches" / source
        archive_dir.mkdir(parents=True, exist_ok=True)
        path = archive_dir / f"{date}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def normalize_file(source: str, input_path: Path, output_path: Path, *, collected_at: str | None = None) -> int:
    rows = _read_rows(input_path)
    records = normalize_vendor_rows(source, rows, collected_at=collected_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    output_path.write_text(text + ("\n" if records else ""), encoding="utf-8")
    return len(records)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        return rows
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _extract_rows(payload)
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported vendor evidence input format: {path.suffix}")


def _extract_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "articles", "data", "records", "news", "stories", "items"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return [item for item in rows if isinstance(item, dict)]
        return [payload]
    return []


def _vendor_headers_and_auth(source: str) -> tuple[dict[str, str], tuple[str, str] | None]:
    key, key_env, header = _vendor_key(source)
    headers: dict[str, str] = {}
    auth = None
    if key:
        if header.lower() == "authorization":
            headers[header] = f"Bearer {key}"
        else:
            headers[header] = key
    if source == "wrds" and not key:
        username = os.environ.get("WRDS_USERNAME")
        password = os.environ.get("WRDS_PASSWORD")
        if username and password:
            auth = (username, password)
    if key_env:
        headers["X-Prophet-Vendor-Key-Env"] = key_env
    return headers, auth


def _vendor_url(source: str) -> tuple[str | None, str | None]:
    for env_name in _url_envs(source):
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, None


def _vendor_key(source: str) -> tuple[str | None, str | None, str]:
    explicit_header = os.environ.get(f"{source.upper()}_API_KEY_HEADER")
    for env_name in _key_envs(source):
        value = os.environ.get(env_name)
        if not value:
            continue
        if explicit_header:
            return value, env_name, explicit_header
        if env_name.startswith("LSEG_APP_KEY") or env_name.startswith("LSED_APP_KEY"):
            return value, env_name, "App-Key"
        return value, env_name, "Authorization"
    return None, None, explicit_header or "Authorization"


def _url_envs(source: str) -> list[str]:
    prefix = source.upper()
    if source == "lseg":
        return ["LSEG_NEWS_API_URL", "LSEG_API_URL"]
    if source == "wrds":
        return ["WRDS_NEWS_API_URL", "WRDS_API_URL"]
    return [f"{prefix}_NEWS_API_URL", f"{prefix}_API_URL"]


def _key_envs(source: str) -> list[str]:
    if source == "lseg":
        return [
            "LSEG_API_KEY",
            "LSEG_APP_KEY",
            "LSEG_APP_KEY_EIKON",
            "LSEG_APP_KEY_SIDE_BY_SIDE",
            "LSED_APP_KEY_SIDE_BY_SIDE",
        ]
    if source == "wrds":
        return ["WRDS_API_KEY", "WRDS_ACCESS_TOKEN"]
    return [f"{source.upper()}_API_KEY", f"{source.upper()}_ACCESS_TOKEN"]


def _canonical_vendor(source: str) -> str:
    normalized = source.strip().lower()
    if normalized not in VENDOR_SOURCES:
        raise ValueError(f"unsupported vendor source: {source}")
    return normalized


def _source_timeout(cfg: ForecastConfig, deadline_at: float | None) -> float:
    configured = float(os.environ.get("ARENA_EVIDENCE_SOURCE_TIMEOUT_SECONDS", cfg.arena.evidence_source_timeout_seconds))
    if deadline_at is None:
        return max(0.1, configured)
    return max(0.1, min(configured, deadline_at - time.monotonic()))


def _run_native_fetch_with_timeout(
    source: str,
    fetch_fn: Any,
    *,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            records, errors = fetch_fn()
            result["records"] = records
            result["errors"] = errors
        except Exception as exc:  # noqa: BLE001
            result["records"] = []
            result["errors"] = [{"source": source, "error": f"{type(exc).__name__}:{exc}"}]

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(max(0.1, timeout_seconds))
    if thread.is_alive():
        return [], [{"source": source, "error": f"native_vendor_timeout:{timeout_seconds:.2f}s"}]
    return result.get("records") or [], result.get("errors") or []


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lower_lookup = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            value = lower_lookup.get(key.lower())
        if value is not None and value != "":
            return value
    return None


def _render_wrds_sql(sql: str, as_of_dt: datetime) -> str:
    return sql.replace("{year}", str(as_of_dt.year)).replace("{previous_year}", str(as_of_dt.year - 1))


def _vendor_backend(source: str, url: str | None) -> str:
    if not url:
        return "missing"
    scheme = urlparse(url).scheme.lower()
    if scheme in {"http", "https"}:
        return "http"
    if source == "lseg" and scheme in {"lseg-data-library", "lseg"}:
        return "lseg_data_library"
    if source == "wrds" and scheme in {"wrds-postgres", "wrds"}:
        return "wrds_postgres"
    return "unsupported"


def _native_url_schemes(source: str) -> list[str]:
    if source == "lseg":
        return ["lseg-data-library://news"]
    if source == "wrds":
        return ["wrds-postgres://news"]
    return []


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return False
    return True


def _wrds_sql() -> str | None:
    sql = os.environ.get("WRDS_NEWS_SQL")
    if sql:
        return sql
    sql_file = os.environ.get("WRDS_NEWS_SQL_FILE")
    if sql_file:
        candidates = [Path(sql_file), PACKAGE_ROOT / sql_file, PACKAGE_ROOT.parent / sql_file]
        for path in candidates:
            if path.exists():
                return path.read_text(encoding="utf-8")
    return None


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _lseg_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _dataframe_rows(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "reset_index"):
        value = value.reset_index()
    if hasattr(value, "to_dict"):
        rows = value.to_dict("records")
        return [dict(row) for row in rows if isinstance(row, dict)]
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return _extract_rows(value)
    return []


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _as_iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    status = sub.add_parser("status")
    status.add_argument("--source", choices=sorted(VENDOR_SOURCES))
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--source", choices=sorted(VENDOR_SOURCES), required=True)
    fetch.add_argument("--query", required=True)
    fetch.add_argument("--as-of", required=True)
    fetch.add_argument("--output", type=Path)
    normalize = sub.add_parser("normalize")
    normalize.add_argument("--source", choices=sorted(VENDOR_SOURCES), required=True)
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--collected-at")
    args = parser.parse_args()

    load_local_env()
    cfg = load_config()
    if args.cmd == "status":
        sources = [args.source] if args.source else sorted(VENDOR_SOURCES)
        print(json.dumps({source: vendor_env_status(source) for source in sources}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "fetch":
        records, errors = fetch_vendor_records(args.source, args.query, args.as_of, cfg)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                "\n".join(json.dumps(record, sort_keys=True) for record in records) + ("\n" if records else ""),
                encoding="utf-8",
            )
        print(json.dumps({
            "errors": errors,
            "n_records": len(records),
            "output": str(args.output) if args.output else None,
            "source": args.source,
        }, indent=2, sort_keys=True))
        return 0
    if args.cmd == "normalize":
        n = normalize_file(args.source, args.input, args.output, collected_at=args.collected_at)
        print(json.dumps({"output": str(args.output), "n_records": n}, indent=2, sort_keys=True))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
