"""Load the public Prophet-Arena-Subset-100 dataset into the same event
shape the hackathon's `--local` predict function receives.

The HuggingFace CSV stores one row per *event*, where each event has 1+
binary *markets*. The production agent contract (see
`ai-prophet/packages/cli/ai_prophet/forecast/example_agent.py`) takes one
market at a time. We flatten accordingly: N rows → M >= N (event, market)
pairs.

Each returned `event` dict matches the production EventRequest schema, so
a predict_fn written here will work unchanged against the live server.
We expose the market price snapshot separately, since the production
EventRequest does *not* include it — but it's a free signal to anchor on.
"""

from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "reference" / "subset_data_100.csv"


@dataclass
class Sample:
    """One (event, market) pair to predict on.

    `event` matches the production EventRequest dict. `market_info` is the
    Kalshi snapshot for that market (last_price, yes_ask, no_ask, ...) —
    use it for the market-price baseline. `outcome` is the binary ground
    truth (1 if the market resolved YES, else 0).
    """

    event: dict
    market_info: dict
    outcome: int


def _safe_literal_eval(s):
    if pd.isna(s) or s == "":
        return None
    try:
        return literal_eval(s)
    except Exception:
        return None


def _market_info_to_event(event_row: pd.Series, market_name: str, market_info: dict) -> dict:
    return {
        "event_ticker": event_row["event_ticker"],
        "market_ticker": market_info.get("ticker", f"{event_row['event_ticker']}-{market_name}"),
        "title": market_info.get("title") or event_row["title"],
        "subtitle": market_info.get("subtitle") or None,
        "description": None,
        "category": event_row["category"],
        "rules": market_info.get("rules_primary") or None,
        "close_time": market_info.get("close_time") or event_row["close_time"],
    }


def load_subset_100(csv_path: Path = DEFAULT_CSV) -> list[Sample]:
    df = pd.read_csv(csv_path)
    samples: list[Sample] = []
    for _, row in df.iterrows():
        outcomes = _safe_literal_eval(row["market_outcome"]) or {}
        market_info_all = _safe_literal_eval(row["market_info"]) or {}
        for market_name, outcome in outcomes.items():
            mi = market_info_all.get(market_name, {})
            event = _market_info_to_event(row, market_name, mi)
            samples.append(Sample(event=event, market_info=mi, outcome=int(outcome)))
    return samples


SUBSET_1200_CSV = Path(__file__).resolve().parents[2] / "data" / "external" / "subset_1200.csv"


def load_subset_1200(csv_path: Path = SUBSET_1200_CSV) -> list[Sample]:
    """Load the OFFICIAL hackathon-hosts benchmark (Prophet Arena Subset 1200).

    Same structure as the 100-subset but 12x bigger (1,200 submissions
    spanning 897 unique events, June–Nov 2025). The 'market_data' column
    has full bid/ask/liquidity for each market in an event.

    This is the most authoritative backtest we have — curated by the
    actual organizers, representing exactly the distribution they think
    is fair to evaluate on.
    """
    df = pd.read_csv(csv_path)
    samples: list[Sample] = []
    for _, row in df.iterrows():
        outcomes = _safe_literal_eval(row["market_outcome"]) or {}
        market_data_all = _safe_literal_eval(row["market_data"]) or {}
        for market_name, outcome in outcomes.items():
            md = market_data_all.get(market_name) or {}
            event = {
                "event_ticker": row["event_ticker"],
                "market_ticker": f"{row['event_ticker']}-{market_name.replace(' ', '_')}",
                "title": row.get("title") or "",
                "subtitle": market_name,
                "description": None,
                "category": row.get("category") or "Other",
                "rules": row.get("rules") or None,
                "close_time": row.get("close_time") or "",
            }
            samples.append(Sample(event=event, market_info=md, outcome=int(outcome)))
    return samples


def filter_by_category(samples: Iterable[Sample], category: str) -> list[Sample]:
    return [s for s in samples if s.event["category"] == category]


def load_hf_eval_set(min_snapshots: int = 2, use_earliest: bool = True) -> list[Sample]:
    """Load the HF-ingested Kalshi trades dataset (eval_pack_hf.jsonl).

    Uses real trade prices, not bid/ask snapshots. Median trajectory length
    is ~11 snapshots per market — orders of magnitude richer than our
    self-polled data. Covers May-Jul 2025.

    Set use_earliest=True (default) to use the first snapshot as the
    "agent's view at start" — simulates the live trading scenario where
    the agent sees a market when it first appears in the candidate set.
    Set use_earliest=False to use the latest snapshot (near-close prices).
    """
    import json
    pack_path = PREP_ROOT / "data" / "eval_pack_hf.jsonl"
    if not pack_path.exists():
        return []
    samples: list[Sample] = []
    for line in pack_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        snaps = row.get("snapshots") or []
        if len(snaps) < min_snapshots:
            continue
        snap = snaps[0] if use_earliest else snaps[-1]
        info = {
            "yes_ask": (snap.get("yes_ask") or 0) * 100 if snap.get("yes_ask") is not None else None,
            "no_ask": (snap.get("no_ask") or 0) * 100 if snap.get("no_ask") is not None else None,
            "last_price": (snap.get("last_price") or 0) * 100 if snap.get("last_price") is not None else None,
        }
        samples.append(Sample(event=row["event"], market_info=info, outcome=int(row["outcome"])))
    return samples


def load_clean_eval_set(min_snapshots: int = 2) -> list[Sample]:
    """Return Samples derived from the cleaner subset of the eval pack:
    only markets where we captured >=2 snapshots during their lifecycle.

    Each sample's market_info uses the EARLIEST snapshot's prices —
    simulating what a trading agent sees when a market first appears in
    its candidate set, not the converged-near-close price.

    Use this instead of load_local_snapshots() when measuring strategy
    edge. The full eval pack inflates apparent performance because the
    backfill captures prices near settlement.
    """
    import json
    pack_path = PREP_ROOT / "data" / "eval_pack.jsonl"
    if not pack_path.exists():
        return []

    samples: list[Sample] = []
    for line in pack_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        snaps = row.get("snapshots") or []
        if len(snaps) < min_snapshots:
            continue
        first = snaps[0]
        # The snapshots in eval_pack.jsonl already have prices in dollars (0–1).
        # Build a market_info dict in cents (0–100) so the harness's
        # _normalize() round-trip still works.
        info = {
            "yes_ask": (first.get("yes_ask") or 0) * 100 if first.get("yes_ask") is not None else None,
            "no_ask": (first.get("no_ask") or 0) * 100 if first.get("no_ask") is not None else None,
            "last_price": (first.get("last_price") or 0) * 100 if first.get("last_price") is not None else None,
        }
        samples.append(Sample(event=row["event"], market_info=info, outcome=int(row["outcome"])))
    return samples


# ---------------------------------------------------------------------------
# Local snapshot loader — fresh, contamination-free eval data we collect
# ourselves via scripts/snapshot.py + scripts/resolve.py.
# ---------------------------------------------------------------------------

PREP_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = PREP_ROOT / "data" / "snapshots"
OUTCOMES_PATH = PREP_ROOT / "data" / "outcomes.jsonl"


def _load_outcomes() -> dict[str, int]:
    if not OUTCOMES_PATH.exists():
        return {}
    out: dict[str, int] = {}
    import json
    for line in OUTCOMES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("market_ticker") and row.get("outcome") is not None:
                out[row["market_ticker"]] = int(row["outcome"])
        except Exception:
            continue
    return out


def _category_label(event_ticker: str) -> str:
    """Best-effort category from Kalshi event-ticker prefix. Mirrors the
    mapping in scripts/consolidate.py so the live loader and consolidated
    pack agree on category."""
    if not event_ticker:
        return "Other"
    p = event_ticker.split("-")[0].upper()
    crypto = ("KXBTC", "KXETH", "KXSOL", "KXXRP", "KXHYPE", "KXBNB", "KXDOGE",
              "KXSUI", "KXAVAX", "KXLTC", "KXLINK", "KXBCH")
    if any(p.startswith(c) for c in crypto):
        return "Crypto"
    sports = ("KXATPMATCH", "KXNBA", "KXNFL", "KXMLB", "KXNHL", "KXR6MAP",
              "KXSOCC", "KXCS", "KXLOL", "KXDOTA", "KXVAL", "KXGOLF",
              "KXTEN", "KXWTAMATCH", "KXUEFA", "KXNCAA")
    if any(p.startswith(c) for c in sports):
        return "Sports"
    if p.startswith(("KXPRES", "KXSENATE", "KXHOUSE", "KXGOV", "KXELEC")):
        return "Politics"
    if p.startswith(("KXTEMP", "KXRAIN", "KXSNOW", "KXHURR")):
        return "Weather"
    if p.startswith(("KX30Y", "KXFED", "KXCPI", "KXJOBS", "KXGDP", "KXNATGAS",
                     "KXOIL", "KXJETFUEL")):
        return "Economics"
    if p.startswith(("KXOSCAR", "KXBOX", "KXMOVIE", "KXMUSIC")):
        return "Entertainment"
    return "Other"


def _market_to_event(market: dict) -> dict:
    event_ticker = market.get("event_ticker") or ""
    return {
        "event_ticker": event_ticker,
        "market_ticker": market.get("ticker") or "",
        "title": market.get("title") or "",
        "subtitle": market.get("subtitle") or market.get("yes_sub_title") or None,
        "description": None,
        "category": market.get("category") or _category_label(event_ticker),
        "rules": market.get("rules_primary") or None,
        "close_time": market.get("close_time") or "",
    }


def _normalize_market_info(market: dict) -> dict:
    """Map both old (cents) and new (_dollars) Kalshi schemas into a common
    dict with float probabilities in [0, 1] for yes_ask / no_ask / last_price.
    """
    def _pick(key: str) -> float | None:
        # Modern API uses *_dollars (0-1 range); legacy uses cents (0-100).
        # Check for None, not falsiness — a price of 0.0 is valid info.
        d = market.get(f"{key}_dollars")
        if d is not None:
            try:
                return float(d)
            except Exception:
                pass
        c = market.get(key)
        if c is not None:
            try:
                return float(c) / 100.0
            except Exception:
                pass
        return None

    info = dict(market)
    info["yes_ask"] = _pick("yes_ask")
    info["no_ask"] = _pick("no_ask")
    info["last_price"] = _pick("last_price")
    # market.py expects 0–100 cent ranges — keep that contract by scaling
    for k in ("yes_ask", "no_ask", "last_price"):
        if info.get(k) is not None:
            info[k] = info[k] * 100
    return info


def load_local_snapshots(*, snapshot_dir: Path | None = None) -> list[Sample]:
    """Load every resolved market from our local snapshot collection.

    By default uses the *most recent* snapshot per market (so prices reflect
    the latest pre-resolution state we captured). Override `snapshot_dir` to
    use one specific snapshot instead.
    """
    import json

    outcomes = _load_outcomes()
    if not outcomes:
        return []

    # ticker -> (snapshot_time, market)
    latest: dict[str, tuple[str, dict]] = {}
    dirs = [snapshot_dir] if snapshot_dir else sorted(SNAPSHOT_ROOT.iterdir()) if SNAPSHOT_ROOT.exists() else []
    for snap_dir in dirs:
        if not snap_dir or not snap_dir.is_dir():
            continue
        for fp in snap_dir.glob("*.json"):
            if fp.name == "_meta.json":
                continue
            try:
                data = json.loads(fp.read_text())
            except Exception:
                continue
            snap_time = data.get("snapshot_time", "")
            for m in data.get("markets", []):
                ticker = m.get("ticker")
                if not ticker or ticker not in outcomes:
                    continue
                prev = latest.get(ticker)
                if prev is None or snap_time > prev[0]:
                    latest[ticker] = (snap_time, m)

    samples: list[Sample] = []
    for ticker, (_, market) in latest.items():
        samples.append(Sample(
            event=_market_to_event(market),
            market_info=_normalize_market_info(market),
            outcome=outcomes[ticker],
        ))
    return samples
