"""Consolidate accumulated snapshots + outcomes into a single eval pack
ready for handoff to teammates.

Produces:
- data/eval_pack.jsonl       one row per (market, outcome), with the full
                             snapshot trajectory inlined (price over time)
- data/eval_pack_latest.csv  flat CSV — one row per market, latest snapshot
                             only. Easy to open in a spreadsheet.
- data/summary.md            human-readable summary: counts, categories,
                             temporal coverage, class balance

Run after each resolve. Idempotent and fast (reads from disk, no API).

Usage:
    python scripts/consolidate.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PREP_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_ROOT = PREP_ROOT / "data" / "snapshots"
OUTCOMES_PATH = PREP_ROOT / "data" / "outcomes.jsonl"
EVAL_PACK_PATH = PREP_ROOT / "data" / "eval_pack.jsonl"
EVAL_PACK_CSV = PREP_ROOT / "data" / "eval_pack_latest.csv"
SUMMARY_PATH = PREP_ROOT / "data" / "summary.md"


def _load_outcomes() -> dict[str, dict]:
    if not OUTCOMES_PATH.exists():
        return {}
    out: dict[str, dict] = {}
    for line in OUTCOMES_PATH.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row.get("market_ticker"):
                out[row["market_ticker"]] = row
        except Exception:
            continue
    return out


def _normalize_price(market: dict, key: str) -> float | None:
    """Return price as a probability in [0, 1], handling both old (cents)
    and new (_dollars) Kalshi schemas."""
    dollars = market.get(f"{key}_dollars")
    if dollars is not None:
        try:
            return float(dollars)
        except Exception:
            pass
    cents = market.get(key)
    if cents is not None:
        try:
            return float(cents) / 100.0
        except Exception:
            pass
    return None


def _event_prefix(event_ticker: str) -> str:
    """Best-effort category-ish prefix (e.g. KXBTC, KXNBA)."""
    if not event_ticker:
        return ""
    parts = event_ticker.split("-")
    return parts[0] if parts else ""


def _category_label(prefix: str) -> str:
    """Map Kalshi event prefixes to friendly categories. Best effort."""
    p = prefix.upper()
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
    if p.startswith(("KXBTCD", "KXETHD")):
        return "Crypto"
    if p.startswith(("KX30Y", "KXFED", "KXCPI", "KXJOBS", "KXGDP", "KXNATGAS",
                     "KXOIL", "KXJETFUEL")):
        return "Economics"
    if p.startswith(("KXOSCAR", "KXBOX", "KXMOVIE", "KXMUSIC")):
        return "Entertainment"
    return "Other"


def main() -> int:
    if not OUTCOMES_PATH.exists() or OUTCOMES_PATH.stat().st_size == 0:
        print("No outcomes yet — run scripts/resolve.py first.")
        return 0

    outcomes = _load_outcomes()
    print(f"Outcomes on disk: {len(outcomes)}")

    # Walk all snapshots in chronological order so per-market trajectory
    # comes out time-sorted.
    snap_dirs = sorted(SNAPSHOT_ROOT.iterdir()) if SNAPSHOT_ROOT.exists() else []

    # ticker -> list of (snapshot_time, market_dict)
    trajectory: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    # ticker -> latest snapshot dict
    latest_by_ticker: dict[str, tuple[str, dict]] = {}

    for snap_dir in snap_dirs:
        if not snap_dir.is_dir():
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
                yes_ask = _normalize_price(m, "yes_ask")
                no_ask = _normalize_price(m, "no_ask")
                yes_bid = _normalize_price(m, "yes_bid")
                no_bid = _normalize_price(m, "no_bid")
                last_price = _normalize_price(m, "last_price")
                trajectory[ticker].append((snap_time, {
                    "t": snap_time,
                    "yes_ask": yes_ask,
                    "no_ask": no_ask,
                    "yes_bid": yes_bid,
                    "no_bid": no_bid,
                    "last_price": last_price,
                }))
                prev = latest_by_ticker.get(ticker)
                if prev is None or snap_time > prev[0]:
                    latest_by_ticker[ticker] = (snap_time, m)

    # Sort trajectories.
    for ticker, traj in trajectory.items():
        traj.sort(key=lambda x: x[0])

    # Write the JSONL eval pack — one row per market with full trajectory.
    pack_count = 0
    with EVAL_PACK_PATH.open("w") as fh:
        for ticker, (_, latest) in latest_by_ticker.items():
            traj = [pt[1] for pt in trajectory[ticker]]
            outcome_row = outcomes[ticker]
            event = {
                "event_ticker": latest.get("event_ticker") or "",
                "market_ticker": ticker,
                "title": latest.get("title") or "",
                "subtitle": latest.get("subtitle") or latest.get("yes_sub_title") or None,
                "description": None,
                "category": _category_label(_event_prefix(latest.get("event_ticker") or "")),
                "rules": latest.get("rules_primary") or None,
                "close_time": latest.get("close_time") or "",
            }
            fh.write(json.dumps({
                "event": event,
                "snapshots": traj,
                "outcome": outcome_row["outcome"],
                "result": outcome_row["result"],
                "settled_at": outcome_row.get("settled_at"),
            }) + "\n")
            pack_count += 1

    # Write a flat CSV — one row per market, latest snapshot only.
    with EVAL_PACK_CSV.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "market_ticker", "event_ticker", "category", "title", "close_time",
            "latest_snapshot_time", "latest_yes_ask", "latest_no_ask",
            "latest_last_price", "n_snapshots", "outcome",
            "latest_yes_bid", "latest_no_bid",
        ])
        for ticker, (snap_time, latest) in latest_by_ticker.items():
            cat = _category_label(_event_prefix(latest.get("event_ticker") or ""))
            writer.writerow([
                ticker,
                latest.get("event_ticker") or "",
                cat,
                (latest.get("title") or "")[:200],
                latest.get("close_time") or "",
                snap_time,
                _normalize_price(latest, "yes_ask"),
                _normalize_price(latest, "no_ask"),
                _normalize_price(latest, "last_price"),
                len(trajectory[ticker]),
                outcomes[ticker]["outcome"],
                _normalize_price(latest, "yes_bid"),
                _normalize_price(latest, "no_bid"),
            ])

    # Summary.md
    by_cat = Counter(
        _category_label(_event_prefix(latest.get("event_ticker") or ""))
        for _, latest in latest_by_ticker.values()
    )
    by_outcome = Counter(outcomes[t]["outcome"] for t in latest_by_ticker)
    snapshots_per_market = Counter(len(trajectory[t]) for t in latest_by_ticker)
    snap_dir_count = sum(1 for d in snap_dirs if d.is_dir())

    # Baseline scores — the numbers any new agent should beat.
    # We compute these inline (rather than importing from prep) so consolidate
    # stays self-contained and won't break if package layout changes.
    def _brier(preds, outs):
        return sum((p - o) ** 2 for p, o in zip(preds, outs)) / len(preds) if preds else float("nan")

    def _ece(preds, outs, n_bins=10):
        if not preds:
            return float("nan")
        bins = [[] for _ in range(n_bins)]
        for p, o in zip(preds, outs):
            bins[min(int(p * n_bins), n_bins - 1)].append((p, o))
        total = 0.0
        for b in bins:
            if not b:
                continue
            ap = sum(p for p, _ in b) / len(b)
            ao = sum(o for _, o in b) / len(b)
            total += (len(b) / len(preds)) * abs(ap - ao)
        return total

    def _market_p(market: dict) -> float:
        ya = _normalize_price(market, "yes_ask")
        na = _normalize_price(market, "no_ask")
        if ya is not None and na is not None:
            return max(0.01, min(0.99, (ya + (1 - na)) / 2))
        lp = _normalize_price(market, "last_price")
        if lp is not None:
            return max(0.01, min(0.99, lp))
        return 0.5

    # Group (preds, outcomes) by category and overall.
    cat_data: dict[str, tuple[list[float], list[int]]] = {}
    all_market_preds: list[float] = []
    all_half_preds: list[float] = []
    all_outs: list[int] = []
    for ticker, (_, latest) in latest_by_ticker.items():
        cat = _category_label(_event_prefix(latest.get("event_ticker") or ""))
        out = outcomes[ticker]["outcome"]
        mp = _market_p(latest)
        cat_data.setdefault(cat, ([], [], []))[0].append(mp)
        cat_data[cat][1].append(0.5)
        cat_data[cat][2].append(out)
        all_market_preds.append(mp)
        all_half_preds.append(0.5)
        all_outs.append(out)

    lines = []
    lines.append("# Local eval pack summary")
    lines.append("")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append(f"- **Resolved markets in pack: {pack_count}**")
    lines.append(f"- Snapshot dirs scanned: {snap_dir_count}")
    lines.append(f"- Outcomes on disk: {len(outcomes)}")
    lines.append(f"- Class balance — YES: {by_outcome.get(1, 0)}, NO: {by_outcome.get(0, 0)}  "
                 f"(YES rate {by_outcome.get(1, 0) / max(1, pack_count):.3f})")
    lines.append("")
    lines.append("## Category breakdown")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for cat, n in by_cat.most_common():
        lines.append(f"| {cat} | {n} |")
    lines.append("")
    lines.append("## Baselines — the numbers your agent must beat")
    lines.append("")
    lines.append("Brier scoring rewards calibration over confidence. Random = 0.25, perfect = 0.")
    lines.append("ECE measures whether predicted probabilities match actual frequencies. Lower is better.")
    lines.append("")
    lines.append("| Category | N | always_half Brier | **market Brier** | market ECE |")
    lines.append("|---|---|---|---|---|")
    overall_market_brier = _brier(all_market_preds, all_outs)
    overall_half_brier = _brier(all_half_preds, all_outs)
    overall_market_ece = _ece(all_market_preds, all_outs)
    lines.append(f"| **all** | {len(all_outs)} | {overall_half_brier:.4f} | **{overall_market_brier:.4f}** | {overall_market_ece:.4f} |")
    for cat in sorted(cat_data.keys(), key=lambda c: -len(cat_data[c][2])):
        mp, hp, outs = cat_data[cat]
        lines.append(
            f"| {cat} | {len(outs)} | {_brier(hp, outs):.4f} | **{_brier(mp, outs):.4f}** | {_ece(mp, outs):.4f} |"
        )
    lines.append("")
    lines.append("**Reading this table:** if your agent can't beat `market Brier` on a category,")
    lines.append("you'd be better off just returning the market price for those events. The aggregate")
    lines.append("number is dominated by Crypto (which is near-deterministic). **Sports is the meaningful")
    lines.append("regression suite** — that's where agent skill differentiates.")
    lines.append("")
    lines.append("## Snapshots per market (trajectory length)")
    lines.append("")
    lines.append("| #snapshots | markets |")
    lines.append("|---|---|")
    for n, k in sorted(snapshots_per_market.items()):
        lines.append(f"| {n} | {k} |")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `data/eval_pack.jsonl` — one row per market with full price trajectory and outcome")
    lines.append(f"- `data/eval_pack_latest.csv` — flat CSV, latest snapshot per market")
    lines.append(f"- `data/outcomes.jsonl` — raw outcomes log")
    lines.append("")
    lines.append("Load via `prep.data.load_local_snapshots()` or just read `eval_pack.jsonl` directly.")

    SUMMARY_PATH.write_text("\n".join(lines))

    print(f"Wrote {pack_count} markets to eval_pack.jsonl")
    print(f"Wrote {pack_count} rows to eval_pack_latest.csv")
    print(f"Wrote summary to {SUMMARY_PATH}")
    print()
    print("Quick stats:")
    print(f"  Category breakdown: {dict(by_cat.most_common())}")
    print(f"  Class balance: YES={by_outcome.get(1, 0)}, NO={by_outcome.get(0, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
