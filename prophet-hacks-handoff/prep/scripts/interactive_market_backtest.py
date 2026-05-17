#!/usr/bin/env python3
"""Interactive point-in-time market backtest.

Randomly samples a historical market state, asks for a prediction, then compares
the prediction against the market-implied probability at that same timestamp.

Examples:
    python scripts/interactive_market_backtest.py --mode mixed --rounds 10
    python scripts/interactive_market_backtest.py --mode binary --rounds 5 --seed 7
    python scripts/interactive_market_backtest.py --mode nonbinary --auto-market
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY_ROOT = REPO_ROOT / "Kalshitopvolmarkets"
DEFAULT_NONBINARY_ROOT = REPO_ROOT / "NonBinaryMarkets"


@dataclass(frozen=True)
class Quote:
    yes_bid: float
    yes_ask: float
    last_price: float | None
    volume: float | None
    open_interest: float | None
    time: datetime
    raw: dict[str, Any]

    @property
    def market_mid(self) -> float:
        return clamp_prob((self.yes_bid + self.yes_ask) / 2.0)

    @property
    def spread(self) -> float:
        return max(0.0, self.yes_ask - self.yes_bid)


@dataclass(frozen=True)
class BinarySample:
    market: dict[str, Any]
    quote: Quote
    history: list[Quote]
    outcome_yes: int
    minutes_to_close: float
    week: str


@dataclass(frozen=True)
class ComponentState:
    market: dict[str, Any]
    quote: Quote
    outcome_yes: int


@dataclass(frozen=True)
class NonBinarySample:
    group: dict[str, Any]
    components: list[ComponentState]
    as_of: datetime
    minutes_to_close: float
    week: str


def clamp_prob(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_result_yes(value: str | None) -> int | None:
    result = (value or "").strip().lower()
    if result == "yes":
        return 1
    if result == "no":
        return 0
    return None


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def quote_from_candle(row: dict[str, Any]) -> Quote | None:
    yes_bid = parse_float(row.get("yes_bid_close"))
    yes_ask = parse_float(row.get("yes_ask_close"))
    if yes_bid is None or yes_ask is None:
        return None
    if not (0.0 <= yes_bid <= 1.0 and 0.0 <= yes_ask <= 1.0):
        return None
    if yes_ask <= 0.0 or yes_bid >= 1.0 or yes_bid > yes_ask:
        return None
    ts = parse_iso(row.get("end_period_time"))
    if ts is None:
        return None
    return Quote(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        last_price=parse_float(row.get("price_close")),
        volume=parse_float(row.get("volume")),
        open_interest=parse_float(row.get("open_interest")),
        time=ts,
        raw=row,
    )


def load_quotes(path: Path) -> list[Quote]:
    if not path.exists():
        return []
    quotes: list[Quote] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            quote = quote_from_candle(row)
            if quote is not None:
                quotes.append(quote)
    quotes.sort(key=lambda q: q.time)
    return quotes


def quote_at_or_before(quotes: list[Quote], as_of: datetime) -> Quote | None:
    best: Quote | None = None
    for quote in quotes:
        if quote.time > as_of:
            break
        best = quote
    return best


def binary_candle_path(root: Path, week: str, ticker: str, period: int) -> Path:
    return root / "ohlcv" / f"period_{period}m" / f"week={week}" / f"{ticker}.csv.gz"


def nonbinary_candle_path(root: Path, week: str, ticker: str, period: int) -> Path:
    return root / "ohlcv" / f"period_{period}m" / f"week={week}" / f"{ticker}.csv.gz"


def load_binary_markets(root: Path) -> list[dict[str, Any]]:
    path = root / "weekly_top_markets.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing binary market index: {path}")
    return [
        row for row in read_csv(path)
        if (row.get("status") or "").lower() == "finalized"
        and parse_result_yes(row.get("result")) is not None
    ]


def choose_binary_sample(
    *,
    rng: random.Random,
    root: Path,
    markets: list[dict[str, Any]],
    period: int,
    min_time_to_close_minutes: float,
    history_rows: int,
    max_attempts: int = 400,
) -> BinarySample:
    for _ in range(max_attempts):
        market = rng.choice(markets)
        week = (market.get("week_start") or "")[:10]
        ticker = market.get("ticker") or ""
        close_time = parse_iso(market.get("close_time"))
        outcome = parse_result_yes(market.get("result"))
        if not week or not ticker or close_time is None or outcome is None:
            continue
        quotes = load_quotes(binary_candle_path(root, week, ticker, period))
        eligible = [
            q for q in quotes
            if (close_time - q.time).total_seconds() / 60.0 >= min_time_to_close_minutes
        ]
        if not eligible:
            continue
        quote = rng.choice(eligible)
        idx = quotes.index(quote)
        history = quotes[max(0, idx - history_rows + 1):idx + 1]
        return BinarySample(
            market=market,
            quote=quote,
            history=history,
            outcome_yes=outcome,
            minutes_to_close=(close_time - quote.time).total_seconds() / 60.0,
            week=week,
        )
    raise RuntimeError("could not find an eligible binary market-time sample")


def load_nonbinary_groups(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    groups_path = root / "weekly_top_groups.csv"
    if not groups_path.exists():
        raise FileNotFoundError(f"missing nonbinary group index: {groups_path}")
    groups = read_csv(groups_path)
    components: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((root / "markets").glob("*_component_markets.jsonl")):
        week = path.name[:10]
        for row in read_jsonl(path):
            group_key = row.get("_context_group_key") or row.get("group_key")
            if group_key:
                components[(week, group_key)].append(row)
    return groups, components


def selected_component_tickers(group: dict[str, Any]) -> list[str]:
    raw = group.get("component_tickers") or ""
    if isinstance(raw, list):
        return [str(x) for x in raw if x]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def valid_nonbinary_group(
    group: dict[str, Any],
    components: list[dict[str, Any]],
    *,
    min_components: int,
    require_complete: bool,
) -> bool:
    selected = selected_component_tickers(group)
    if len(selected) < min_components:
        return False
    if require_complete:
        total = int(float(group.get("component_count") or len(selected)))
        if total != len(selected):
            return False
    component_by_ticker = {c.get("ticker"): c for c in components}
    selected_components = [component_by_ticker.get(t) for t in selected]
    if any(c is None for c in selected_components):
        return False
    outcomes = [parse_result_yes(c.get("result")) for c in selected_components if c is not None]
    return len(outcomes) == len(selected) and sum(outcomes) == 1


def choose_nonbinary_sample(
    *,
    rng: random.Random,
    root: Path,
    groups: list[dict[str, Any]],
    components_by_group: dict[tuple[str, str], list[dict[str, Any]]],
    period: int,
    min_time_to_close_minutes: float,
    min_components: int,
    require_complete: bool,
    max_attempts: int = 400,
) -> NonBinarySample:
    candidates = list(groups)
    for _ in range(max_attempts):
        group = rng.choice(candidates)
        week = (group.get("week_start") or "")[:10]
        group_key = group.get("group_key") or ""
        close_time = parse_iso(group.get("max_close_time") or group.get("min_close_time"))
        if not week or not group_key or close_time is None:
            continue
        components = components_by_group.get((week, group_key), [])
        if not valid_nonbinary_group(
            group,
            components,
            min_components=min_components,
            require_complete=require_complete,
        ):
            continue

        component_by_ticker = {c.get("ticker"): c for c in components}
        selected = selected_component_tickers(group)
        quote_lists: dict[str, list[Quote]] = {}
        timestamp_sets: list[set[datetime]] = []
        for ticker in selected:
            quotes = load_quotes(nonbinary_candle_path(root, week, ticker, period))
            quotes = [
                q for q in quotes
                if (close_time - q.time).total_seconds() / 60.0 >= min_time_to_close_minutes
            ]
            if not quotes:
                break
            quote_lists[ticker] = quotes
            timestamp_sets.append({q.time for q in quotes})
        if len(quote_lists) != len(selected):
            continue
        common_times = set.intersection(*timestamp_sets) if timestamp_sets else set()
        if not common_times:
            continue
        as_of = rng.choice(sorted(common_times))
        states: list[ComponentState] = []
        for ticker in selected:
            quote = next(q for q in quote_lists[ticker] if q.time == as_of)
            market = component_by_ticker[ticker]
            outcome = parse_result_yes(market.get("result"))
            if outcome is None:
                break
            states.append(ComponentState(market=market, quote=quote, outcome_yes=outcome))
        if len(states) != len(selected):
            continue
        return NonBinarySample(
            group=group,
            components=states,
            as_of=as_of,
            minutes_to_close=(close_time - as_of).total_seconds() / 60.0,
            week=week,
        )
    raise RuntimeError("could not find an eligible nonbinary group-time sample")


def normalize_distribution(values: list[float]) -> list[float]:
    cleaned = [max(0.0, float(v)) for v in values]
    total = sum(cleaned)
    if total <= 0:
        return [1.0 / len(cleaned)] * len(cleaned)
    return [v / total for v in cleaned]


def market_distribution(states: list[ComponentState]) -> list[float]:
    return normalize_distribution([state.quote.market_mid for state in states])


def binary_brier(p_yes: float, outcome_yes: int) -> float:
    return (p_yes - outcome_yes) ** 2


def multiclass_brier(probs: list[float], outcomes: list[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes))


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def prompt_binary(sample: BinarySample, *, history_rows: int) -> None:
    market = sample.market
    print("\n" + "=" * 80)
    print("BINARY MARKET")
    print(f"Ticker: {market.get('ticker')}")
    print(f"Title:  {market.get('title')}")
    subtitle = market.get("subtitle")
    if subtitle:
        print(f"Sub:    {subtitle}")
    print(f"Series: {market.get('series_ticker')}   Week: {sample.week}")
    print(f"As of:  {sample.quote.time.isoformat()}   Close: {market.get('close_time')}")
    print(f"Time to close: {sample.minutes_to_close:.1f} minutes")
    rules = market.get("rules_primary")
    if rules:
        print(f"Rules:  {rules[:500]}")
    print("\nMarket at as-of:")
    print(
        f"  YES bid {fmt_pct(sample.quote.yes_bid)} | YES ask {fmt_pct(sample.quote.yes_ask)} | "
        f"mid {fmt_pct(sample.quote.market_mid)} | spread {fmt_pct(sample.quote.spread)}"
    )
    print("\nRecent history:")
    for quote in sample.history[-history_rows:]:
        print(
            f"  {quote.time.isoformat()}  mid={fmt_pct(quote.market_mid)} "
            f"bid={fmt_pct(quote.yes_bid)} ask={fmt_pct(quote.yes_ask)} "
            f"vol={quote.volume if quote.volume is not None else ''}"
        )


def parse_probability(text: str) -> float:
    text = text.strip().replace("%", "")
    value = float(text)
    if value > 1.0:
        value = value / 100.0
    if not (0.0 <= value <= 1.0):
        raise ValueError("probability must be in [0, 1] or [0, 100]")
    return value


def ask_binary_prediction(sample: BinarySample, auto: str | None) -> float:
    if auto == "market":
        return sample.quote.market_mid
    if auto == "uniform":
        return 0.5
    while True:
        raw = input("Your P(YES) [0-1 or %]: ")
        try:
            return parse_probability(raw)
        except ValueError as exc:
            print(f"Invalid probability: {exc}")


def score_binary(sample: BinarySample, pred: float) -> dict[str, float]:
    market_p = sample.quote.market_mid
    user_brier = binary_brier(pred, sample.outcome_yes)
    market_brier = binary_brier(market_p, sample.outcome_yes)
    print("\nResult:")
    print(f"  Outcome: {'YES' if sample.outcome_yes else 'NO'}")
    print(f"  Your p_yes:   {fmt_pct(pred)}  Brier={user_brier:.4f}")
    print(f"  Market p_yes: {fmt_pct(market_p)}  Brier={market_brier:.4f}")
    print(f"  Delta vs market: {user_brier - market_brier:+.4f} ({'beat' if user_brier < market_brier else 'lost to' if user_brier > market_brier else 'tied'} market)")
    return {"user_brier": user_brier, "market_brier": market_brier}


def prompt_nonbinary(sample: NonBinarySample) -> None:
    group = sample.group
    market_probs = market_distribution(sample.components)
    print("\n" + "=" * 80)
    print("NONBINARY / SIBLING OUTCOME SET")
    print(f"Group:  {group.get('group_key')}")
    print(f"Title:  {group.get('representative_title')}")
    print(f"Series: {group.get('series_tickers')}   Week: {sample.week}")
    print(f"As of:  {sample.as_of.isoformat()}   Close: {group.get('max_close_time') or group.get('min_close_time')}")
    print(f"Time to close: {sample.minutes_to_close:.1f} minutes")
    print("\nComponents at as-of:")
    for idx, (state, market_p) in enumerate(zip(sample.components, market_probs), 1):
        ticker = state.market.get("ticker")
        title = state.market.get("title") or ""
        subtitle = state.market.get("subtitle") or ""
        label = subtitle or title.replace(str(sample.group.get("representative_title") or ""), "").strip() or ticker
        print(
            f"  {idx:2d}. {ticker:42s} market={fmt_pct(market_p)} "
            f"mid={fmt_pct(state.quote.market_mid)} spread={fmt_pct(state.quote.spread)} "
            f"label={label[:80]}"
        )


def parse_distribution(text: str, n: int) -> list[float]:
    parts = [p.strip().replace("%", "") for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) != n:
        raise ValueError(f"expected {n} comma-separated probabilities")
    values = [float(p) for p in parts]
    if any(v > 1.0 for v in values):
        values = [v / 100.0 for v in values]
    if any(v < 0.0 for v in values):
        raise ValueError("probabilities cannot be negative")
    return normalize_distribution(values)


def ask_nonbinary_prediction(sample: NonBinarySample, auto: str | None) -> list[float]:
    n = len(sample.components)
    if auto == "market":
        return market_distribution(sample.components)
    if auto == "uniform":
        return [1.0 / n] * n
    print("\nEnter probabilities in displayed order.")
    print("They can be decimals or percentages and do not need to sum exactly to 1; I will normalize.")
    print("Shortcuts: `market`, `uniform`.")
    while True:
        raw = input(f"Your distribution ({n} comma-separated values): ").strip()
        try:
            if raw.lower() == "market":
                return market_distribution(sample.components)
            if raw.lower() == "uniform":
                return [1.0 / n] * n
            return parse_distribution(raw, n)
        except ValueError as exc:
            print(f"Invalid distribution: {exc}")


def score_nonbinary(sample: NonBinarySample, pred: list[float]) -> dict[str, float]:
    market_probs = market_distribution(sample.components)
    outcomes = [state.outcome_yes for state in sample.components]
    user_brier = multiclass_brier(pred, outcomes)
    market_brier = multiclass_brier(market_probs, outcomes)
    winner = outcomes.index(1)
    print("\nResult:")
    print(f"  Winner: #{winner + 1} {sample.components[winner].market.get('ticker')}")
    print(f"  Your Brier:   {user_brier:.4f}")
    print(f"  Market Brier: {market_brier:.4f}")
    print(f"  Delta vs market: {user_brier - market_brier:+.4f} ({'beat' if user_brier < market_brier else 'lost to' if user_brier > market_brier else 'tied'} market)")
    print("\nPredictions:")
    for idx, (state, user_p, market_p, outcome) in enumerate(zip(sample.components, pred, market_probs, outcomes), 1):
        marker = "YES" if outcome else "no"
        print(
            f"  {idx:2d}. {state.market.get('ticker'):42s} "
            f"you={fmt_pct(user_p)} market={fmt_pct(market_p)} outcome={marker}"
        )
    return {"user_brier": user_brier, "market_brier": market_brier}


def choose_mode(rng: random.Random, mode: str) -> str:
    if mode == "mixed":
        return rng.choice(["binary", "nonbinary"])
    return mode


def summarize(scores: list[dict[str, float]]) -> None:
    if not scores:
        return
    user = sum(s["user_brier"] for s in scores) / len(scores)
    market = sum(s["market_brier"] for s in scores) / len(scores)
    wins = sum(1 for s in scores if s["user_brier"] < s["market_brier"])
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Rounds: {len(scores)}")
    print(f"Your average Brier:   {user:.4f}")
    print(f"Market average Brier: {market:.4f}")
    print(f"Average delta:        {user - market:+.4f}")
    print(f"Rounds beat market:   {wins}/{len(scores)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["binary", "nonbinary", "mixed"], default="mixed")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--binary-root", type=Path, default=DEFAULT_BINARY_ROOT)
    parser.add_argument("--nonbinary-root", type=Path, default=DEFAULT_NONBINARY_ROOT)
    parser.add_argument("--binary-period", type=int, default=1)
    parser.add_argument("--nonbinary-period", type=int, default=1)
    parser.add_argument("--history", type=int, default=8, help="number of recent binary candles to show")
    parser.add_argument("--min-time-to-close-minutes", type=float, default=30.0)
    parser.add_argument("--nonbinary-min-components", type=int, default=3)
    parser.add_argument("--allow-truncated-nonbinary", action="store_true")
    parser.add_argument("--auto-market", action="store_true", help="non-interactive smoke mode: use market probabilities")
    parser.add_argument("--auto-uniform", action="store_true", help="non-interactive smoke mode: use uniform probabilities")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    auto = "market" if args.auto_market else "uniform" if args.auto_uniform else None
    if args.auto_market and args.auto_uniform:
        raise SystemExit("--auto-market and --auto-uniform are mutually exclusive")

    binary_markets: list[dict[str, Any]] = []
    nonbinary_groups: list[dict[str, Any]] = []
    nonbinary_components: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if args.mode in {"binary", "mixed"}:
        binary_markets = load_binary_markets(args.binary_root)
    if args.mode in {"nonbinary", "mixed"}:
        nonbinary_groups, nonbinary_components = load_nonbinary_groups(args.nonbinary_root)

    print("Point-in-time backtest")
    print(f"Mode: {args.mode} | rounds: {args.rounds} | seed: {args.seed}")
    print("No final result fields are shown until after your prediction.")

    scores: list[dict[str, float]] = []
    for round_idx in range(1, args.rounds + 1):
        mode = choose_mode(rng, args.mode)
        print(f"\nRound {round_idx}/{args.rounds}")
        if mode == "binary":
            sample = choose_binary_sample(
                rng=rng,
                root=args.binary_root,
                markets=binary_markets,
                period=args.binary_period,
                min_time_to_close_minutes=args.min_time_to_close_minutes,
                history_rows=args.history,
            )
            prompt_binary(sample, history_rows=args.history)
            pred = ask_binary_prediction(sample, auto)
            scores.append(score_binary(sample, pred))
        else:
            sample = choose_nonbinary_sample(
                rng=rng,
                root=args.nonbinary_root,
                groups=nonbinary_groups,
                components_by_group=nonbinary_components,
                period=args.nonbinary_period,
                min_time_to_close_minutes=args.min_time_to_close_minutes,
                min_components=args.nonbinary_min_components,
                require_complete=not args.allow_truncated_nonbinary,
            )
            prompt_nonbinary(sample)
            pred = ask_nonbinary_prediction(sample, auto)
            scores.append(score_nonbinary(sample, pred))

    summarize(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
