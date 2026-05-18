"""Polymarket cross-venue price prior for Kalshi markets.

Offline bulk matching: scripts/poly_match.py. At runtime get_market_priors()
uses map.csv, or resolve_mapping() on cache miss (Gamma public-search).
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import requests

from .schemas import KalshiQuote, MarketPacket

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "kalshi_polymarket"
MAP_CSV = DATA_DIR / "map.csv"
NEG_CSV = DATA_DIR / "rejected.csv"
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"

_MAPPING_MEM: dict[str, tuple[str, str] | None] = {}

FAMILY_QUERY = {
    "KXSB":                 "Super Bowl",
    "KXMENWORLDCUP":        "FIFA World Cup",
    "KXMLB":                "World Series",
    "KXMLBNL":              "National League Championship Series",
    "KXNBA":                "NBA Finals",
    "KXNBAEAST":            "NBA Eastern Conference Finals",
    "KXNBAWEST":            "NBA Western Conference Finals",
    "KXNHL":                "Stanley Cup",
    "KXNFLMVP":             "NFL MVP",
    "KXWNBAMVP":            "WNBA MVP",
    "KXWNBA":               "WNBA Championship",
    "KXUCL":                "Champions League",
    "KXOSCARPIC":           "Best Picture Oscar",
    "KXOSCARACTO":          "Best Actor Oscar",
    "KXOSCARNOMACTO":       "Best Actor Oscar nominee",
    "KXOSCARNOMPIC":        "Best Picture Oscar nominee",
    "KXOSCARNOMDIR":        "Best Director Oscar nominee",
    "KXOSCARNOMSPLAY":      "Best Original Screenplay Oscar nominee",
    "KXOSCARNOMINTERFILM":  "Best International Feature Film Oscar nominee",
    "KXOSCARNOMBCASTING":   "Best Casting Oscar nominee",
    "KXGAMEAWARDS":         "Game of the Year",
    "KXPRESNOMR":           "2028 Republican presidential nomination",
    "KXPRESPERSON":         "2028 US Presidential Election",
    "KXNEXTAG":             "next Attorney General",
    "KXSPACEXBANKPUBLIC":   "SpaceX IPO underwriter",
    "KXVPRESNOMD":          "2028 Democratic Vice Presidential nominee",
    "KXSTATE51":            "51st state",
    "KXROLEATEVENTCOACHELLA": "Coachella 2027",
}

SHORT_LABEL_ALIASES = {
    "man city":        "manchester city",
    "man u":           "manchester united",
    "a's":             "athletics",
    "chicago ws":      "chicago white sox",
    "donald j. trump": "donald trump",
}

SENATE_STATE = {
    "SENATEAK":    "Alaska",
    "SENATEME":    "Maine",
    "SENATETX":    "Texas",
    "SENATENE":    "Nebraska",
    "KXSENATELAD": "Louisiana",
}

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}

STOP = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "by", "for", "with", "or", "and", "not", "vs", "than",
    "more", "less", "over", "under", "winner", "win", "wins", "won", "this",
    "it", "who", "what", "which", "that", "any", "part", "some", "next",
    "his", "her", "their", "before", "after",
}

_DEADLINE_RE = re.compile(r"\b(?:by|before)\s+(?:\w+\s+){0,2}\d", re.IGNORECASE)


class MarketPrior(NamedTuple):
    quote: KalshiQuote
    exchange: str
    title: str


def _append_row(path: Path, header: list[str], row: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


def _load_map() -> dict[str, tuple[str, str]]:
    if not MAP_CSV.exists():
        return {}
    with MAP_CSV.open() as f:
        return {
            row["kalshi_ticker"]: (row["poly_condition_id"], row["poly_outcome"])
            for row in csv.DictReader(f)
        }


def _load_negative() -> set[str]:
    if not NEG_CSV.exists():
        return set()
    with NEG_CSV.open() as f:
        return {row["kalshi_ticker"] for row in csv.DictReader(f)}


def _parse_outcomes(market: dict) -> list[str]:
    raw = market.get("outcomes")
    if isinstance(raw, str):
        try:
            return [str(o) for o in json.loads(raw)]
        except json.JSONDecodeError:
            return []
    return [str(o) for o in raw] if isinstance(raw, list) else []


def meta_from_packet(packet: MarketPacket) -> dict:
    desc = packet.rules or ""
    extra = (packet.retrieval or {}).get("description")
    if extra:
        desc = f"{desc} {extra}".strip()
    ticker = packet.market_ticker or ""
    return {
        "ticker": ticker,
        "question": packet.title or "",
        "short_label": packet.subtitle or "",
        "description": desc,
        "family": ticker.split("-")[0] if ticker else "",
    }


def meta_from_snapshot_row(m: dict) -> dict:
    ticker = m["market_id"].removeprefix("kalshi:")
    return {
        "ticker": ticker,
        "question": m.get("question") or m.get("title") or "",
        "description": m.get("description") or "",
        "short_label": (
            m.get("short_label") or m.get("yes_sub_title") or m.get("subtitle") or ""
        ),
        "family": m.get("family") or (ticker.split("-")[0] if ticker else ""),
    }


def _keywords(title: str, drop: str | None = None) -> str:
    t = title.rstrip("?").strip()
    if drop and drop.lower() in t.lower():
        t = re.sub(re.escape(drop), "", t, count=1, flags=re.I)
    words = re.findall(r"[A-Za-z0-9'.]+", t)
    return " ".join(w for w in words if w.lower() not in STOP and len(w) > 1)


def active_search(query: str, limit: int = 200) -> list[dict]:
    try:
        rs = requests.get(
            GAMMA_SEARCH,
            params={
                "q": query,
                "limit_per_type": 100,
                "search_profiles": "false",
                "search_tags": "false",
            },
            timeout=15,
        )
        rs.raise_for_status()
    except requests.RequestException:
        return []

    now = datetime.now(timezone.utc)
    out: list[dict] = []
    for ev in rs.json().get("events", []):
        for m in ev.get("markets", []):
            if not m.get("active") or m.get("closed") or m.get("archived"):
                continue
            ed = m.get("endDateIso") or (m.get("endDate") or "")[:10]
            if ed:
                try:
                    if datetime.fromisoformat(ed).replace(tzinfo=timezone.utc) < now:
                        continue
                except ValueError:
                    pass
            out.append(m)
            if len(out) >= limit:
                return out
    return out


def _end_year(m: dict) -> int | None:
    ed = m.get("endDateIso") or (m.get("endDate") or "")[:10]
    try:
        return datetime.fromisoformat(ed).year if ed else None
    except ValueError:
        return None


def _has_explicit_deadline(question: str) -> bool:
    return bool(_DEADLINE_RE.search(question or ""))


def _all_years(text: str) -> set[int]:
    years: set[int] = set()
    for full, suffix in re.findall(r"\b(20[2-3]\d)(?:[-–](\d{2}))?\b", text or ""):
        years.add(int(full))
        if suffix:
            years.add(int(full[:2] + suffix))
    return years


def _ticker_year(ticker: str) -> int | None:
    m = re.search(r"-(\d{2})(?:-|$)", ticker)
    return 2000 + int(m.group(1)) if m else None


def _state_mention(text: str) -> str | None:
    t = (text or "").lower()
    for s in US_STATES:
        if re.search(r"\b" + re.escape(s) + r"\b", t):
            return s
    return None


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def _senate_match(
    candidates: list[dict], ticker: str, short_label: str,
) -> tuple[dict, str] | None:
    family = ticker.split("-")[0]
    state = SENATE_STATE.get(family)
    if not state:
        return None
    s_lower = state.lower()
    sl = _norm(short_label)
    pool = [
        m for m in candidates
        if s_lower in _norm(m.get("question") or "")
        and "senate" in _norm(m.get("question") or "")
    ]
    if not pool:
        return None
    pool.sort(key=lambda m: (
        "control the senate" in _norm(m.get("question") or ""),
        -float(m.get("volume24hr") or 0),
    ))
    chosen = pool[0]
    poly_q = _norm(chosen.get("question") or "")
    if "democrat" in sl:
        outcome = (
            "Yes" if "democrats win" in poly_q
            else ("No" if "republicans win" in poly_q else None)
        )
    elif "republican" in sl:
        outcome = (
            "Yes" if "republicans win" in poly_q
            else ("No" if "democrats win" in poly_q else None)
        )
    elif sl in poly_q:
        outcome = "Yes"
    else:
        outcome = None
    if outcome is None:
        return None
    return chosen, outcome


def _match(
    candidates: list[dict],
    kalshi_question: str,
    short_label: str,
    ticker: str,
    kalshi_description: str,
) -> dict | None:
    if not short_label:
        return None
    sl = _norm(short_label).strip()
    sl = SHORT_LABEL_ALIASES.get(sl, sl)
    hits = [m for m in candidates if sl in _norm(m.get("question") or "")]

    kal_years = _all_years(kalshi_question)
    ty = _ticker_year(ticker)
    fam = ticker.split("-")[0]
    relax_year = fam in {"KXNEXTAG", "KX2028RRUN", "KXNEXTUKPM"}

    def _year_ok(m: dict) -> bool:
        poly_years = _all_years(m.get("question", ""))
        if not poly_years:
            return True
        if kal_years:
            return bool(poly_years & kal_years)
        if relax_year:
            return True
        if ty is not None:
            return bool(poly_years & {ty - 1, ty, ty + 1})
        return True

    hits = [m for m in hits if _year_ok(m)]

    if ty is not None and fam not in {"KXNEXTAG", "KX2028RRUN", "KXNEXTUKPM"}:
        new_hits = []
        for m in hits:
            if _has_explicit_deadline(m.get("question", "")):
                ey = _end_year(m)
                if ey is not None and ey < ty - 1:
                    continue
            new_hits.append(m)
        hits = new_hits

    state = _state_mention(kalshi_description) or _state_mention(kalshi_question)
    if state:
        hits = [m for m in hits if state in (m.get("question") or "").lower()]

    if not hits:
        return None
    hits.sort(key=lambda m: float(m.get("volume24hr") or 0), reverse=True)
    return hits[0]


def resolve_mapping(
    meta: dict,
) -> tuple[dict | None, str, str, int]:
    """Returns (chosen_market, outcome, query, n_candidates)."""
    ticker = meta["ticker"]
    family = meta.get("family") or ticker.split("-")[0]
    short_label = meta.get("short_label") or ""
    question = meta.get("question") or ""
    description = meta.get("description") or ""

    senate_outcome: str | None = None
    state = SENATE_STATE.get(family)
    if state:
        q = f"{state} Senate race 2026"
        cands = active_search(q)
        sm = _senate_match(cands, ticker, short_label)
        chosen = sm[0] if sm else None
        senate_outcome = sm[1] if sm else None
    else:
        family_q = FAMILY_QUERY.get(family)
        q = (
            family_q
            or _keywords(question, drop=short_label)
            or _keywords(question)
            or question
        )
        cands = active_search(q)
        chosen = _match(cands, question, short_label, ticker, description)

    if chosen is None:
        return None, "", q, len(cands)

    outcomes = _parse_outcomes(chosen)
    if senate_outcome is not None:
        outcome = senate_outcome
    else:
        outcome = "Yes" if "Yes" in outcomes else (outcomes[0] if outcomes else "Yes")
    return chosen, outcome, q, len(cands)


def write_match(meta: dict, chosen: dict, outcome: str) -> None:
    ticker = meta["ticker"]
    cid = chosen.get("conditionId", "")
    _append_row(
        MAP_CSV,
        [
            "kalshi_ticker", "poly_condition_id", "poly_outcome",
            "kalshi_question", "poly_question", "poly_end_date", "poly_vol_24h",
        ],
        [
            ticker, cid, outcome,
            meta.get("question", ""), chosen.get("question", ""),
            chosen.get("endDateIso", ""),
            f"{float(chosen.get('volume24hr') or 0):.2f}",
        ],
    )
    _MAPPING_MEM[ticker] = (cid, outcome)


def write_reject(meta: dict, query: str, n_candidates: int) -> None:
    _append_row(
        NEG_CSV,
        ["kalshi_ticker", "kalshi_question", "short_label", "n_candidates", "query"],
        [
            meta["ticker"], meta.get("question", ""), meta.get("short_label", ""),
            str(n_candidates), query,
        ],
    )


def _lookup_mapping(ticker: str, meta: dict) -> tuple[str, str] | None:
    if ticker in _MAPPING_MEM:
        return _MAPPING_MEM[ticker]

    disk = _load_map().get(ticker)
    if disk:
        _MAPPING_MEM[ticker] = disk
        return disk

    if ticker in _load_negative():
        _MAPPING_MEM[ticker] = None
        return None

    chosen, outcome, _query, _n = resolve_mapping(meta)
    if chosen is None:
        _MAPPING_MEM[ticker] = None
        return None

    write_match(meta, chosen, outcome)
    return _MAPPING_MEM[ticker]


def _fetch_gamma_market(poly_cid: str) -> dict | None:
    try:
        rs = requests.get(GAMMA_MARKETS, params={"condition_ids": poly_cid}, timeout=15)
        rs.raise_for_status()
        markets = rs.json()
    except requests.RequestException:
        return None
    if not isinstance(markets, list) or not markets:
        return None
    return markets[0]


def _quote_from_gamma(poly_cid: str, outcome: str) -> tuple[KalshiQuote | None, str]:
    m = _fetch_gamma_market(poly_cid)
    if m is None:
        return None, ""

    if not m.get("active") or m.get("closed") or m.get("archived"):
        return None, m.get("question", "")

    outcomes = _parse_outcomes(m)
    if outcome not in outcomes:
        return None, m.get("question", "")

    try:
        bid = float(m.get("bestBid") or 0)
        ask = float(m.get("bestAsk") or 0)
        last = float(m.get("lastTradePrice") or 0)
    except (TypeError, ValueError):
        return None, m.get("question", "")

    if outcomes.index(outcome) == 1:
        bid, ask = 1.0 - ask, 1.0 - bid
        last = 1.0 - last if last else 0.0

    if ask <= 0.01 or ask >= 0.99:
        return None, m.get("question", "")

    quote = KalshiQuote(
        yes_bid=bid,
        yes_ask=ask,
        no_bid=1.0 - ask,
        no_ask=1.0 - bid,
        last_price=last or None,
        snapshot_time=datetime.now(timezone.utc).isoformat(),
    )
    return quote, m.get("question", "")


def get_market_priors(packet: MarketPacket) -> list[MarketPrior]:
    """Return at most one MarketPrior, or []. Resolves on cache miss via packet metadata."""
    ticker = packet.market_ticker
    if not ticker:
        return []

    mapping = _lookup_mapping(ticker, meta_from_packet(packet))
    if not mapping:
        return []

    poly_cid, outcome = mapping
    quote, title = _quote_from_gamma(poly_cid, outcome)
    if quote is None:
        return []

    return [MarketPrior(quote=quote, exchange="polymarket", title=title)]