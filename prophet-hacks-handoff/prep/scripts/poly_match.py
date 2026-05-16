"""Match Kalshi -> Polymarket via Gamma public-search.

Pipeline (deterministic, no LLM):
  1. Build a topical query from the Kalshi market (keywords minus stopwords,
     minus short_label so we match the event, not just the outcome).
  2. Hit Gamma /public-search.
  3. Filter to markets where active=true, closed=false, endDate in the future.
  4. Among those, keep ones whose question contains the Kalshi short_label
     (case-insensitive substring). Tiebreak by volume24hr (most liquid wins).
  5. Write to data/kalshi_polymarket/map.csv or rejected.csv.

Usage:
  python scripts/poly_match.py SNAPSHOT.json [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests

PREP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PREP_ROOT / "src"))

from prep.polymarket import MAP_CSV, NEG_CSV, _append_row, _load_map, _load_negative  # noqa: E402

GAMMA_SEARCH = "https://gamma-api.polymarket.com/public-search"

# Kalshi avoids trademarked names; Polymarket uses them. Override the query
# for families where the Kalshi question wording wouldn't surface the right
# Polymarket event (e.g., "Pro Football Championship" -> "Super Bowl").
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

# Kalshi short_labels that need an alias to match Polymarket phrasing.
SHORT_LABEL_ALIASES = {
    "man city":       "manchester city",
    "man u":          "manchester united",
    "a's":            "athletics",
    "chicago ws":     "chicago white sox",
    "donald j. trump": "donald trump",
}

SENATE_STATE = {
    "SENATEAK":      "Alaska",
    "SENATEME":      "Maine",
    "SENATETX":      "Texas",
    "SENATENE":      "Nebraska",
    "KXSENATELAD":   "Louisiana",
}

US_STATES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut",
    "delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa",
    "kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan",
    "minnesota","mississippi","missouri","montana","nebraska","nevada",
    "new hampshire","new jersey","new mexico","new york","north carolina",
    "north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont",
    "virginia","washington","west virginia","wisconsin","wyoming",
}

STOP = {
    "will", "the", "a", "an", "be", "is", "are", "was", "were", "to", "of",
    "in", "on", "at", "by", "for", "with", "or", "and", "not", "vs", "than",
    "more", "less", "over", "under", "winner", "win", "wins", "won", "this",
    "it", "who", "what", "which", "that", "any", "part", "some", "next",
    "his", "her", "their", "before", "after",
}


def _keywords(title: str, drop: str | None = None) -> str:
    t = title.rstrip("?").strip()
    if drop and drop.lower() in t.lower():
        t = re.sub(re.escape(drop), "", t, count=1, flags=re.I)
    words = re.findall(r"[A-Za-z0-9'.]+", t)
    return " ".join(w for w in words if w.lower() not in STOP and len(w) > 1)


def _active_search(query: str, limit: int = 200) -> list[dict]:
    """Return active, future-dated Polymarket markets matching `query`."""
    try:
        rs = requests.get(
            GAMMA_SEARCH,
            params={"q": query, "limit_per_type": 100,
                    "search_profiles": "false", "search_tags": "false"},
            timeout=15,
        )
        rs.raise_for_status()
    except requests.RequestException:
        return []

    now = datetime.now(timezone.utc)
    out = []
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


def _parse_outcomes(market: dict) -> list[str]:
    raw = market.get("outcomes")
    if isinstance(raw, str):
        try:
            return list(json.loads(raw))
        except json.JSONDecodeError:
            return []
    return list(raw) if isinstance(raw, list) else []


def _end_year(m: dict) -> int | None:
    ed = m.get("endDateIso") or (m.get("endDate") or "")[:10]
    try:
        return datetime.fromisoformat(ed).year if ed else None
    except ValueError:
        return None


_DEADLINE_RE = re.compile(r"\b(?:by|before)\s+(?:\w+\s+){0,2}\d", re.IGNORECASE)


def _has_explicit_deadline(question: str) -> bool:
    return bool(_DEADLINE_RE.search(question or ""))


def _all_years(text: str) -> set[int]:
    """Extract years. Expands "2025-26" / "2025–26" ranges to {2025, 2026}."""
    years: set[int] = set()
    for full, suffix in re.findall(r"\b(20[2-3]\d)(?:[-–](\d{2}))?\b", text or ""):
        years.add(int(full))
        if suffix:
            years.add(int(full[:2] + suffix))  # "2025"+"26" -> 2026
    return years


def _ticker_year(ticker: str) -> int | None:
    """KXPREMIERLEAGUE-26-ARS -> 2026; KXBOND-30-JACO -> 2030."""
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


def _senate_match(candidates: list[dict], ticker: str, short_label: str) -> tuple[dict, str] | None:
    """Handle SENATE state-races. Polymarket has one binary market per state
    ('Will the Democrats win the X Senate race in YYYY?'); Kalshi has separate
    R/D/candidate tickers that map to YES or NO on that market.
    Returns (chosen_market, poly_outcome) or None.
    """
    family = ticker.split("-")[0]
    state = SENATE_STATE.get(family)
    if not state:
        return None
    s_lower = state.lower()
    sl = _norm(short_label)
    pool = [m for m in candidates
            if s_lower in _norm(m.get("question") or "")
            and "senate" in _norm(m.get("question") or "")]
    if not pool:
        return None
    # Prefer state-specific markets ("X Senate race") over national ("control the Senate").
    pool.sort(key=lambda m: (
        "control the senate" in _norm(m.get("question") or ""),
        -float(m.get("volume24hr") or 0),
    ))
    chosen = pool[0]
    poly_q = _norm(chosen.get("question") or "")
    if "democrat" in sl:
        outcome = "Yes" if "democrats win" in poly_q else ("No" if "republicans win" in poly_q else None)
    elif "republican" in sl:
        outcome = "Yes" if "republicans win" in poly_q else ("No" if "democrats win" in poly_q else None)
    elif sl in poly_q:
        outcome = "Yes"
    else:
        outcome = None
    if outcome is None:
        return None
    return chosen, outcome


def _match(candidates: list[dict], kalshi_question: str, short_label: str,
           ticker: str, kalshi_description: str) -> dict | None:
    if not short_label:
        return None
    sl = _norm(short_label).strip()
    sl = SHORT_LABEL_ALIASES.get(sl, sl)
    hits = [m for m in candidates if sl in _norm(m.get("question") or "")]

    # If Polymarket's question mentions a year, that's a hard constraint we
    # must align with Kalshi (via question or ticker year). If Polymarket has
    # no year in its question, treat it as unconstrained -- accept the match.
    kal_years = _all_years(kalshi_question)
    ty = _ticker_year(ticker)

    fam = ticker.split("-")[0]
    # Families where Polymarket only has shorter-horizon markets ("in 2026",
    # "by June 30") for a question Kalshi asks with a longer deadline (2029/2030).
    # Treat those as biased lower-bound priors rather than rejecting.
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

    # If Polymarket question has an explicit short deadline ("by/before <date>")
    # AND that deadline (endDate year) is well before the Kalshi ticker year,
    # it's a different, tighter market — reject as a biased lower-bound.
    # Exception: families where the Polymarket short-window market is the only
    # source of price signal (KXNEXTAG, KX2028RRUN); accept as low-confidence prior.
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


def _adapt(m: dict) -> dict:
    return {
        "ticker": m["market_id"].removeprefix("kalshi:"),
        "question": m["question"],
        "description": m.get("description") or "",
        "short_label": m.get("short_label") or "",
        "family": m.get("family") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="seconds between Gamma calls (default 0.3)")
    args = parser.parse_args()

    snap = json.loads(args.snapshot.read_text())
    kalshi = [m for m in snap["markets"] if m.get("source") == "kalshi"]
    if args.limit:
        kalshi = kalshi[: args.limit]

    cached = _load_map()
    rejected = _load_negative()

    n_match = n_reject = n_skip = 0
    for i, m in enumerate(kalshi, 1):
        a = _adapt(m)
        if a["ticker"] in cached or a["ticker"] in rejected:
            n_skip += 1
            continue

        # SENATE state races use a dedicated matcher (state-aware query +
        # complement-outcome logic for "Democrats win X" markets vs Kalshi
        # "Republicans win X" sub-markets).
        state = SENATE_STATE.get(a["family"])
        senate_outcome = None
        if state:
            q = f"{state} Senate race 2026"
            cands = _active_search(q)
            sm = _senate_match(cands, a["ticker"], a["short_label"])
            chosen = sm[0] if sm else None
            senate_outcome = sm[1] if sm else None
        else:
            # If we have a family override, use it exclusively (manual mapping
            # of Kalshi euphemisms to Polymarket's brand-name search terms).
            family_q = FAMILY_QUERY.get(a["family"])
            q = family_q or _keywords(a["question"], drop=a["short_label"]) \
                         or _keywords(a["question"]) or a["question"]
            cands = _active_search(q)
            chosen = _match(cands, a["question"], a["short_label"], a["ticker"], a["description"])

        if chosen is None:
            _append_row(
                NEG_CSV,
                ["kalshi_ticker", "kalshi_question", "short_label", "n_candidates", "query"],
                [a["ticker"], a["question"], a["short_label"], str(len(cands)), q],
            )
            n_reject += 1
            print(f"  [{i:3}/{len(kalshi)}] {a['ticker']:32} no-match  ({len(cands)} cand)  q={q!r}")
        else:
            outcomes = _parse_outcomes(chosen)
            if senate_outcome is not None:
                yes = senate_outcome
            else:
                yes = "Yes" if "Yes" in outcomes else (outcomes[0] if outcomes else "Yes")
            _append_row(
                MAP_CSV,
                ["kalshi_ticker", "poly_condition_id", "poly_outcome",
                 "kalshi_question", "poly_question", "poly_end_date", "poly_vol_24h"],
                [a["ticker"], chosen.get("conditionId", ""), yes,
                 a["question"], chosen.get("question", ""),
                 chosen.get("endDateIso", ""),
                 f"{float(chosen.get('volume24hr') or 0):.2f}"],
            )
            n_match += 1
            v24 = float(chosen.get("volume24hr") or 0)
            print(f"  [{i:3}/{len(kalshi)}] {a['ticker']:32} MATCH [{a['short_label']}] -> "
                  f"{str(chosen.get('question'))[:70]} (vol24h={v24:.0f})")
        time.sleep(args.sleep)

    print(f"\ntotal: {n_match} matched, {n_reject} rejected, {n_skip} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
