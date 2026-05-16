"""Claude tool-using research agent for Prophet Arena binary markets.

Provider name : "claude_agent"
Required env  : ANTHROPIC_API_KEY  (or whatever api_key_env points to)
Optional env  : PERPLEXITY_API_KEY  (web_search backend; degraded without it)

Config extras (beyond base ForecasterConfig):
  backtest_mode    : bool   – inject evidence-cutoff block; date-restrict searches
  evidence_cutoff  : str    – ISO-8601 UTC; "auto" uses packet.as_of
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import anthropic
import requests

from ..calibration import time_to_close_hours
from ..schemas import (
    ForecastDiagnostics,
    ForecastValues,
    MarketPacket,
    ModelForecast,
    ReasoningTrack,
    clamp_prob,
)
from .base import ForecasterConfig

PROMPT_PATH = Path(__file__).parent / "agents" / "claude.md"
MAX_TURNS = 14


# ---------------------------------------------------------------------------
# Prompt template loading
# ---------------------------------------------------------------------------

def _load_template() -> str:
    raw = PROMPT_PATH.read_text()
    if raw.startswith("<!--"):
        end = raw.find("-->")
        if end >= 0:
            raw = raw[end + 3:].lstrip()
    return raw


def _render(template: str, **slots: Any) -> str:
    """Substitute only known {slot} names; leave all other {…} untouched."""
    for key, value in slots.items():
        template = template.replace("{" + key + "}", str(value))
    # Handle format-spec slots like {ttc_hours:.1f} — replace with formatted value
    import re as _re
    def _replace_spec(m: re.Match) -> str:
        name, spec = m.group(1), m.group(2)
        if name in slots:
            return format(slots[name], spec)
        return m.group(0)  # leave unknown specs alone
    template = _re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*):([\w.,<>^+\-#]*)\}", _replace_spec, template)
    return template


# ---------------------------------------------------------------------------
# Mode blocks (rendered strings, not template slots)
# ---------------------------------------------------------------------------

_LIVE_MODE = """\
## Mode: LIVE
Use information through the current date. Recency matters — fresh news,
polls, and announcements the market may not have fully absorbed are
exactly where research-driven edge comes from. Prefer sources from the
last 48 hours when the question is news-sensitive."""

_BACKTEST_MODE = """\
## Mode: BACKTEST — evidence cutoff {cutoff}
You are replaying this market **as of {cutoff}**. Search queries are
date-restricted to this cutoff server-side. The Kalshi prior was assembled
from a pre-cutoff snapshot.

You must internally constrain your reasoning:
- Do not reference world events you recall from after {cutoff}, even if
  known from training.
- Do not anchor on what "actually happened" — you do not know it.
- When uncertain whether a fact is pre- or post-cutoff, omit it.
- If a tool result leaks post-cutoff info, discard it and note it in
  `information_gaps`.

Contamination invalidates the backtest. Treat the cutoff as a hard wall."""


# ---------------------------------------------------------------------------
# Prior assembly: Kalshi microprice + Polymarket cross-venue
# ---------------------------------------------------------------------------

def _microprice(bid: float, ask: float, bid_sz: float, ask_sz: float) -> float:
    if bid_sz + ask_sz < 1.0:
        return (bid + ask) / 2.0
    spread = ask - bid
    raw = (bid_sz * ask + ask_sz * bid) / (bid_sz + ask_sz)
    mid = (bid + ask) / 2.0
    return mid + (raw - mid) * math.exp(-spread / 0.05)


def _compute_prior(packet: MarketPacket, backtest_mode: bool) -> tuple[float, float, str]:
    """Returns (p_prior, sigma, poly_block_text)."""
    q = packet.kalshi
    bid = q.yes_bid or 0.0
    ask = q.yes_ask or q.market_mid
    bid_sz = getattr(q, "yes_bid_size", None) or 0.0
    ask_sz = getattr(q, "yes_ask_size", None) or 0.0
    spread = q.spread or 0.0

    k_micro = _microprice(bid, ask, bid_sz, ask_sz) if (bid and ask) else q.market_mid
    k_micro = clamp_prob(k_micro)
    depth = bid_sz + ask_sz
    k_weight = math.exp(-spread / 0.05) * (min(1.0, depth / 5000) if depth > 0 else 0.5)
    k_weight = max(0.1, k_weight)

    poly_block = "no Polymarket match"
    poly_mid = 0.0
    poly_weight = 0.0

    if not backtest_mode:
        try:
            from ..polymarket import get_market_priors
            priors = get_market_priors(packet)
            if priors:
                pq = priors[0].quote
                pm_mid = pq.market_mid
                pm_spread = pq.spread or 0.20
                alignment = abs(pm_mid - k_micro)
                pw = math.exp(-pm_spread / 0.05) * math.exp(-(alignment ** 2) / 0.02)
                if pw > 0.05:
                    poly_mid = pm_mid
                    poly_weight = pw
                    poly_block = (
                        f"mid {pm_mid:.3f}, spread {pm_spread * 100:.1f}pp, "
                        f"alignment weight {pw:.2f}"
                    )
                else:
                    poly_block = f"match found but low alignment/liquidity (weight {pw:.3f}); excluded"
        except Exception as exc:
            poly_block = f"lookup failed: {exc}"
    else:
        poly_block = "skipped in backtest mode"

    total_w = k_weight + poly_weight
    p_prior = clamp_prob((k_weight * k_micro + poly_weight * poly_mid) / total_w)
    sigma = min(max(spread / 2.0, 0.02) + abs(poly_mid - k_micro) * 0.4, 0.45)
    return p_prior, sigma, poly_block


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------

_LIQ_GRID: dict[tuple[str, str], tuple[str, float]] = {
    ("liquid",    "far"):      ("shallow", 5.0),
    ("liquid",    "near"):     ("shallow", 4.0),
    ("liquid",    "close"):    ("none",    2.0),
    ("liquid",    "imminent"): ("none",    1.0),
    ("mid",       "far"):      ("shallow", 10.0),
    ("mid",       "near"):     ("shallow", 8.0),
    ("mid",       "close"):    ("shallow", 5.0),
    ("mid",       "imminent"): ("none",    3.0),
    ("illiquid",  "far"):      ("deep",    60.0),
    ("illiquid",  "near"):     ("deep",    50.0),
    ("illiquid",  "close"):    ("deep",    25.0),
    ("illiquid",  "imminent"): ("shallow", 8.0),
    ("no_market", "far"):      ("deep",    95.0),
    ("no_market", "near"):     ("deep",    90.0),
    ("no_market", "close"):    ("deep",    50.0),
    ("no_market", "imminent"): ("shallow", 12.0),
}

_LIQ_EXPLAIN = {
    "liquid":    "Deep, tight book. Prior is high-confidence. Defer unless you find evidence the market has missed.",
    "mid":       "Moderate depth and spread. Prior is moderately reliable. Research if the category warrants it.",
    "illiquid":  (
        "Thin book. The prior price reflects very limited trading and may be far from fair value. "
        "Base rates, mechanistic reasoning, and primary-source research dominate the prior here. "
        "Move freely toward the answer your evidence supports."
    ),
    "no_market": (
        "No meaningful order book. **The prior is a numerical placeholder, not a market signal — "
        "do not anchor on it.** Treat this as a clean forecasting question and estimate from first "
        "principles: rules text, base rates, primary sources, and (if any) historical context. "
        "Your research-derived estimate is the primary output; the gate is intentionally wide so "
        "you can express it. Do not artificially shrink toward the prior."
    ),
}

_TTC_EXPLAIN = {
    "far":      "More than 72 hours to close. Full research depth appropriate.",
    "near":     "24–72 hours. Markets are still updating; research can add edge for news-sensitive questions.",
    "close":    "Under 24 hours. Only strong, specific, recent evidence justifies deviation.",
    "imminent": "Under 3 hours. Markets outperform LLMs in this window. Default action is to defer.",
}


def _classify(packet: MarketPacket) -> dict[str, Any]:
    q = packet.kalshi
    spread = q.spread
    oi = q.open_interest or 0
    depth = (getattr(q, "yes_bid_size", 0) or 0) + (getattr(q, "yes_ask_size", 0) or 0)

    if spread is None:
        liq = "no_market"
    elif spread <= 0.04 and (depth > 200 or oi > 500):
        liq = "liquid"
    elif spread <= 0.10 and (depth > 50 or oi > 100):
        liq = "mid"
    elif spread <= 0.20:
        liq = "illiquid"
    else:
        liq = "no_market"

    hours = time_to_close_hours(packet)
    if hours is None or hours > 72:
        ttc = "far"
    elif hours > 24:
        ttc = "near"
    elif hours > 3:
        ttc = "close"
    else:
        ttc = "imminent"

    triage, max_delta = _LIQ_GRID.get((liq, ttc), ("shallow", 10.0))

    recency_block = ""
    if hours is not None and hours < 6 and liq in ("liquid", "mid"):
        recency_block = (
            "## News-driven imminent carve-out\n"
            "If you detect ALL THREE: price moved >10pp in last hour, TTC <6h, "
            "spread widened — override triage to `shallow` and run one recency "
            "search for news in the last 6 hours."
        )

    return {
        "liquidity": liq,
        "ttc_band": ttc,
        "ttc_hours": hours or 999.0,
        "triage_default": triage,
        "max_delta_pp": max_delta,
        "regime_explanation": _LIQ_EXPLAIN[liq],
        "ttc_explanation": _TTC_EXPLAIN[ttc],
        "recency_carveout_block": recency_block,
        "depth_total": depth,
        "spread_pp": (spread or 0.0) * 100,
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for recent information. Returns a synthesized answer "
            "with citations. Frame queries as specific questions, not keyword dumps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the text content of a URL. Use for primary sources: official "
            "statements, government data, court documents, stats pages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "get_kalshi_price",
        "description": "Re-fetch current Kalshi price and depth. Call before submitting if >60s have elapsed.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_history",
        "description": "Get recent price/depth snapshots for a Kalshi ticker. Useful for detecting recent moves.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "n": {"type": "integer", "default": 5},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "submit_forecast",
        "description": "Submit your final forecast. Terminal — call once when done.",
        "input_schema": {
            "type": "object",
            "properties": {
                "forecast": {
                    "type": "object",
                    "properties": {
                        "p_yes": {"type": "number"},
                        "confidence": {"type": "number"},
                        "uncertainty": {"type": "number"},
                        "fair_yes_price": {"type": "number"},
                        "max_yes_buy_price": {"type": "number"},
                        "max_no_buy_price": {"type": "number"},
                        "trade_recommendation": {"type": "string"},
                    },
                    "required": ["p_yes"],
                },
                "reasoning_track": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "base_rate": {"type": "string"},
                        "market_analysis": {"type": "string"},
                        "key_evidence": {"type": "array", "items": {"type": "object"}},
                        "counterarguments": {"type": "array", "items": {"type": "object"}},
                        "assumptions": {"type": "array", "items": {"type": "string"}},
                        "information_gaps": {"type": "array", "items": {"type": "string"}},
                        "what_would_change_my_mind": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["summary"],
                },
                "diagnostics": {
                    "type": "object",
                    "properties": {
                        "evidence_quality": {"type": "string"},
                        "rules_clarity": {"type": "string"},
                        "liquidity_quality": {"type": "string"},
                        "market_disagreement_reason": {"type": "string"},
                        "should_defer_to_market": {"type": "boolean"},
                    },
                },
            },
            "required": ["forecast", "reasoning_track"],
        },
    },
    {
        "name": "abandon_research",
        "description": "Abandon research and return the prior unchanged. Use when you have no edge.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _web_search(query: str, cutoff: str | None = None) -> str:
    key = os.environ.get("PERPLEXITY_API_KEY")
    if not key:
        return f"[No PERPLEXITY_API_KEY set. Cannot search: {query}]"
    try:
        payload: dict[str, Any] = {
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
        }
        if cutoff:
            # Perplexity search_before_date is YYYY/MM/DD
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
                payload["search_before_date"] = dt.strftime("%m/%d/%Y")
            except Exception:
                pass
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        citations = data.get("citations") or []
        if citations:
            text += "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations[:6])
        return text
    except Exception as exc:
        return f"[Search failed: {exc}]"


def _fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:5000] + ("…" if len(text) > 5000 else "")
    except Exception as exc:
        return f"[Fetch failed: {exc}]"


def _get_kalshi_price(ticker: str, packet: MarketPacket) -> str:
    q = packet.kalshi
    return json.dumps({
        "ticker": ticker,
        "yes_bid": q.yes_bid,
        "yes_ask": q.yes_ask,
        "no_bid": q.no_bid,
        "no_ask": q.no_ask,
        "market_mid": q.market_mid,
        "spread": q.spread,
        "last_price": q.last_price,
        "snapshot_time": q.snapshot_time,
    }, indent=2)


def _get_history(ticker: str, n: int, cutoff: str | None) -> str:
    from ..data import SNAPSHOT_ROOT
    if not SNAPSHOT_ROOT.exists():
        return "[No local snapshots available]"
    rows = []
    for snap_dir in sorted(SNAPSHOT_ROOT.iterdir()):
        if not snap_dir.is_dir():
            continue
        for fp in snap_dir.glob("*.json"):
            if fp.name == "_meta.json":
                continue
            try:
                data = json.loads(fp.read_text())
                snap_time = data.get("snapshot_time", "")
                if cutoff and snap_time > cutoff:
                    continue
                for m in data.get("markets", []):
                    if m.get("ticker") == ticker:
                        rows.append({"t": snap_time, "yes_ask": m.get("yes_ask"), "no_ask": m.get("no_ask")})
            except Exception:
                continue
    rows.sort(key=lambda r: r["t"])
    return json.dumps(rows[-n:], indent=2) if rows else f"[No history found for {ticker}]"


def _handle_tool(name: str, inp: dict, packet: MarketPacket, cutoff: str | None) -> str:
    if name == "web_search":
        return _web_search(inp.get("query", ""), cutoff=cutoff)
    if name == "fetch_url":
        return _fetch_url(inp.get("url", ""))
    if name == "get_kalshi_price":
        return _get_kalshi_price(inp.get("ticker", packet.market_ticker), packet)
    if name == "get_history":
        return _get_history(inp.get("ticker", packet.market_ticker), int(inp.get("n", 5)), cutoff)
    return f"[Unknown tool: {name}]"


# ---------------------------------------------------------------------------
# CLI path — no API key needed, uses Claude Code session auth
# ---------------------------------------------------------------------------

_CLI_USER_PROMPT = """\
Analyze this market following the Procedure in the system prompt.
You are in CLI mode — no tool calls are available. Reason directly from
your training knowledge (respecting the evidence cutoff if in BACKTEST mode)
and the prior provided.

After your internal reasoning, output ONLY a single valid JSON object matching
the ModelForecast schema (forecast / reasoning_track / diagnostics).
No other text before or after the JSON."""


def _extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    raise ValueError("No JSON object found in output")


def _run_via_cli(system: str, model: str) -> dict[str, Any]:
    """Run a zero-shot forecast via the claude CLI (Claude Code auth, no API key)."""
    cmd = [
        "claude",
        "--print",
        "--output-format", "text",
        "--model", model,
        "--system-prompt", system,
        _CLI_USER_PROMPT,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    text = proc.stdout.strip()
    if proc.returncode != 0 or not text:
        stderr = proc.stderr.strip()[:300]
        return {"terminal": "abandon_research", "args": {"reason": f"CLI failed: {stderr}"}}
    try:
        return {"terminal": "submit_forecast", "args": _extract_json(text)}
    except Exception as exc:
        return {"terminal": "abandon_research", "args": {"reason": f"CLI parse failed ({exc}): {text[:200]}"}}


# ---------------------------------------------------------------------------
# Tool-use loop (Anthropic API)
# ---------------------------------------------------------------------------

def _run_loop(
    system: str,
    packet: MarketPacket,
    config: ForecasterConfig,
    cutoff: str | None,
) -> dict[str, Any]:
    api_key = os.environ.get(config.api_key_env or "ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(f"API key not set: {config.api_key_env or 'ANTHROPIC_API_KEY'}")

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = [
        {"role": "user", "content": "Begin your analysis. Start with Phase 1 — Triage."}
    ]

    for _turn in range(MAX_TURNS):
        kwargs: dict[str, Any] = dict(
            model=config.model or "claude-opus-4-7",
            max_tokens=config.max_tokens or 4000,
            temperature=0.0,
            system=system,
            tools=_TOOLS,
            messages=messages,
        )
        resp = client.messages.create(**kwargs)
        messages.append({"role": "assistant", "content": resp.content})

        tool_results = []
        terminal: tuple[str, dict] | None = None

        for block in resp.content:
            if block.type != "tool_use":
                continue
            if block.name in ("submit_forecast", "abandon_research"):
                terminal = (block.name, block.input)
                break
            result = _handle_tool(block.name, block.input, packet, cutoff)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        if terminal:
            return {"terminal": terminal[0], "args": terminal[1]}

        if resp.stop_reason == "end_turn" and not tool_results:
            break

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return {"terminal": "abandon_research", "args": {"reason": "turn budget exhausted"}}


# ---------------------------------------------------------------------------
# ModelForecast builders
# ---------------------------------------------------------------------------

def _prior_result(config: ForecasterConfig, packet: MarketPacket, p_prior: float, reason: str) -> ModelForecast:
    return ModelForecast(
        model_id=config.model,
        provider="claude_agent",
        as_of=packet.as_of,
        market_ticker=packet.market_ticker,
        forecast=ForecastValues(
            p_yes=clamp_prob(p_prior),
            confidence=0.3,
            uncertainty=0.5,
            trade_recommendation="NO_TRADE",
        ),
        reasoning_track=ReasoningTrack(
            summary=f"Deferred to market prior. {reason}",
            market_analysis="No deviation from prior.",
        ),
        diagnostics=ForecastDiagnostics(
            should_defer_to_market=True,
            evidence_quality="low",
        ),
    )


def _parse_submit(config: ForecasterConfig, packet: MarketPacket, args: dict, p_prior: float) -> ModelForecast:
    fc = args.get("forecast") or {}
    rt = args.get("reasoning_track") or {}
    dx = args.get("diagnostics") or {}

    p_yes = clamp_prob(float(fc.get("p_yes", p_prior)))
    conf = float(fc.get("confidence", 0.5))
    unc = float(fc.get("uncertainty", 0.5))

    return ModelForecast(
        model_id=config.model,
        provider="claude_agent",
        as_of=packet.as_of,
        market_ticker=packet.market_ticker,
        forecast=ForecastValues(
            p_yes=p_yes,
            confidence=conf,
            uncertainty=unc,
            fair_yes_price=float(fc.get("fair_yes_price", p_yes)),
            max_yes_buy_price=clamp_prob(float(fc.get("max_yes_buy_price", p_yes - unc * 0.25))),
            max_no_buy_price=clamp_prob(float(fc.get("max_no_buy_price", (1 - p_yes) - unc * 0.25))),
            trade_recommendation="NO_TRADE",
        ),
        reasoning_track=ReasoningTrack(
            summary=str(rt.get("summary", "")),
            base_rate=str(rt.get("base_rate", "")),
            market_analysis=str(rt.get("market_analysis", "")),
            key_evidence=list(rt.get("key_evidence") or []),
            counterarguments=list(rt.get("counterarguments") or []),
            assumptions=list(rt.get("assumptions") or []),
            information_gaps=list(rt.get("information_gaps") or []),
            what_would_change_my_mind=list(rt.get("what_would_change_my_mind") or []),
        ),
        diagnostics=ForecastDiagnostics(
            evidence_quality=dx.get("evidence_quality", "medium"),
            rules_clarity=dx.get("rules_clarity", "medium"),
            liquidity_quality=dx.get("liquidity_quality", "medium"),
            market_disagreement_reason=str(dx.get("market_disagreement_reason", "")),
            should_defer_to_market=bool(dx.get("should_defer_to_market", False)),
        ),
        raw_response=args,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def forecast(config: ForecasterConfig, packet: MarketPacket) -> ModelForecast:
    backtest_mode: bool = getattr(config, "backtest_mode", False)
    evidence_cutoff: str | None = getattr(config, "evidence_cutoff", None)
    if evidence_cutoff == "auto":
        evidence_cutoff = packet.as_of

    p_prior, sigma, poly_block = _compute_prior(packet, backtest_mode)
    regime = _classify(packet)

    mode_block = (
        _BACKTEST_MODE.format(cutoff=evidence_cutoff)
        if backtest_mode and evidence_cutoff
        else _LIVE_MODE
    )

    template = _load_template()
    system = _render(
        template,
        event_title=packet.title or "",
        event_subtitle=packet.subtitle or "(none)",
        event_category=packet.category or "Other",
        event_description=packet.retrieval.get("description") or "(none)",
        event_rules=packet.rules or "(none provided)",
        event_close_time=packet.close_time or "(unknown)",
        ttc_hours=regime["ttc_hours"],
        prior_p_yes=p_prior,
        prior_sigma=sigma,
        kalshi_microprice=p_prior,
        kalshi_depth_total=regime["depth_total"],
        kalshi_spread_pp=regime["spread_pp"],
        poly_block=poly_block,
        regime=regime["liquidity"],
        regime_explanation=regime["regime_explanation"],
        ttc_band=regime["ttc_band"],
        ttc_explanation=regime["ttc_explanation"],
        max_delta_pp=regime["max_delta_pp"],
        triage_default=regime["triage_default"],
        recency_carveout_block=regime["recency_carveout_block"],
        mode_block=mode_block,
    )

    api_key = os.environ.get(config.api_key_env or "ANTHROPIC_API_KEY", "")
    if api_key:
        result = _run_loop(system, packet, config, cutoff=evidence_cutoff if backtest_mode else None)
    else:
        # Fall back to claude CLI (works with Claude Code session auth, no key needed)
        result = _run_via_cli(system, config.model or "claude-opus-4-7")

    if result["terminal"] == "abandon_research" or "forecast" not in result.get("args", {}):
        return _prior_result(config, packet, p_prior, result["args"].get("reason", ""))

    return _parse_submit(config, packet, result["args"], p_prior)
