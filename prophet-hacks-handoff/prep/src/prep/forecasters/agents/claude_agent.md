<!--
Claude grounded forecaster — rules-first, current-evidence assembly, market comparison.

Template slots: {mode_block}, {market_json}
Intended provider: claude_grounded
-->

{mode_block}

## Market
```json
{market_json}
```

---

## Phase 0 — Rules & Research Target  *(no tools)*

Read the `rules`, `outcomes`, and `description` in the market data above.

```
[RULES]
Outcomes            : <list from market data>
Exclusive?          : yes (exactly one resolves) | no (each resolves independently)
Trigger per outcome : <what must be true for each outcome to resolve>
Resolving source    : <official body/site that publishes the answer>
Measurement         : <exact series/vintage — e.g. "initial BLS release, not revised">
Key ambiguities     : <timing edge cases, exclusions, measurement disputes>
```

```
[RESEARCH TARGET]
Core question : <one sentence — the specific fact you need to know>
Data type     : price_threshold | data_release | game_result | event_occurrence | other
Best source   : <the one place that most directly answers the question>
```

---

## Phase 1 — Evidence Assembly  *(targeted, not exhaustive — aim for 2–4 calls)*

Fetch sources that directly answer your research target. Stop when you have enough
signal. Prioritize **current** over historical, **specific** over general, **primary**
over commentary.

| Category | What to fetch |
|----------|---------------|
| Crypto | Spot price vs threshold (CoinGecko/CMC), recent volatility |
| Weather | NWS forecast discussion for location + date, NOAA ensemble output |
| Economics | Consensus forecast median, most recent leading indicator, CME FedWatch for rate markets |
| Sports | Current betting line, official injury report, recent head-to-head and last 5 form |
| Politics | Latest poll aggregate with methodology, official vote count or legislative status |
| Legal | Most recent docket entry or official ruling |

After every call:
```
[ESTIMATE] p ≈ {Outcome: X.XX, ...}  |  finding: <one sentence>  |  next: <source or "done">
```

End Phase 1 with:
```
[p_research]
probabilities   : {"Outcome A": X.XX, ...}  ← normalize only if mutually exclusive
evidence_quality: low | medium | high
summary         : <what you found and why it moves you where it does>
```

**evidence_quality rubric:**
- **high**: ≥ 2 independent primary sources agree, clear mechanistic link to resolution
- **medium**: 1 primary source, or 2 sources that partially agree
- **low**: secondary/commentary only, conflicting signals, or sources older than a few days

---

## Phase 2 — Market Comparison  *(no tools)*

```
[COMPARISON]
p_market (YES) = <prior.p_yes from market data>
                 binary: NO implied as 1 - p_market_YES
                 non-binary: use market prices per outcome if available

outcome     p_research   p_market   gap
-------     ----------   --------   ---
<outcome>   0.XX         0.XX       ±XXpp
Largest gap : ±XXpp on [outcome]
```

| Gap | Action |
|-----|--------|
| < 5pp | Aligned — go to Phase 4, submit near market |
| 5–15pp | Notable — corroborate before deciding (Phase 3) |
| > 15pp | Significant — must corroborate |

Before Phase 3, ask:
- Are your sources more recent than the market snapshot?
- Is there a simpler explanation — thin book, stale snapshot, bad cross-match?

---

## Phase 3 — Corroboration & Red Team  *(1–3 calls; only if gap ≥ 5pp)*

**Pass A — Corroborate**: find one independent source on the largest-gap outcome.
Independent = different publisher, different data series, different methodology.

```
[CORROBORATION]
source  : <URL or search>
finding : <what it says>
updated : {"Outcome": X.XX}
```

**Pass B — Red team**: search specifically for what would make you wrong.
Do not search neutrally — search for the best case against your current lean.

```
[RED TEAM]
query   : <search targeting the opposite case>
finding : <what you found, or "nothing substantive found">
impact  : <does this weaken p_research? by how much?>
```

If corroboration and red team both support your direction → confidence goes up.
If they conflict → widen uncertainty, pull back toward market on the contested outcome.

---

## Phase 4 — Submit

**evidence_quality = high** (≥ 2 independent primary sources, mechanistic link):
→ deviate from market toward p_research by the full amount your evidence supports.
  Own the call. Set confidence proportionally.

**evidence_quality = medium** (1 primary source, partial corroboration):
→ deviate toward p_research by half the gap. Widen uncertainty.
  State what would push you to the full deviation.

**evidence_quality = low** (secondary only, conflicting, or stale):
→ return market distribution. `should_defer_to_market = true`.

**Definitive** (resolving source has published the actual answer, or you have an exact
numeric value from an official source that unambiguously clears the threshold):
→ winning outcome = 0.95, remainder split among others. No cap.
  This overrides the evidence_quality gate — a published result is definitive
  regardless of how many sources you found.

Final checks:
1. Every outcome gets an explicit probability — do not omit any.
2. Normalize only if outcomes are mutually exclusive and exhaustive.
3. Every deviation > 5pp from market has a citation.
4. `counterarguments` non-empty if `confidence ≥ 0.6`.
5. `market_analysis` states p_research, p_market, gap, and decision rationale.

---

## Calibration reminders
- **Commit.** Deviate the full amount evidence supports, or return market cleanly. Never half-deviate.
- **Most of the time you confirm the market.** Returning market distribution is correct when evidence is weak.
- **Extremes need mechanistic stories.** Any outcome above 0.90 or below 0.10 requires a specific, citable reason.
- **Normalize only when outcomes are mutually exclusive.**

---

## Output

```json
{
  "forecast": {
    "probabilities": {"YES": 0.62, "NO": 0.38},
    "confidence": 0.70,
    "uncertainty": 0.30
  },
  "reasoning_track": {
    "summary": "p_research=... from [source]. Market=... Gap=±Xpp. [Deviated / Returned market] because ...",
    "base_rate": "optional sanity check from training knowledge",
    "market_analysis": "p_research=... | market=... | gap=±Xpp | evidence_quality=... | decision: ...",
    "context_market_analysis": "",
    "key_evidence": [{"claim": "...", "source": "...", "impact": "..."}],
    "counterarguments": [{"claim": "...", "impact": "..."}],
    "assumptions": ["..."],
    "information_gaps": ["..."],
    "what_would_change_my_mind": ["..."]
  },
  "diagnostics": {
    "evidence_quality": "low | medium | high",
    "rules_clarity": "low | medium | high",
    "liquidity_quality": "low | medium | high",
    "should_defer_to_market": false,
    "market_disagreement_reason": "what drove the gap, or '' if returning market"
  }
}
```
