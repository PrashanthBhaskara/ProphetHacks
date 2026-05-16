"""Data-fair-price baselines.

The premise (from the team's iteration on May 16):
    The simplest competitive benchmark is "trade off of fair price
    learned from historical data alone — no LLM, no search." If our
    LLM agent can't beat this, we should just ship the recalibration
    model.

We fit a recalibration map on a TRAIN split of resolved markets
(market_price → P(YES)) and apply it to a TEST split. Four variants:

    mean_bias_market         — q + (mean(y_train) − mean(q_train))
    platt_market             — sigmoid(a·logit(q) + b)
    decile_isotonic_market   — q binned to nearest train-decile YES rate
    category_platt_market    — per-category Platt, falls back to global

All four return predict(event, market_info) → {p_yes, rationale} so they
plug into the existing harness exactly like market.py or always_half.py.

These are *closures* over the fitted parameters — call `fit_*(train_samples)`
to build them, then call the returned predictor on each test sample.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Sequence

import numpy as np

from ..data import Sample
from ..score import platt_fit
from ..trade import market_mid_forecast

PredictFn = Callable[[dict, dict], dict]


def _q_of(sample: Sample) -> float | None:
    """Return the market mid-price (0-1) for a sample, or None."""
    try:
        q = market_mid_forecast(sample.event, sample.market_info)
        return max(0.01, min(0.99, q))
    except Exception:
        return None


def _qy_pairs(samples: Sequence[Sample]) -> tuple[list[float], list[int]]:
    q_list: list[float] = []
    y_list: list[int] = []
    for s in samples:
        q = _q_of(s)
        if q is None:
            continue
        q_list.append(q)
        y_list.append(int(s.outcome))
    return q_list, y_list


# ---------------------------------------------------------------------------
# 1. Mean-bias correction
# ---------------------------------------------------------------------------


def fit_mean_bias(train_samples: Sequence[Sample]) -> PredictFn:
    """Learn constant shift = mean(y_train) − mean(q_train) and apply it.

    This is the principled version of `market_minus_0.10` — the shift
    is *learned* from data rather than hard-coded, and it goes the
    correct direction even if the imbalance flips."""
    q_train, y_train = _qy_pairs(train_samples)
    if not q_train:
        shift = 0.0
    else:
        shift = sum(y_train) / len(y_train) - sum(q_train) / len(q_train)

    def predict(event: dict, market_info: dict) -> dict:
        q = market_mid_forecast(event, market_info)
        p = max(0.01, min(0.99, q + shift))
        return {"p_yes": p, "rationale": f"market + {shift:+.4f} (mean bias)"}

    predict.shift = shift  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 2. Platt-recalibrated market
# ---------------------------------------------------------------------------


def fit_platt_market(train_samples: Sequence[Sample]) -> PredictFn:
    """Fit y ~ sigmoid(a·logit(q) + b) on training data. Two-parameter
    rescale — captures both a slope (calibration sharpness) and an
    intercept (systematic YES/NO bias).
    """
    q_train, y_train = _qy_pairs(train_samples)
    fit = platt_fit(q_train, y_train)
    a, b = fit["slope"], fit["intercept"]
    if math.isnan(a) or math.isnan(b):
        a, b = 1.0, 0.0

    def predict(event: dict, market_info: dict) -> dict:
        q = market_mid_forecast(event, market_info)
        q = max(1e-4, min(1 - 1e-4, q))
        z = a * math.log(q / (1 - q)) + b
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return {"p_yes": p, "rationale": f"platt(a={a:.3f}, b={b:.3f}) of market"}

    predict.slope = a  # type: ignore[attr-defined]
    predict.intercept = b  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 3. Decile-binned isotonic
# ---------------------------------------------------------------------------


def fit_decile_isotonic(train_samples: Sequence[Sample], n_bins: int = 10) -> PredictFn:
    """Non-parametric recalibration via equal-count quantile binning.

    Bin train markets by market price into `n_bins` deciles. Each bin's
    fair price is its empirical YES rate. To predict, find which bin the
    test q falls into, return that bin's rate.

    Robust to weird shapes in the (q, y) relationship that Platt can't
    capture (e.g. systematic miscalibration only in the [0.7, 0.9] range).
    """
    q_train, y_train = _qy_pairs(train_samples)
    if not q_train:
        return lambda e, m: {"p_yes": 0.5, "rationale": "fallback"}

    sorted_pairs = sorted(zip(q_train, y_train), key=lambda x: x[0])
    n = len(sorted_pairs)
    bin_size = max(1, n // n_bins)

    edges: list[float] = []  # right edge of each bin, ascending
    rates: list[float] = []
    for i in range(0, n, bin_size):
        bucket = sorted_pairs[i : i + bin_size]
        if not bucket:
            continue
        edge = bucket[-1][0]
        rate = sum(y for _, y in bucket) / len(bucket)
        edges.append(edge)
        rates.append(rate)

    # Enforce monotonicity (pool-adjacent-violators, light version):
    # if a later bucket has a lower rate, average it with the prior.
    for _ in range(len(rates)):
        adjusted = False
        for i in range(len(rates) - 1):
            if rates[i] > rates[i + 1]:
                avg = (rates[i] + rates[i + 1]) / 2
                rates[i] = rates[i + 1] = avg
                adjusted = True
        if not adjusted:
            break

    def predict(event: dict, market_info: dict) -> dict:
        q = market_mid_forecast(event, market_info)
        # Find the smallest edge >= q
        idx = 0
        for i, e in enumerate(edges):
            if q <= e:
                idx = i
                break
        else:
            idx = len(edges) - 1
        p = max(0.01, min(0.99, rates[idx]))
        return {"p_yes": p, "rationale": f"decile bin {idx}/{len(edges)} rate={p:.3f}"}

    predict.edges = edges  # type: ignore[attr-defined]
    predict.rates = rates  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 4. Per-category Platt with global fallback
# ---------------------------------------------------------------------------


def fit_category_platt(train_samples: Sequence[Sample], min_per_cat: int = 50) -> PredictFn:
    """Fit Platt per category; fall back to a global Platt for categories
    with too few train samples. Captures the paper's finding that
    miscalibration is category-dependent (Sports tight, Politics loose)."""
    by_cat: dict[str, list[Sample]] = {}
    for s in train_samples:
        by_cat.setdefault(s.event["category"], []).append(s)

    global_fit = fit_platt_market(train_samples)

    cat_fits: dict[str, PredictFn] = {}
    for cat, sub in by_cat.items():
        if len(sub) < min_per_cat:
            continue
        cat_fits[cat] = fit_platt_market(sub)

    def predict(event: dict, market_info: dict) -> dict:
        cat = event.get("category") or "Other"
        f = cat_fits.get(cat, global_fit)
        out = f(event, market_info)
        out["rationale"] = f"category={cat} {out['rationale']}"
        return out

    predict.cat_fits = cat_fits  # type: ignore[attr-defined]
    predict.global_fit = global_fit  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 5. Multi-feature recalibrator
# ---------------------------------------------------------------------------


def _spread_of(market_info: dict) -> float | None:
    """yes_ask + no_ask in dollars (0–2 range; ~1 for tight markets)."""
    ya = market_info.get("yes_ask")
    na = market_info.get("no_ask")
    if ya is None or na is None:
        return None
    ya = ya / 100.0 if ya > 1 else ya
    na = na / 100.0 if na > 1 else na
    return ya + na


def _liquidity_of(market_info: dict) -> float | None:
    """`liquidity` field from subset_1200 (cents · contracts)."""
    v = market_info.get("liquidity")
    if v is None:
        return None
    try:
        return max(0.0, float(v))
    except Exception:
        return None


def _categories_seen(train_samples: Sequence[Sample], min_count: int = 30) -> list[str]:
    by_cat: dict[str, int] = {}
    for s in train_samples:
        c = s.event.get("category") or "Other"
        by_cat[c] = by_cat.get(c, 0) + 1
    return sorted(c for c, n in by_cat.items() if n >= min_count)


def _feature_row(market_info: dict, event: dict, categories: list[str]) -> list[float] | None:
    """Build the feature vector for one market.

    Features:
      1. logit(q)
      2. spread − 1.0  (centered around an at-the-money market)
      3. log1p(liquidity)
      4...K. one-hot for category (drop one as baseline)
    Returns None if any required price is missing.
    """
    try:
        q = market_mid_forecast(event, market_info)
    except Exception:
        return None
    q = max(1e-4, min(1 - 1e-4, q))
    logit_q = math.log(q / (1 - q))

    sp = _spread_of(market_info)
    if sp is None:
        return None
    spread_centered = sp - 1.0

    liq = _liquidity_of(market_info) or 0.0
    log_liq = math.log1p(liq)

    cat = event.get("category") or "Other"
    # drop the first category as baseline; remaining are one-hot
    cat_features = [1.0 if cat == c else 0.0 for c in categories[1:]]

    return [1.0, logit_q, spread_centered, log_liq, *cat_features]


def fit_multi_feature(
    train_samples: Sequence[Sample],
    *,
    l2: float = 0.5,
    max_iter: int = 100,
) -> PredictFn:
    """Logistic regression on (logit_q, spread, log_liquidity, category).

    L2-regularised so the per-category coefficients can't explode with
    small per-cat training counts. The "right" multi-feature extension of
    Platt — captures the things that vary across markets but a fixed
    logit-shift can't.
    """
    categories = _categories_seen(train_samples)
    X: list[list[float]] = []
    y: list[float] = []
    for s in train_samples:
        row = _feature_row(s.market_info, s.event, categories)
        if row is None:
            continue
        X.append(row)
        y.append(float(s.outcome))

    if not X:
        return fit_platt_market(train_samples)

    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float)
    n_feat = Xa.shape[1]

    # IRLS with L2 (Newton-Raphson with damping)
    beta = np.zeros(n_feat)
    beta[1] = 1.0  # init slope on logit_q to 1
    L2_diag = np.full(n_feat, l2)
    L2_diag[0] = 0.0  # don't regularise the bias term
    for _ in range(max_iter):
        z = np.clip(Xa @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-z))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        r = ya - mu
        g = -Xa.T @ r + L2_diag * beta
        H = (Xa.T * W) @ Xa + np.diag(L2_diag)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = max(1.0, float(np.max(np.abs(delta))))
        delta /= step
        beta -= delta
        if float(np.max(np.abs(delta))) < 1e-7:
            break

    def predict(event: dict, market_info: dict) -> dict:
        row = _feature_row(market_info, event, categories)
        if row is None:
            # fallback: use logit_q with intercept of beta only
            try:
                q = market_mid_forecast(event, market_info)
            except Exception:
                return {"p_yes": 0.5, "rationale": "fallback"}
            return {"p_yes": max(0.01, min(0.99, q)), "rationale": "fallback (no spread)"}
        z = float(np.dot(beta, np.asarray(row)))
        z = max(-30.0, min(30.0, z))
        p = 1.0 / (1.0 + math.exp(-z))
        return {"p_yes": max(0.01, min(0.99, p)),
                "rationale": f"multi-feature (cat={event.get('category')})"}

    predict.beta = beta  # type: ignore[attr-defined]
    predict.categories = categories  # type: ignore[attr-defined]
    predict.feature_names = ["bias", "logit_q", "spread_centered", "log1p_liquidity",
                             *[f"cat={c}" for c in categories[1:]]]  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 6. Trading-objective Platt — fits (a, b) to maximize TRAINING P&L
# ---------------------------------------------------------------------------


def fit_platt_max_pnl(
    train_samples: Sequence[Sample],
    *,
    strategy_max_spread: float = 1.02,
    strategy_min_spread: float = 0.95,
    contracts_per_unit: float = 100.0,
) -> PredictFn:
    """Grid-search (a, b) ∈ a-grid × b-grid to maximise training P&L
    under the tight_band strategy. Direct attack on the §B.4 paradox:
    instead of fitting log-likelihood (which doesn't track P&L), fit
    the actual trading objective.

    Caveats:
        - Grid search (not gradient) because the trading P&L surface
          has plateaus and discontinuities (strategy band rejections)
          that gradient methods misbehave on.
        - The strategy thresholds are baked in. A different strategy
          would have a different optimum.
    """
    pairs: list[tuple[float, float, float, int]] = []  # (q, yes_ask, no_ask, y)
    for s in train_samples:
        ya = s.market_info.get("yes_ask")
        na = s.market_info.get("no_ask")
        if ya is None or na is None:
            continue
        ya_d = ya / 100.0 if ya > 1 else ya
        na_d = na / 100.0 if na > 1 else na
        if ya_d + na_d <= 0:
            continue
        q = max(1e-4, min(1 - 1e-4, (ya_d + (1 - na_d)) / 2))
        pairs.append((q, ya_d, na_d, int(s.outcome)))

    if not pairs:
        return fit_platt_market(train_samples)

    def pnl_for(a: float, b: float) -> float:
        total = 0.0
        for q, ya, na, y in pairs:
            sp = ya + na
            if sp > strategy_max_spread or sp < strategy_min_spread:
                continue
            z = a * math.log(q / (1 - q)) + b
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            lo, hi = 1 - na, ya
            if lo <= p <= hi:
                continue
            diff = p - ya
            if diff > 0:
                shares, side, price = diff, "yes", ya
            elif diff < 0:
                shares, side, price = abs(diff), "no", na
            else:
                continue
            cost = shares * price * contracts_per_unit
            won = (side == "yes" and y == 1) or (side == "no" and y == 0)
            payoff = shares * contracts_per_unit if won else 0.0
            total += payoff - cost
        return total

    a_grid = [0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
    b_grid = [-1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4]

    best = (1.0, 0.0, -float("inf"))
    for a in a_grid:
        for b in b_grid:
            p = pnl_for(a, b)
            if p > best[2]:
                best = (a, b, p)
    a, b, train_pnl = best

    def predict(event: dict, market_info: dict) -> dict:
        q = market_mid_forecast(event, market_info)
        q = max(1e-4, min(1 - 1e-4, q))
        z = a * math.log(q / (1 - q)) + b
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        return {"p_yes": p, "rationale": f"pnl-max platt (a={a}, b={b}, train_pnl=${train_pnl:.0f})"}

    predict.slope = a  # type: ignore[attr-defined]
    predict.intercept = b  # type: ignore[attr-defined]
    predict.train_pnl = train_pnl  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 7. Event-size Platt — captures the multi-outcome structural NO-bias
# ---------------------------------------------------------------------------


def _event_size_index(samples: Sequence[Sample]) -> dict[str, int]:
    """Build `event_ticker -> count_of_markets` lookup for one set of samples."""
    counts: dict[str, int] = {}
    for s in samples:
        et = s.event.get("event_ticker") or ""
        counts[et] = counts.get(et, 0) + 1
    return counts


def fit_event_size_platt(train_samples: Sequence[Sample], *, l2: float = 0.5) -> PredictFn:
    """Three-parameter logistic on (bias, logit_q, log(event_size)).

    Captures the Mentions / multi-outcome structural NO-bias in a way
    that transfers across categories. YES rate drops monotonically with
    event size (50% at 2-outcome → 28% at 11+ outcome events), so a
    market in a 20-way event is structurally NO-biased regardless of
    whether it's a "Mentions" category or a Crypto strike grid.
    """
    sizes_train = _event_size_index(train_samples)

    rows: list[list[float]] = []
    y: list[float] = []
    for s in train_samples:
        try:
            q = market_mid_forecast(s.event, s.market_info)
        except Exception:
            continue
        q = max(1e-4, min(1 - 1e-4, q))
        n_event = sizes_train.get(s.event.get("event_ticker") or "", 1)
        rows.append([1.0, math.log(q / (1 - q)), math.log(max(1, n_event))])
        y.append(float(s.outcome))

    if not rows:
        return fit_platt_market(train_samples)

    X = np.asarray(rows)
    ya = np.asarray(y)
    n_feat = X.shape[1]
    L2 = np.array([0.0, l2, l2])  # don't regularize bias
    beta = np.array([0.0, 1.0, 0.0])
    for _ in range(100):
        z = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-z))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        g = -X.T @ (ya - mu) + L2 * beta
        H = (X.T * W) @ X + np.diag(L2)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = max(1.0, float(np.max(np.abs(delta))))
        delta /= step
        beta -= delta
        if float(np.max(np.abs(delta))) < 1e-7:
            break

    # event sizes get re-counted on the test set at predict time — see closure
    # variables below. We need the *test* set's event counts to be honest.
    train_size_lookup = sizes_train

    def predict_with_sizes(event: dict, market_info: dict, test_sizes: dict[str, int]) -> dict:
        try:
            q = market_mid_forecast(event, market_info)
        except Exception:
            return {"p_yes": 0.5, "rationale": "fallback"}
        q = max(1e-4, min(1 - 1e-4, q))
        et = event.get("event_ticker") or ""
        n_event = test_sizes.get(et, train_size_lookup.get(et, 1))
        x = np.array([1.0, math.log(q / (1 - q)), math.log(max(1, n_event))])
        z = float(np.dot(beta, x))
        z = max(-30, min(30, z))
        p = 1.0 / (1.0 + math.exp(-z))
        return {"p_yes": max(0.01, min(0.99, p)),
                "rationale": f"event_size_platt (n_event={n_event})"}

    def predict(event: dict, market_info: dict) -> dict:
        # Default: use train sizes (call attach_test_sizes() to update)
        return predict_with_sizes(event, market_info, train_size_lookup)

    def attach_test_sizes(test_samples: Sequence[Sample]) -> None:
        nonlocal predict
        new_sizes = _event_size_index(test_samples)
        def _p(e, m):
            return predict_with_sizes(e, m, new_sizes)
        predict = _p

    predict.beta = beta  # type: ignore[attr-defined]
    predict.attach_test_sizes = attach_test_sizes  # type: ignore[attr-defined]
    predict.train_size_lookup = train_size_lookup  # type: ignore[attr-defined]
    predict.feature_names = ["bias", "logit_q", "log_event_size"]  # type: ignore[attr-defined]
    # Hack so the dispatch picks up attach_test_sizes mutation:
    class _Predictor:
        def __init__(self, p): self._p = p; self.beta = beta; self.feature_names = predict.feature_names; self.train_size_lookup = train_size_lookup
        def __call__(self, e, m): return self._p(e, m)
        def attach_test_sizes(self, test):
            new_sizes = _event_size_index(test)
            self._p = lambda e, m, _s=new_sizes: predict_with_sizes(e, m, _s)
    return _Predictor(predict)


# ---------------------------------------------------------------------------
# 7b. Event-size Platt with interaction (logit_q × log_event_size)
# ---------------------------------------------------------------------------


def fit_event_size_platt_v2(train_samples: Sequence[Sample], *, l2: float = 0.5) -> PredictFn:
    """Four-parameter logistic: bias, logit_q, log_event_size, AND their
    interaction. The interaction lets the slope on q vary with event size:
    well-calibrated 2-market events keep slope ~1, large multi-outcome
    events get a flatter slope (less trust in q).
    """
    sizes_train = _event_size_index(train_samples)

    def _row(s, sizes):
        try:
            q = market_mid_forecast(s.event, s.market_info)
        except Exception:
            return None
        q = max(1e-4, min(1 - 1e-4, q))
        n_event = sizes.get(s.event.get("event_ticker") or "", 1)
        lq = math.log(q / (1 - q))
        les = math.log(max(1, n_event))
        return [1.0, lq, les, lq * les]

    rows = []
    y = []
    for s in train_samples:
        r = _row(s, sizes_train)
        if r is None:
            continue
        rows.append(r)
        y.append(float(s.outcome))

    if not rows:
        return fit_platt_market(train_samples)

    X = np.asarray(rows)
    ya = np.asarray(y)
    L2 = np.array([0.0, l2, l2, l2])
    beta = np.array([0.0, 1.0, 0.0, 0.0])
    for _ in range(100):
        z = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-z))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        g = -X.T @ (ya - mu) + L2 * beta
        H = (X.T * W) @ X + np.diag(L2)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = max(1.0, float(np.max(np.abs(delta))))
        delta /= step
        beta -= delta
        if float(np.max(np.abs(delta))) < 1e-7:
            break

    def predict_with_sizes(event, market_info, test_sizes):
        try:
            q = market_mid_forecast(event, market_info)
        except Exception:
            return {"p_yes": 0.5, "rationale": "fallback"}
        q = max(1e-4, min(1 - 1e-4, q))
        et = event.get("event_ticker") or ""
        n_event = test_sizes.get(et, sizes_train.get(et, 1))
        lq = math.log(q / (1 - q))
        les = math.log(max(1, n_event))
        x = np.array([1.0, lq, les, lq * les])
        z = float(np.dot(beta, x))
        z = max(-30, min(30, z))
        p = 1.0 / (1.0 + math.exp(-z))
        return {"p_yes": max(0.01, min(0.99, p)),
                "rationale": f"event_size_v2 (n_event={n_event})"}

    class _Predictor:
        def __init__(self):
            self._sizes = sizes_train
            self.beta = beta
            self.feature_names = ["bias", "logit_q", "log_event_size", "logit_q*log_event_size"]
        def __call__(self, e, m): return predict_with_sizes(e, m, self._sizes)
        def attach_test_sizes(self, test):
            self._sizes = _event_size_index(test)
    return _Predictor()


# ---------------------------------------------------------------------------
# 8. Hierarchical Platt — per-category fits shrunk to global posterior
# ---------------------------------------------------------------------------


def fit_hierarchical_platt(
    train_samples: Sequence[Sample],
    *,
    prior_strength: int = 200,
) -> PredictFn:
    """Per-category Platt fit, then shrink (a_cat, b_cat) toward
    (a_global, b_global) with weight n_cat / (n_cat + prior_strength).

    Concretely: each category gets its own slope/intercept, but small
    categories are pulled hard toward the global fit. Captures the
    paper's "calibration differs by category" hypothesis without the
    overfit of plain per-cat fits.

    `prior_strength` is the equivalent training-sample count of the
    global prior. 200 is a reasonable default for our scale; with
    train=5k markets, a category with 50 markets gets ~20% weight on
    its own fit, 80% on global.
    """
    global_fit = fit_platt_market(train_samples)
    a_g, b_g = global_fit.slope, global_fit.intercept

    by_cat: dict[str, list[Sample]] = {}
    for s in train_samples:
        by_cat.setdefault(s.event.get("category") or "Other", []).append(s)

    shrunk: dict[str, tuple[float, float]] = {}
    for cat, sub in by_cat.items():
        cat_fit = fit_platt_market(sub)
        w = len(sub) / (len(sub) + prior_strength)
        a = w * cat_fit.slope + (1 - w) * a_g
        b = w * cat_fit.intercept + (1 - w) * b_g
        shrunk[cat] = (a, b)

    def predict(event: dict, market_info: dict) -> dict:
        cat = event.get("category") or "Other"
        a, b = shrunk.get(cat, (a_g, b_g))
        q = market_mid_forecast(event, market_info)
        q = max(1e-4, min(1 - 1e-4, q))
        z = a * math.log(q / (1 - q)) + b
        p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
        return {"p_yes": p, "rationale": f"hier platt cat={cat} (a={a:.3f}, b={b:.3f})"}

    predict.shrunk = shrunk  # type: ignore[attr-defined]
    predict.global_a_b = (a_g, b_g)  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 9b. Beta calibration — strictly better than Platt for skewed data
# ---------------------------------------------------------------------------


def fit_beta_calibration(train_samples: Sequence[Sample], *, l2: float = 0.5) -> PredictFn:
    """Beta calibration (Kull, Silva Filho & Flach 2017).

    Fits y ~ sigmoid(a·log(q) + b·log(1−q) + c). Generalizes Platt:
    when a + b = 0 it reduces to Platt; when a ≠ −b it captures
    asymmetric miscalibration (e.g. accurate on low probs but
    overconfident on high probs), which is exactly what subset_1200's
    33% YES rate creates.
    """
    rows, ys = [], []
    for s in train_samples:
        try:
            q = market_mid_forecast(s.event, s.market_info)
        except Exception:
            continue
        q = max(1e-4, min(1 - 1e-4, q))
        rows.append([1.0, math.log(q), math.log(1 - q)])
        ys.append(float(s.outcome))
    if not rows:
        return fit_platt_market(train_samples)
    X = np.asarray(rows)
    y = np.asarray(ys)
    L2 = np.array([0.0, l2, l2])
    beta = np.array([0.0, 0.5, -0.5])  # init to Platt-ish
    for _ in range(100):
        z = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-z))
        W = np.clip(mu * (1 - mu), 1e-6, None)
        g = -X.T @ (y - mu) + L2 * beta
        H = (X.T * W) @ X + np.diag(L2)
        try:
            d = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        step = max(1.0, float(np.max(np.abs(d))))
        d /= step
        beta -= d
        if float(np.max(np.abs(d))) < 1e-7:
            break

    def predict(event, market_info):
        try:
            q = market_mid_forecast(event, market_info)
        except Exception:
            return {"p_yes": 0.5, "rationale": "fallback"}
        q = max(1e-4, min(1 - 1e-4, q))
        x = np.array([1.0, math.log(q), math.log(1 - q)])
        z = float(np.dot(beta, x))
        p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
        return {"p_yes": max(0.01, min(0.99, p)),
                "rationale": f"beta calibration"}

    predict.beta = beta  # type: ignore[attr-defined]
    return predict


# ---------------------------------------------------------------------------
# 9. Gated Platt — only trade categories that were profitable in training
# ---------------------------------------------------------------------------


def fit_gated_platt(
    train_samples: Sequence[Sample],
    *,
    strategy=None,
) -> PredictFn:
    """Fit Platt on train, then evaluate per-category P&L on train.
    For categories where train P&L < 0, predict the market price
    (which causes the strategy to take no action). For profitable
    categories, predict Platt.

    Uses TRAINING data only to determine the gate — no test leakage.
    """
    from ..trade import default_strategy as _default
    strategy = strategy or _default

    platt = fit_platt_market(train_samples)
    pred = lambda e, m: platt(e, m)["p_yes"]

    from ..trade import backtest
    res = backtest(train_samples, forecast_fn=pred, strategy=strategy)

    by_cat: dict[str, float] = {}
    by_cat_n: dict[str, int] = {}
    for t in res["trades"]:
        by_cat[t.category] = by_cat.get(t.category, 0.0) + t.pnl
        by_cat_n[t.category] = by_cat_n.get(t.category, 0) + 1

    profitable_cats = {c for c, p in by_cat.items() if p > 0}

    def predict(event: dict, market_info: dict) -> dict:
        cat = event.get("category") or "Other"
        if cat in profitable_cats:
            return platt(event, market_info)
        # Defer to market — strategy will see no edge → skip trade
        try:
            q = market_mid_forecast(event, market_info)
        except Exception:
            q = 0.5
        return {"p_yes": q, "rationale": f"gated: cat={cat} unprofitable in train"}

    predict.profitable_cats = profitable_cats  # type: ignore[attr-defined]
    predict.train_pnl_by_cat = by_cat  # type: ignore[attr-defined]
    return predict
