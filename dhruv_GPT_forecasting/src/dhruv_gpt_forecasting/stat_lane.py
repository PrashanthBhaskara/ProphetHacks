"""Market-anchored statistical forecast lane."""

from __future__ import annotations

import math
from statistics import mean

from .config import StatConfig
from .constraints import enforce_constraints
from .schemas import FeaturePacket, StatForecast, clamp_prob


def logit(p: float) -> float:
    p = clamp_prob(p)
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return clamp_prob(1.0 / (1.0 + math.exp(-x)))


def _platt_probability(pred: float, *, intercept: float, slope: float) -> float:
    p = clamp_prob(pred, lo=1e-6, hi=1.0 - 1e-6)
    return inv_logit(intercept + slope * math.log(p / (1.0 - p)))


def _near_close_brier_forecast(packet: FeaturePacket, cfg: StatConfig) -> float | None:
    if not cfg.near_close_brier_enabled:
        return None
    if packet.horizon_hours is None or packet.horizon_hours > cfg.near_close_max_horizon_hours:
        return None
    points = packet.price_trajectory
    if len(points) < 2:
        return None
    market = packet.quote.market_mid
    move = float(points[-1]["market_mid"] - points[0]["market_mid"])
    adjustment = max(
        -cfg.near_close_momentum_cap,
        min(cfg.near_close_momentum_cap, cfg.near_close_momentum_scale * move),
    )
    momentum = clamp_prob(market + adjustment)
    return _platt_probability(
        momentum,
        intercept=cfg.near_close_platt_a,
        slope=cfg.near_close_platt_b,
    )


def _kalman_logit_forecast(packet: FeaturePacket, cfg: StatConfig) -> float:
    points = packet.price_trajectory
    if not points:
        return packet.quote.market_mid
    state = logit(points[0]["market_mid"])
    state_var = 1.0
    prev_state = state
    for point in points:
        obs = logit(point["market_mid"])
        spread = point.get("spread")
        obs_var = cfg.kalman_base_obs_var + max(0.0, float(spread or 0.0)) * 0.8
        pred_var = state_var + cfg.kalman_process_var
        gain = pred_var / (pred_var + obs_var)
        prev_state = state
        state = state + gain * (obs - state)
        state_var = (1.0 - gain) * pred_var
    momentum = 0.25 * (state - prev_state) if len(points) >= 2 else 0.0
    return inv_logit(state + momentum)


def _ar_bic_logit_forecast(packet: FeaturePacket, cfg: StatConfig) -> float | None:
    ys = [logit(point["market_mid"]) for point in packet.price_trajectory]
    n = len(ys)
    if n < cfg.min_ar_snapshots:
        return None
    best_bic = float("inf")
    best_forecast: float | None = None
    max_lag = min(cfg.max_ar_lag, max(1, n // 3))
    for lag in range(1, max_lag + 1):
        rows = []
        target = []
        for i in range(lag, n):
            rows.append([1.0, *ys[i - lag:i][::-1]])
            target.append(ys[i])
        beta = _ols(rows, target)
        if beta is None:
            continue
        residuals = []
        for x, y in zip(rows, target):
            pred = sum(b * xx for b, xx in zip(beta, x))
            residuals.append(y - pred)
        sigma2 = max(1e-8, mean(r * r for r in residuals))
        bic = math.log(sigma2) + lag * math.log(len(target)) / len(target)
        forecast_x = [1.0, *ys[-lag:][::-1]]
        forecast = sum(b * xx for b, xx in zip(beta, forecast_x))
        if bic < best_bic:
            best_bic = bic
            best_forecast = forecast
    return inv_logit(best_forecast) if best_forecast is not None else None


def _ols(rows: list[list[float]], target: list[float]) -> list[float] | None:
    try:
        import numpy as np

        x = np.array(rows, dtype=float)
        y = np.array(target, dtype=float)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return [float(v) for v in beta]
    except Exception:
        return None


def _uncertainty(packet: FeaturePacket) -> float:
    spread = float(packet.quote.spread or 0.10)
    n = int(packet.features.get("n_snapshots") or 0)
    horizon = packet.horizon_hours
    u = 0.20 + min(0.35, spread) + (0.10 if n < 3 else 0.0)
    if horizon is None:
        u += 0.05
    elif horizon < 3:
        u += 0.08
    elif horizon > 24 * 7:
        u += 0.05
    if packet.category in {"Sports", "Politics", "Elections"}:
        u += 0.05
    return max(0.05, min(0.95, u))


def forecast_stat(packet: FeaturePacket, cfg: StatConfig | None = None) -> StatForecast:
    cfg = cfg or StatConfig()
    market = packet.quote.market_mid
    kalman = _kalman_logit_forecast(packet, cfg)
    ar_forecast = _ar_bic_logit_forecast(packet, cfg)
    raw_stat = kalman if ar_forecast is None else 0.75 * kalman + 0.25 * ar_forecast
    max_dev = cfg.max_market_deviation if len(packet.price_trajectory) >= cfg.min_ar_snapshots else cfg.default_market_deviation
    brier_forecast = _near_close_brier_forecast(packet, cfg)
    if brier_forecast is not None:
        calibrated = brier_forecast
    else:
        edge = max(-max_dev, min(max_dev, raw_stat - market))
        calibrated = clamp_prob(market + edge)
    uncertainty = _uncertainty(packet)
    confidence = max(0.0, min(1.0, 1.0 - uncertainty))
    reason_codes = ["market_anchor"]
    if packet.price_trajectory:
        reason_codes.append("kalman_logit_price")
    if ar_forecast is not None:
        reason_codes.append("ar_bic_blend")
    if brier_forecast is not None:
        reason_codes.append("near_close_brier_platt")
    if packet.quote.spread is not None and packet.quote.spread > 0.15:
        reason_codes.append("wide_spread")
    if packet.horizon_hours is not None and packet.horizon_hours < 3:
        reason_codes.append("near_close")
    probs = {"YES": calibrated, "NO": 1.0 - calibrated} if packet.is_binary_yes_no else {
        outcome: 1.0 / max(1, len(packet.outcomes)) for outcome in packet.outcomes
    }
    probs = enforce_constraints(probs, packet.outcomes, packet.event_structure)
    return StatForecast(
        probabilities=probs,
        market_prior=market,
        calibrated_probability=calibrated,
        uncertainty=uncertainty,
        confidence=confidence,
        reason_codes=reason_codes,
        diagnostics={
            "raw_stat_probability": raw_stat,
            "kalman_probability": kalman,
            "ar_probability": ar_forecast,
            "near_close_brier_probability": brier_forecast,
            "max_market_deviation": max_dev,
        },
    )
