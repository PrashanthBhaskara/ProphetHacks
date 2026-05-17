"""Configuration and local environment loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "default.json"
ENV_ALIASES = {
    "GEMINI_API_KEY_1": "GEMINI_API_KEY",
    "GOOGLE_GEMINI_API_KEY": "GEMINI_API_KEY",
    # Temporary compatibility for the current local .env, where the personal
    # Gemini key was originally placed under the old OpenRouter slot.
    "OPENROUTER_API_KEY_1": "GEMINI_API_KEY",
    "KALSHI-ACCESS-KEY": "KALSHI_API_KEY_ID",
    "KALSHI_ACCESS_KEY": "KALSHI_API_KEY_ID",
    "KALSHI_API_KEY": "KALSHI_API_KEY_ID",
    "KALSHI-ACCESS-KEY-DEMO": "KALSHI_DEMO_API_KEY_ID",
    "KALSHI-PRIVATE-KEY-B64": "KALSHI_PRIVATE_KEY_B64",
    "KALSHI-PRIVATE-KEY": "KALSHI_PRIVATE_KEY",
    "KALSHI_PRIVATE_KEY_PEM": "KALSHI_PRIVATE_KEY",
    "KALSHI_PRIVATE_KEY_BASE64": "KALSHI_PRIVATE_KEY_B64",
    "PROPHET_API_KEY": "PA_SERVER_API_KEY",
    "AIPROPHET_API_KEY": "PA_SERVER_API_KEY",
}


@dataclass
class ModelConfig:
    name: str
    provider: str
    model: str
    api_key_env: str
    api_key_fallback_envs: list[str] = field(default_factory=list)
    temperature: float = 0.1
    max_tokens: int = 900
    enabled: bool = True
    native_search_grounding_enabled: bool = False
    native_search_grounding_live_only: bool = True
    search_grounding_engine: str = "native"
    search_grounding_max_results: int = 5
    search_grounding_max_total_results: int = 8
    search_grounding_context_size: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        search = data.get("search_grounding") or {}
        return cls(
            name=str(data["name"]),
            provider=str(data.get("provider", "openrouter")),
            model=str(data["model"]),
            api_key_env=str(data["api_key_env"]),
            api_key_fallback_envs=list(data.get("api_key_fallback_envs") or []),
            temperature=float(data.get("temperature", 0.1)),
            max_tokens=int(data.get("max_tokens", 900)),
            enabled=bool(data.get("enabled", True)),
            native_search_grounding_enabled=bool(search.get("enabled", False)),
            native_search_grounding_live_only=bool(search.get("live_only", True)),
            search_grounding_engine=str(search.get("engine", "native")),
            search_grounding_max_results=int(search.get("max_results", 5)),
            search_grounding_max_total_results=int(search.get("max_total_results", 8)),
            search_grounding_context_size=(
                str(search["search_context_size"]) if search.get("search_context_size") else None
            ),
        )


@dataclass
class GateConfig:
    cheap_delta_pp: float = 0.025
    cheap_tight_spread: float = 0.05
    cheap_high_spread: float = 0.15
    cheap_categories: set[str] = field(default_factory=lambda: {
        "Sports",
        "Economics",
        "Politics",
        "Elections",
        "Climate and Weather",
        "Weather",
    })
    supervisor_disagreement_pp: float = 0.05
    supervisor_min_edge: float = 0.06
    high_notional_liquidity: float = 100000.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GateConfig":
        cfg = cls()
        if not data:
            return cfg
        cfg.cheap_delta_pp = float(data.get("cheap_delta_pp", cfg.cheap_delta_pp))
        cfg.cheap_tight_spread = float(data.get("cheap_tight_spread", cfg.cheap_tight_spread))
        cfg.cheap_high_spread = float(data.get("cheap_high_spread", cfg.cheap_high_spread))
        cfg.cheap_categories = set(data.get("cheap_categories") or cfg.cheap_categories)
        cfg.supervisor_disagreement_pp = float(
            data.get("supervisor_disagreement_pp", cfg.supervisor_disagreement_pp)
        )
        cfg.supervisor_min_edge = float(data.get("supervisor_min_edge", cfg.supervisor_min_edge))
        cfg.high_notional_liquidity = float(data.get("high_notional_liquidity", cfg.high_notional_liquidity))
        return cfg


@dataclass
class RiskConfig:
    min_edge: float = 0.06
    hard_no_trade_spread: float = 0.30
    wide_spread: float = 0.15
    spread_buffer: float = 0.50
    uncertainty_buffer: float = 0.03
    fee_buffer: float = 0.0
    kelly_fraction: float = 0.08
    max_equity_fraction_per_market: float = 0.01
    starting_equity: float = 10000.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RiskConfig":
        cfg = cls()
        if not data:
            return cfg
        for field_name in cfg.__dataclass_fields__:
            if field_name in data:
                setattr(cfg, field_name, float(data[field_name]))
        return cfg


@dataclass
class StatConfig:
    max_market_deviation: float = 0.06
    default_market_deviation: float = 0.03
    kalman_process_var: float = 0.04
    kalman_base_obs_var: float = 0.03
    min_ar_snapshots: int = 8
    max_ar_lag: int = 3
    near_close_brier_enabled: bool = False
    near_close_max_horizon_hours: float = 0.5
    near_close_momentum_scale: float = 0.10
    near_close_momentum_cap: float = 0.04
    near_close_platt_a: float = 0.09
    near_close_platt_b: float = 1.21

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StatConfig":
        cfg = cls()
        if not data:
            return cfg
        cfg.max_market_deviation = float(data.get("max_market_deviation", cfg.max_market_deviation))
        cfg.default_market_deviation = float(data.get("default_market_deviation", cfg.default_market_deviation))
        cfg.kalman_process_var = float(data.get("kalman_process_var", cfg.kalman_process_var))
        cfg.kalman_base_obs_var = float(data.get("kalman_base_obs_var", cfg.kalman_base_obs_var))
        cfg.min_ar_snapshots = int(data.get("min_ar_snapshots", cfg.min_ar_snapshots))
        cfg.max_ar_lag = int(data.get("max_ar_lag", cfg.max_ar_lag))
        cfg.near_close_brier_enabled = bool(data.get("near_close_brier_enabled", cfg.near_close_brier_enabled))
        cfg.near_close_max_horizon_hours = float(
            data.get("near_close_max_horizon_hours", cfg.near_close_max_horizon_hours)
        )
        cfg.near_close_momentum_scale = float(
            data.get("near_close_momentum_scale", cfg.near_close_momentum_scale)
        )
        cfg.near_close_momentum_cap = float(data.get("near_close_momentum_cap", cfg.near_close_momentum_cap))
        cfg.near_close_platt_a = float(data.get("near_close_platt_a", cfg.near_close_platt_a))
        cfg.near_close_platt_b = float(data.get("near_close_platt_b", cfg.near_close_platt_b))
        return cfg


@dataclass
class BudgetConfig:
    dry_run_default: bool = True
    log_dir: str = "dhruv_GPT_forecasting/logs"
    estimated_prices_per_1m_tokens: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "BudgetConfig":
        cfg = cls()
        if not data:
            return cfg
        cfg.dry_run_default = bool(data.get("dry_run_default", cfg.dry_run_default))
        cfg.log_dir = str(data.get("log_dir", cfg.log_dir))
        cfg.estimated_prices_per_1m_tokens = dict(data.get("estimated_prices_per_1m_tokens") or {})
        return cfg


@dataclass
class ArenaConfig:
    gpt_enabled_default: bool = True
    live_data_enabled_default: bool = False
    second_pass_enabled: bool = True
    probability_floor: float = 0.001
    probability_ceiling: float = 0.999
    prior_shrink_weight: float = 0.00
    second_pass_delta_pp: float = 0.12
    second_pass_low_confidence: float = 0.40
    second_pass_high_entropy: float = 0.92
    live_cache_ttl_seconds: int = 900
    max_historical_analogs: int = 8
    max_live_evidence: int = 12
    pit_external_enabled_default: bool = False
    pit_external_root: str = "dhruv_GPT_forecasting/data/external_evidence"
    pit_external_sources: list[str] = field(default_factory=lambda: ["local_jsonl", "reddit", "gdelt", "espn", "wrds", "lseg"])
    pit_external_strict_collected_at: bool = True
    pit_external_live_lookback_hours: int = 24
    pit_external_max_records: int = 12
    pit_external_max_live_age_minutes: int = 10
    pit_external_clock_tolerance_seconds: int = 300
    pit_external_archive_live_fetches: bool = True
    response_deadline_seconds: float = 480.0
    evidence_source_timeout_seconds: float = 5.0
    total_evidence_timeout_seconds: float = 45.0
    llm_timeout_seconds: float = 90.0
    deadline_reserve_seconds: float = 30.0
    min_gpt_call_seconds: float = 55.0
    grounded_research_enabled_default: bool = True
    grounded_research_live_only: bool = True
    grounded_research_backtest_enabled: bool = False
    grounded_research_timeout_seconds: float = 45.0
    grounded_research_min_seconds: float = 20.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ArenaConfig":
        cfg = cls()
        if not data:
            return cfg
        cfg.gpt_enabled_default = bool(data.get("gpt_enabled_default", cfg.gpt_enabled_default))
        cfg.live_data_enabled_default = bool(data.get("live_data_enabled_default", cfg.live_data_enabled_default))
        cfg.second_pass_enabled = bool(data.get("second_pass_enabled", cfg.second_pass_enabled))
        cfg.probability_floor = float(data.get("probability_floor", cfg.probability_floor))
        cfg.probability_ceiling = float(data.get("probability_ceiling", cfg.probability_ceiling))
        cfg.prior_shrink_weight = float(data.get("prior_shrink_weight", cfg.prior_shrink_weight))
        cfg.second_pass_delta_pp = float(data.get("second_pass_delta_pp", cfg.second_pass_delta_pp))
        cfg.second_pass_low_confidence = float(
            data.get("second_pass_low_confidence", cfg.second_pass_low_confidence)
        )
        cfg.second_pass_high_entropy = float(data.get("second_pass_high_entropy", cfg.second_pass_high_entropy))
        cfg.live_cache_ttl_seconds = int(data.get("live_cache_ttl_seconds", cfg.live_cache_ttl_seconds))
        cfg.max_historical_analogs = int(data.get("max_historical_analogs", cfg.max_historical_analogs))
        cfg.max_live_evidence = int(data.get("max_live_evidence", cfg.max_live_evidence))
        cfg.pit_external_enabled_default = bool(
            data.get("pit_external_enabled_default", cfg.pit_external_enabled_default)
        )
        cfg.pit_external_root = str(data.get("pit_external_root", cfg.pit_external_root))
        cfg.pit_external_sources = list(data.get("pit_external_sources") or cfg.pit_external_sources)
        cfg.pit_external_strict_collected_at = bool(
            data.get("pit_external_strict_collected_at", cfg.pit_external_strict_collected_at)
        )
        cfg.pit_external_live_lookback_hours = int(
            data.get("pit_external_live_lookback_hours", cfg.pit_external_live_lookback_hours)
        )
        cfg.pit_external_max_records = int(data.get("pit_external_max_records", cfg.pit_external_max_records))
        cfg.pit_external_max_live_age_minutes = int(
            data.get("pit_external_max_live_age_minutes", cfg.pit_external_max_live_age_minutes)
        )
        cfg.pit_external_clock_tolerance_seconds = int(
            data.get("pit_external_clock_tolerance_seconds", cfg.pit_external_clock_tolerance_seconds)
        )
        cfg.pit_external_archive_live_fetches = bool(
            data.get("pit_external_archive_live_fetches", cfg.pit_external_archive_live_fetches)
        )
        cfg.response_deadline_seconds = float(data.get("response_deadline_seconds", cfg.response_deadline_seconds))
        cfg.evidence_source_timeout_seconds = float(
            data.get("evidence_source_timeout_seconds", cfg.evidence_source_timeout_seconds)
        )
        cfg.total_evidence_timeout_seconds = float(
            data.get("total_evidence_timeout_seconds", cfg.total_evidence_timeout_seconds)
        )
        cfg.llm_timeout_seconds = float(data.get(
            "llm_timeout_seconds",
            data.get("openrouter_timeout_seconds", cfg.llm_timeout_seconds),
        ))
        cfg.deadline_reserve_seconds = float(data.get("deadline_reserve_seconds", cfg.deadline_reserve_seconds))
        cfg.min_gpt_call_seconds = float(data.get("min_gpt_call_seconds", cfg.min_gpt_call_seconds))
        cfg.grounded_research_enabled_default = bool(
            data.get("grounded_research_enabled_default", cfg.grounded_research_enabled_default)
        )
        cfg.grounded_research_live_only = bool(
            data.get("grounded_research_live_only", cfg.grounded_research_live_only)
        )
        cfg.grounded_research_backtest_enabled = bool(
            data.get("grounded_research_backtest_enabled", cfg.grounded_research_backtest_enabled)
        )
        cfg.grounded_research_timeout_seconds = float(
            data.get("grounded_research_timeout_seconds", cfg.grounded_research_timeout_seconds)
        )
        cfg.grounded_research_min_seconds = float(
            data.get("grounded_research_min_seconds", cfg.grounded_research_min_seconds)
        )
        return cfg


@dataclass
class ForecastConfig:
    cheap_model: ModelConfig
    supervisor_model: ModelConfig
    gates: GateConfig
    risk: RiskConfig
    stat: StatConfig
    budget: BudgetConfig
    arena: ArenaConfig


def load_local_env(path: Path | None = None) -> None:
    """Load key=value pairs from a local .env without printing secrets."""
    candidates = []
    if path is not None:
        candidates.append(path)
    candidates.extend([
        Path.cwd() / ".env",
        PACKAGE_ROOT / ".env",
        PACKAGE_ROOT.parent / "prophet-hacks-handoff" / "prep" / ".env",
    ])
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
            alias = ENV_ALIASES.get(key)
            if alias and alias not in os.environ:
                os.environ[alias] = value


def resolve_api_key(model: ModelConfig) -> tuple[str | None, str | None]:
    for env_name in [model.api_key_env, *model.api_key_fallback_envs]:
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, None


def load_config(path: Path | str | None = None) -> ForecastConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    models = raw.get("models") or {}
    return ForecastConfig(
        cheap_model=ModelConfig.from_dict(models["cheap"]),
        supervisor_model=ModelConfig.from_dict(models["supervisor"]),
        gates=GateConfig.from_dict(raw.get("gates")),
        risk=RiskConfig.from_dict(raw.get("risk")),
        stat=StatConfig.from_dict(raw.get("stat")),
        budget=BudgetConfig.from_dict(raw.get("budget")),
        arena=ArenaConfig.from_dict(raw.get("arena")),
    )
