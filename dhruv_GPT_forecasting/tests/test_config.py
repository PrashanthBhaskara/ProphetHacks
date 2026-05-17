from dhruv_gpt_forecasting.config import load_config
from dhruv_gpt_forecasting.config import ENV_ALIASES


def test_default_models_are_direct_gemini_3_flash_with_native_grounding():
    cfg = load_config()
    assert cfg.cheap_model.provider == "gemini"
    assert cfg.supervisor_model.provider == "gemini"
    assert cfg.cheap_model.model == "gemini-3-flash-preview"
    assert cfg.supervisor_model.model == "gemini-3-flash-preview"
    assert cfg.cheap_model.api_key_env == "GEMINI_API_KEY"
    assert cfg.supervisor_model.api_key_env == "GEMINI_API_KEY"
    assert cfg.cheap_model.native_search_grounding_enabled is True
    assert cfg.cheap_model.native_search_grounding_live_only is True
    assert cfg.cheap_model.search_grounding_engine == "native"
    assert "OPENROUTER_API_KEY_2" not in cfg.cheap_model.api_key_fallback_envs
    assert "OPENROUTER_API_KEY_3" not in cfg.cheap_model.api_key_fallback_envs
    assert "OPENROUTER_API_KEY_4" not in cfg.cheap_model.api_key_fallback_envs
    assert "OPENROUTER_API_KEY_1" not in cfg.cheap_model.api_key_fallback_envs
    assert cfg.arena.gpt_enabled_default is True
    assert cfg.arena.probability_floor == 0.001
    assert cfg.stat.near_close_brier_enabled is True
    assert cfg.arena.prior_shrink_weight == 0.0
    assert cfg.arena.pit_external_enabled_default is False
    assert "reddit" in cfg.arena.pit_external_sources
    assert "gdelt" in cfg.arena.pit_external_sources
    assert "espn" in cfg.arena.pit_external_sources
    assert "wrds" in cfg.arena.pit_external_sources
    assert "lseg" in cfg.arena.pit_external_sources
    assert cfg.arena.pit_external_archive_live_fetches is True
    assert cfg.arena.response_deadline_seconds == 480
    assert cfg.arena.evidence_source_timeout_seconds == 5
    assert cfg.arena.total_evidence_timeout_seconds == 45
    assert cfg.arena.llm_timeout_seconds == 90
    assert cfg.arena.deadline_reserve_seconds == 30


def test_env_aliases_include_kalshi_access_names():
    assert ENV_ALIASES["OPENROUTER_API_KEY_1"] == "GEMINI_API_KEY"
    assert ENV_ALIASES["KALSHI-ACCESS-KEY"] == "KALSHI_API_KEY_ID"
    assert ENV_ALIASES["KALSHI-ACCESS-KEY-DEMO"] == "KALSHI_DEMO_API_KEY_ID"
