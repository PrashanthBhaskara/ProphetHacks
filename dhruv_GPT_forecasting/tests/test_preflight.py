import math

import pytest

from dhruv_gpt_forecasting.config import BudgetConfig, ModelConfig, load_config
from dhruv_gpt_forecasting.key_utils import api_key_metadata, key_fingerprint
from dhruv_gpt_forecasting.openrouter import call_openrouter_json
from dhruv_gpt_forecasting.preflight import _gemini_status, _openrouter_status, run_preflight


def test_key_fingerprint_is_nonsecret():
    secret = "sk-or-test-secret-value"
    fingerprint = key_fingerprint(secret)
    metadata = api_key_metadata(value=secret, env_name="OPENROUTER_API_KEY_1", expected_prefix="sk-or-")

    assert fingerprint
    assert secret not in str(metadata)
    assert metadata["api_key_fingerprint"] == fingerprint
    assert metadata["api_key_env"] == "OPENROUTER_API_KEY_1"
    assert metadata["key_length"] == len(secret)
    assert metadata["prefix_valid"] is True


def test_call_openrouter_log_includes_key_fingerprint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "sk-or-test-secret-value")

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"probabilities":{"YES":0.6,"NO":0.4}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    monkeypatch.setattr("dhruv_gpt_forecasting.openrouter.requests.post", lambda *args, **kwargs: Response())
    model = ModelConfig(
        name="test",
        provider="openrouter",
        model="openai/gpt-5-nano",
        api_key_env="OPENROUTER_API_KEY_1",
    )

    _, call_log = call_openrouter_json(
        model=model,
        messages=[{"role": "user", "content": "Return JSON"}],
        budget=BudgetConfig(),
        cache_key="test",
    )

    assert call_log.api_key_env == "OPENROUTER_API_KEY_1"
    assert call_log.api_key_fingerprint == key_fingerprint("sk-or-test-secret-value")
    assert call_log.input_tokens == 10
    assert call_log.output_tokens == 5
    assert call_log.search_grounding_enabled is False


def test_call_openrouter_can_enable_native_search_grounding(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "sk-or-test-secret-value")
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": '{"probabilities":{"YES":0.55,"NO":0.45}}',
                        "annotations": [{"type": "url_citation"}],
                    }
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    def fake_post(*args, **kwargs):
        captured["payload"] = kwargs["json"]
        return Response()

    monkeypatch.setattr("dhruv_gpt_forecasting.openrouter.requests.post", fake_post)
    model = ModelConfig(
        name="test",
        provider="openrouter",
        model="openai/gpt-5-nano",
        api_key_env="OPENROUTER_API_KEY_1",
        native_search_grounding_enabled=True,
        search_grounding_engine="native",
        search_grounding_max_results=5,
        search_grounding_max_total_results=8,
    )

    _, call_log = call_openrouter_json(
        model=model,
        messages=[{"role": "user", "content": "Return JSON"}],
        budget=BudgetConfig(),
        cache_key="test",
        search_grounding=True,
    )

    assert captured["payload"]["tools"] == [{
        "type": "openrouter:web_search",
        "parameters": {"engine": "native", "max_results": 5, "max_total_results": 8},
    }]
    assert call_log.search_grounding_enabled is True
    assert call_log.search_grounding_engine == "native"
    assert call_log.response_annotation_count == 1


def test_call_direct_gemini_uses_google_search_grounding(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "responseId": "gemini-response-id",
                "candidates": [{
                    "content": {"parts": [{"text": '{"probabilities":{"YES":0.55,"NO":0.45}}'}]},
                    "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://example.com"}}]},
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            }

    def fake_post(*args, **kwargs):
        captured["url"] = args[0]
        captured["headers"] = kwargs["headers"]
        captured["payload"] = kwargs["json"]
        return Response()

    monkeypatch.setattr("dhruv_gpt_forecasting.openrouter.requests.post", fake_post)
    model = ModelConfig(
        name="test",
        provider="gemini",
        model="gemini-3-flash-preview",
        api_key_env="GEMINI_API_KEY",
        native_search_grounding_enabled=True,
    )

    _, call_log = call_openrouter_json(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": "Forecast."},
        ],
        budget=BudgetConfig(),
        cache_key="test",
        search_grounding=True,
    )

    assert captured["url"].endswith("/models/gemini-3-flash-preview:generateContent")
    assert captured["headers"]["x-goog-api-key"] == "AIza-test-secret-value"
    assert captured["payload"]["tools"] == [{"google_search": {}}]
    assert "responseMimeType" not in captured["payload"]["generationConfig"]
    assert call_log.provider == "gemini"
    assert call_log.api_key_env == "GEMINI_API_KEY"
    assert call_log.provider_response_id == "gemini-response-id"
    assert call_log.search_grounding_enabled is True
    assert call_log.search_grounding_engine == "google_search"
    assert call_log.response_annotation_count == 1


def test_call_direct_gemini_repairs_malformed_grounded_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")
    calls = []

    class Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(*args, **kwargs):
        calls.append(kwargs["json"])
        if len(calls) == 1:
            return Response({
                "responseId": "grounded-response-id",
                "candidates": [{
                    "content": {"parts": [{"text": '{"summary":"partial'}]},
                    "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://example.com"}}]},
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
            })
        return Response({
            "responseId": "repair-response-id",
            "candidates": [{"content": {"parts": [{"text": '{"summary":"partial"}'}]}}],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 3},
        })

    monkeypatch.setattr("dhruv_gpt_forecasting.openrouter.requests.post", fake_post)
    model = ModelConfig(
        name="test",
        provider="gemini",
        model="gemini-3-flash-preview",
        api_key_env="GEMINI_API_KEY",
        native_search_grounding_enabled=True,
    )

    payload, call_log = call_openrouter_json(
        model=model,
        messages=[{"role": "user", "content": "Return JSON"}],
        budget=BudgetConfig(),
        cache_key="test",
        search_grounding=True,
    )

    assert payload == {"summary": "partial"}
    assert calls[0]["tools"] == [{"google_search": {}}]
    assert "tools" not in calls[1]
    assert calls[1]["generationConfig"]["responseMimeType"] == "application/json"
    assert call_log.input_tokens == 30
    assert call_log.output_tokens == 8
    assert call_log.fallback_path == "gemini_json_repair_after_parse_error"
    assert call_log.provider_response_id == "grounded-response-id,repair-response-id"


def test_preflight_offline_does_not_call_network(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")

    def fail_network(*args, **kwargs):
        raise AssertionError("offline preflight should not call network")

    monkeypatch.setattr("dhruv_gpt_forecasting.preflight.requests.get", fail_network)
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path)

    report = run_preflight(config=cfg, offline=True)

    assert report["ok"] is True
    assert report["checks"]["cache_writable"] is True
    assert report["checks"]["offline_prediction_valid"] is True
    assert report["llm"]["auth"]["checked"] is False


def test_openrouter_status_handles_auth_and_model_success():
    cfg = load_config()

    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.ok = 200 <= status_code < 300
            self.text = "text"

        def json(self):
            return self._payload

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith("/auth/key"):
            return Response(200, {"usage": 1, "label": "dev-key", "secret": "must-not-appear"})
        return Response(200, {"data": [{"id": cfg.model.model}]})

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("dhruv_gpt_forecasting.preflight.requests.get", fake_get)
    try:
        status = _openrouter_status(
            cfg,
            key="sk-or-test-secret-value",
            key_env="OPENROUTER_API_KEY_1",
            offline=False,
            timeout_seconds=1,
        )
    finally:
        monkeypatch.undo()

    assert len(calls) == 2
    assert status["auth"]["ok"] is True
    assert status["auth"]["safe_json"]["label"] == "dev-key"
    assert "secret" not in status["auth"]["safe_json"]
    assert status["model_availability"]["available"] is True


def test_openrouter_status_reads_nested_auth_key_usage():
    cfg = load_config()

    class Response:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.ok = 200 <= status_code < 300
            self.text = "text"

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if url.endswith("/auth/key"):
            return Response(200, {"data": {"usage": 1.25, "limit": 50.0, "limit_remaining": 48.75}})
        return Response(200, {"data": [{"id": cfg.model.model}]})

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("dhruv_gpt_forecasting.preflight.requests.get", fake_get)
    try:
        status = _openrouter_status(
            cfg,
            key="sk-or-test-secret-value",
            key_env="OPENROUTER_API_KEY_1",
            offline=False,
            timeout_seconds=1,
        )
    finally:
        monkeypatch.undo()

    assert status["auth"]["safe_json"]["usage"] == 1.25
    assert status["auth"]["safe_json"]["limit"] == 50.0
    assert status["auth"]["safe_json"]["limit_remaining"] == 48.75


def test_openrouter_status_handles_401_timeout_and_malformed(monkeypatch):
    cfg = load_config()

    class Response:
        def __init__(self, status_code, payload=None, json_error=False):
            self.status_code = status_code
            self._payload = payload
            self.ok = 200 <= status_code < 300
            self.text = "not-json"
            self.json_error = json_error

        def json(self):
            if self.json_error:
                raise ValueError("bad json")
            return self._payload

    def auth_401_then_malformed(url, **kwargs):
        if url.endswith("/auth/key"):
            return Response(401, {"error": {"message": "User not found."}})
        return Response(200, json_error=True)

    monkeypatch.setattr("dhruv_gpt_forecasting.preflight.requests.get", auth_401_then_malformed)
    status = _openrouter_status(
        cfg,
        key="sk-or-test-secret-value",
        key_env="OPENROUTER_API_KEY_1",
        offline=False,
        timeout_seconds=1,
    )
    assert status["auth"]["ok"] is False
    assert status["auth"]["error"] == "User not found."
    assert status["model_availability"]["available"] is None

    def timeout_get(url, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("dhruv_gpt_forecasting.preflight.requests.get", timeout_get)
    status = _openrouter_status(
        cfg,
        key="sk-or-test-secret-value",
        key_env="OPENROUTER_API_KEY_1",
        offline=False,
        timeout_seconds=1,
    )
    assert status["auth"]["ok"] is False
    assert "TimeoutError" in status["auth"]["error"]


def test_gemini_status_handles_model_list_success(monkeypatch):
    cfg = load_config()

    class Response:
        status_code = 200
        ok = True
        text = "text"

        def json(self):
            return {"models": [{"name": "models/gemini-3-flash-preview"}]}

    def fake_get(url, **kwargs):
        return Response()

    monkeypatch.setattr("dhruv_gpt_forecasting.preflight.requests.get", fake_get)
    status = _gemini_status(
        cfg,
        key="AIza-test-secret-value",
        key_env="GEMINI_API_KEY",
        offline=False,
        timeout_seconds=1,
    )

    assert status["provider"] == "gemini"
    assert status["key"]["prefix_valid"] is True
    assert status["auth"]["ok"] is True
    assert status["model_availability"]["available"] is True


def test_preflight_prediction_probabilities_sum(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-secret-value")
    cfg = load_config()
    cfg.budget.log_dir = str(tmp_path)

    report = run_preflight(config=cfg, offline=True)

    assert math.isclose(report["offline_prediction"]["probability_sum"], 1.0)
