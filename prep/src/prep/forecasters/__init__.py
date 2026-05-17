"""Model adapter entrypoints."""

from .base import ForecasterConfig, forecast_from_config

__all__ = ["ForecasterConfig", "forecast_from_config"]
