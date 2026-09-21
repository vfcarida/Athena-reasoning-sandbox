"""Utilities subpackage — Shared statistical functions, configuration, and metrics."""

from .config import AppConfig, config
from .metrics import elo_rating, overthinking_index, shannon_entropy

Config = AppConfig

__all__ = ["AppConfig", "Config", "config", "shannon_entropy", "elo_rating", "overthinking_index"]
