"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file and return an OmegaConf DictConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = OmegaConf.load(path)
    return cfg


def merge_configs(base: DictConfig, override: DictConfig) -> DictConfig:
    """Merge two configs, with override taking precedence."""
    return OmegaConf.merge(base, override)
