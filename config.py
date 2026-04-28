"""Configuration loader — reads a YAML file and exposes values as
a nested namespace with attribute access.  CLI arguments can override
any top-level YAML key via ``--key value`` pairs.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Any, Dict

import yaml


def _namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace."""
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = _namespace(val)
    return SimpleNamespace(**d)


def load_config(yaml_path: str = "config.yaml") -> SimpleNamespace:
    """Load experiment configuration from a YAML file."""
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh)

    cfg = _namespace(raw)
    return cfg


def load_config_with_cli(yaml_default: str = "config.yaml") -> SimpleNamespace:
    """Load config from YAML, allowing a ``--config`` CLI override."""
    parser = argparse.ArgumentParser(
        description="Static DP-SGD and MIA Evaluation"
    )
    parser.add_argument(
        "--config", type=str, default=yaml_default,
        help="Path to YAML configuration file",
    )
    # Allow selective CLI overrides for convenience
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--target_epochs", type=int, default=None)
    parser.add_argument("--use_dp", action="store_true", default=None)
    parser.add_argument("--no_dp", dest="use_dp", action="store_false")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_subset_size", type=int, default=None)

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.device is not None:
        cfg.training.device = args.device
    if args.dataset is not None:
        cfg.training.dataset = args.dataset
    if args.target_epochs is not None:
        cfg.training.target_epochs = args.target_epochs
    if args.use_dp is not None:
        cfg.dp.use_dp = args.use_dp
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.train_subset_size is not None:
        cfg.training.train_subset_size = args.train_subset_size

    return cfg