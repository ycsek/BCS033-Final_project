# logger.py
"""Experiment logger — structured, thread-safe logging with ISO timestamps.

Records hyperparameters, per-epoch metrics, MIA results, and timing
information in a structured JSON file alongside a human-readable log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Thread-safe experiment logger with structured JSON output.

    Parameters
    ----------
    cfg : SimpleNamespace
        Full experiment configuration loaded from YAML.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        self._lock = threading.Lock()

        # ── Build log directory name ────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "DP" if cfg.dp.use_dp else "Base"
        dataset = cfg.training.dataset
        log_base = getattr(cfg.paths, "log_base", "./logs")
        self.log_dir = os.path.join(
            log_base, f"Static_{prefix}_{dataset}_{timestamp}"
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # ── Set up Python logger ────────────────────────────────────
        self.logger = logging.getLogger(f"experiment.{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # avoid duplicate output

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

        # File handler
        fh = logging.FileHandler(
            os.path.join(self.log_dir, "training.log"), encoding="utf-8"
        )
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        # ── Results store ───────────────────────────────────────────
        self._cfg_dict = self._namespace_to_dict(cfg)
        self.results: Dict[str, Any] = {
            "hyperparameters": self._cfg_dict,
            "training_start_time": None,
            "training_end_time": None,
            "target_epsilon": cfg.dp.target_epsilon if cfg.dp.use_dp else None,
            "final_epsilon_spent": None,
            "trajectory": [],
            "final_mia_metrics": None,
        }

    # ── Public API ──────────────────────────────────────────────────

    def info(self, msg: str) -> None:
        """Log an INFO-level message (thread-safe)."""
        with self._lock:
            self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a WARNING-level message (thread-safe)."""
        with self._lock:
            self.logger.warning(msg)

    def log_training_start(self) -> str:
        """Record the training start time. Returns ISO timestamp."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.results["training_start_time"] = ts
        self.info(f"Training started at {ts}")
        return ts

    def log_training_end(self) -> str:
        """Record the training end time. Returns ISO timestamp."""
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.results["training_end_time"] = ts
        self.info(f"Training ended at {ts}")
        return ts

    def log_epoch_metrics(
        self,
        epoch: int,
        target_loss: float,
        train_acc: float,
        target_acc: float,
        epsilon: Optional[float] = None,
    ) -> None:
        """Append one epoch's metrics to the trajectory."""
        record: Dict[str, Any] = {
            "epoch": epoch,
            "target_loss": target_loss,
            "train_acc": train_acc,
            "target_acc": target_acc,
        }
        if epsilon is not None:
            record["epsilon"] = epsilon
        with self._lock:
            self.results["trajectory"].append(record)

    def log_final_epsilon(self, epsilon: float) -> None:
        """Record the final privacy budget spent."""
        with self._lock:
            self.results["final_epsilon_spent"] = epsilon
        self.info(f"Final privacy budget spent: ε = {epsilon:.4f}")

    def log_final_metrics(self, metrics: Dict[str, Any]) -> None:
        """Store the final MIA evaluation metrics."""
        with self._lock:
            self.results["final_mia_metrics"] = metrics

    def save_results(self) -> str:
        """Persist the results dict to ``results.json``.

        Returns
        -------
        str
            Path to the saved JSON file.
        """
        result_path = os.path.join(self.log_dir, "results.json")
        with self._lock:
            with open(result_path, "w", encoding="utf-8") as fh:
                json.dump(self.results, fh, indent=4, ensure_ascii=False)
        self.info(f">>> Experiment data saved to {result_path} <<<")
        return result_path

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _namespace_to_dict(ns: Any) -> Any:
        """Recursively convert SimpleNamespace to dict for JSON serialization."""
        if isinstance(ns, SimpleNamespace):
            return {k: ExperimentLogger._namespace_to_dict(v)
                    for k, v in vars(ns).items()}
        return ns
