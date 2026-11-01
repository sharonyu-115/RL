"""Append scalar metrics locally so monitoring does not depend on W&B availability."""

import json
import math
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from nemo_rl.utils.logger import LoggerInterface


class ScalarJsonlLogger(LoggerInterface):
    def __init__(self) -> None:
        self.path = Path(os.environ["SC_METRICS_PATH"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str | None = "",
        step_metric: str | None = None,
        step_finished: bool = False,
    ) -> None:
        scalars = {}
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            key = f"{prefix}/{name}" if prefix else name
            scalars[key] = scalar if math.isfinite(scalar) else str(scalar)
        record = {"step": step, "metrics": scalars, "step_finished": step_finished}
        with self.lock, self.path.open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        return None

    def log_histogram(self, histogram: list[Any], step: int, name: str) -> None:
        return None

    def log_plot(self, figure: Any, step: int, name: str) -> None:
        return None
