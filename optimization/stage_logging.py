"""Human-readable, thread-safe operational logging for Phase 4 runs."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Callable


StageLogger = Callable[..., None]


class IterationStageLogger:
    """Append one complete line atomically and mirror it to stdout."""

    schema_version = "phase4_iteration_stage_log_v1"

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()
        self._sequence = 0
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def emit(
        self, *, event: str, stage: str, message: str, **details: Any,
    ) -> None:
        with self._lock:
            self._sequence += 1
            timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
            suffix = ""
            if details:
                suffix = " | " + json.dumps(
                    details, sort_keys=True, ensure_ascii=False, default=str)
            line = (
                f"{timestamp} [{event.upper()}] [{stage}] "
                f"{message}{suffix}\n")
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
            print(line, end="", flush=True)


def emit_stage(
    stage_logger: StageLogger | None, *, event: str, stage: str,
    message: str, **details: Any,
) -> None:
    if stage_logger is not None:
        stage_logger(
            event=event, stage=stage, message=message, **details)
