"""
Buffered writer for the inference log.

Request threads enqueue rows; a daemon thread bulk-inserts them every
``flush_seconds`` (or sooner when a batch fills), keeping the DB write off the
request path. If the buffer is full the row is dropped and counted.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import insert
from sqlalchemy.orm import Session, sessionmaker

from app.db import InferenceLog
from app.features import ModelInput
from app.metrics import INFERENCE_LOG_DROPPED

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


class InferenceLogger:
    def __init__(self, session_factory: sessionmaker[Session], flush_seconds: float, buffer_size: int) -> None:
        self._session_factory = session_factory
        self._flush_seconds = flush_seconds
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=buffer_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._flush_lock = threading.Lock()

    def record(
        self, features: ModelInput, model_version: int, predicted: float, cached: bool, source: str
    ) -> None:
        row = {
            **features.as_row(),
            "created_at": datetime.now(UTC),
            "model_version": model_version,
            "predicted": predicted,
            "cached": cached,
            "source": source,
        }
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            INFERENCE_LOG_DROPPED.inc()

    def flush(self) -> int:
        """Write everything buffered so far. Returns rows written."""
        written = 0
        with self._flush_lock:
            while True:
                batch: list[dict[str, Any]] = []
                while len(batch) < BATCH_SIZE:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                if not batch:
                    return written
                try:
                    with self._session_factory() as db:
                        db.execute(insert(InferenceLog), batch)
                        db.commit()
                    written += len(batch)
                except Exception:
                    INFERENCE_LOG_DROPPED.inc(len(batch))
                    logger.exception("inference log flush failed; dropped %d rows", len(batch))

    def _run(self) -> None:
        while not self._stop.wait(self._flush_seconds):
            self.flush()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="inference-log-writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.flush()
