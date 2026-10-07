"""Opt-in, bounded operational events. Never attach general loggers or input payloads."""

import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from logging.handlers import RotatingFileHandler
from pathlib import Path

MAX_BYTES, BACKUPS = 2 * 1024 * 1024, 3


class Diagnostics:
    def __init__(self):
        from ito.desktop.settings import settings_path

        self.enabled = False
        override = os.environ.get("ITO_DEBUG", "").lower()
        self.override = None if not override else override not in {"0", "false", "off", "no"}
        self.run_id = uuid.uuid4().hex
        self.build = None
        self.path = settings_path().parent / "diagnostics.jsonl"
        self.jobs = queue.Queue(maxsize=1024)
        self.thread = None
        self.error = None
        self.dropped = 0
        self.lock = threading.RLock()
        self.last = {}
        self.set_enabled(bool(self.override))

    def set_enabled(self, enabled):
        with self.lock:
            enabled = self.override if self.override is not None else enabled
            if enabled == self.enabled:
                return
            if enabled and self.thread is None:
                try:
                    digest = hashlib.sha256()
                    for path in sorted(Path(__file__).parent.rglob("*.py")):
                        digest.update(path.read_bytes())
                    self.build = version("ito") + "+" + digest.hexdigest()[:12]
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handler = RotatingFileHandler(
                        self.path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8"
                    )
                except OSError:
                    self.error = "Could not open diagnostic log"
                    return
                handler.handleError = self._write_failed
                self.thread = threading.Thread(
                    target=self._write, args=(handler,), name="ito-diagnostics", daemon=True
                )
                self.thread.start()
            self.error = None
            self.enabled = True
            self.event("diagnostics", enabled=enabled)
            self.enabled = enabled
            self.last.clear()

    def event(self, event, *, interval=0, **fields):
        # Enabling may open a file and hash source code. Input and display threads
        # must not wait for that work, even when a pilot toggles logging mid-drive.
        if not self.enabled:
            return
        with self.lock:
            if not self.enabled:
                return
            now = time.monotonic()
            if interval and now - self.last.get(event, 0) < interval:
                return
            self.last[event] = now
            record = dict(
                timestamp=datetime.now(UTC).isoformat(),
                monotonic=now,
                run_id=self.run_id,
                build=self.build,
                event=event,
                dropped_events=self.dropped,
                **fields,
            )
            try:
                self.jobs.put_nowait(record)
            except queue.Full:
                self.dropped += 1

    def _write_failed(self, record):
        # Disk-full/permission errors must not flood stderr or interrupt piloting.
        with self.lock:
            self.enabled = False
            self.error = "Diagnostic log unavailable; check disk space and permissions"

    def _write(self, handler):
        try:
            while (record := self.jobs.get()) is not None:
                handler.emit(
                    logging.LogRecord(
                        "ito.diagnostics",
                        logging.INFO,
                        "",
                        0,
                        json.dumps(record, separators=(",", ":")),
                        (),
                        None,
                    )
                )
        finally:
            handler.close()

    def close(self):
        self.event("diagnostics_closed")
        with self.lock:
            self.enabled = False
            if self.thread:
                # Shutdown never waits for queue space on a stalled disk. Producers
                # share this lock, so freeing one slot guarantees the sentinel fits.
                try:
                    self.jobs.put_nowait(None)
                except queue.Full:
                    try:
                        self.jobs.get_nowait()
                    except queue.Empty:
                        pass
                    self.jobs.put_nowait(None)
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None


_current = None


def current():
    global _current
    if _current is None:
        _current = Diagnostics()
    return _current


def event(name, **fields):
    # Disabled hot paths do not allocate a logger, open files or hash the build.
    if _current is not None:
        _current.event(name, **fields)


@contextmanager
def stage(name):
    started = time.monotonic()
    event("shutdown_stage", stage=name, state="begin")
    try:
        yield
    finally:
        event(
            "shutdown_stage",
            stage=name,
            state="end",
            elapsed_ms=(time.monotonic() - started) * 1000,
        )


def close():
    global _current
    if _current is not None:
        _current.close()
        _current = None
