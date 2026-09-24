"""
Reports enforcement-bypass windows to the platform.

The SDK defaults to ``fail_open``: when the governance plane is unreachable, or
the plan quota is exhausted, an action proceeds with no server enforcement
decision. ``on_enforcement_bypassed`` surfaces that locally, but until it was
also sent to the server the platform's audit trail could not tell "all calls
governed" from "the SDK lost contact for two hours" -- no bypass record read as
fully governed.

Mirrors ``sdk-js/src/bypassReporter.ts`` (the two cannot import from each other;
``backend/tests/unit/sdkBypassParity.test.ts`` holds them together). Design:
``docs/superpowers/specs/2026-09-21-sdk-bypass-reporting-design.md``.

* Bypasses are COALESCED into windows keyed by ``(agent_id, reason)``. A per-call
  record would put thousands of rows a minute into a per-org hash chain.
* Only CLOSED windows are sent, so a window is immutable once reported. A window
  closes on recovery (a governed decision for the agent), after an idle gap, or
  at a maximum duration -- a long outage is many records, not one unbounded,
  mutable one.
* Nothing here is on the enforcement path. Recording is cheap and in-memory;
  sending happens from a background thread, and every public method swallows its
  own errors.
* Evidence is kept until the server acknowledges it, retried with backoff, and
  the buffer is bounded. When it overflows the oldest window is dropped and the
  LOSS is reported too, so the trail says "N bypasses were not reported" instead
  of staying silent.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

BYPASS_REASONS = (
    "circuit_breaker_open",
    "network_error",
    "server_error",
    "plan_limit_exceeded",
)

# Bounds mirror the server's schema (backend/src/schema/sdkBypass.ts). A field
# over its limit gets the WHOLE report rejected, so it is truncated here. The
# server counts UTF-16 code units (JavaScript string length), not Python code
# points, so truncation is by UTF-16 units too: 500 emoji are 1000 units.
MAX_AGENT_ID = 256
MAX_SOURCE = 64
MAX_MESSAGE = 500
MAX_WINDOWS_PER_REQUEST = 100
MAX_COUNT = 1_000_000_000
MAX_CONSECUTIVE_FAILURES = 1_000_000

BACKOFF_BASE_SECONDS = 10.0
BACKOFF_MAX_SECONDS = 300.0

# Statuses that mean "this body will never be accepted": resending is a loop, so
# the batch is discarded (and logged). Everything else -- 5xx, 429, 408, auth
# errors, and 404 from a backend that does not have the endpoint yet -- is kept
# and retried: the buffer is bounded, and the evidence is worth more than a
# request every few minutes.
NON_RETRYABLE = frozenset({400, 413, 422})

# The server requires min length 1 for these; an empty value would get the
# whole report rejected.
_UNKNOWN = "unknown"

SendFn = Callable[[Dict[str, Any]], int]

# Postgres text/jsonb cannot store NUL or a lone UTF-16 surrogate. One such
# character makes the server's audit write fail, which reads as a retryable 5xx,
# so the window would be retried for ever and never recorded. Python strings
# hold a lone surrogate as one character (a valid emoji is a single non-
# surrogate character), so this range only ever matches the unstorable ones.
_REPLACEMENT = chr(0xFFFD)
_UNSTORABLE = re.compile("[" + chr(0) + chr(0xD800) + "-" + chr(0xDFFF) + "]")


def _iso(epoch_seconds: float) -> str:
    """ISO-8601 UTC with milliseconds and a ``Z`` suffix, like the JS SDK."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _truncate(value: str, limit: int) -> str:
    """Cut ``value`` to at most ``limit`` UTF-16 code units."""
    if len(value) * 2 <= limit:
        return value
    units = 0
    for index, char in enumerate(value):
        units += 2 if ord(char) > 0xFFFF else 1
        if units > limit:
            return value[:index]
    return value


def _text(value: Any, limit: int) -> str:
    """Make ``value`` storable, then cut it to ``limit`` UTF-16 units."""
    return _truncate(_UNSTORABLE.sub(_REPLACEMENT, str(value)), limit)


class BypassWindowReporter:
    """Coalesces bypass observations into windows and delivers them.

    Thread-safe. ``send`` receives the request body and returns the HTTP status;
    it raises on a network failure. It is only ever called from ``flush`` /
    ``drain``, never from ``record``.
    """

    def __init__(
        self,
        send: SendFn,
        sdk: Dict[str, str],
        *,
        idle_seconds: float = 60.0,
        max_duration_seconds: float = 15 * 60.0,
        max_buffered_windows: int = 500,
        now: Callable[[], float] = time.time,
        new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._send = send
        self._sdk = sdk
        self._idle = idle_seconds
        self._max_duration = max_duration_seconds
        self._max_buffered = max_buffered_windows
        self._now = now
        self._new_id = new_id
        self._log = logger or logging.getLogger("Execlave")

        # Guards every field below. Never held across a send.
        self._lock = threading.RLock()
        # Serialises sends: at most one request in flight.
        self._send_lock = threading.Lock()

        self._open: Dict[tuple, Dict[str, Any]] = {}
        self._closed: List[Dict[str, Any]] = []

        # Loss accounting. ``_pending_drop`` is the block currently awaiting
        # acknowledgement: immutable, so a retry after a lost response reuses the
        # same dropId and the server can recognise it. Drops that happen
        # meanwhile accumulate in ``_live_drop``.
        self._live_drop: Dict[str, Any] = {"windows": 0, "bypasses": 0, "since": None}
        self._pending_drop: Optional[Dict[str, Any]] = None

        self._failures = 0
        self._next_attempt_at = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        reason: str,
        source: str,
        agent_id: str,
        message: Optional[str] = None,
        status: Optional[int] = None,
        consecutive_failures: Optional[int] = None,
    ) -> None:
        """Record one bypass. Cheap, in-memory, never raises."""
        try:
            agent = _text(agent_id, MAX_AGENT_ID) or _UNKNOWN
            src = _text(source, MAX_SOURCE) or _UNKNOWN
            with self._lock:
                t = self._now()
                key = (agent, reason)
                existing = self._open.get(key)
                if existing and (
                    t - existing["last"] >= self._idle
                    or t - existing["first"] >= self._max_duration
                ):
                    self._close_locked(key)

                w = self._open.get(key)
                if w:
                    wire = w["wire"]
                    wire["count"] = min(wire["count"] + 1, MAX_COUNT)
                    wire["lastAt"] = _iso(t)
                    wire["source"] = src
                    w["last"] = t
                    self._apply_optional(wire, message, status, consecutive_failures)
                    return

                wire = {
                    "windowId": self._new_id(),
                    "agentId": agent,
                    "reason": reason,
                    "source": src,
                    "firstAt": _iso(t),
                    "lastAt": _iso(t),
                    "count": 1,
                }
                self._apply_optional(wire, message, status, consecutive_failures)
                self._open[key] = {"wire": wire, "first": t, "last": t}
        except Exception as exc:  # noqa: BLE001 - reporting must never break enforcement
            self._log.debug("bypass record failed: %s", exc)

    @staticmethod
    def _apply_optional(
        wire: Dict[str, Any],
        message: Optional[str],
        status: Optional[int],
        consecutive_failures: Optional[int],
    ) -> None:
        if message is not None:
            wire["message"] = _text(message, MAX_MESSAGE)
        if status is not None:
            wire["status"] = int(status)
        if consecutive_failures is not None:
            wire["consecutiveFailures"] = max(
                0, min(int(consecutive_failures), MAX_CONSECUTIVE_FAILURES)
            )

    def recover(self, agent_id: str) -> None:
        """The agent got a governed decision from the server: its windows are over."""
        try:
            agent = _text(agent_id, MAX_AGENT_ID) or _UNKNOWN
            with self._lock:
                for key in [k for k in self._open if k[0] == agent]:
                    self._close_locked(key)
        except Exception as exc:  # noqa: BLE001
            self._log.debug("bypass recover failed: %s", exc)

    def tick(self) -> None:
        """Close windows that have gone idle or reached the maximum duration."""
        try:
            with self._lock:
                t = self._now()
                for key, w in list(self._open.items()):
                    if (
                        t - w["last"] >= self._idle
                        or t - w["first"] >= self._max_duration
                    ):
                        self._close_locked(key)
        except Exception as exc:  # noqa: BLE001
            self._log.debug("bypass tick failed: %s", exc)

    def _close_locked(self, key: tuple) -> None:
        w = self._open.pop(key, None)
        if w is None:
            return
        self._closed.append(w["wire"])

        # Bounded: never let an outage grow memory without limit. The oldest
        # goes first, and the loss is counted so it can be reported.
        while len(self._closed) > self._max_buffered:
            dropped = self._closed.pop(0)
            self._live_drop["windows"] += 1
            self._live_drop["bypasses"] += dropped["count"]
            if self._live_drop["since"] is None:
                self._live_drop["since"] = dropped["firstAt"]

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    @property
    def has_pending(self) -> bool:
        """True when anything closed (or a loss) is waiting to be acknowledged."""
        with self._lock:
            return self._has_pending_locked()

    def _has_pending_locked(self) -> bool:
        return (
            bool(self._closed)
            or self._pending_drop is not None
            or self._live_drop["windows"] > 0
        )

    def flush(self, *, force: bool = False) -> None:
        """Send everything closed until the server has acknowledged it, or a
        request fails. ``force`` skips the retry backoff (shutdown). If another
        thread is already sending, returns immediately: that send drains the same
        buffer. Never raises."""
        if not self._send_lock.acquire(blocking=False):
            return
        try:
            self._run(force)
        finally:
            self._send_lock.release()

    def drain(self, timeout_seconds: float) -> None:
        """Shutdown: close every open window and make a bounded, best-effort
        attempt to deliver. Gives up at the deadline rather than hanging process
        exit on an endpoint that is down -- which is exactly when there is
        something to report. Never raises."""
        try:
            with self._lock:
                for key in list(self._open):
                    self._close_locked(key)
                if not self._has_pending_locked():
                    return
            worker = threading.Thread(
                target=self._flush_blocking, daemon=True, name="Execlave-bypass-drain"
            )
            worker.start()
            worker.join(timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            self._log.debug("bypass drain failed: %s", exc)

    def _flush_blocking(self) -> None:
        with self._send_lock:
            self._run(True)

    def _run(self, force: bool) -> None:
        try:
            while True:
                with self._lock:
                    if not self._has_pending_locked():
                        return
                    if not force and self._now() < self._next_attempt_at:
                        return

                    if self._pending_drop is None and self._live_drop["windows"] > 0:
                        self._pending_drop = {"dropId": self._new_id(), **self._live_drop}
                        self._live_drop = {"windows": 0, "bypasses": 0, "since": None}

                    windows = [dict(w) for w in self._closed[:MAX_WINDOWS_PER_REQUEST]]
                    body: Dict[str, Any] = {
                        "sdk": dict(self._sdk),
                        "sentAt": _iso(self._now()),
                        "windows": windows,
                    }
                    if self._pending_drop is not None:
                        # `since` is optional on the wire; null would be rejected.
                        body["dropped"] = {
                            k: v for k, v in self._pending_drop.items() if v is not None
                        }

                try:
                    status = self._send(body)
                except Exception as exc:  # noqa: BLE001
                    self._fail(f"bypass report failed: {exc}")
                    return

                with self._lock:
                    if 200 <= status < 300:
                        self._failures = 0
                        self._next_attempt_at = 0.0
                        self._acknowledge(windows)
                    elif status in NON_RETRYABLE:
                        # The body will never be accepted; resending is a loop.
                        self._log.warning(
                            "bypass report rejected (HTTP %s); discarding %d window(s)",
                            status,
                            len(windows),
                        )
                        self._acknowledge(windows)
                    else:
                        self._fail_locked(f"bypass report failed (HTTP {status})")
                        return
        except Exception as exc:  # noqa: BLE001
            self._fail(f"bypass report failed: {exc}")

    def _acknowledge(self, sent: List[Dict[str, Any]]) -> None:
        ids = {w["windowId"] for w in sent}
        self._closed = [w for w in self._closed if w["windowId"] not in ids]
        self._pending_drop = None

    def _fail(self, message: str) -> None:
        with self._lock:
            self._fail_locked(message)

    def _fail_locked(self, message: str) -> None:
        self._failures += 1
        delay = min(
            BACKOFF_BASE_SECONDS * (2 ** min(self._failures - 1, 16)), BACKOFF_MAX_SECONDS
        )
        self._next_attempt_at = self._now() + delay
        self._log.debug(message)
