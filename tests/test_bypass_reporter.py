"""BypassWindowReporter: coalescing, closing, delivery, overflow.

The SDK defaults to fail_open, so an outage or an exhausted plan quota lets
actions proceed with no server enforcement decision. The callback surfaces that
locally; this reporter puts it on the server's audit trail, so "no bypass
record" stops reading as "fully governed". Mirrors
sdk-js/src/__tests__/bypassReporter.test.ts so the two do not drift.

Bypasses are coalesced into windows -- a per-call record would put thousands of
rows a minute into a per-org hash chain during an outage. Only CLOSED windows are
sent, so a window is immutable once reported.
"""

import re
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from execlave.bypass_reporter import BypassWindowReporter, _truncate

IDLE = 60.0
MAX = 15 * 60.0
START = datetime(2026, 9, 21, 12, 0, 0, tzinfo=timezone.utc).timestamp()
SDK = {"name": "execlave-sdk", "language": "python", "version": "1.8.0"}
UUID_RE = re.compile(r"^[0-9a-f-]{36}$")


class Clock:
    def __init__(self):
        self.t = START

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


@pytest.fixture()
def clock():
    return Clock()


def make(clock, send, **opts):
    counter = iter(range(1, 10_000))
    return BypassWindowReporter(
        send,
        SDK,
        now=clock,
        new_id=lambda: f"00000000-0000-4000-8000-{next(counter):012d}",
        idle_seconds=IDLE,
        max_duration_seconds=MAX,
        **opts,
    )


def ok():
    return MagicMock(return_value=200)


def bypass(**over):
    return {
        "reason": "network_error",
        "source": "fail_open_network_error",
        "agent_id": "agent-1",
        **over,
    }


def bodies(send):
    return [c.args[0] for c in send.call_args_list]


class TestCoalescing:
    def test_folds_repeated_bypasses_for_one_agent_and_reason_into_one_window(self, clock):
        send = ok()
        r = make(clock, send)

        r.record(**bypass(message="first"))
        clock.advance(5)
        r.record(**bypass(message="last", status=503))
        clock.advance(IDLE)
        r.tick()
        r.flush()

        assert send.call_count == 1
        (w,) = bodies(send)[0]["windows"]
        assert w == {
            "windowId": "00000000-0000-4000-8000-000000000001",
            "agentId": "agent-1",
            "reason": "network_error",
            "source": "fail_open_network_error",
            "count": 2,
            "firstAt": "2026-09-21T12:00:00.000Z",
            "lastAt": "2026-09-21T12:00:05.000Z",
            "message": "last",
            "status": 503,
        }

    def test_keeps_different_agents_and_different_reasons_apart(self, clock):
        send = ok()
        r = make(clock, send)

        r.record(**bypass())
        r.record(**bypass(agent_id="agent-2"))
        r.record(**bypass(reason="plan_limit_exceeded", source="fail_open_plan_limit"))
        clock.advance(IDLE)
        r.tick()
        r.flush()

        assert len(bodies(send)[0]["windows"]) == 3

    def test_every_window_has_its_own_id_the_idempotency_key(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())
        r.record(**bypass(agent_id="agent-2"))
        clock.advance(IDLE)
        r.tick()
        r.flush()

        ids = [w["windowId"] for w in bodies(send)[0]["windows"]]
        assert len(set(ids)) == 2

    def test_default_window_ids_are_uuids(self, clock):
        send = ok()
        r = BypassWindowReporter(send, SDK, now=clock)
        r.record(**bypass())
        r.recover("agent-1")
        r.flush()

        assert UUID_RE.match(bodies(send)[0]["windows"][0]["windowId"])


class TestWhenAWindowCloses:
    def test_does_not_send_an_open_window(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())
        clock.advance(10)  # well inside the idle gap
        r.tick()
        r.flush()

        send.assert_not_called()

    def test_closes_after_an_idle_gap(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())
        clock.advance(IDLE)
        r.tick()
        r.flush()

        assert send.call_count == 1

    def test_a_bypass_after_the_idle_gap_opens_a_new_window(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())
        clock.advance(IDLE + 1)
        r.record(**bypass())  # no tick in between: record itself must notice
        clock.advance(IDLE)
        r.tick()
        r.flush()

        ws = bodies(send)[0]["windows"]
        assert [w["count"] for w in ws] == [1, 1]

    def test_caps_a_window_at_the_max_duration(self, clock):
        send = ok()
        r = make(clock, send)
        # A bypass every 30 s for 40 minutes: never idle, so only the duration
        # cap can close it.
        for _ in range(81):
            r.record(**bypass())
            clock.advance(30)
        clock.advance(IDLE)
        r.tick()
        r.flush()

        ws = [w for b in bodies(send) for w in b["windows"]]
        assert len(ws) >= 3
        for w in ws:
            first = datetime.fromisoformat(w["firstAt"].replace("Z", "+00:00"))
            last = datetime.fromisoformat(w["lastAt"].replace("Z", "+00:00"))
            assert (last - first).total_seconds() <= MAX
        assert sum(w["count"] for w in ws) == 81

    def test_recovery_closes_only_that_agents_windows(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())
        r.record(**bypass(agent_id="agent-2"))

        r.recover("agent-1")
        r.flush()

        assert [w["agentId"] for w in bodies(send)[0]["windows"]] == ["agent-1"]


class TestSending:
    def test_sends_the_sdk_identity_and_the_client_clock_at_send_time(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")
        clock.advance(1.234)
        r.flush()

        body = bodies(send)[0]
        assert body["sdk"] == SDK
        assert body["sentAt"] == "2026-09-21T12:00:01.234Z"

    def test_does_not_send_again_once_acknowledged(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")
        r.flush()
        r.flush()

        assert send.call_count == 1

    def test_sends_at_most_100_windows_per_request(self, clock):
        send = ok()
        r = make(clock, send)
        for i in range(230):
            r.record(**bypass(agent_id=f"agent-{i}"))
        clock.advance(IDLE)
        r.tick()
        r.flush(force=True)

        assert [len(b["windows"]) for b in bodies(send)] == [100, 100, 30]

    def test_truncates_fields_to_what_the_server_accepts(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass(message="x" * 2_000, agent_id="a" * 400, source="s" * 200))
        r.recover("a" * 256)
        r.flush()

        w = bodies(send)[0]["windows"][0]
        assert len(w["message"]) == 500
        assert len(w["agentId"]) == 256
        assert len(w["source"]) == 64

    def test_truncates_by_utf16_units_because_the_server_counts_those(self, clock):
        # 300 emoji are 300 Python code points but 600 UTF-16 units: a 500-code-
        # point cut would still be over the server's `max(500)` and get the whole
        # report rejected.
        send = ok()
        r = make(clock, send)
        r.record(**bypass(message="\U0001f600" * 300))
        r.recover("agent-1")
        r.flush()

        message = bodies(send)[0]["windows"][0]["message"]
        assert len(message.encode("utf-16-le")) // 2 == 500
        assert message == "\U0001f600" * 250

    def test_never_splits_a_surrogate_pair(self):
        assert _truncate("a\U0001f600", 2) == "a"
        assert _truncate("a\U0001f600", 3) == "a\U0001f600"

    def test_an_empty_agent_id_is_reported_not_rejected(self, clock):
        # The server requires agentId min length 1; one empty id would get every
        # other window in the batch rejected with it.
        send = ok()
        r = make(clock, send)
        r.record(**bypass(agent_id=""))
        r.recover("")
        r.flush()

        assert bodies(send)[0]["windows"][0]["agentId"] == "unknown"

    def test_never_runs_two_sends_at_once(self, clock):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def slow_send(body):
            calls.append(body)
            entered.set()
            release.wait(5)
            return 200

        r = make(clock, slow_send)
        r.record(**bypass())
        r.recover("agent-1")

        first = threading.Thread(target=r.flush)
        first.start()
        assert entered.wait(5)
        r.flush()  # second caller while the first is mid-send
        release.set()
        first.join(5)

        assert len(calls) == 1

    def test_is_safe_to_record_from_many_threads(self, clock):
        send = ok()
        r = make(clock, send)

        def worker():
            for _ in range(500):
                r.record(**bypass())

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        r.recover("agent-1")
        r.flush()

        assert sum(w["count"] for w in bodies(send)[0]["windows"]) == 4_000


class TestFailureKeepsTheEvidence:
    def test_keeps_the_window_and_retries_with_the_same_id_when_the_network_fails(self, clock):
        send = MagicMock(side_effect=[ConnectionError("ECONNREFUSED"), 200])
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")

        r.flush()
        r.flush(force=True)

        assert send.call_count == 2
        first, second = bodies(send)
        assert second["windows"][0]["windowId"] == first["windows"][0]["windowId"]
        r.flush(force=True)  # acknowledged now: nothing left
        assert send.call_count == 2

    @pytest.mark.parametrize("status", [500, 502, 503, 429, 408, 401, 403, 404])
    def test_keeps_the_windows_on_retryable_statuses(self, clock, status):
        send = MagicMock(return_value=status)
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")

        r.flush()
        r.flush(force=True)

        assert send.call_count == 2

    @pytest.mark.parametrize("status", [400, 413, 422])
    def test_discards_a_body_the_server_will_never_accept(self, clock, status, caplog):
        send = MagicMock(return_value=status)
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")

        with caplog.at_level("WARNING", logger="Execlave"):
            r.flush()
            r.flush(force=True)

        assert send.call_count == 1
        assert str(status) in caplog.text

    def test_backs_off_instead_of_hammering_a_failing_endpoint(self, clock):
        send = MagicMock(side_effect=ConnectionError("down"))
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")

        r.flush()  # attempt 1 fails
        clock.advance(1)
        r.flush()  # inside the backoff: must not send
        assert send.call_count == 1

        clock.advance(5 * 60)  # past any backoff
        r.flush()
        assert send.call_count == 2

    def test_a_forced_flush_ignores_backoff(self, clock):
        send = MagicMock(side_effect=ConnectionError("down"))
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")
        r.flush()

        r.flush(force=True)
        assert send.call_count == 2

    def test_never_raises_even_if_the_transport_does(self, clock):
        send = MagicMock(side_effect=RuntimeError("boom"))
        r = make(clock, send)

        r.record(**bypass())
        r.recover("agent-1")
        r.tick()
        r.flush()
        r.drain(1)

    def test_a_non_numeric_status_is_treated_as_a_failure_not_a_crash(self, clock):
        send = MagicMock(return_value=None)
        r = make(clock, send)
        r.record(**bypass())
        r.recover("agent-1")

        r.flush()

        assert r.has_pending


class TestOverflowStatesItsOwnLoss:
    def test_drops_the_oldest_past_the_cap_and_reports_what_was_dropped(self, clock):
        send = ok()
        r = make(clock, send, max_buffered_windows=3)
        for i in range(1, 6):
            r.record(**bypass(agent_id=f"agent-{i}"))
            r.record(**bypass(agent_id=f"agent-{i}"))  # count 2 each
            r.recover(f"agent-{i}")

        r.flush(force=True)

        body = bodies(send)[0]
        assert [w["agentId"] for w in body["windows"]] == ["agent-3", "agent-4", "agent-5"]
        # agent-1 and agent-2 were lost: 2 windows, 4 bypasses. The trail must say so.
        assert body["dropped"]["windows"] == 2
        assert body["dropped"]["bypasses"] == 4
        assert UUID_RE.match(body["dropped"]["dropId"])
        assert body["dropped"]["since"] == "2026-09-21T12:00:00.000Z"

    def test_keeps_the_same_drop_id_across_retries_and_clears_it_once_acknowledged(self, clock):
        send = MagicMock(side_effect=[ConnectionError("down"), 200, 200])
        r = make(clock, send, max_buffered_windows=1)
        for i in range(1, 4):
            r.record(**bypass(agent_id=f"agent-{i}"))
            r.recover(f"agent-{i}")

        r.flush(force=True)
        r.flush(force=True)
        first, second = bodies(send)
        assert second["dropped"]["dropId"] == first["dropped"]["dropId"]

        r.flush(force=True)
        # Acknowledged: no further request, and no stale dropped block.
        assert send.call_count == 2

    def test_sends_a_loss_only_report_when_nothing_buffered_survived(self, clock):
        send = ok()
        r = make(clock, send, max_buffered_windows=0)
        r.record(**bypass())
        r.recover("agent-1")
        r.flush(force=True)

        body = bodies(send)[0]
        assert body["windows"] == []
        assert body["dropped"]["windows"] == 1
        assert body["dropped"]["bypasses"] == 1

    def test_omits_since_rather_than_sending_null_when_unknown(self, clock):
        # `since` is optional on the wire; the server's strict schema rejects null.
        send = ok()
        r = make(clock, send, max_buffered_windows=0)
        r.record(**bypass())
        r.recover("agent-1")
        r._live_drop["since"] = None
        r.flush(force=True)

        assert "since" not in bodies(send)[0]["dropped"]


class TestShutdownDrain:
    def test_closes_open_windows_and_sends_them(self, clock):
        send = ok()
        r = make(clock, send)
        r.record(**bypass())  # still open, nowhere near idle
        r.drain(1)

        assert send.call_count == 1
        assert bodies(send)[0]["windows"][0]["count"] == 1

    def test_does_nothing_when_there_is_nothing_to_send(self, clock):
        send = ok()
        r = make(clock, send)
        r.drain(1)

        send.assert_not_called()

    def test_gives_up_at_the_deadline_instead_of_hanging_on_a_dead_endpoint(self, clock):
        release = threading.Event()

        def hung(_body):
            release.wait(10)
            return 200

        r = make(clock, hung)
        r.record(**bypass())

        started = time.monotonic()
        r.drain(0.1)
        elapsed = time.monotonic() - started
        release.set()

        assert elapsed < 2

    def test_stops_after_a_failed_attempt_rather_than_looping_until_the_deadline(self, clock):
        send = MagicMock(side_effect=ConnectionError("down"))
        r = make(clock, send)
        r.record(**bypass())

        r.drain(5)

        assert send.call_count == 1

    def test_waits_for_an_in_flight_send_then_delivers_what_is_left(self, clock):
        gate = threading.Event()
        started = threading.Event()
        sent = []

        def gated(body):
            sent.append(body)
            started.set()
            gate.wait(5)
            return 200

        r = make(clock, gated)
        r.record(**bypass())
        r.recover("agent-1")
        timer_thread = threading.Thread(target=r.flush)
        timer_thread.start()
        assert started.wait(5)

        r.record(**bypass(agent_id="agent-2"))  # arrives mid-send
        threading.Timer(0.05, gate.set).start()
        r.drain(3)
        timer_thread.join(5)

        delivered = [w["agentId"] for b in sent for w in b["windows"]]
        assert sorted(delivered) == ["agent-1", "agent-2"]


class TestOnlySendsTextTheServerCanStore:
    """Postgres text/jsonb reject NUL and a lone UTF-16 surrogate. The server's
    audit write would throw, which reads as a retryable 5xx -- so one such
    character would make that window fail on every retry and never be recorded."""

    LONE = chr(0xD83D)
    NUL = chr(0)
    REPLACEMENT = chr(0xFFFD)
    EMOJI = chr(0x1F600)

    def _sent(self, clock, **over):
        send = ok()
        r = make(clock, send)
        r.record(**bypass(**over))
        r.recover(over.get("agent_id", "agent-1"))
        r.flush(force=True)
        return bodies(send)[0]["windows"][0]

    def test_replaces_nul_and_stray_surrogates(self, clock):
        w = self._sent(
            clock,
            message=f"a{self.NUL}b{self.LONE}c",
            source=f"s{self.NUL}",
        )

        assert w["message"] == f"a{self.REPLACEMENT}b{self.REPLACEMENT}c"
        assert w["source"] == f"s{self.REPLACEMENT}"

    def test_keeps_a_whole_emoji(self, clock):
        w = self._sent(clock, message="a" + self.EMOJI + "b")

        assert w["message"] == "a" + self.EMOJI + "b"

    def test_the_body_serialises_without_surrogate_escapes(self, clock):
        # json.dumps would emit a lone surrogate as an escape, which Postgres
        # jsonb refuses -- this is the exact bytes the server would receive.
        import json

        w = self._sent(clock, message=f"x{self.LONE}", agent_id=f"a{self.LONE}")

        wire = json.dumps(w)
        assert "ud83d" not in wire.lower()

    def test_a_nul_only_agent_id_is_still_a_valid_non_empty_id(self, clock):
        w = self._sent(clock, agent_id=self.NUL)

        assert w["agentId"] == self.REPLACEMENT
