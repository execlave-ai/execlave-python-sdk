"""Bypass reporting to the platform, end to end through a real client.

``on_enforcement_bypassed`` only tells the caller's own process. Until bypasses
also reached the server, the platform's audit trail could not tell "all calls
governed" from "the SDK lost contact for two hours" -- an absent record read as
fully governed. These pin what a real client sends, when, and that none of it can
touch the enforcement path. Mirrors
sdk-js/src/__tests__/enforcementBypassReporting.test.ts.
"""

import threading
import time
from unittest.mock import patch

import pytest
import requests

import execlave
import execlave.client as client_module
from execlave.client import Execlave
from execlave.errors import EnforcementUnavailableError

REPORT_PATH = "/sdk/bypass-windows"


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.ok = status_code < 400

    def json(self):
        return self._payload


def _quota_body():
    return {
        "error": {
            "resource": "maxTracesPerMonth",
            "current": 10,
            "max": 10,
            "message": "Plan limit reached",
        }
    }


class Harness:
    """Routes the session's requests: enforcement calls and report calls are
    handled by separately settable functions, and every report is recorded."""

    def __init__(self):
        self.clients = []
        self.report_calls = []
        self._lock = threading.Lock()
        self.enforce = self._down
        self.report = lambda kwargs: _Resp(200, {"accepted": 1, "duplicates": 0})

    @staticmethod
    def _down(_kwargs):
        raise requests.ConnectionError("socket hang up")

    def route(self, method, url, **kwargs):
        if url.endswith(REPORT_PATH):
            with self._lock:
                self.report_calls.append({"method": method, "url": url, **kwargs})
            return self.report(kwargs)
        return self.enforce(kwargs)

    def client(self, **over):
        kwargs = dict(
            api_key="exe_prod_test",
            base_url="https://api.test",
            enable_control_channel=False,
            async_mode=False,
        )
        kwargs.update(over)
        exe = Execlave(**kwargs)
        patcher = patch.object(exe._session, "request", side_effect=self.route)
        patcher.start()
        self.clients.append((exe, patcher))
        return exe

    @property
    def bodies(self):
        return [c["json"] for c in self.report_calls]

    @property
    def windows(self):
        return [w for b in self.bodies for w in b["windows"]]


@pytest.fixture()
def harness():
    h = Harness()
    yield h
    for exe, patcher in h.clients:
        exe.shutdown()  # while the patch is still active: nothing reaches a real network
        patcher.stop()


def enforce(exe, text="x", agent_id="agent-1"):
    return exe.enforce_policy(agent_id=agent_id, input=text)


class TestReportsToThePlatform:
    def test_reports_a_network_outage_as_one_coalesced_window_on_shutdown(self, harness):
        exe = harness.client()
        enforce(exe, "one")
        enforce(exe, "two")
        exe.shutdown()

        (w,) = harness.windows
        assert w["agentId"] == "agent-1"
        assert w["reason"] == "network_error"
        assert w["source"] == "fail_open_network_error"
        assert w["count"] == 2
        assert len(w["windowId"]) == 36

    def test_posts_to_the_versioned_endpoint_and_identifies_the_sdk(self, harness):
        exe = harness.client()
        enforce(exe)
        exe.shutdown()

        (call,) = harness.report_calls
        assert call["method"] == "POST"
        assert call["url"] == "https://api.test/api/v1/sdk/bypass-windows"
        assert call["timeout"] > 0
        assert call["json"]["sdk"] == {
            "name": "execlave-sdk",
            "language": "python",
            "version": execlave.__version__,
        }

    def test_reports_a_plan_limit_bypass(self, harness):
        harness.enforce = lambda _kw: _Resp(402, _quota_body())
        exe = harness.client()
        enforce(exe)
        exe.shutdown()

        (w,) = harness.windows
        assert w["reason"] == "plan_limit_exceeded"
        assert w["source"] == "fail_open_plan_limit"
        assert w["status"] == 402

    def test_reports_a_5xx_bypass_with_its_status(self, harness):
        harness.enforce = lambda _kw: _Resp(503)
        exe = harness.client()
        enforce(exe)
        exe.shutdown()

        (w,) = harness.windows
        assert w["reason"] == "server_error"
        assert w["status"] == 503

    def test_reports_even_when_no_listener_is_wired(self, harness):
        exe = harness.client()  # no on_enforcement_bypassed
        enforce(exe)
        exe.shutdown()

        assert len(harness.windows) == 1

    def test_still_calls_the_local_listener_once_per_bypass(self, harness):
        events = []
        exe = harness.client(on_enforcement_bypassed=events.append)
        enforce(exe, "a")
        enforce(exe, "b")
        exe.shutdown()

        # The listener keeps its per-call granularity; the server gets the window.
        assert len(events) == 2
        assert harness.windows[0]["count"] == 2

    def test_a_throwing_listener_cannot_stop_the_report(self, harness):
        def boom(_event):
            raise RuntimeError("listener exploded")

        exe = harness.client(on_enforcement_bypassed=boom)
        enforce(exe)
        exe.shutdown()

        assert len(harness.windows) == 1

    def test_opting_out_keeps_only_the_local_callback(self, harness):
        events = []
        exe = harness.client(
            report_bypasses_to_platform=False, on_enforcement_bypassed=events.append
        )
        enforce(exe)
        exe.shutdown()

        assert len(events) == 1
        assert harness.report_calls == []
        assert exe._bypass_thread is None

    def test_reports_nothing_under_fail_closed_because_nothing_was_bypassed(self, harness):
        exe = harness.client(enforcement_on_outage="fail_closed")
        with pytest.raises(EnforcementUnavailableError):
            enforce(exe)
        exe.shutdown()

        assert harness.report_calls == []

    def test_reports_nothing_for_a_governed_allow(self, harness):
        harness.enforce = lambda _kw: _Resp(200, {"allowed": True})
        exe = harness.client()
        enforce(exe)
        exe.shutdown()

        assert harness.report_calls == []

    def test_a_client_that_never_bypasses_starts_no_thread(self, harness):
        harness.enforce = lambda _kw: _Resp(200, {"allowed": True})
        exe = harness.client()
        enforce(exe)

        assert exe._bypass_thread is None


class TestWindowLifecycle:
    def test_a_governed_decision_ends_the_window_so_two_outages_are_two_records(self, harness):
        exe = harness.client()
        enforce(exe, "down 1")
        enforce(exe, "down 2")

        # The platform is back and decides.
        harness.enforce = lambda _kw: _Resp(200, {"allowed": True})
        enforce(exe, "up")

        # ...and goes away again.
        harness.enforce = Harness._down
        enforce(exe, "down 3")
        exe.shutdown()

        assert [w["count"] for w in harness.windows] == [2, 1]

    def test_a_fail_open_plan_limit_402_keeps_its_window_open(self, harness):
        harness.enforce = lambda _kw: _Resp(402, _quota_body())
        exe = harness.client()
        for text in "abc":
            enforce(exe, text)
        exe.shutdown()

        # A 402 is itself a bypass, not a recovery.
        assert [w["count"] for w in harness.windows] == [3]

    def test_recovery_only_closes_that_agents_windows(self, harness):
        exe = harness.client()
        enforce(exe, "x", agent_id="agent-a")
        enforce(exe, "x", agent_id="agent-b")

        harness.enforce = lambda _kw: _Resp(200, {"allowed": True})
        enforce(exe, "up", agent_id="agent-a")

        exe._bypass_reporter.flush(force=True)
        assert [w["agentId"] for w in harness.windows] == ["agent-a"]


class TestNeverOnTheEnforcementPath:
    def test_a_hung_report_endpoint_does_not_slow_enforcement_and_shutdown_gives_up(
        self, harness, monkeypatch
    ):
        monkeypatch.setattr(client_module, "_BYPASS_SHUTDOWN_TIMEOUT_SECONDS", 0.2)
        release = threading.Event()

        def hung(_kwargs):
            release.wait(10)
            return _Resp(200)

        harness.report = hung
        exe = harness.client()

        started = time.monotonic()
        for i in range(5):
            enforce(exe, f"call {i}")
        assert time.monotonic() - started < 1
        assert harness.report_calls == []  # enforcement never touched the endpoint

        started = time.monotonic()
        exe.shutdown()
        elapsed = time.monotonic() - started
        release.set()

        assert elapsed < 3

    def test_the_background_thread_sends_an_idle_window_without_a_shutdown(
        self, harness, monkeypatch
    ):
        monkeypatch.setattr(client_module, "_BYPASS_REPORT_INTERVAL_SECONDS", 0.02)
        exe = harness.client()
        exe._bypass_reporter._idle = 0.05
        enforce(exe)

        deadline = time.monotonic() + 5
        while not harness.report_calls and time.monotonic() < deadline:
            time.sleep(0.01)

        assert len(harness.windows) == 1

    def test_keeps_an_unacknowledged_window_and_retries_it_with_the_same_id(
        self, harness, monkeypatch
    ):
        monkeypatch.setattr(client_module, "_BYPASS_REPORT_INTERVAL_SECONDS", 0.02)
        attempts = []

        def flaky(_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise requests.ConnectionError("down")
            return _Resp(200)

        harness.report = flaky
        exe = harness.client()
        exe._bypass_reporter._idle = 0.05
        enforce(exe)

        deadline = time.monotonic() + 5
        while not harness.report_calls and time.monotonic() < deadline:
            time.sleep(0.01)
        exe.shutdown()  # forced flush ignores the backoff

        assert len(harness.bodies) == 2
        first, second = harness.bodies
        assert second["windows"][0]["windowId"] == first["windows"][0]["windowId"]

    def test_a_failing_report_does_not_touch_the_circuit_breaker(self, harness):
        harness.report = lambda _kw: _Resp(500)
        exe = harness.client()
        enforce(exe)
        failures_before = exe._cb_failures
        exe.shutdown()

        assert exe._cb_failures == failures_before == 1

    def test_a_402_on_the_report_endpoint_does_not_arm_the_quota_cache(self, harness):
        harness.report = lambda _kw: _Resp(402, _quota_body())
        exe = harness.client()
        enforce(exe)
        exe.shutdown()

        assert exe._quota_exceeded is None
