"""Workload identity federation in the Python SDK: the credential source
(cache, single exchange under concurrency, refresh, validation) and the client
wiring (config guards, register_agent without POST /agents, bearer and
signing, refusal vs outage, traces, bypass reports, WebSocket handshake).

A real requests.Session runs with a routing adapter mounted, so the session's
auth hook (where the credential is attached) is exercised, not mocked away.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import pytest
import requests

from execlave import client as client_module
from execlave.client import Execlave
from execlave.credential_provider import (
    CredentialExchangeUnavailable,
    FederatedCredentialSource,
)
from execlave.errors import EnforcementUnavailableError, ExeclaveAuthError, ExeclaveError

AGENT_UUID = "11111111-2222-4333-8444-555555555555"
OTHER_UUID = "99999999-2222-4333-8444-555555555555"
# Shaped like a compact JWS; never a real token.
IDP_TOKEN = "aaa.bbb.ccc"
CREDENTIAL_TTL = timedelta(minutes=15)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def credential(n: int = 1, ttl: timedelta = CREDENTIAL_TTL, agent_id: str = AGENT_UUID) -> dict:
    return {
        "credential": f"exe_agt_test{n}",
        "expiresAt": _iso(datetime.now(timezone.utc) + ttl),
        "agentId": agent_id,
    }


# ---------------------------------------------------------------------------
# FederatedCredentialSource
# ---------------------------------------------------------------------------


class TestFederatedCredentialSource:
    def test_exchanges_once_for_concurrent_callers_then_serves_the_cache(self):
        provider = MagicMock(return_value=IDP_TOKEN)
        started = threading.Event()

        def slow_exchange(token: str) -> dict:
            started.wait(0.2)
            return credential(1)

        exchange = MagicMock(side_effect=slow_exchange)
        source = FederatedCredentialSource(provider, exchange)

        results: list[dict] = []
        threads = [threading.Thread(target=lambda: results.append(source.get())) for _ in range(5)]
        for t in threads:
            t.start()
        started.set()
        for t in threads:
            t.join()
        source.get()

        assert len({r["credential"] for r in results}) == 1
        assert provider.call_count == 1
        exchange.assert_called_once_with(IDP_TOKEN)
        assert source.bound_agent_id == AGENT_UUID

    def test_exchanges_again_once_less_than_the_refresh_margin_is_left(self):
        now = [1_000_000.0]
        base = datetime.fromtimestamp(now[0], tz=timezone.utc)
        exchange = MagicMock(
            side_effect=[
                {"credential": "exe_agt_a", "expiresAt": _iso(base + timedelta(seconds=120)), "agentId": AGENT_UUID},
                {"credential": "exe_agt_b", "expiresAt": _iso(base + timedelta(seconds=900)), "agentId": AGENT_UUID},
            ]
        )
        source = FederatedCredentialSource(lambda: IDP_TOKEN, exchange, clock=lambda: now[0])

        assert source.get()["credential"] == "exe_agt_a"
        now[0] += 59  # 61 s left: still cached
        assert source.get()["credential"] == "exe_agt_a"
        now[0] += 2  # 59 s left: refresh
        assert source.get()["credential"] == "exe_agt_b"
        assert exchange.call_count == 2

    def test_surfaces_a_failed_exchange_and_tries_again_on_the_next_call(self):
        exchange = MagicMock(side_effect=[requests.ConnectionError("ECONNRESET"), credential(2)])
        source = FederatedCredentialSource(lambda: IDP_TOKEN, exchange)

        with pytest.raises(requests.ConnectionError):
            source.get()
        assert source.get()["credential"] == "exe_agt_test2"

    @pytest.mark.parametrize(
        "answer",
        [
            {**credential(), "credential": "exe_prod_key"},
            {**credential(), "agentId": ""},
            {**credential(), "expiresAt": "soon"},
            None,
        ],
        ids=["not exe_agt_", "no agentId", "unparseable expiry", "no body"],
    )
    def test_refuses_an_unexpected_exchange_answer(self, answer):
        source = FederatedCredentialSource(lambda: IDP_TOKEN, MagicMock(return_value=answer))
        with pytest.raises(ExeclaveError, match="unexpected response"):
            source.get()

    @pytest.mark.parametrize("token", ["", None])
    def test_refuses_an_empty_token_without_calling_the_exchange(self, token):
        exchange = MagicMock()
        source = FederatedCredentialSource(lambda: token, exchange)
        with pytest.raises(ExeclaveError, match="credential_provider returned no token"):
            source.get()
        exchange.assert_not_called()

    def test_stops_when_the_identity_is_rebound_to_a_different_agent(self):
        now = [1_000_000.0]
        base = datetime.fromtimestamp(now[0], tz=timezone.utc)
        exchange = MagicMock(
            side_effect=[
                {"credential": "exe_agt_a", "expiresAt": _iso(base + timedelta(seconds=120)), "agentId": AGENT_UUID},
                {"credential": "exe_agt_b", "expiresAt": _iso(base + timedelta(seconds=900)), "agentId": OTHER_UUID},
            ]
        )
        source = FederatedCredentialSource(lambda: IDP_TOKEN, exchange, clock=lambda: now[0])

        source.get()
        now[0] += 100
        with pytest.raises(ExeclaveAuthError):
            source.get()
        assert source.bound_agent_id == AGENT_UUID


# ---------------------------------------------------------------------------
# Client wiring
# ---------------------------------------------------------------------------

Handler = Callable[[requests.PreparedRequest], "tuple[int, Any]"]


class _Router(requests.adapters.BaseAdapter):
    """Answers requests by path suffix and records them (after session auth ran)."""

    def __init__(self, handlers: dict[str, Handler]):
        super().__init__()
        self.handlers = handlers
        self.calls: list[requests.PreparedRequest] = []

    def send(self, request, **kwargs):  # noqa: ANN001 - requests adapter signature
        self.calls.append(request)
        path = urlsplit(request.url).path
        for suffix, handler in self.handlers.items():
            if path.endswith(suffix):
                status, body = handler(request)
                resp = requests.Response()
                resp.status_code = status
                resp._content = json.dumps(body).encode("utf-8")
                resp.headers["Content-Type"] = "application/json"
                resp.url = request.url
                resp.request = request
                return resp
        raise requests.ConnectionError(f"unexpected request {request.method} {path}")

    def close(self) -> None:
        pass

    def find(self, suffix: str) -> requests.PreparedRequest:
        return next(c for c in self.calls if urlsplit(c.url).path.endswith(suffix))


def exchange_ok(_req):
    return 201, {"data": credential(1)}


def body_of(req: requests.PreparedRequest) -> Any:
    raw = req.body.decode("utf-8") if isinstance(req.body, bytes) else req.body
    return json.loads(raw)


def federated_client(handlers: dict[str, Handler], **overrides) -> tuple[Execlave, _Router]:
    options: dict[str, Any] = dict(
        credential_provider=lambda: IDP_TOKEN,
        base_url="http://localhost:4000",
        async_mode=False,
        enable_control_channel=False,
        enable_injection_scan=False,
    )
    options.update(overrides)
    exe = Execlave(**options)
    router = _Router(handlers)
    exe._session.mount("http://", router)
    return exe, router


class TestConfiguration:
    def test_refuses_an_api_key_together_with_a_credential_provider(self):
        with pytest.raises(ValueError, match="not both"):
            Execlave(api_key="exe_prod_x", credential_provider=lambda: IDP_TOKEN)

    def test_refuses_otlp_mode(self):
        with pytest.raises(ValueError, match="otlp"):
            Execlave(
                credential_provider=lambda: IDP_TOKEN,
                mode="otlp",
                otlp_endpoint="http://localhost:4317",
            )

    def test_ignores_execlave_api_key_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("EXECLAVE_API_KEY", "exe_prod_from_env")
        exe, router = federated_client(
            {
                "/agent-credentials/exchange": exchange_ok,
                "/policies/enforce": lambda _r: (200, {"allowed": True}),
            }
        )
        exe.enforce_policy(AGENT_UUID, "hi")
        assert all("exe_prod_from_env" not in str(c.headers) for c in router.calls)


class TestRegisterAgent:
    def test_names_the_bound_agent_without_creating_one(self):
        exe, router = federated_client({"/agent-credentials/exchange": exchange_ok})

        agent = exe.register_agent(agent_id="support-bot", name="Support bot")

        assert agent.id == AGENT_UUID
        assert agent.agent_id == "support-bot"
        assert len(router.calls) == 1
        assert body_of(router.calls[0]) == {"token": IDP_TOKEN}
        # The exchange is unauthenticated: the token is the proof.
        assert "Authorization" not in router.calls[0].headers
        assert exe.register_agent(agent_id="support-bot", name="Support bot") is agent

    def test_reports_an_unavailable_exchange_as_a_network_error(self):
        exe, _ = federated_client({"/agent-credentials/exchange": lambda _r: (503, {})})
        with pytest.raises(ExeclaveError, match="Network error") as raised:
            exe.register_agent(agent_id="support-bot", name="Support bot")
        assert not isinstance(raised.value, ExeclaveAuthError)
        assert not isinstance(raised.value, requests.RequestException)

    def test_refuses_a_second_agent_name(self):
        exe, _ = federated_client({"/agent-credentials/exchange": exchange_ok})
        exe.register_agent(agent_id="support-bot", name="Support bot")
        with pytest.raises(ExeclaveError, match="already registered as 'support-bot'"):
            exe.register_agent(agent_id="billing-bot", name="Billing")


class TestEnforcePolicy:
    def test_sends_the_credential_as_bearer_the_agent_uuid_and_signs_with_the_credential(self):
        exe, router = federated_client(
            {
                "/agent-credentials/exchange": exchange_ok,
                "/policies/enforce": lambda _r: (200, {"allowed": True}),
            },
            sign_requests=True,
        )
        exe.register_agent(agent_id="support-bot", name="Support bot")

        assert exe.enforce_policy("support-bot", "hi")["allowed"] is True

        enforce = router.find("/policies/enforce")
        assert enforce.headers["Authorization"] == "Bearer exe_agt_test1"
        assert body_of(enforce)["agentId"] == AGENT_UUID
        ts = enforce.headers["X-Execlave-Timestamp"]
        raw = enforce.body if isinstance(enforce.body, bytes) else enforce.body.encode("utf-8")
        expected = "sha256=" + hmac.new(
            b"exe_agt_test1", ts.encode() + b"." + raw, hashlib.sha256
        ).hexdigest()
        assert enforce.headers["X-Execlave-Signature"] == expected

    def test_does_not_fail_open_when_the_exchange_refuses_the_token(self):
        bypassed = MagicMock()
        exe, _ = federated_client(
            {
                "/agent-credentials/exchange": lambda _r: (
                    401,
                    {"error": {"message": "Federation token rejected"}},
                )
            },
            on_enforcement_bypassed=bypassed,
        )
        with pytest.raises(ExeclaveAuthError, match="refused"):
            exe.enforce_policy(AGENT_UUID, "hi")
        bypassed.assert_not_called()

    def test_treats_an_unavailable_exchange_as_an_outage_under_fail_open(self):
        bypassed = MagicMock()
        exe, _ = federated_client(
            {"/agent-credentials/exchange": lambda _r: (503, {})},
            on_enforcement_bypassed=bypassed,
            report_bypasses_to_platform=False,
        )
        result = exe.enforce_policy(AGENT_UUID, "hi")
        assert result == {"allowed": True, "source": "fail_open_network_error"}
        bypassed.assert_called_once()

    def test_treats_a_failing_identity_provider_as_an_outage(self):
        def provider() -> str:
            raise RuntimeError("IMDS endpoint unreachable")

        bypassed = MagicMock()
        exe, router = federated_client(
            {}, credential_provider=provider, on_enforcement_bypassed=bypassed,
            report_bypasses_to_platform=False,
        )
        assert exe.enforce_policy(AGENT_UUID, "hi")["source"] == "fail_open_network_error"
        bypassed.assert_called_once()
        assert router.calls == []

    def test_blocks_on_an_unavailable_exchange_under_fail_closed(self):
        exe, _ = federated_client(
            {"/agent-credentials/exchange": lambda _r: (429, {})},
            enforcement_on_outage="fail_closed",
        )
        with pytest.raises(EnforcementUnavailableError):
            exe.enforce_policy(AGENT_UUID, "hi")


def test_unavailable_exchange_is_a_connection_error():
    assert issubclass(CredentialExchangeUnavailable, requests.ConnectionError)


def test_sends_traces_with_the_agent_uuid():
    exe, router = federated_client(
        {
            "/agent-credentials/exchange": exchange_ok,
            "/traces/ingest": lambda _r: (201, {"data": {"accepted": 1}}),
        }
    )
    exe.register_agent(agent_id="support-bot", name="Support bot")

    exe.start_trace(agent_id="support-bot").set_input("q").set_output("a").finish()
    exe.flush()

    ingest = router.find("/traces/ingest")
    assert ingest.headers["Authorization"] == "Bearer exe_agt_test1"
    trace = body_of(ingest)["traces"][0]
    assert trace["agentId"] == "support-bot"
    assert trace["agentUuid"] == AGENT_UUID


def test_records_bypass_windows_under_the_agent_uuid():
    exe, _ = federated_client(
        {
            "/agent-credentials/exchange": exchange_ok,
            "/policies/enforce": lambda _r: (502, {}),
        }
    )
    exe.register_agent(agent_id="support-bot", name="Support bot")
    with patch.object(exe._bypass_reporter, "record") as record, patch.object(
        exe, "_ensure_bypass_thread"
    ):
        exe.enforce_policy("support-bot", "hi")

    assert record.call_args.kwargs["agent_id"] == AGENT_UUID


def test_authorize_sends_the_caller_as_its_uuid():
    exe, router = federated_client(
        {
            "/agent-credentials/exchange": exchange_ok,
            "/agents/authorize": lambda _r: (200, {"allowed": True}),
        }
    )
    exe.register_agent(agent_id="support-bot", name="Support bot")
    exe.authorize_agent_call("support-bot", "billing-bot", "delegate")
    assert body_of(router.find("/agents/authorize"))["callerAgentId"] == AGENT_UUID


def test_pings_health_without_exchanging_a_credential():
    exe, router = federated_client({"/health": lambda _r: (200, {"status": "ok"})})
    assert exe.ping() is True
    assert len(router.calls) == 1
    assert "Authorization" not in router.calls[0].headers


# ---------------------------------------------------------------------------
# WebSocket control channel
# ---------------------------------------------------------------------------


class _FakeSio:
    """Stands in for socketio.Client: records handlers and connect calls."""

    instances: list["_FakeSio"] = []

    class reason:  # noqa: N801 - mirrors python-socketio's attribute
        SERVER_DISCONNECT = "server disconnect"
        TRANSPORT_ERROR = "transport error"

    def __init__(self, **_kwargs):
        self.handlers: dict[str, Callable[..., Any]] = {}
        self.connect_calls: list[dict] = []
        self.connected_event = threading.Event()
        _FakeSio.instances.append(self)

    def on(self, event, namespace=None):
        def register(fn):
            self.handlers[event] = fn
            return fn

        return register

    def event(self, fn=None, namespace=None):
        if fn is None:
            return lambda f: self.event(f)
        self.handlers[fn.__name__] = fn
        return fn

    def connect(self, url, **kwargs):
        self.connect_calls.append(kwargs)
        self.connected_event.set()

    def disconnect(self):
        pass


@pytest.fixture()
def fake_socketio(monkeypatch):
    _FakeSio.instances.clear()
    monkeypatch.setattr(client_module, "_socketio_mod", MagicMock(Client=_FakeSio))
    monkeypatch.setattr(client_module, "_HAS_SOCKETIO", True)
    # Keep the status-poll and heartbeat threads idle for the test.
    monkeypatch.setattr(client_module.Execlave, "_status_poll_loop", lambda self: None)
    monkeypatch.setattr(client_module.Execlave, "_heartbeat_loop", lambda self: None)
    return _FakeSio


def _wait_for_connects(sio: _FakeSio, count: int) -> None:
    for _ in range(100):
        if len(sio.connect_calls) >= count:
            return
        threading.Event().wait(0.01)
    raise AssertionError(f"expected {count} connect calls, saw {len(sio.connect_calls)}")


class TestWebSocket:
    def test_authenticates_each_handshake_with_a_current_credential(self, fake_socketio):
        exe, _ = federated_client(
            {"/agent-credentials/exchange": exchange_ok}, enable_control_channel=True
        )
        # The session (and its router) only exists after construction, so the
        # handshake callable is invoked here, as python-socketio would on connect.
        sio = fake_socketio.instances[0]
        _wait_for_connects(sio, 1)
        auth = sio.connect_calls[0]["auth"]
        assert callable(auth)
        assert auth() == {"agentCredential": "exe_agt_test1"}
        exe.shutdown()

    def test_reconnects_after_a_server_side_disconnect_only(self, fake_socketio):
        exe, _ = federated_client(
            {"/agent-credentials/exchange": exchange_ok}, enable_control_channel=True
        )
        sio = fake_socketio.instances[0]
        _wait_for_connects(sio, 1)

        sio.handlers["disconnect"]("transport error")  # the client retries these itself
        threading.Event().wait(0.05)
        assert len(sio.connect_calls) == 1

        sio.handlers["disconnect"]("server disconnect")
        _wait_for_connects(sio, 2)
        threading.Event().wait(0.05)
        assert len(sio.connect_calls) == 2
        exe.shutdown()

    def test_pause_event_naming_the_agent_uuid_pauses_the_client(self, fake_socketio):
        exe, _ = federated_client(
            {"/agent-credentials/exchange": exchange_ok}, enable_control_channel=True
        )
        agent = exe.register_agent(agent_id="support-bot", name="Support bot")

        fake_socketio.instances[0].handlers["agent.status_updated"](
            {"agentId": AGENT_UUID, "status": "paused"}
        )

        assert agent.status == "paused"
        assert exe._state == client_module._STATE_PAUSED
        exe.shutdown()
