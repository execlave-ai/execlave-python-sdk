"""Tests for optional HMAC request signing (sign_requests)."""

import hashlib
import hmac

import requests

from execlave.client import _HmacSigner, Execlave


def _make_prepared(body):
    req = requests.Request(
        method="POST",
        url="http://localhost:4000/api/v1/policies/enforce",
        json=body,
    )
    return req.prepare()


def test_signer_adds_correct_hmac_headers():
    key = "exe_test_key"
    signer = _HmacSigner(key)
    prepared = _make_prepared({"agentId": "a", "input": "hi"})

    signer(prepared)

    ts = prepared.headers["X-Execlave-Timestamp"]
    sig = prepared.headers["X-Execlave-Signature"]
    assert ts.isdigit()

    body = prepared.body
    if isinstance(body, str):
        body = body.encode("utf-8")
    expected = (
        "sha256="
        + hmac.new(key.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    )
    assert sig == expected


def test_signer_handles_empty_body():
    signer = _HmacSigner("k")
    req = requests.Request(method="GET", url="http://localhost:4000/health").prepare()
    signer(req)
    assert "X-Execlave-Signature" in req.headers
    assert req.headers["X-Execlave-Timestamp"].isdigit()


def test_client_enables_session_auth_when_sign_requests_true():
    exe = Execlave(api_key="exe_test_key", sign_requests=True, enable_control_channel=False)
    try:
        assert isinstance(exe._session.auth, _HmacSigner)
    finally:
        # best-effort cleanup of background timers if any
        try:
            exe.shutdown()
        except Exception:
            pass


def test_client_no_session_auth_by_default():
    exe = Execlave(api_key="exe_test_key", enable_control_channel=False)
    try:
        assert exe._session.auth is None
    finally:
        try:
            exe.shutdown()
        except Exception:
            pass
