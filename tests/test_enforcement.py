"""
Tests for enforcement and authorization methods on Execlave client.

Covers enforce_policy, authorize_agent_call, and discover_agents.
"""

import pytest
from unittest.mock import MagicMock

from execlave.client import Execlave
from execlave.errors import (
    ExeclaveError,
    ExeclaveAuthError,
    PolicyBlockedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    QuotaExceededError,
    PlanLimitExceededError,
)

from tests.helpers import make_mock_response


# =========================================================================
# PolicyBlockedError
# =========================================================================


class TestPolicyBlockedError:

    def test_inherits_from_base(self):
        assert issubclass(PolicyBlockedError, ExeclaveError)

    def test_carries_violations(self):
        violations = [{"policyType": "injection_scan", "message": "SQL detected"}]
        err = PolicyBlockedError(violations)
        assert err.violations == violations

    def test_message_format(self):
        violations = [
            {"policyType": "injection_scan", "message": "SQL detected"},
            {"policyType": "data_restriction", "message": "PII found"},
        ]
        err = PolicyBlockedError(violations)
        assert "[injection_scan] SQL detected" in str(err)
        assert "[data_restriction] PII found" in str(err)
        assert str(err).startswith("Execution blocked by policy:")

    def test_empty_violations(self):
        err = PolicyBlockedError([])
        assert "Execution blocked by policy:" in str(err)
        assert err.violations == []

    def test_caught_as_base(self):
        with pytest.raises(ExeclaveError):
            raise PolicyBlockedError([{"policyType": "test", "message": "fail"}])


# =========================================================================
# enforce_policy
# =========================================================================


class TestEnforcePolicy:

    def test_allowed_returns_result(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(
            200, {"allowed": True}
        )
        result = exe_client.enforce_policy("my-agent", "hello world")
        assert result["allowed"] is True

        # Verify request shape
        args, kwargs = mock_session.request.call_args
        assert args[0] == "POST"
        assert "/api/v1/policies/enforce" in args[1]
        assert kwargs["json"]["agentId"] == "my-agent"
        assert kwargs["json"]["input"] == "hello world"

    def test_allowed_with_warnings(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(
            200,
            {
                "allowed": True,
                "warnings": [
                    {"policyType": "data_restriction", "message": "Sensitive content"}
                ],
            },
        )
        result = exe_client.enforce_policy("my-agent", "show me SSN data")
        assert result["allowed"] is True
        assert len(result["warnings"]) == 1

    def test_blocked_raises_policy_blocked_error(self, exe_client, mock_session):
        violations = [
            {"policyType": "injection_scan", "message": "SQL injection detected"}
        ]
        mock_session.request.return_value = make_mock_response(
            403,
            {"allowed": False, "violations": violations},
        )
        with pytest.raises(PolicyBlockedError) as exc_info:
            exe_client.enforce_policy("my-agent", "DROP TABLE users;")
        assert exc_info.value.violations == violations

    def test_optional_params_sent(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(
            200, {"allowed": True}
        )
        exe_client.enforce_policy(
            "my-agent",
            "hello",
            environment="production",
            metadata={"user": "test"},
            estimated_cost=0.05,
            tools=["web_search"],
        )
        payload = mock_session.request.call_args.kwargs["json"]
        assert payload["environment"] == "production"
        assert payload["metadata"] == {"user": "test"}
        assert payload["estimatedCost"] == 0.05
        assert payload["tools"] == ["web_search"]

    def test_require_approval_polls_until_approved(self, exe_client, mock_session):
        mock_session.request.side_effect = [
            make_mock_response(
                202,
                {"allowed": False, "requiresApproval": True, "approvalRequestId": "apr_1"},
            ),
            make_mock_response(200, {"data": {"id": "apr_1", "status": "approved"}}),
        ]

        result = exe_client.enforce_policy("my-agent", "delete 100 customer records")
        assert result["allowed"] is True
        assert result["approvalRequestId"] == "apr_1"

    def test_require_approval_denied_raises(self, exe_client, mock_session):
        mock_session.request.side_effect = [
            make_mock_response(
                202,
                {"allowed": False, "requiresApproval": True, "approvalRequestId": "apr_2"},
            ),
            make_mock_response(
                200,
                {"data": {"id": "apr_2", "status": "denied", "decisionReason": "Denied"}},
            ),
        ]

        with pytest.raises(PolicyDeniedError):
            exe_client.enforce_policy("my-agent", "dangerous")

    def test_require_approval_expired_raises(self, exe_client, mock_session):
        mock_session.request.side_effect = [
            make_mock_response(
                202,
                {"allowed": False, "requiresApproval": True, "approvalRequestId": "apr_3"},
            ),
            make_mock_response(200, {"data": {"id": "apr_3", "status": "expired"}}),
        ]

        with pytest.raises(ApprovalTimeoutError):
            exe_client.enforce_policy("my-agent", "dangerous")

    def test_401_raises_auth_error(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(401, {})
        with pytest.raises(ExeclaveAuthError):
            exe_client.enforce_policy("my-agent", "hello")

    def test_402_fail_open_returns_allowed_with_warning(self, exe_client, mock_session):
        """Default fail_open: 402 returns allowed=True with a plan_limit warning."""
        mock_session.request.return_value = make_mock_response(
            402,
            {
                "error": {
                    "code": "PLAN_LIMIT_EXCEEDED",
                    "message": "Your plan limit for maxTracesPerMonth has been reached (10000/10000).",
                    "resource": "maxTracesPerMonth",
                    "current": 10000,
                    "max": 10000,
                }
            },
        )

        result = exe_client.enforce_policy("my-agent", "hello")
        assert result["allowed"] is True
        assert len(result["warnings"]) == 1
        assert result["warnings"][0]["policyType"] == "plan_limit"

    def test_402_fail_closed_raises_plan_limit_exceeded(self, mock_session):
        """fail_closed: 402 raises PlanLimitExceededError."""
        exe = Execlave(
            api_key="exe_test_key_0123456789",
            base_url="http://mock-server:4000",
            async_mode=False,
            enable_control_channel=False,
            enable_injection_scan=False,
            plan_limit_behavior="fail_closed",
        )
        mock_session.request.return_value = make_mock_response(
            402,
            {
                "error": {
                    "resource": "maxTracesPerMonth",
                    "current": 10000,
                    "max": 10000,
                }
            },
        )

        with pytest.raises(PlanLimitExceededError) as exc_info:
            exe.enforce_policy("my-agent", "hello")

        assert exc_info.value.resource == "maxTracesPerMonth"
        assert exc_info.value.current == 10000
        assert exc_info.value.max == 10000

    def test_402_does_not_trip_circuit_breaker(self, mock_session):
        """402 should not increment circuit breaker failures."""
        exe = Execlave(
            api_key="exe_test_key_0123456789",
            base_url="http://mock-server:4000",
            async_mode=False,
            enable_control_channel=False,
            enable_injection_scan=False,
            enforcement_on_outage="fail_open",
        )
        mock_session.request.return_value = make_mock_response(
            402,
            {
                "error": {
                    "resource": "maxTracesPerMonth",
                    "current": 10000,
                    "max": 10000,
                }
            },
        )

        # fail_open default — returns result, no throw
        exe.enforce_policy("my-agent", "hello")
        assert exe._cb_failures == 0
        assert exe._cb_open is False

    def test_500_raises_generic_error(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(
            500, {"error": {"message": "Internal error"}}, text="fail"
        )
        with pytest.raises(ExeclaveError, match="Internal error"):
            exe_client.enforce_policy("my-agent", "hello")

    def test_network_error_fail_open_returns_allowed(self, exe_client, mock_session):
        import requests

        mock_session.request.side_effect = requests.ConnectionError("refused")
        result = exe_client.enforce_policy("my-agent", "hello")
        assert result["allowed"] is True


# =========================================================================
# authorize_agent_call
# =========================================================================


class TestAuthorizeAgentCall:

    def test_authorized_returns_grant(self, exe_client, mock_session):
        grant = {
            "data": {
                "id": "grant-1",
                "callerAgentId": "agent-a",
                "calleeAgentId": "agent-b",
                "action": "summarize",
            }
        }
        mock_session.request.return_value = make_mock_response(200, grant)
        result = exe_client.authorize_agent_call("agent-a", "agent-b", "summarize")
        assert result["data"]["callerAgentId"] == "agent-a"

        payload = mock_session.request.call_args.kwargs["json"]
        assert payload["callerAgentId"] == "agent-a"
        assert payload["calleeAgentId"] == "agent-b"
        assert payload["action"] == "summarize"

    def test_unauthorized_raises_auth_error(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(403, {})
        with pytest.raises(ExeclaveAuthError):
            exe_client.authorize_agent_call("agent-a", "agent-b", "summarize")


# =========================================================================
# discover_agents
# =========================================================================


class TestDiscoverAgents:

    def test_returns_agent_list(self, exe_client, mock_session):
        agents = [
            {"id": "1", "agentId": "bot-a", "name": "Bot A", "capabilities": ["search"]},
            {"id": "2", "agentId": "bot-b", "name": "Bot B", "capabilities": ["summarize"]},
        ]
        mock_session.request.return_value = make_mock_response(200, {"data": agents})
        result = exe_client.discover_agents()
        assert len(result) == 2
        assert result[0]["agentId"] == "bot-a"

    def test_filters_by_capability(self, exe_client, mock_session):
        agents = [{"id": "1", "agentId": "bot-a", "name": "Bot A", "capabilities": ["search"]}]
        mock_session.request.return_value = make_mock_response(200, {"data": agents})
        result = exe_client.discover_agents(capability="search")
        assert len(result) == 1

        url = mock_session.request.call_args[0][1]
        assert "capability=search" in url

    def test_empty_result(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(200, {"data": []})
        result = exe_client.discover_agents(capability="nonexistent")
        assert result == []
