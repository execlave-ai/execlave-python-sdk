"""
Tests for Execlave.errors — exception hierarchy and attributes.
"""

import pytest

from execlave.errors import ExeclaveError, ExeclaveAuthError, AgentPausedError, QuotaExceededError, PlanLimitExceededError


# =========================================================================
# Inheritance
# =========================================================================


class TestInheritance:

    def test_base_error_is_exception(self):
        assert issubclass(ExeclaveError, Exception)

    def test_auth_error_is_execlave_error(self):
        assert issubclass(ExeclaveAuthError, ExeclaveError)

    def test_paused_error_is_execlave_error(self):
        assert issubclass(AgentPausedError, ExeclaveError)

    def test_quota_error_is_execlave_error(self):
        assert issubclass(QuotaExceededError, ExeclaveError)

    def test_catch_base_catches_auth(self):
        with pytest.raises(ExeclaveError):
            raise ExeclaveAuthError("bad key")

    def test_catch_base_catches_paused(self):
        with pytest.raises(ExeclaveError):
            raise AgentPausedError("bot1")


# =========================================================================
# ExeclaveError
# =========================================================================


class TestExeclaveError:

    def test_message(self):
        err = ExeclaveError("something broke")
        assert str(err) == "something broke"

    def test_empty_message(self):
        err = ExeclaveError()
        assert str(err) == ""


# =========================================================================
# ExeclaveAuthError
# =========================================================================


class TestExeclaveAuthError:

    def test_message(self):
        err = ExeclaveAuthError("invalid key")
        assert str(err) == "invalid key"


# =========================================================================
# AgentPausedError
# =========================================================================


class TestAgentPausedError:

    def test_message_without_reason(self):
        err = AgentPausedError("my-bot")
        assert "my-bot" in str(err)
        assert "paused" in str(err).lower()

    def test_message_with_reason(self):
        err = AgentPausedError("my-bot", reason="safety violation")
        assert "my-bot" in str(err)
        assert "safety violation" in str(err)

    def test_agent_id_attribute(self):
        err = AgentPausedError("my-bot")
        assert err.agent_id == "my-bot"

    def test_reason_attribute_none(self):
        err = AgentPausedError("my-bot")
        assert err.reason is None

    def test_reason_attribute_set(self):
        err = AgentPausedError("my-bot", reason="policy breach")
        assert err.reason == "policy breach"

    def test_catch_with_except(self):
        """Ensure typical try/except usage works."""
        try:
            raise AgentPausedError("bot1", reason="kill switch")
        except AgentPausedError as e:
            assert e.agent_id == "bot1"
            assert e.reason == "kill switch"


# =========================================================================
# QuotaExceededError
# =========================================================================


class TestQuotaExceededError:

    def test_fields_are_exposed(self):
        err = QuotaExceededError("maxTracesPerMonth", 10000, 10000)
        assert err.resource == "maxTracesPerMonth"
        assert err.current == 10000
        assert err.max == 10000

    def test_default_message_contains_upgrade_url(self):
        err = QuotaExceededError("traces", 10000, 10000)
        assert "Plan limit reached for traces (10000/10000)" in str(err)
        assert "https://www.execlave.com/dashboard/billing" in str(err)

    def test_custom_message_is_respected(self):
        err = QuotaExceededError("traces", 10000, 10000, "custom quota message")
        assert str(err) == "custom quota message"


# =========================================================================
# PlanLimitExceededError
# =========================================================================


class TestPlanLimitExceededError:

    def test_is_subclass_of_quota_exceeded(self):
        assert issubclass(PlanLimitExceededError, QuotaExceededError)
        assert issubclass(PlanLimitExceededError, ExeclaveError)

    def test_fields_are_exposed(self):
        err = PlanLimitExceededError("maxAgents", 3, 3)
        assert err.resource == "maxAgents"
        assert err.current == 3
        assert err.max == 3

    def test_catch_with_quota_exceeded(self):
        with pytest.raises(QuotaExceededError):
            raise PlanLimitExceededError("maxAgents", 3, 3)


# =========================================================================
# ValidatorDeniedError (BYOV)
# =========================================================================


class TestValidatorDeniedError:
    from execlave.errors import PolicyBlockedError, ValidatorDeniedError

    VALIDATOR_VIOLATION = {
        "policyId": "pol-v1",
        "policyName": "Finance Validator",
        "policyType": "custom_validator",
        "message": "Exceeds finance team spend limit",
        "enforcementMode": "block",
    }
    INJECTION_VIOLATION = {
        "policyId": "pol-i1",
        "policyName": "Injection Scan",
        "policyType": "injection_scan",
        "message": "Prompt injection detected",
        "enforcementMode": "block",
    }

    def test_is_subclass_of_policy_blocked(self):
        from execlave.errors import PolicyBlockedError, ValidatorDeniedError
        assert issubclass(ValidatorDeniedError, PolicyBlockedError)

    def test_exposes_validator_violations(self):
        from execlave.errors import ValidatorDeniedError
        err = ValidatorDeniedError([self.VALIDATOR_VIOLATION, self.INJECTION_VIOLATION])
        assert len(err.violations) == 2
        assert len(err.validator_violations) == 1
        assert err.validator_violations[0]["policyId"] == "pol-v1"

    def test_from_violations_returns_validator_when_present(self):
        from execlave.errors import ValidatorDeniedError
        err = ValidatorDeniedError.from_violations([self.INJECTION_VIOLATION, self.VALIDATOR_VIOLATION])
        assert isinstance(err, ValidatorDeniedError)

    def test_from_violations_returns_policy_blocked_when_no_validator(self):
        from execlave.errors import PolicyBlockedError, ValidatorDeniedError
        err = ValidatorDeniedError.from_violations([self.INJECTION_VIOLATION])
        assert isinstance(err, PolicyBlockedError)
        assert not isinstance(err, ValidatorDeniedError)
