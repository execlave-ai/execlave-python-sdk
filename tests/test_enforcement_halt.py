"""
Regression guard: every governance decision that must STOP execution has to
be classified as an enforcement error, so integration wrappers re-raise it
instead of swallowing it as "non-fatal".

This locks the fix for the bug where CertificateMismatchError (post-approval
action drift) was NOT in any integration's hand-maintained _ENFORCEMENT_ERRORS
tuple, so a drift rejection was caught, logged, and execution continued —
silently turning a fail-closed guarantee into fail-open on every adapter path.
"""

import pytest

from execlave.errors import (
    is_enforcement_error,
    EnforcementHaltError,
    ExeclaveError,
    ExeclaveAuthError,
    AgentPausedError,
    PolicyBlockedError,
    ValidatorDeniedError,
    ToolIntegrityError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    CertificateMismatchError,
    ApprovalVerificationError,
    EnforcementUnavailableError,
    QuotaExceededError,
    PlanLimitExceededError,
)

VIOLATION = [
    {
        "policyId": "p1",
        "policyName": "P",
        "policyType": "custom_validator",
        "message": "no",
        "enforcementMode": "block",
    }
]


class TestEnforcementHaltClassification:
    def test_every_halt_decision_error_is_classified(self):
        halt_errors = [
            AgentPausedError("agent_1", "paused"),
            PolicyBlockedError(VIOLATION),
            ValidatorDeniedError(VIOLATION),
            ToolIntegrityError(VIOLATION),
            PolicyDeniedError("apr_1", "denied"),
            ApprovalTimeoutError("apr_1"),
            CertificateMismatchError("apr_1", "action_context_mismatch"),
            ApprovalVerificationError("apr_1", "ECONNRESET"),
            EnforcementUnavailableError(3, "boom"),
        ]
        for err in halt_errors:
            assert is_enforcement_error(err) is True
            assert isinstance(err, EnforcementHaltError)

    def test_post_approval_drift_errors_are_halt_decisions(self):
        # The two that were missing from every integration's allowlist.
        assert (
            is_enforcement_error(CertificateMismatchError("apr_1", "certificate_expired"))
            is True
        )
        assert is_enforcement_error(ApprovalVerificationError("apr_1")) is True

    def test_non_halt_errors_are_not_classified_as_enforcement(self):
        non_halt = [
            ExeclaveError("generic"),
            ExeclaveAuthError(),
            QuotaExceededError("traces", 10, 10),
            PlanLimitExceededError("traces", 10, 10),
            ValueError("plain"),
        ]
        for err in non_halt:
            assert is_enforcement_error(err) is False

    def test_halt_errors_keep_their_own_identity(self):
        denied = ValidatorDeniedError(VIOLATION)
        assert isinstance(denied, ValidatorDeniedError)
        assert isinstance(denied, PolicyBlockedError)
        assert isinstance(denied, EnforcementHaltError)
        assert isinstance(denied, ExeclaveError)

    def test_except_enforcement_halt_error_catches_all_halt_types(self):
        for err_instance in [
            CertificateMismatchError("apr_1"),
            ApprovalVerificationError("apr_1"),
            PolicyDeniedError("apr_1"),
            ApprovalTimeoutError("apr_1"),
        ]:
            with pytest.raises(EnforcementHaltError):
                raise err_instance
