"""Execlave SDK exceptions."""


class ExeclaveError(Exception):
    """Base exception for Execlave SDK errors."""
    pass


class ExeclaveAuthError(ExeclaveError):
    """Raised when authentication fails (invalid API key or insufficient permissions)."""
    pass


class AgentPausedError(ExeclaveError):
    """
    Raised when an agent is paused via the kill switch.
    
    New trace calls return this immediately without hitting the LLM.
    In-flight traces complete naturally (no mid-execution termination).
    
    Your application should catch this and return a graceful message to users.
    """
    def __init__(self, agent_id: str, reason: str | None = None):
        self.agent_id = agent_id
        self.reason = reason
        msg = f"Agent '{agent_id}' is paused"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PolicyBlockedError(ExeclaveError):
    """
    Raised when pre-execution policy enforcement blocks an agent call.

    Contains the list of policy violations that caused the block.
    Your application should catch this and prevent the LLM call from proceeding.
    """
    def __init__(self, violations: list[dict]):
        self.violations = violations
        parts = [f"[{v.get('policyType', 'unknown')}] {v.get('message', '')}" for v in violations]
        msg = "Execution blocked by policy: " + "; ".join(parts)
        super().__init__(msg)


class ValidatorDeniedError(PolicyBlockedError):
    """
    Raised when a PolicyBlockedError is caused by a ``custom_validator`` policy
    — the decision came from a customer-hosted HTTP endpoint (BYOV).

    Exposes ``validator_violations`` so callers can distinguish validator
    denials from built-in policy blocks via ``isinstance``.
    """
    def __init__(self, violations: list[dict]):
        super().__init__(violations)
        self.validator_violations = [
            v for v in violations if v.get("policyType") == "custom_validator"
        ]

    @classmethod
    def from_violations(cls, violations: list[dict]) -> "PolicyBlockedError":
        """Return ValidatorDeniedError if any violation is a custom_validator,
        otherwise a plain PolicyBlockedError."""
        if any(v.get("policyType") == "custom_validator" for v in violations):
            return cls(violations)
        return PolicyBlockedError(violations)


class PolicyDeniedError(ExeclaveError):
    """Raised when a human approver explicitly denies an approval request."""

    def __init__(self, approval_request_id: str, reason: str | None = None):
        self.approval_request_id = approval_request_id
        self.reason = reason
        msg = f"Approval request '{approval_request_id}' was denied"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ApprovalTimeoutError(ExeclaveError):
    """Raised when an approval request expires or polling times out."""

    def __init__(self, approval_request_id: str):
        self.approval_request_id = approval_request_id
        super().__init__(f"Approval request '{approval_request_id}' timed out")


class EnforcementUnavailableError(ExeclaveError):
    """
    Raised when the Execlave enforcement endpoint is unreachable
    and the SDK is configured with enforcement_on_outage='fail_closed'.

    The circuit breaker trips after 3 consecutive failures. When this error
    is raised, your application should halt agent execution to maintain the
    security posture.
    """

    def __init__(self, consecutive_failures: int, last_error: str | None = None):
        self.consecutive_failures = consecutive_failures
        self.last_error = last_error
        msg = (
            f"Enforcement unavailable after {consecutive_failures} consecutive failures. "
            f"SDK is in fail_closed mode — agent execution is blocked."
        )
        if last_error:
            msg += f" Last error: {last_error}"
        super().__init__(msg)


class QuotaExceededError(ExeclaveError):
    """Raised when the organization's plan quota is exhausted."""

    def __init__(self, resource: str, current: int, max: int, message: str = ""):
        self.resource = resource
        self.current = current
        self.max = max
        super().__init__(
            message
            or f"Plan limit reached for {resource} ({current}/{max}). "
            "Upgrade at https://www.execlave.com/dashboard/billing"
        )


class PlanLimitExceededError(QuotaExceededError):
    """
    Raised when the organization's plan limit is exceeded and the SDK is
    configured with plan_limit_behavior='fail_closed'.

    When plan_limit_behavior is 'fail_open' (default), the SDK logs a
    warning and allows execution to continue unmonitored instead of raising.
    """
    pass
