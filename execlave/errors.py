"""Execlave SDK exceptions."""


class ExeclaveError(Exception):
    """Base exception for Execlave SDK errors."""
    pass


class ExeclaveAuthError(ExeclaveError):
    """Raised when authentication fails (invalid API key or insufficient permissions)."""
    pass


class EnforcementHaltError(ExeclaveError):
    """
    Base class for every error that represents a governance decision to STOP
    agent execution — a policy block, an approval denial/timeout, a
    certificate that does not bind the action, an unverifiable approval, a
    paused agent, or a fail-closed enforcement outage.

    Integrations MUST re-raise these rather than swallowing them (see
    :func:`is_enforcement_error`). Deriving from a common base is deliberate:
    a new enforcement error added below automatically halts execution
    everywhere, instead of silently failing open until every wrapper's
    ``except`` clause is updated. Errors that are NOT halt decisions (auth
    failures, plan/quota limits, which have their own fail-open handling)
    intentionally do not extend this.
    """
    pass


def is_enforcement_error(err: BaseException) -> bool:
    """
    True when an error is a governance decision to halt execution.

    Single source of truth — integrations catch ``EnforcementHaltError``
    directly (or call this) instead of maintaining their own tuple of
    exception classes to re-raise (the duplication that previously let
    certificate-mismatch rejections be swallowed as "non-fatal").
    """
    return isinstance(err, EnforcementHaltError)


class AgentPausedError(EnforcementHaltError):
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


class PolicyBlockedError(EnforcementHaltError):
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


class ToolIntegrityError(PolicyBlockedError):
    """
    Raised when a PolicyBlockedError is caused by a ``tool_integrity`` policy —
    an MCP tool descriptor drifted from its pinned baseline, an unapproved
    tool/server was used, or a tool description matched a poisoning pattern.

    Exposes ``tool_violations`` so callers can react to supply-chain tampering
    (e.g. quarantine the agent, alert security) via ``isinstance``.
    """

    def __init__(self, violations: list[dict]):
        super().__init__(violations)
        self.tool_violations = [
            v for v in violations if v.get("policyType") == "tool_integrity"
        ]


def policy_blocked_error_from_violations(violations: list[dict]) -> PolicyBlockedError:
    """Return the most specific PolicyBlockedError subclass for a violation set:
    a ``custom_validator`` denial -> ValidatorDeniedError; otherwise a
    ``tool_integrity`` block -> ToolIntegrityError; otherwise PolicyBlockedError."""
    if any(v.get("policyType") == "custom_validator" for v in violations):
        return ValidatorDeniedError(violations)
    if any(v.get("policyType") == "tool_integrity" for v in violations):
        return ToolIntegrityError(violations)
    return PolicyBlockedError(violations)


class PolicyDeniedError(EnforcementHaltError):
    """Raised when a human approver explicitly denies an approval request."""

    def __init__(self, approval_request_id: str, reason: str | None = None):
        self.approval_request_id = approval_request_id
        self.reason = reason
        msg = f"Approval request '{approval_request_id}' was denied"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ApprovalTimeoutError(EnforcementHaltError):
    """Raised when an approval request expires or polling times out."""

    def __init__(self, approval_request_id: str):
        self.approval_request_id = approval_request_id
        super().__init__(f"Approval request '{approval_request_id}' timed out")


class CertificateMismatchError(EnforcementHaltError):
    """
    Raised when an approval was granted but its authorization certificate does
    NOT bind to the action the SDK is about to execute — i.e. the server
    returned ``valid: False`` from the verify step.

    This is the closed-loop, fail-closed guard: an approval only authorizes
    the exact action a human approved. A mismatch means the presented action
    context differs from the approved one (``action_context_mismatch``), the
    certificate is expired (``certificate_expired``), already consumed
    (``certificate_already_used``), unanchored in the audit chain
    (``certificate_unanchored``), or tampered (``certificate_tampered``).
    Execution MUST NOT proceed — treat like a denial.
    """

    def __init__(self, approval_request_id: str, reason: str | None = None):
        self.approval_request_id = approval_request_id
        self.reason = reason
        msg = (
            f"Authorization certificate for approval '{approval_request_id}' "
            "does not bind to the action being executed"
        )
        if reason:
            msg += f" ({reason})"
        msg += " — execution blocked."
        super().__init__(msg)


class ApprovalVerificationError(EnforcementHaltError):
    """
    Raised when the SDK could not positively confirm an approval's
    certificate — the verify call itself failed (network error, 5xx,
    malformed response), as opposed to a definitive ``valid: False``
    (:class:`CertificateMismatchError`).

    Fail-closed by design: no confirmation means no execution. Catch this to
    halt the agent (and optionally retry) rather than proceeding on an
    unverified approval.
    """

    def __init__(self, approval_request_id: str, cause: str | None = None):
        self.approval_request_id = approval_request_id
        self.cause = cause
        msg = (
            "Could not verify the authorization certificate for approval "
            f"'{approval_request_id}'"
        )
        if cause:
            msg += f": {cause}"
        msg += " — execution blocked (fail-closed)."
        super().__init__(msg)


class EnforcementUnavailableError(EnforcementHaltError):
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


class MetadataContractError(EnforcementHaltError):
    """
    Raised when a framework adapter calls the internal ``enforce_policy_bound``
    boundary with ``metadata`` that was not produced by
    ``seal_for_metadata``/``seal_metadata_entry``/``seal_metadata`` (see
    ``_action_binding.py``).

    This is a contract violation in our own adapter code, not an operational
    failure or a policy decision — it means a material field could have been
    silently omitted from the certificate's canonical action context. Python
    has no compiler to catch this at call sites the way TypeScript does, so
    ``enforce_policy_bound`` checks the runtime marker on every call instead;
    treated fail-closed for the same reason an unbound certificate is worse
    than a blocked call.
    """

    def __init__(self, detail: str | None = None):
        msg = (
            "enforce_policy metadata must be sealed via "
            "seal_for_metadata/seal_metadata_entry/seal_metadata before crossing "
            "the enforce_policy_bound boundary"
        )
        if detail:
            msg += f" ({detail})"
        msg += " — execution blocked."
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
