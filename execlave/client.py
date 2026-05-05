"""
Execlave SDK — Main client.

Provides the Execlave class for agent registration, tracing, and governance.
Implements non-blocking trace ingestion with an in-memory circular buffer
and a background flush thread.  Includes client-side PII scrubbing and
prompt-injection pre-screening.
"""

import hashlib
import os
import re
import time
import uuid
import logging
import functools
import threading
from collections import deque
from typing import Any, Callable, Optional, Dict, List

import requests  # type: ignore[import-untyped]

from .errors import (
    ExeclaveError,
    ExeclaveAuthError,
    AgentPausedError,
    PolicyBlockedError,
    ValidatorDeniedError,
    PolicyDeniedError,
    ApprovalTimeoutError,
    EnforcementUnavailableError,
    QuotaExceededError,
    PlanLimitExceededError,
)
from .agent import Agent
from .trace import Trace

logger = logging.getLogger("Execlave")

# ---------------------------------------------------------------------------
# Optional WebSocket support (python-socketio)
# ---------------------------------------------------------------------------
try:
    import socketio as _socketio_mod  # type: ignore[import-not-found]
    _HAS_SOCKETIO = True
except ImportError:
    _socketio_mod = None  # type: ignore[assignment]
    _HAS_SOCKETIO = False

# ---------------------------------------------------------------------------
# SDK State enum
# ---------------------------------------------------------------------------
_STATE_INITIALIZING = "INITIALIZING"
_STATE_ACTIVE = "ACTIVE"
_STATE_PAUSED = "PAUSED"
_STATE_SHUTDOWN = "SHUTDOWN"

# ---------------------------------------------------------------------------
# PII patterns  (same as the processing service, but usable client-side)
# ---------------------------------------------------------------------------
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "api_key": re.compile(r"\b(?:sk|pk|ag)_[a-zA-Z0-9]{20,}\b"),
}

# ---------------------------------------------------------------------------
# Injection patterns  (common prompt-injection signatures)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?(?:DAN|evil|unrestricted)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(?:previous|earlier|your)\s+(?:instructions|rules|guidelines)", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"\[SYSTEM\]|\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"<\|(?:system|im_start|im_end)\|>", re.IGNORECASE),
    re.compile(r"(?:reveal|show|display|print|output)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"(?:act|behave|respond)\s+as\s+(?:if|though)\s+(?:you\s+(?:are|were|have))", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"bypass\s+(?:your\s+)?(?:filters?|restrictions?|safety|guidelines?)", re.IGNORECASE),
]


class Execlave:
    """
    Main entry point for the Execlave SDK.

    Usage::

        ag = Execlave(api_key="exe_prod_xxx", environment="production")
        agent = ag.register_agent(agent_id="my-bot", name="My Bot", ...)

        @ag.trace
        def answer(question):
            return llm.call(question)

        with ag.trace(session_id="sess_1") as t:
            result = do_work()
            t.set_output(result)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        environment: str = "production",
        async_mode: bool = True,
        mode: str = "native",
        otlp_endpoint: str | None = None,
        batch_size: int = 100,
        flush_interval_seconds: int = 10,
        debug: bool = False,
        privacy: dict | None = None,
        enable_control_channel: bool = True,
        enable_injection_scan: bool = True,
        enforcement_on_outage: str = "fail_open",
        plan_limit_behavior: str = "fail_open",
        heartbeat_interval_seconds: int = 600,
        policy_cache_ttl_seconds: int = 60,
        api_version: str | None = "v1",
    ):
        self.api_key = api_key or os.environ.get("EXECLAVE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "api_key must be provided or EXECLAVE_API_KEY env var must be set"
            )

        self.base_url = (
            base_url or os.environ.get("EXECLAVE_BASE_URL") or "https://api.execlave.com"
        ).rstrip("/")
        self.api_version = api_version if api_version else None
        self.environment = environment
        self.async_mode = async_mode
        self.mode = mode
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.debug = debug
        self.privacy = privacy or {}
        self.enable_control_channel = enable_control_channel
        self.enable_injection_scan = enable_injection_scan
        self.enforcement_on_outage = enforcement_on_outage  # 'fail_open' or 'fail_closed'
        self.plan_limit_behavior = plan_limit_behavior  # 'fail_open' or 'fail_closed'
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.policy_cache_ttl_seconds = policy_cache_ttl_seconds

        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # HTTP session
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Execlave-python-sdk/1.0.0",
        })

        # State machine: INITIALIZING → ACTIVE → PAUSED → ACTIVE / SHUTDOWN
        self._state = _STATE_INITIALIZING

        # In-memory circular buffer (max 10,000 traces)
        self._buffer: deque = deque(maxlen=10_000)
        self._buffer_lock = threading.Lock()

        # Background flush thread
        self._flush_event = threading.Event()
        self._flush_thread: threading.Thread | None = None
        if async_mode:
            self._flush_thread = threading.Thread(
                target=self._flush_loop, daemon=True, name="Execlave-flush"
            )
            self._flush_thread.start()

        # Track registered agents for status polling
        self._agents: dict[str, Agent] = {}

        # Circuit breaker state
        self._cb_failures: int = 0
        self._cb_threshold: int = 3
        self._cb_open: bool = False
        self._cb_open_at: float = 0.0
        self._cb_reset_after: float = 60.0  # seconds before retrying after circuit opens
        self._cb_last_error: str | None = None
        self._cb_lock = threading.Lock()

        # Policy decision cache: {cache_key: {"response": dict, "expires_at": float}}
        self._policy_cache: dict[str, dict] = {}
        self._policy_cache_lock = threading.Lock()

        # Quota-exhausted cache for trace-related operations (60s fail-fast)
        self._quota_exceeded: QuotaExceededError | None = None
        self._quota_expires_at: float = 0.0
        self._quota_cache_ttl_seconds: float = 60.0

        # Background status polling thread (control channel fallback)
        self._poll_interval_seconds = 15
        self._poll_thread: threading.Thread | None = None
        if enable_control_channel:
            self._poll_thread = threading.Thread(
                target=self._status_poll_loop, daemon=True, name="Execlave-poll"
            )
            self._poll_thread.start()

        # WebSocket control channel (real-time kill-switch, <500ms latency)
        self._sio: Any = None
        if enable_control_channel and _HAS_SOCKETIO:
            self._connect_websocket()

        self._state = _STATE_ACTIVE

        # Heartbeat background thread
        self._heartbeat_thread: threading.Thread | None = None
        if enable_control_channel:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True, name="Execlave-heartbeat"
            )
            self._heartbeat_thread.start()

        # OTel exporter (initialized only when mode is "otlp")
        self._otel_exporter = None
        if self.mode == "otlp":
            if not otlp_endpoint:
                raise ValueError("otlp_endpoint is required when mode='otlp'")
            from .otel import OTelExporter
            self._otel_exporter = OTelExporter(
                endpoint=otlp_endpoint,
                api_key=self.api_key,
                service_name=f"Execlave-{self.environment}",
            )
            logger.info("OTel OTLP exporter initialized (endpoint=%s)", otlp_endpoint)

        logger.debug("Execlave SDK initialized (env=%s, async=%s)", environment, async_mode)

    # ------------------------------------------------------------------
    # API path helper
    # ------------------------------------------------------------------

    def _api_path(self, path: str) -> str:
        """Prepend the versioned API prefix to a resource path.

        If ``api_version`` is set (e.g. ``'v1'``), returns ``/api/v1{path}``.
        Otherwise falls back to the legacy ``/api{path}`` format.
        """
        if self.api_version:
            return f"/api/{self.api_version}{path}"
        return f"/api{path}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check if the Execlave API is reachable and the API key is valid."""
        try:
            # Use unversioned /health (the versioned /api/v1/health doesn't exist)
            # Include auth headers so invalid keys are detected early
            resp = self._session.get(
                f"{self.base_url}/health",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _extract_agent_payload(
        self,
        response: Any,
        agent_id: str,
        environment: str | None,
    ) -> dict[str, Any]:
        """Normalize agent create/search responses into a single matching agent object."""
        payload = response.get("data", response) if isinstance(response, dict) else response

        if isinstance(payload, dict):
            response_agent_id = payload.get("agentId")
            if response_agent_id is None or response_agent_id == agent_id:
                return payload
            raise ExeclaveError(
                f"Agent registration returned agentId '{response_agent_id}' instead of '{agent_id}'"
            )

        if isinstance(payload, list):
            agents = [item for item in payload if isinstance(item, dict)]
            matches = [item for item in agents if item.get("agentId") == agent_id]
            if environment:
                exact_environment_matches = [
                    item for item in matches if item.get("environment") == environment
                ]
                if exact_environment_matches:
                    return exact_environment_matches[0]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ExeclaveError(
                    f"Agent registration returned multiple entries for agentId '{agent_id}'"
                )
            raise ExeclaveError(
                f"Agent registration response did not include agentId '{agent_id}'"
            )

        raise ExeclaveError(
            f"Agent registration returned unexpected response shape: {type(payload).__name__}"
        )

    def register_agent(
        self,
        agent_id: str,
        name: str,
        type: str = "chatbot",
        platform: str = "custom",
        environment: str | None = None,
        description: str | None = None,
        owner_email: str | None = None,
        allowed_data_sources: list[str] | None = None,
        allowed_actions: list[str] | None = None,
        requires_human_approval_for: list[str] | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> Agent:
        """
        Register (or re-register) an AI agent. Idempotent — call on startup.

        Returns an Agent object with prompt management methods.
        """
        resolved_environment = environment or self.environment
        payload: dict[str, Any] = {
            "agentId": agent_id,
            "name": name,
            "type": type,
            "platform": platform,
            "environment": resolved_environment,
        }
        if description:
            payload["description"] = description
        if owner_email:
            payload["ownerEmail"] = owner_email
        if allowed_data_sources is not None:
            payload["allowedDataSources"] = allowed_data_sources
        if allowed_actions is not None:
            payload["allowedActions"] = allowed_actions
        if requires_human_approval_for is not None:
            payload["requiresHumanApprovalFor"] = requires_human_approval_for
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata

        try:
            data = self._request("POST", self._api_path("/agents"), json=payload)
        except ExeclaveError as registration_error:
            # If agent already exists, try to fetch it by agent_id or name
            try:
                agents_resp = self._request("GET", self._api_path(f"/agents?search={agent_id}"))
            except ExeclaveError:
                raise registration_error

            try:
                agent_data = self._extract_agent_payload(
                    agents_resp,
                    agent_id,
                    resolved_environment,
                )
                agent = Agent(self, agent_data)
                self._agents[agent_id] = agent
                return agent
            except ExeclaveError:
                pass

            # Fallback: search by name if agent_id search succeeded but did not match.
            if name:
                try:
                    agents_resp = self._request("GET", self._api_path(f"/agents?search={name}"))
                except ExeclaveError:
                    raise registration_error

                try:
                    agent_data = self._extract_agent_payload(
                        agents_resp,
                        agent_id,
                        resolved_environment,
                    )
                    agent = Agent(self, agent_data)
                    self._agents[agent_id] = agent
                    return agent
                except ExeclaveError:
                    pass
            raise registration_error

        agent = Agent(self, self._extract_agent_payload(data, agent_id, resolved_environment))
        self._agents[agent_id] = agent
        return agent

    # ------------------------------------------------------------------
    # Tracing — can be used as decorator or context manager
    # ------------------------------------------------------------------

    def trace(
        self,
        func: Callable | None = None,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        agent_id: str | None = None,
        tags: list[str] | None = None,
        environment: str | None = None,
        parent_trace_id: str | None = None,
        span_type: str | None = None,
    ):
        """
        Trace an execution. Can be used three ways:

        1. **Decorator** (no args):  ``@ag.trace``
        2. **Decorator** (with args):  ``@ag.trace(session_id="s1")``
        3. **Context manager**:   ``with ag.trace(session_id="s1") as t: ...``
        """
        if self._state == _STATE_PAUSED:
            # If used as decorator call, we need to figure out which agent
            raise AgentPausedError(agent_id or "unknown")

        if self._state == _STATE_SHUTDOWN:
            raise ExeclaveError("SDK has been shut down. Call not allowed.")

        # Resolve the agent_id from the first registered agent if not provided
        resolved_agent_id = agent_id
        if not resolved_agent_id and self._agents:
            first_agent = next(iter(self._agents.values()))
            resolved_agent_id = first_agent.agent_id

        resolved_environment = environment or self.environment

        # Case 1: @ag.trace  (func is the decorated function)
        if func is not None and callable(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if self._state == _STATE_PAUSED:
                    raise AgentPausedError(resolved_agent_id or "unknown")
                t = Trace(
                    self,
                    agent_id=resolved_agent_id,
                    session_id=session_id,
                    user_id=user_id,
                    metadata=metadata,
                    tags=tags,
                    environment=resolved_environment,
                    parent_trace_id=parent_trace_id,
                    span_type=span_type,
                )
                try:
                    result = func(*args, **kwargs)
                    t.set_output(result)
                    t.finish(status="success")
                    return result
                except Exception as e:
                    t.finish(status="error", error_message=str(e), error_type=type(e).__name__)
                    raise
            return wrapper

        # Case 2/3: ag.trace(session_id=...) — returns decorator or context manager
        # If called with no func arg, we return a Trace (context manager) or a decorator
        trace_obj = Trace(
            self,
            agent_id=resolved_agent_id,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            environment=resolved_environment,
            parent_trace_id=parent_trace_id,
            span_type=span_type,
        )
        return trace_obj

    # Cross-language parity: ``wrap`` is the JS SDK's name for the same role.
    def wrap(
        self,
        func: Callable | None = None,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        agent_id: str | None = None,
    ):
        """Alias for :meth:`trace`. Provided for parity with the JS SDK ``wrap()``."""
        return self.trace(
            func,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            agent_id=agent_id,
        )

    def start_trace(
        self,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        environment: str | None = None,
        parent_trace_id: str | None = None,
        span_type: str | None = None,
    ) -> Trace:
        """
        Start a manual trace. You must call ``trace.finish()`` when done.
        """
        if self._state == _STATE_PAUSED:
            raise AgentPausedError(agent_id or "unknown")

        resolved_agent_id = agent_id
        if not resolved_agent_id and self._agents:
            first_agent = next(iter(self._agents.values()))
            resolved_agent_id = first_agent.agent_id

        return Trace(
            self,
            agent_id=resolved_agent_id,
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            environment=environment or self.environment,
            parent_trace_id=parent_trace_id,
            span_type=span_type,
        )

    # ------------------------------------------------------------------
    # Flush / Shutdown
    # ------------------------------------------------------------------
    # Enforcement & Authorization
    # ------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Circuit breaker helpers
    # -----------------------------------------------------------------------

    def _cb_record_success(self) -> None:
        with self._cb_lock:
            self._cb_failures = 0
            self._cb_open = False
            self._cb_last_error = None

    def _cb_record_failure(self, error_msg: str) -> None:
        with self._cb_lock:
            self._cb_failures += 1
            self._cb_last_error = error_msg
            if self._cb_failures >= self._cb_threshold:
                self._cb_open = True
                self._cb_open_at = time.time()
                logger.warning(
                    "Circuit breaker OPEN after %d consecutive failures (mode=%s)",
                    self._cb_failures,
                    self.enforcement_on_outage,
                )

    def _cb_is_open(self) -> bool:
        with self._cb_lock:
            if not self._cb_open:
                return False
            # Half-open: allow one retry after reset period
            if time.time() - self._cb_open_at > self._cb_reset_after:
                logger.info("Circuit breaker half-open — retrying enforcement")
                return False
            return True

    # -----------------------------------------------------------------------
    # Policy cache helpers
    # -----------------------------------------------------------------------

    def _cache_key(self, agent_id: str, input_text: str) -> str:
        h = hashlib.sha256(f"{agent_id}:{input_text}".encode()).hexdigest()[:16]
        return f"policy:{h}"

    def _cache_get(self, key: str) -> dict | None:
        with self._policy_cache_lock:
            entry = self._policy_cache.get(key)
            if entry and entry["expires_at"] > time.time():
                return entry["response"]
            if entry:
                del self._policy_cache[key]  # expired
            return None

    def _cache_set(self, key: str, response: dict) -> None:
        with self._policy_cache_lock:
            self._policy_cache[key] = {
                "response": response,
                "expires_at": time.time() + self.policy_cache_ttl_seconds,
            }
            # Evict old entries (keep max 500)
            if len(self._policy_cache) > 500:
                oldest = sorted(self._policy_cache, key=lambda k: self._policy_cache[k]["expires_at"])
                for old_key in oldest[:100]:
                    del self._policy_cache[old_key]

    def _set_quota_exceeded(self, error: QuotaExceededError) -> None:
        # Cache only trace quota, since we use this as fail-fast for enforce/trace paths.
        if error.resource != "maxTracesPerMonth":
            return
        self._quota_exceeded = error
        self._quota_expires_at = time.time() + self._quota_cache_ttl_seconds

    def _get_cached_quota_error(self) -> QuotaExceededError | None:
        if not self._quota_exceeded:
            return None
        if time.time() >= self._quota_expires_at:
            self._quota_exceeded = None
            self._quota_expires_at = 0.0
            return None
        return self._quota_exceeded

    def _raise_if_quota_exceeded(self) -> None:
        cached = self._get_cached_quota_error()
        if not cached:
            return
        if self.plan_limit_behavior == "fail_open":
            return
        raise PlanLimitExceededError(
            resource=cached.resource,
            current=cached.current,
            max=cached.max,
            message=str(cached),
        )

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _quota_error_from_response(self, resp: requests.Response) -> QuotaExceededError:
        resource = "unknown"
        current = 0
        max_value = 0
        message = ""

        try:
            body = resp.json()
            err = body.get("error", {}) if isinstance(body, dict) else {}
            resource = str(err.get("resource", "unknown"))
            current = self._to_int(err.get("current", 0), 0)
            max_value = self._to_int(err.get("max", 0), 0)
            message = str(err.get("message", ""))
        except ValueError:
            message = ""

        return QuotaExceededError(
            resource=resource,
            current=current,
            max=max_value,
            message=message,
        )

    # -----------------------------------------------------------------------
    # Heartbeat loop
    # -----------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Background thread: pings backend heartbeat every 10 minutes."""
        while self._state not in (_STATE_SHUTDOWN,):
            time.sleep(self.heartbeat_interval_seconds)
            if self._state == _STATE_SHUTDOWN:
                break
            for agent_id in list(self._agents.keys()):
                try:
                    agent = self._agents.get(agent_id)
                    if not agent:
                        continue
                    self._session.post(
                        f"{self.base_url}{self._api_path(f'/agents/{agent.id}/heartbeat')}",
                        json={"lastPolicyCheckAt": None},
                        timeout=10,
                    )
                    logger.debug("Heartbeat sent for agent %s", agent_id)
                except Exception as e:
                    logger.debug("Heartbeat failed for agent %s: %s", agent_id, e)

    # -----------------------------------------------------------------------
    # Agent ID Resolution
    # -----------------------------------------------------------------------

    def _resolve_agent_id(self, agent_id: str) -> str:
        """Resolve an external agentId string to the internal UUID.

        The API endpoints like /policies/enforce expect the internal UUID,
        but users naturally pass the external agentId (e.g. "my-bot").
        This looks up the cached Agent and returns its UUID (.id).
        If no match is found, returns the original value unchanged.
        """
        agent = self._agents.get(agent_id)
        if agent:
            return agent.id  # internal UUID
        # Maybe the caller already passed a UUID — return as-is
        return agent_id

    # -----------------------------------------------------------------------
    # enforce_policy — with circuit breaker + cache
    # -----------------------------------------------------------------------

    def enforce_policy(
        self,
        agent_id: str,
        input: str,
        *,
        environment: str | None = None,
        metadata: dict | None = None,
        estimated_cost: float | None = None,
        tools: list[str] | None = None,
    ) -> dict:
        """
        Pre-execution policy enforcement. Call BEFORE sending a prompt to the LLM.

        Raises ``PolicyBlockedError`` if any policy in ``block`` mode fires.
        Raises ``EnforcementUnavailableError`` if circuit breaker trips in fail_closed.
        Returns a dict with ``allowed`` (bool) and optional ``warnings`` list.

        Features:
        - **Circuit breaker**: After 3 consecutive network failures, the circuit opens.
          In ``fail_open`` mode, execution is allowed. In ``fail_closed``, an error is raised.
        - **Cache**: Successful responses are cached for 60s (configurable).
          ``require_approval`` (202) responses are never cached.

        Example::

            try:
                result = guard.enforce_policy("my-agent", user_input)
                # result["allowed"] is True; check result.get("warnings")
            except PolicyBlockedError as e:
                print("Blocked:", e.violations)
        """
        self._raise_if_quota_exceeded()

        # 1. Check cache first
        cache_key = self._cache_key(agent_id, input)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug("Policy cache hit for %s", agent_id)
            return cached

        # 2. Check circuit breaker
        if self._cb_is_open():
            if self.enforcement_on_outage == "fail_closed":
                raise EnforcementUnavailableError(
                    self._cb_failures, self._cb_last_error
                )
            # fail_open: allow execution
            logger.warning(
                "Circuit breaker open — fail_open mode, allowing execution for %s",
                agent_id,
            )
            return {"allowed": True, "source": "circuit_breaker_fail_open"}

        # 3. Build payload and make HTTP call
        # Resolve external agentId to internal UUID if we have a cached agent
        resolved_agent_id = self._resolve_agent_id(agent_id)

        payload: dict[str, Any] = {"agentId": resolved_agent_id, "input": input}
        if environment:
            payload["environment"] = environment
        if metadata:
            payload["metadata"] = metadata
        if estimated_cost is not None:
            payload["estimatedCost"] = estimated_cost
        if tools:
            payload["tools"] = tools

        url = f"{self.base_url}{self._api_path('/policies/enforce')}"
        try:
            resp = self._session.request("POST", url, json=payload, timeout=30)
        except requests.RequestException as e:
            # Network failure — circuit breaker
            self._cb_record_failure(str(e))
            if self.enforcement_on_outage == "fail_closed":
                if self._cb_is_open():
                    raise EnforcementUnavailableError(
                        self._cb_failures, str(e)
                    ) from e
            logger.warning("Network error in enforce_policy (fail_open): %s", e)
            return {"allowed": True, "source": "fail_open_network_error"}

        # 4. Record success in circuit breaker
        self._cb_record_success()

        # 5. Handle response codes
        if resp.status_code == 403:
            try:
                body = resp.json()
            except ValueError:
                raise ExeclaveAuthError("Policy enforcement denied (unable to parse response)")
            if not body.get("allowed", True):
                raise ValidatorDeniedError.from_violations(body.get("violations", []))
            raise ExeclaveAuthError("Insufficient permissions")

        if resp.status_code == 202:
            # Never cache require_approval
            body = resp.json()
            approval_request_id = body.get("approvalRequestId")
            if approval_request_id:
                return self._poll_approval_decision(approval_request_id)

        if resp.status_code == 401:
            raise ExeclaveAuthError("Invalid API key or insufficient permissions")

        if resp.status_code == 402:
            quota_error = self._quota_error_from_response(resp)
            self._set_quota_exceeded(quota_error)
            if self.plan_limit_behavior == "fail_open":
                logger.warning(
                    "Plan limit exceeded for %s (%d/%d) — continuing unmonitored",
                    quota_error.resource, quota_error.current, quota_error.max,
                )
                return {
                    "allowed": True,
                    "warnings": [{
                        "policyId": "plan_limit",
                        "policyName": "Plan Limit",
                        "policyType": "plan_limit",
                        "message": str(quota_error),
                        "enforcementMode": "warn",
                    }],
                }
            raise PlanLimitExceededError(
                resource=quota_error.resource,
                current=quota_error.current,
                max=quota_error.max,
                message=str(quota_error),
            )

        if not resp.ok:
            msg = f"API request failed ({resp.status_code})"
            try:
                err = resp.json()
                if "error" in err:
                    msg += f": {err['error'].get('message', '')}"
            except ValueError:
                msg += f": {resp.text[:200]}"
            raise ExeclaveError(msg)

        # 6. Cache the successful result
        result = resp.json()
        self._cache_set(cache_key, result)
        return result

    def _poll_approval_decision(self, approval_request_id: str) -> dict:
        timeout_seconds = 30 * 60
        poll_interval_seconds = 5
        started_at = time.time()

        while time.time() - started_at < timeout_seconds:
            try:
                resp = self._session.request(
                    "GET",
                    f"{self.base_url}{self._api_path(f'/approvals/{approval_request_id}')}",
                    timeout=30,
                )
            except requests.RequestException as e:
                raise ExeclaveError(f"Network error: {e}") from e

            if not resp.ok:
                raise ExeclaveError(f"Approval polling failed ({resp.status_code})")

            body = resp.json()
            approval = body.get("data", {})
            status = approval.get("status")

            if status == "approved":
                return {"allowed": True, "approvalRequestId": approval_request_id}
            if status == "denied":
                raise PolicyDeniedError(approval_request_id, approval.get("decisionReason"))
            if status == "expired":
                raise ApprovalTimeoutError(approval_request_id)

            time.sleep(poll_interval_seconds)

        raise ApprovalTimeoutError(approval_request_id)

    def authorize_agent_call(
        self,
        caller_agent_id: str,
        callee_agent_id: str,
        action: str,
    ) -> dict:
        """
        Check whether *caller* is authorized to invoke *callee* for *action*.

        Raises ``ExeclaveAuthError`` (403) when the grant does not exist.
        Returns the grant record on success.
        """
        return self._request(
            "POST",
            self._api_path("/agents/authorize"),
            json={
                "callerAgentId": caller_agent_id,
                "calleeAgentId": callee_agent_id,
                "action": action,
            },
        )

    def discover_agents(self, capability: str | None = None) -> list[dict]:
        """
        Discover agents available to the organization, optionally filtered by
        *capability*.

        Returns a list of agent dicts with ``id``, ``agentId``, ``name``,
        and ``capabilities``.
        """
        path = self._api_path("/agents/discover")
        if capability:
            path += f"?capability={capability}"
        result = self._request("GET", path)
        return result.get("data", result) if isinstance(result, dict) else result

    def check_usage(self) -> dict:
        """Return current plan usage and limits from the billing usage endpoint."""
        result = self._request("GET", self._api_path("/billing/usage"))
        data = result.get("data", result) if isinstance(result, dict) else {}

        usage_block = data.get("usage") if isinstance(data, dict) else None

        def pick_usage(resource: str) -> dict:
            if isinstance(usage_block, dict) and isinstance(usage_block.get(resource), dict):
                bucket = usage_block.get(resource, {})
                return {
                    "current": self._to_int(bucket.get("current", 0), 0),
                    "max": self._to_int(bucket.get("max", 0), 0),
                }

            fallback = data.get(resource, {}) if isinstance(data, dict) else {}
            return {
                "current": self._to_int(fallback.get("current", 0), 0),
                "max": self._to_int(fallback.get("max", 0), 0),
            }

        return {
            "plan": data.get("plan", "unknown") if isinstance(data, dict) else "unknown",
            "agents": pick_usage("agents"),
            "traces": pick_usage("traces"),
            "users": pick_usage("users"),
            "policies": pick_usage("policies"),
            "upgradeUrl": (
                data.get("upgradeUrl")
                if isinstance(data, dict)
                else None
            )
            or "https://www.execlave.com/dashboard/billing",
        }

    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Flush all buffered traces to the Execlave API.
        Call this before shutdown (or register with ``atexit``).
        """
        self._do_flush(raise_quota_error=False)

    def shutdown(self) -> None:
        """Flush and shut down the SDK."""
        self._state = _STATE_SHUTDOWN
        self.flush()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_event.set()
            self._flush_thread.join(timeout=5)
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=3)
        # Disconnect WebSocket
        if self._sio:
            try:
                self._sio.disconnect()
            except Exception:
                pass
            self._sio = None
        if self._otel_exporter:
            self._otel_exporter.shutdown()
        logger.debug("Execlave SDK shut down")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an authenticated HTTP request to the Execlave API."""
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.request(method, url, timeout=30, **kwargs)
        except requests.RequestException as e:
            raise ExeclaveError(f"Network error: {e}") from e

        if resp.status_code == 402:
            quota_error = self._quota_error_from_response(resp)
            self._set_quota_exceeded(quota_error)
            raise quota_error

        if resp.status_code in (401, 403):
            raise ExeclaveAuthError("Invalid API key or insufficient permissions")

        if not resp.ok:
            msg = f"API request failed ({resp.status_code})"
            try:
                err = resp.json()
                if "error" in err:
                    msg += f": {err['error'].get('message', '')}"
            except ValueError:
                msg += f": {resp.text[:200]}"
            raise ExeclaveError(msg)

        return resp.json()

    def _buffer_trace(self, payload: dict) -> None:
        """Add a trace payload to the in-memory buffer, applying privacy/injection pre-processing."""
        self._raise_if_quota_exceeded()

        # Client-side PII scrubbing
        if self.privacy.get("enabled", False):
            payload = self._apply_privacy(payload)

        # Client-side injection scanning
        if self.enable_injection_scan:
            injection = self._scan_injection(payload)
            if injection["detected"]:
                if not payload.get("metadata"):
                    payload["metadata"] = {}
                payload["metadata"]["injection_scan"] = injection

        with self._buffer_lock:
            self._buffer.append(payload)
            logger.debug("Buffered trace %s (buffer size: %d)", payload.get("traceId"), len(self._buffer))

        # If sync mode or buffer full, flush immediately
        if not self.async_mode or len(self._buffer) >= self.batch_size:
            self._do_flush(raise_quota_error=True)

    def _do_flush(self, raise_quota_error: bool = False) -> None:
        """Flush buffered traces to the API."""
        with self._buffer_lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        # Send in batches of batch_size
        for i in range(0, len(batch), self.batch_size):
            chunk = batch[i : i + self.batch_size]

            # Route through OTel exporter when in OTLP mode
            if self.mode == "otlp" and self._otel_exporter:
                try:
                    self._otel_exporter.export_traces(chunk)
                    logger.debug("Exported %d traces via OTLP", len(chunk))
                except Exception as e:
                    logger.warning("Failed to export %d traces via OTLP: %s", len(chunk), e)
                continue

            retries = 0
            while retries < 3:
                try:
                    self._request("POST", self._api_path("/traces/ingest"), json={"traces": chunk})
                    logger.debug("Flushed %d traces", len(chunk))
                    break
                except QuotaExceededError as e:
                    self._set_quota_exceeded(e)
                    if raise_quota_error:
                        raise
                    logger.warning("Trace quota exceeded while flushing %d traces: %s", len(chunk), e)
                    break
                except ExeclaveError as e:
                    retries += 1
                    if retries >= 3:
                        logger.warning("Failed to flush %d traces after 3 retries: %s", len(chunk), e)
                    else:
                        time.sleep(2 ** retries * 0.5)  # exponential backoff

    def _flush_loop(self) -> None:
        """Background thread: periodically flush the buffer."""
        while self._state != _STATE_SHUTDOWN:
            self._flush_event.wait(timeout=self.flush_interval_seconds)
            if self._state == _STATE_SHUTDOWN:
                break
            self._do_flush()

    def _connect_websocket(self) -> None:
        """Connect to Socket.IO /sdk namespace for real-time agent control."""
        try:
            sio = _socketio_mod.Client(reconnection=True, reconnection_delay=2)  # type: ignore[union-attr]

            @sio.on('agent.status_updated', namespace='/sdk')
            def on_status_update(data: dict) -> None:
                agent_id = data.get('agentId')
                new_status = data.get('status')

                if agent_id and agent_id in self._agents:
                    agent = self._agents[agent_id]

                    if new_status == 'paused' and self._state == _STATE_ACTIVE:
                        self._state = _STATE_PAUSED
                        agent.status = 'paused'
                        logger.warning(
                            'Agent %s PAUSED via WebSocket kill switch (reason: %s)',
                            agent_id, data.get('reason', 'none'),
                        )
                    elif new_status == 'active' and self._state == _STATE_PAUSED:
                        self._state = _STATE_ACTIVE
                        agent.status = 'active'
                        logger.info('Agent %s RESUMED via WebSocket', agent_id)

            @sio.event
            def connect() -> None:
                logger.info('WebSocket control channel connected')

            @sio.event
            def connect_error(data: Any) -> None:
                logger.debug('WebSocket connect error: %s — falling back to HTTP polling', data)

            # Connect in background thread
            def _ws_connect() -> None:
                try:
                    sio.connect(
                        self.base_url,
                        namespaces=['/sdk'],
                        auth={'apiKey': self.api_key},
                        transports=['websocket'],
                    )
                except Exception as e:
                    logger.debug('WebSocket connection failed: %s — HTTP polling continues', e)

            ws_thread = threading.Thread(target=_ws_connect, daemon=True, name='Execlave-ws')
            ws_thread.start()
            self._sio = sio

        except Exception as e:
            logger.debug('WebSocket setup failed: %s — HTTP polling continues', e)

    def _status_poll_loop(self) -> None:
        """Background thread: poll agent status for kill-switch / control channel."""
        while self._state != _STATE_SHUTDOWN:
            time.sleep(self._poll_interval_seconds)
            if self._state == _STATE_SHUTDOWN:
                break
            for agent_id, agent in list(self._agents.items()):
                try:
                    resp = self._request("GET", self._api_path(f"/agents/{agent.id}/status-poll"))
                    data = resp.get("data", {})
                    new_status = data.get("status", "active")

                    if new_status == "paused" and self._state == _STATE_ACTIVE:
                        self._state = _STATE_PAUSED
                        agent.status = "paused"
                        logger.warning(
                            "Agent %s has been PAUSED via kill switch", agent_id
                        )
                    elif new_status == "active" and self._state == _STATE_PAUSED:
                        self._state = _STATE_ACTIVE
                        agent.status = "active"
                        logger.info("Agent %s has been RESUMED", agent_id)

                    agent.status = new_status
                except ExeclaveError:
                    logger.debug("Status poll failed for agent %s", agent_id)
                except Exception:
                    logger.debug("Unexpected error polling agent %s", agent_id, exc_info=True)

    def check_agent_status(self, agent_id: str | None = None) -> str:
        """
        Check the current status of a registered agent.

        Returns 'active', 'paused', or 'error'.
        """
        agent = None
        if agent_id and agent_id in self._agents:
            agent = self._agents[agent_id]
        elif self._agents:
            agent = next(iter(self._agents.values()))

        if not agent:
            return "unknown"

        try:
            resp = self._request("GET", self._api_path(f"/agents/{agent.id}/status-poll"))
            status = resp.get("data", {}).get("status", "active")
            agent.status = status
            return status
        except ExeclaveError:
            return "error"

    # ------------------------------------------------------------------
    # Privacy & Injection Scanning
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_pii(value: str) -> str:
        """SHA-256 hash a PII value for redacted storage."""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _to_text(data: Any) -> str:
        """Convert arbitrary data to searchable text."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return " ".join(str(v) for v in data.values())
        if isinstance(data, (list, tuple)):
            return " ".join(str(v) for v in data)
        return str(data)

    @staticmethod
    def _scrub_text(text: str) -> str:
        """Replace PII in text with type-labeled placeholders."""
        if not text:
            return text or ""
        result = text
        for pii_type, pattern in _PII_PATTERNS.items():
            result = pattern.sub(f"[{pii_type.upper()}_REDACTED]", result)
        return result

    def _apply_privacy(self, payload: dict) -> dict:
        """
        Scrub PII from input/output fields of a trace payload.

        Privacy config:
            privacy = {
                "enabled": True,
                "scrub_fields": ["input", "output"],  # fields to scrub
                "hash_pii": True,  # include hashed PII in metadata
            }
        """
        fields_to_scrub = self.privacy.get("scrub_fields", ["input", "output"])
        hash_pii = self.privacy.get("hash_pii", True)

        pii_summary: dict = {}

        for field in fields_to_scrub:
            value = payload.get(field)
            if not value:
                continue
            text = self._to_text(value)
            if not text:
                continue

            # Detect PII
            for pii_type, pattern in _PII_PATTERNS.items():
                matches = pattern.findall(text)
                if matches:
                    if pii_type not in pii_summary:
                        pii_summary[pii_type] = {"count": 0, "hashes": []}
                    pii_summary[pii_type]["count"] += len(matches)
                    if hash_pii:
                        pii_summary[pii_type]["hashes"].extend(
                            self._hash_pii(m) for m in matches
                        )

            # Replace PII with placeholders
            if isinstance(value, str):
                payload[field] = self._scrub_text(value)
            elif isinstance(value, dict):
                payload[field] = {
                    k: self._scrub_text(str(v)) if isinstance(v, str) else v
                    for k, v in value.items()
                }

        if pii_summary:
            if not payload.get("metadata"):
                payload["metadata"] = {}
            payload["metadata"]["pii_detected"] = pii_summary
            payload["metadata"]["pii_scrubbed"] = True

        return payload

    def _scan_injection(self, payload: dict) -> dict:
        """
        Scan the input field for prompt injection patterns.

        Returns: { "detected": bool, "risk_level": str, "patterns_matched": list }
        """
        text = self._to_text(payload.get("input"))
        if not text:
            return {"detected": False, "risk_level": "none", "patterns_matched": []}

        matched: list[str] = []
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                matched.append(pattern.pattern)

        count = len(matched)
        if count == 0:
            risk = "none"
        elif count == 1:
            risk = "low"
        elif count <= 3:
            risk = "medium"
        elif count <= 5:
            risk = "high"
        else:
            risk = "critical"

        return {
            "detected": count > 0,
            "risk_level": risk,
            "patterns_matched": matched,
        }

    # ------------------------------------------------------------------
    # Backward compat — expose old methods
    # ------------------------------------------------------------------

    def _register_agent_compat(self, **kwargs) -> dict:
        """Legacy method — returns raw dict instead of Agent object."""
        agent = self.register_agent(**kwargs)
        return agent._data


# Backward-compatible alias
ExeclaveClient = Execlave

