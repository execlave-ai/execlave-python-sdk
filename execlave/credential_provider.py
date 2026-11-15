"""Workload identity federation: turns identity provider tokens into the
short-lived ``exe_agt_`` API credential the SDK authenticates with."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable

import requests  # type: ignore[import-untyped]

from .errors import ExeclaveAuthError, ExeclaveError

_DEFAULT_REFRESH_BEFORE_SECONDS = 60.0


class CredentialExchangeUnavailable(requests.exceptions.ConnectionError):
    """No credential could be had: the identity provider callable failed, or
    the exchange endpoint could not answer (rate limited or a server error).

    A ``requests`` connection error on purpose: every call site already treats
    those as an outage (``enforcement_on_outage``, retries), which is what an
    unavailable exchange is. A refused token is an ``ExeclaveAuthError``
    instead, so a withdrawn identity is never mistaken for an outage.
    """


def _parse_expiry(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class FederatedCredentialSource:
    """Caches the exchanged credential and exchanges again shortly before it
    expires.

    Thread-safe: callers that find the cache stale wait on one lock, so a burst
    of calls at startup or at expiry costs one identity provider token and one
    exchange. A failed refresh is not hidden: callers get the error, and the
    next call tries again. A credential about to expire is never handed out.
    """

    def __init__(
        self,
        provider: Callable[[], str],
        exchange: Callable[[str], Any],
        refresh_before_seconds: float = _DEFAULT_REFRESH_BEFORE_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        self._provider = provider
        self._exchange = exchange
        self._refresh_before = refresh_before_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._current: dict[str, str] | None = None
        self._expires_at = 0.0
        self._bound_agent_id: str | None = None

    @property
    def bound_agent_id(self) -> str | None:
        """UUID of the agent the credentials act for, once an exchange succeeded."""
        return self._bound_agent_id

    def get(self) -> dict[str, str]:
        """A credential (``credential``, ``expiresAt``, ``agentId``) with more than
        ``refresh_before_seconds`` left, exchanging a new one if needed."""
        with self._lock:
            if self._current and self._expires_at - self._clock() > self._refresh_before:
                return self._current
            return self._refresh()

    def _refresh(self) -> dict[str, str]:
        try:
            idp_token = self._provider()
        except Exception as e:  # noqa: BLE001 - any provider failure is an outage
            # Only the exception type: provider errors can quote request details.
            raise CredentialExchangeUnavailable(
                f"credential_provider failed ({type(e).__name__})"
            ) from e
        if not isinstance(idp_token, str) or not idp_token:
            raise ExeclaveError("credential_provider returned no token")

        exchanged = self._exchange(idp_token)
        if not isinstance(exchanged, dict):
            raise ExeclaveError("Credential exchange returned an unexpected response")
        credential = exchanged.get("credential")
        agent_id = exchanged.get("agentId")
        expires_at = _parse_expiry(exchanged.get("expiresAt"))
        if (
            not isinstance(credential, str)
            or not credential.startswith("exe_agt_")
            or not isinstance(agent_id, str)
            or not agent_id
            or expires_at is None
        ):
            raise ExeclaveError("Credential exchange returned an unexpected response")

        # Everything registered locally points at the first agent. If the
        # identity provider subject is now bound to a different agent, carrying
        # on would send that agent's credential with the old agent's ID.
        if self._bound_agent_id and agent_id != self._bound_agent_id:
            raise ExeclaveAuthError(
                f"Federated identity is now bound to agent {agent_id}, not "
                f"{self._bound_agent_id}; create a new Execlave client"
            )
        self._bound_agent_id = agent_id
        self._current = {
            "credential": credential,
            "expiresAt": exchanged["expiresAt"],
            "agentId": agent_id,
        }
        self._expires_at = expires_at
        return self._current
