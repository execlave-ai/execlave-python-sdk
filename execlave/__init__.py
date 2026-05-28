"""
Execlave Python SDK

Official SDK for integrating AI agents with the Execlave governance platform.
Provides tracing, prompt management, and governance capabilities.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("execlave-sdk")
except PackageNotFoundError:  # running from source without install metadata
    __version__ = "0.0.0+local"

from .client import Execlave, ExeclaveClient
from .agent import Agent
from .trace import Trace
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
from .connectors import run_openai_chat, run_langchain

__all__ = [
    "__version__",
    "Execlave",
    "ExeclaveClient",  # backward compat alias
    "Agent",
    "Trace",
    "ExeclaveError",
    "ExeclaveAuthError",
    "AgentPausedError",
    "PolicyBlockedError",
    "ValidatorDeniedError",
    "PolicyDeniedError",
    "ApprovalTimeoutError",
    "EnforcementUnavailableError",
    "QuotaExceededError",
    "PlanLimitExceededError",
    "run_openai_chat",
    "run_langchain",
]
