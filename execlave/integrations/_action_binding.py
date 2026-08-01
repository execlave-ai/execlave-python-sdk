"""
Shared action-context binding helpers for framework adapters.

Single source of truth for the truncate-for-classifier / seal-for-metadata
split every adapter uses when enforcing a policy on a value that may be
large or structured (tool arguments, chain inputs, a message array, ...):

  - ``input`` sent to ``enforce_policy`` is a BOUNDED string -- cheap to run
    a classifier/heuristic over, not a governance-complete record.
  - ``metadata`` must carry the FULL, untruncated, JSON-safe value -- that
    is what the server's certificate digest actually covers in whole.

Before this module existed, seven adapters each hand-rolled their own copy
of this split. A prior real incident: several of those copies truncated the
value into ``input`` and sealed nothing else, so the certificate bound to a
4000-char summary instead of the actual payload -- a material field could
change past the truncation point without invalidating the certificate.
Consolidating to one module means there is exactly one place this logic can
be gotten right or wrong, instead of seven, and
``test_action_binding_contract.py`` fails the test suite if any adapter file
reintroduces a local copy instead of importing from here.

Adapters MUST call ``enforce_policy`` through ``enforce_policy_bound`` below,
not ``exe.enforce_policy`` directly -- it requires ``metadata`` to be a
``SealedMetadata`` instance, producible only by ``seal_for_metadata`` /
``seal_metadata_entry`` / ``seal_metadata``. Python has no compiler to reject
a bare ``dict`` at the call site the way TypeScript's ``sdk-js`` counterpart
does, so this is checked with ``isinstance`` at runtime on every call instead
-- still a single enforced boundary rather than trusting each adapter to
remember the convention, but a runtime check rather than a compile error. The
public ``Execlave.enforce_policy`` API is untouched and still accepts a plain
``dict`` -- that boundary is a direct customer supplying their own data, a
different trust relationship than an adapter translating a provider request.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, TYPE_CHECKING

from ..errors import MetadataContractError

if TYPE_CHECKING:
    from ..client import Execlave

_DEFAULT_LIMIT = 4000


class SealedMetadata(dict):
    """A ``metadata`` dict that has passed through ``seal_for_metadata`` /
    ``seal_metadata_entry`` / ``seal_metadata``. A distinct subclass (rather
    than a plain ``dict``) so ``enforce_policy_bound`` can tell a sealed value
    apart from a hand-built dict literal via ``isinstance``."""


def truncate_for_classifier(value: Any, limit: int = _DEFAULT_LIMIT) -> str | None:
    """Truncate a value to a bounded string for the ``input`` field. NOT
    governance-complete on its own -- pair with ``seal_for_metadata`` for
    anything that should be part of the certificate's binding in full."""
    if value is None:
        return None
    try:
        s = value if isinstance(value, str) else str(value)
    except Exception:  # pragma: no cover
        return None
    return s[:limit]


def _safe_serialize(value: Any, seen: frozenset[int] = frozenset()) -> Any:
    """Recursively sanitize a value into something JSON-safe, replacing only
    the individual fields that cannot serialize instead of the whole value.

    A prior version of ``seal_for_metadata`` caught ``json.dumps`` failure at
    the TOP level only: one bad field anywhere in a large dict (bytes, a
    set, a custom class instance a provider SDK attaches) collapsed the
    ENTIRE sealed value to ``{"unserializable": True}``. The certificate
    digest still matched between issuance and re-execution -- both times
    hashing the same placeholder -- so verification silently became a no-op
    for every OTHER field in that payload, not just the one that couldn't
    serialize. Walking key-by-key/element-by-element means a single bad
    field gets a marker in its place while the rest of the structure still
    binds the certificate to real content.

    ``seen`` is passed as a new frozenset per recursion branch (never
    mutated) so a cycle back to a true ancestor is caught, while the same
    object referenced twice by unrelated siblings -- not a cycle -- is not
    falsely flagged.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return f"[unserializable:bytes:{len(value)}]"
    if isinstance(value, (list, tuple)):
        obj_id = id(value)
        if obj_id in seen:
            return "[unserializable:circular]"
        branch = seen | {obj_id}
        return [_safe_serialize(v, branch) for v in value]
    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            return "[unserializable:circular]"
        branch = seen | {obj_id}
        return {
            (k if isinstance(k, str) else str(k)): _safe_serialize(v, branch)
            for k, v in value.items()
        }
    if isinstance(value, (set, frozenset)):
        return f"[unserializable:{type(value).__name__}]"
    if callable(value):
        return "[unserializable:callable]"
    # Arbitrary class instance not covered above -- try a native round-trip;
    # anything json.dumps rejects becomes an explicit marker rather than
    # silently vanishing or collapsing the surrounding structure.
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return f"[unserializable:{type(value).__name__}]"


def seal_for_metadata(value: Any) -> Any:
    """Validate a value is JSON-safe before it is embedded in ``metadata``,
    replacing any individual non-serializable field with a marker rather
    than falling back for the whole value -- see ``_safe_serialize`` for why
    per-field, not whole-object. A raw, unvalidated object here (a circular
    reference, a custom class instance, ...) would otherwise raise inside
    ``json.dumps`` at the HTTP layer -- and ``json.dumps`` raises
    ``TypeError`` for non-JSON-native types, which ``requests`` does NOT wrap
    into a ``RequestException`` the way it does a circular-reference
    ``ValueError``. An uncaught ``TypeError`` propagates past
    ``enforce_policy`` entirely and is swallowed by the integration's generic
    ``except Exception`` handler as "non-fatal" -- silently allowing the
    call, reproducibly, for that payload shape."""
    if value is None:
        return None
    return _safe_serialize(value)


def seal_full_request(request: dict | None, exclude: tuple[str, ...] = ()) -> SealedMetadata:
    """Seal the ACTUAL raw request dict/kwargs an adapter is about to hand
    the provider SDK -- every key, by default, not a hand-picked subset.
    ``exclude`` strips named fields (transport-only concerns) and defaults
    to empty -- start maximal, add an exclusion only when a specific field
    demonstrably needs it.

    This exists because a hand-picked inclusion dict (``{"model": ...,
    "messages": ...}``) silently stops covering a field the day the
    provider SDK adds one and the adapter isn't updated -- the certificate
    binds to less than what actually executes, with no error, no test
    failure, nothing to notice. Sealing the whole request flips the
    default: a NEW provider field is covered automatically, and omitting
    one requires a deliberate, reviewable entry in ``exclude`` instead of an
    easy-to-forget addition to an inclusion list.
    """
    if not isinstance(request, dict):
        return SealedMetadata({})
    excluded = set(exclude)
    return SealedMetadata(
        {k: _safe_serialize(v) for k, v in request.items() if k not in excluded}
    )


def seal_metadata_entry(key: str, value: Any) -> SealedMetadata | None:
    """Convenience: seal ``value`` under ``key`` for a ``metadata`` dict, or
    ``None`` if there is nothing to seal, so callers can pass the result
    straight through -- ``metadata=seal_metadata_entry("toolArguments", args)``
    -- without an extra ``is not None`` check at every call site."""
    sealed = seal_for_metadata(value)
    return SealedMetadata({key: sealed}) if sealed is not None else None


def seal_metadata(value: dict) -> SealedMetadata:
    """Seal a hand-built metadata dict as a whole (e.g. ``{"model": ...,
    "messages": ...}`` where only some fields need sealing) -- for the one
    call site (``openai_chat.py``) that combines a sealed field with plain
    scalars rather than sealing a single field under ``seal_metadata_entry``."""
    sealed = seal_for_metadata(value)
    safe = sealed if isinstance(sealed, dict) else {"unserializable": True}
    return SealedMetadata(safe)


def enforce_policy_bound(
    exe: "Execlave",
    agent_id: str,
    input: str,
    *,
    tools: list[str] | None = None,
    metadata: SealedMetadata | None = None,
    **kwargs: Any,
) -> dict:
    """The only sanctioned way for an adapter to call ``enforce_policy``.
    Requires ``metadata`` to be a ``SealedMetadata`` instance -- raises
    ``MetadataContractError`` (fail-closed) for anything else, including a
    plain ``dict`` literal an adapter built by hand instead of sealing."""
    if metadata is not None and not isinstance(metadata, SealedMetadata):
        raise MetadataContractError(
            f"metadata for agent '{agent_id}' was not produced by "
            "seal_for_metadata/seal_metadata_entry/seal_metadata"
        )
    return exe.enforce_policy(agent_id, input, tools=tools, metadata=metadata, **kwargs)
