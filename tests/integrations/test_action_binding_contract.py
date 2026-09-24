"""Structural contract: every adapter must use the shared action-binding
helpers instead of redefining its own truncate/seal logic, AND must call
``enforce_policy`` only through ``enforce_policy_bound`` (which enforces the
``SealedMetadata`` runtime check) rather than ``exe.enforce_policy`` directly.

This is not a compile-time guarantee -- Python has no compiler to reject a
bare ``dict`` at an adapter's call site the way the sdk-js counterpart does.
What IS guaranteed, automatically and on every run: there is only ONE
implementation of the truncate/seal logic in the package (no adapter
silently reintroduces its own copy); every adapter known to need the split
imports it; and every such adapter routes through ``enforce_policy_bound``,
which raises ``MetadataContractError`` at runtime for any metadata that
isn't a `SealedMetadata` instance. A genuinely new adapter file must be
triaged into ``_KNOWN_ADAPTERS_REQUIRING_SPLIT`` (or explicitly excluded) or
this test fails loudly.
"""

from __future__ import annotations

import os
import re

import execlave.integrations as integrations_pkg

_INTEGRATIONS_DIR = os.path.dirname(integrations_pkg.__file__)

_EXCLUDED = {"_action_binding.py", "__init__.py"}

# Adapters that split truncated-input from sealed-metadata and therefore
# must import the shared helpers. `crewai.py` used to call
# `exe.enforce_policy` directly with no metadata at all (no duplicate-logic
# gap because there was no sealing to duplicate) -- migrated to
# enforce_policy_bound so a require_approval certificate for a tool call
# binds to the full tool input, not just a str()'d, unbounded input string.
_KNOWN_ADAPTERS_REQUIRING_SPLIT = {
    "mcp.py",
    "openai_chat.py",
    "autogen.py",
    "langchain.py",
    "llamaindex.py",
    "openai_agents.py",
    "crewai.py",
}

_FORBIDDEN_LOCAL_DEFS = ("def _safe_str(", "def _safe_metadata_value(")


def _integration_files() -> list[str]:
    return sorted(
        f
        for f in os.listdir(_INTEGRATIONS_DIR)
        if f.endswith(".py") and f not in _EXCLUDED and not f.startswith("__pycache__")
    )


def test_no_adapter_redefines_the_shared_helpers_locally() -> None:
    offenders = []
    for filename in _integration_files():
        path = os.path.join(_INTEGRATIONS_DIR, filename)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        for forbidden in _FORBIDDEN_LOCAL_DEFS:
            if forbidden in content:
                offenders.append(f"{filename}: {forbidden}")
    assert offenders == [], (
        "Adapter(s) locally redefine truncate/seal logic instead of importing "
        f"from _action_binding: {offenders}"
    )


def test_known_adapters_import_the_shared_module() -> None:
    missing = []
    for filename in _KNOWN_ADAPTERS_REQUIRING_SPLIT:
        path = os.path.join(_INTEGRATIONS_DIR, filename)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        if "from ._action_binding import" not in content:
            missing.append(filename)
    assert missing == [], f"Adapter(s) do not import the shared _action_binding module: {missing}"


def test_known_adapters_call_enforce_policy_bound_never_direct() -> None:
    direct_call = re.compile(r"\.enforce_policy\s*\(")
    offenders = []
    for filename in _KNOWN_ADAPTERS_REQUIRING_SPLIT:
        path = os.path.join(_INTEGRATIONS_DIR, filename)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        calls_bound = "enforce_policy_bound(" in content
        calls_direct = bool(direct_call.search(content))
        if not calls_bound or calls_direct:
            offenders.append(filename)
    assert offenders == [], (
        f"Adapter(s) must call enforce_policy_bound and never exe.enforce_policy directly: {offenders}"
    )


_MULTI_FIELD_LITERAL = re.compile(
    r"seal_metadata(?:_entry)?\(\s*(?:[\"'][^\"']+[\"']\s*,\s*)?\{[^{}]*:[^{}]*,[^{}]*:"
)


def test_no_adapter_passes_a_hand_built_multi_field_dict_literal_to_seal_metadata() -> None:
    # Two or more `key:` pairs typed directly inside the seal call's braces
    # is the inclusion-list anti-pattern this repo's incidents keep tracing
    # back to: a hand-picked literal silently stops covering a field the day
    # a provider adds one and the adapter isn't updated. A single value
    # already holding the complete data, or a call to seal_full_request(...),
    # is required instead -- this is exactly the bug that shipped in
    # openai_chat.py's old `seal_metadata({"model": ..., "messages": ...})`.
    offenders = []
    for filename in _integration_files():
        path = os.path.join(_INTEGRATIONS_DIR, filename)
        with open(path, "r", encoding="utf-8") as fh:
            content = fh.read()
        if _MULTI_FIELD_LITERAL.search(content):
            offenders.append(filename)
    assert offenders == [], (
        "Adapter(s) pass a hand-built multi-field dict literal to seal_metadata/"
        f"seal_metadata_entry instead of a single value or seal_full_request(...): {offenders}"
    )


def test_integrations_directory_is_fully_triaged() -> None:
    known = _KNOWN_ADAPTERS_REQUIRING_SPLIT
    actual = set(_integration_files())
    untriaged = actual - known
    assert untriaged == set(), (
        "New/untriaged file(s) in execlave/integrations/ — decide whether they "
        f"need the truncate/seal split and add them to this test: {untriaged}"
    )
