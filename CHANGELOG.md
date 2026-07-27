# Changelog

All notable changes to `execlave-sdk` (Python) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.0] - 2026-07-27

### Fixed

- **Certificate binding no longer stops at a hand-picked field list.** The
  OpenAI Chat adapter sealed only `{"model": ..., "messages": ...}` into the
  approval certificate's action-binding digest — `tools`, `tool_choice`,
  `temperature`, and `response_format` could drift from what a human approved
  without ever invalidating the certificate. It now seals the **entire**
  request kwargs via a new `seal_full_request()` helper (exclusion-list, not
  inclusion-list: every field is covered by default, a field is only dropped
  by explicit, reviewed exclusion).
- **`seal_for_metadata` no longer collapses an entire payload to a placeholder
  over one bad field.** Previously, a single non-serializable value anywhere
  in a metadata dict (a circular reference, a callable, a custom class
  instance) caused the _whole_ sealed value to fall back to
  `{"unserializable": True}` — identical at issuance and at re-execution, so
  certificate verification silently became a no-op for every other field in
  that payload. Sanitization is now per-field: only the offending field is
  replaced with an explicit `[unserializable:...]` marker.
- **CrewAI's tool-call enforcement previously called `exe.enforce_policy`
  directly with no `metadata` at all** — a `require_approval` certificate for
  a CrewAI tool call bound to nothing but a `str()`'d, unbounded input
  string, not the full tool input. Migrated to `enforce_policy_bound` with
  the full tool input sealed, in line with every other adapter.
- Added a structural contract test that fails the build if any adapter passes
  a hand-built, multi-field dict literal to `seal_metadata`/
  `seal_metadata_entry` instead of a single already-complete value or
  `seal_full_request(...)` — closing off the exact pattern that caused the
  OpenAI Chat gap from recurring in a future adapter.

## [1.4.0] - 2026-06-08

### Added

- **Optional HMAC request signing.** New `sign_requests` client argument (default
  `False`). When enabled, every request body is signed with HMAC-SHA256 keyed by
  the API key and sent with `X-Execlave-Timestamp` and `X-Execlave-Signature`
  (`sha256=<hex>`) headers. Implemented as a `requests` session auth callable, so
  it covers every call from a single point and signs the exact serialized body
  bytes (`${timestamp}.${body}`) the server verifies. Defense-in-depth on top of
  TLS + API-key auth; opt-in and fully backward-compatible (unsigned requests are
  unaffected when the server has not made signing mandatory).

## [1.3.0] - 2026-06-03

### Added

- **Agent identity stamping.** New `stamp_identity` client argument (default
  `False`). When enabled, the client issues and caches a short-lived RS256
  `exe_agt_` credential per agent and attaches it to each trace on ingest
  (`agentCredential` field), so the platform can cryptographically stamp who
  produced the trace. Best-effort and non-breaking: if a credential cannot be
  issued, traces are still sent unstamped — stamping never blocks or drops ingest.
- **MCP tool-integrity surfaces.** Optional, backwards-compatible additions for
  MCP tool-supply-chain governance:
  - `tool_descriptor(server, tool, descriptor, description=None)` — computes the
    stable SHA-256 descriptor hash (canonical, key-order independent) used to pin
    and diff a tool.
  - `report_tool_baseline(agent_id, descriptors, reason="manual")` — pins the
    approved set of `(server, tool, descriptorHash)` tuples for an agent; re-pin
    with `reason="baseline_update"` after a reviewed tool update.
  - `enforce_policy()` accepts an optional `tool_descriptors` argument that is
    diffed against the agent's pinned baseline at runtime.
  - New `ToolIntegrityError` (subclass of `PolicyBlockedError`) is raised when an
    enforcement is denied by a `tool_integrity` policy. Callers that do not use
    these fields are unaffected.

### Changed

- No telemetry. The SDK does not phone home, emit anonymous usage events, or
  fetch remote configuration. Every network call goes to the Execlave backend
  URL configured by the caller.

## [1.2.1] - 2026-05-29

### Added

- **AI Agent Management Platform (AMP) surfaces.** Optional, backwards-compatible
  additions for the governance features:
  - `register_agent()` accepts an optional `autonomy_level`
    (`observe` | `advise` | `act_with_approval` | `autonomous`) that maps the
    agent onto a tiered-governance template.
  - New `report_agent_metadata()` method that records a version snapshot in the
    agent registry (`version_label` / `git_commit` / `deployed_at` / `notes` /
    `activate`) — call it from a deploy pipeline to build version history for
    diff/rollback.
  - Available on both `ExeclaveClient` and `AsyncExeclaveClient`. Callers that do
    not set these fields are unaffected.

## [1.2.0] - 2026-05-28

### Added

- **Framework auto-instrumentation modules** (opt-in via `execlave.integrations`,
  each with its own install extra): LangChain (`[langchain]`), OpenAI Agents SDK
  (`[openai-agents]`), CrewAI (`[crewai]`), LlamaIndex (`[llamaindex]`), Model
  Context Protocol (`[mcp]`), OpenAI Chat Completions (`[openai]`), and AutoGen
  (`[autogen]`). Each routes the framework's tool calls / completions through
  policy enforcement and trace ingestion without changing the host app's call
  sites.

## [1.1.5] - 2026-05-05

### Added

- `ValidatorDeniedError` (extends `PolicyBlockedError`) for programmatic handling
  of denials originating from a Custom Validator (BYOV). The `from_violations()`
  factory returns a `ValidatorDeniedError` when any violation is validator-sourced
  and a plain `PolicyBlockedError` otherwise, so existing `except` sites keep working.

### Fixed

- `http://api.execlave.com` is normalized to `https://api.execlave.com` so
  POST-based calls are not downgraded to GET by an HTTP-to-HTTPS redirect.
- `enforce_policy()` now sends the client environment by default, matching
  `register_agent()` and avoiding accidental production-policy enforcement from
  development SDK clients.

## [1.1.4] - 2026-05-05

### Fixed

- `register_agent()` now handles agent responses wrapped as `{ "data": [...] }`
  by selecting the matching `agentId` instead of passing the list into `Agent`.
  Malformed list responses now raise `ExeclaveError` with a clear response-shape
  message instead of surfacing `AttributeError: 'list' object has no attribute 'get'`.

## [1.0.0] — 2026-04

### Added

- Initial public release of `execlave-sdk` on PyPI.
- `ExeclaveClient` class with `enforce()`, `ingest_trace()`, and
  `register_agent()` methods.
- Async variants via `AsyncExeclaveClient` (httpx-based).
- Type hints throughout the public API, verified with `mypy --strict`.
- PEP 621 `pyproject.toml` with a hatchling build backend.
- Python 3.10+ supported; tested against 3.10, 3.11, and 3.12.
- Support for API keys via the `exe_` / `exe_test_` prefix.

### Security

- TLS certificate verification is always enabled. Callers who need to
  target a self-signed local environment must set `verify=False` on the
  client explicitly and are warned on construction.
- The SDK refuses to accept API keys that do not match the `exe_*` prefix,
  preventing accidental use of unrelated credentials.

[Unreleased]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.3.0...HEAD
[1.3.0]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.2.1...sdk-python-v1.3.0
[1.2.1]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.2.0...sdk-python-v1.2.1
[1.2.0]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.1.5...sdk-python-v1.2.0
[1.1.5]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.1.4...sdk-python-v1.1.5
[1.1.4]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.1.3...sdk-python-v1.1.4
[1.0.0]: https://github.com/rishitmavani/agentguard/releases/tag/sdk-python-v1.0.0
