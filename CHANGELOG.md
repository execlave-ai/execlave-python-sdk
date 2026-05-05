# Changelog

All notable changes to `execlave-sdk` (Python) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- No telemetry. The SDK does not phone home, emit anonymous usage events, or
  fetch remote configuration. Every network call goes to the Execlave backend
  URL configured by the caller.

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

[Unreleased]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.1.4...HEAD
[1.1.4]: https://github.com/rishitmavani/agentguard/compare/sdk-python-v1.1.3...sdk-python-v1.1.4
[1.0.0]: https://github.com/rishitmavani/agentguard/releases/tag/sdk-python-v1.0.0
