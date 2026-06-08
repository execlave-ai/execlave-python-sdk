# Contributing to `execlave-sdk`

Thanks for wanting to contribute! This project is the official Python SDK for the Execlave AI Governance Platform.

## Ground rules

1. **By contributing, you agree your work will be released under the MIT licence** (see `LICENSE`).
2. **Be kind.** See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
3. **Security issues go to `security@execlave.com`**, never to a public issue. See [SECURITY.md](SECURITY.md).

## Development setup

```bash
# Fork, then clone
git clone https://github.com/<you>/execlave-python-sdk
cd execlave-python-sdk

# Editable install with dev + test extras
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[test]"

# Run the full test suite
pytest

# Run a single file
pytest tests/test_client.py -v
```

Python 3.11+ is required. The project uses `flit` for packaging and `pytest` for testing.

## Project layout

```
execlave/
├── client.py              # Main Execlave client + enforce_policy
├── trace.py               # Trace class (chainable API)
├── instrumentation/       # Span + event helpers (shared across integrations)
├── integrations/
│   ├── langchain.py       # LangChain callback handler
│   ├── openai_agents.py   # OpenAI Agents SDK tracing processor
│   └── crewai.py          # CrewAI instrument_crew helper
├── otel.py                # OpenTelemetry bridge
└── connectors.py          # (Deprecated) run_langchain
tests/                     # pytest suite
```

## Pull request checklist

- [ ] Tests pass (`pytest`)
- [ ] Type hints added on any new public function signatures
- [ ] Docstrings added or updated
- [ ] `CHANGELOG.md` updated under *Unreleased* if the change is user-visible
- [ ] No unrelated reformatting (keep diffs small)
- [ ] For new integrations: pin the target framework's version range in `pyproject.toml` optional-deps

## Adding a new framework integration

1. Add a new module under `execlave/integrations/`.
2. Import the framework at runtime only (inside `__init__` or a helper). The top-level `execlave` import must never require the framework.
3. Use `execlave.instrumentation` helpers (`record_llm_call`, `record_tool_call`, `record_agent_action`) rather than driving `Trace` directly. This keeps span semantics consistent.
4. Call `Execlave.enforce_policy(...)` on every external action (tool call, outbound HTTP, database write).
5. Add an optional-dependency group in `pyproject.toml` with a pinned range.
6. Add unit tests that mock the framework — CI does not install the real framework.
7. Add a docs page at `frontend/app/docs/integrations/<name>/page.tsx` in the monorepo PR.

## Style

- `black` formatting, `ruff` linting (configured in `pyproject.toml`).
- Prefer composition over inheritance.
- No `print()` in library code; use `logging`.
- Never swallow exceptions without logging — fail-open is acceptable for telemetry paths only, and must be logged at `warning`.

## Release process

Releases are driven from the monorepo. Tag `sdk-python/vX.Y.Z` publishes to PyPI via the `sdk-publish.yml` workflow. Versioning follows SemVer.

## Questions?

Open a discussion at <https://github.com/execlave-ai/execlave-python-sdk/discussions> or email `support@execlave.com`.
