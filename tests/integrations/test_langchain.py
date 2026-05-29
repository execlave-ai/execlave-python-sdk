"""Tests for the LangChain integration.

These tests monkeypatch the BaseCallbackHandler resolution so we do NOT
require the actual langchain-core package in CI.
"""

from __future__ import annotations

import sys
import types
from uuid import uuid4
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake langchain_core BEFORE importing the integration module. This must
# run at module-import time (pytest's collection phase), so we do it at the
# top of the file — fixtures run too late.
# ---------------------------------------------------------------------------
def _install_fake_langchain_core() -> None:
    fake_base_mod = types.ModuleType("langchain_core.callbacks.base")

    class BaseCallbackHandler:  # minimal stand-in
        raise_error: bool = False
        run_inline: bool = False

    fake_base_mod.BaseCallbackHandler = BaseCallbackHandler
    fake_callbacks_pkg = types.ModuleType("langchain_core.callbacks")
    fake_callbacks_pkg.base = fake_base_mod
    fake_root = types.ModuleType("langchain_core")
    fake_root.callbacks = fake_callbacks_pkg

    sys.modules.setdefault("langchain_core", fake_root)
    sys.modules.setdefault("langchain_core.callbacks", fake_callbacks_pkg)
    sys.modules.setdefault("langchain_core.callbacks.base", fake_base_mod)
    # Force re-import of the integration module under the fake.
    sys.modules.pop("execlave.integrations.langchain", None)


_install_fake_langchain_core()

from execlave.errors import PolicyBlockedError  # noqa: E402
from execlave.integrations.langchain import ExeclaveCallbackHandler  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _handler(ag_client, **kwargs):
    return ExeclaveCallbackHandler(ag_client, agent_id="bot", **kwargs)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------
class TestConstructor:
    def test_requires_exe(self):
        with pytest.raises(ValueError):
            ExeclaveCallbackHandler(None, agent_id="bot")  # type: ignore[arg-type]

    def test_requires_agent_id(self, ag_client):
        with pytest.raises(ValueError):
            ExeclaveCallbackHandler(ag_client, agent_id="")


# ---------------------------------------------------------------------------
# Chain lifecycle
# ---------------------------------------------------------------------------
class TestChainLifecycle:
    def test_chain_start_end_closes_span(self, ag_client, monkeypatch):
        monkeypatch.setattr(ag_client, "enforce_policy", lambda *a, **k: {"allowed": True})
        h = _handler(ag_client)
        run_id = uuid4()
        h.on_chain_start({"name": "rag"}, {"input": "hi"}, run_id=run_id)
        assert run_id in h._spans
        h.on_chain_end({"answer": "hello"}, run_id=run_id)
        assert run_id not in h._spans

    def test_chain_start_enforces_only_top_level(self, ag_client, monkeypatch):
        calls: list = []
        monkeypatch.setattr(
            ag_client, "enforce_policy", lambda *a, **k: calls.append(a) or {"allowed": True}
        )
        h = _handler(ag_client)
        parent_id = uuid4()
        child_id = uuid4()
        h.on_chain_start({"name": "top"}, {"input": "hello"}, run_id=parent_id)
        h.on_chain_start({"name": "sub"}, {"input": "nested"}, run_id=child_id, parent_run_id=parent_id)
        # Only the top-level chain triggers enforcement.
        assert len(calls) == 1
        h.on_chain_end({"o": 1}, run_id=child_id)
        h.on_chain_end({"o": 1}, run_id=parent_id)

    def test_chain_error_finishes_as_error(self, ag_client, monkeypatch):
        monkeypatch.setattr(ag_client, "enforce_policy", lambda *a, **k: {"allowed": True})
        h = _handler(ag_client)
        rid = uuid4()
        h.on_chain_start({"name": "c"}, {"input": "x"}, run_id=rid)
        span = h._spans[rid]
        h.on_chain_error(RuntimeError("boom"), run_id=rid)
        assert span._finished
        assert rid not in h._spans

    def test_chain_start_propagates_policy_block(self, ag_client, monkeypatch):
        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "pii", "message": "blocked"}])

        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)
        h = _handler(ag_client)
        with pytest.raises(PolicyBlockedError):
            h.on_chain_start({"name": "c"}, {"input": "secret@example.com"}, run_id=uuid4())


# ---------------------------------------------------------------------------
# LLM lifecycle
# ---------------------------------------------------------------------------
class TestLlmLifecycle:
    def test_llm_start_sets_model(self, ag_client):
        h = _handler(ag_client, enforce=False)
        rid = uuid4()
        h.on_llm_start(
            {"name": "openai"},
            ["what is foo?"],
            run_id=rid,
            invocation_params={"model": "gpt-4o"},
        )
        span = h._spans[rid]
        assert span._trace._model_name == "gpt-4o"
        h.on_llm_end(_mock_llm_result(tokens=(10, 20), text="foo"), run_id=rid)

    def test_llm_end_extracts_tokens(self, ag_client):
        h = _handler(ag_client, enforce=False)
        rid = uuid4()
        h.on_llm_start({"name": "openai"}, ["x"], run_id=rid)
        span = h._spans[rid]
        h.on_llm_end(_mock_llm_result(tokens=(7, 13), text="ok"), run_id=rid)
        assert span._trace._input_tokens == 7
        assert span._trace._output_tokens == 13


# ---------------------------------------------------------------------------
# Tool lifecycle — enforcement must happen with tool allowlist
# ---------------------------------------------------------------------------
class TestToolLifecycle:
    def test_tool_start_enforces_with_tool_allowlist(self, ag_client, monkeypatch):
        captured: dict = {}
        def fake_enforce(agent_id, input, **kwargs):
            captured["tools"] = kwargs.get("tools")
            return {"allowed": True}
        monkeypatch.setattr(ag_client, "enforce_policy", fake_enforce)
        h = _handler(ag_client)
        rid = uuid4()
        h.on_tool_start({"name": "web_search"}, "q=anthropic", run_id=rid)
        assert captured["tools"] == ["web_search"]
        h.on_tool_end("results", run_id=rid)

    def test_tool_error_finishes_span_as_error(self, ag_client, monkeypatch):
        monkeypatch.setattr(ag_client, "enforce_policy", lambda *a, **k: {"allowed": True})
        h = _handler(ag_client)
        rid = uuid4()
        h.on_tool_start({"name": "t"}, "i", run_id=rid)
        span = h._spans[rid]
        h.on_tool_error(ValueError("bad"), run_id=rid)
        assert span._finished

    def test_tool_block_halts_execution(self, ag_client, monkeypatch):
        monkeypatch.setattr(
            ag_client,
            "enforce_policy",
            lambda *a, **k: (_ for _ in ()).throw(
                PolicyBlockedError([{"policyType": "tool", "message": "blocked"}])
            ),
        )
        h = _handler(ag_client)
        with pytest.raises(PolicyBlockedError):
            h.on_tool_start({"name": "rm_rf"}, "/", run_id=uuid4())


# ---------------------------------------------------------------------------
# Retriever lifecycle — must NOT leak page_content
# ---------------------------------------------------------------------------
class TestRetrieverLifecycle:
    def test_retriever_end_records_doc_summary_not_content(self, ag_client):
        h = _handler(ag_client, enforce=False)
        rid = uuid4()
        h.on_retriever_start({"name": "r"}, "my query", run_id=rid)
        span = h._spans[rid]

        doc1 = MagicMock(metadata={"id": "doc1"}, page_content="secret contents here")
        doc2 = MagicMock(metadata={"source": "doc2.md"}, page_content="abc")
        h.on_retriever_end([doc1, doc2], run_id=rid)

        meta = span._trace._metadata.get("retrievedDocs")
        assert meta is not None
        assert meta[0] == {"id": "doc1", "length": len("secret contents here")}
        assert meta[1] == {"id": "doc2.md", "length": 3}
        # Crucially, the page_content itself never lands in metadata.
        assert "secret contents here" not in str(span._trace._metadata)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_llm_result(*, tokens: tuple[int, int] | None = None, text: str = "ok"):
    gen = MagicMock()
    gen.text = text
    gen.message = None
    result = MagicMock()
    result.generations = [[gen]]
    result.llm_output = {"token_usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]}} if tokens else {}
    return result
