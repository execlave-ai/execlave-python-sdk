"""Tests for the OpenAI Chat Completions integration.

No real ``openai`` package needed — we duck-type the client.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from execlave.errors import PolicyBlockedError
from execlave.integrations.openai_chat import instrument_openai


class _Usage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str = "hi", model: str = "gpt-4o-mini") -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage(10, 20)
        self.model = model


class _Completions:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _OpenAI:
    def __init__(self, completions):
        self.chat = _Chat(completions)


class _AsyncCompletions:
    def __init__(self, response):
        self._response = response
        self.calls: list = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


@pytest.fixture()
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestInstrumentOpenAI:
    def test_requires_client(self, ag_client):
        with pytest.raises(ValueError):
            instrument_openai(None, ag_client, agent_id="bot")

    def test_requires_agent_id(self, ag_client):
        with pytest.raises(ValueError):
            instrument_openai(_OpenAI(_Completions(_Resp())), ag_client, agent_id="")

    def test_create_enforces_user_message(self, ag_client, monkeypatch):
        comps = _Completions(_Resp("ok"))
        client = _OpenAI(comps)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["input"] = input
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_openai(client, ag_client, agent_id="bot")
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert captured["input"] == "hello"
        assert comps.calls == [
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}
        ]

    def test_seals_full_model_and_messages_into_metadata(self, ag_client, monkeypatch):
        # `input` is bounded to the extracted latest user turn for policy/
        # classifier read; the certificate's digest must still cover the
        # system prompt, prior turns, and model — otherwise those could drift
        # post-approval undetected.
        comps = _Completions(_Resp("ok"))
        client = _OpenAI(comps)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_openai(client, ag_client, agent_id="bot")
        messages = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hello"},
        ]
        client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        assert captured["metadata"] == {"model": "gpt-4o-mini", "messages": messages}

    def test_seals_the_entire_request_not_just_model_and_messages(self, ag_client, monkeypatch):
        # Before this fix, only {"model": ..., "messages": ...} were
        # hand-picked into the sealed metadata. `tools`, `tool_choice`,
        # `temperature`, and `response_format` could differ from what a
        # human approved without ever invalidating the certificate. Prove
        # the full request survives sealing, and that changing an
        # untouched field changes the sealed value (i.e. would change the
        # certificate digest).
        comps = _Completions(_Resp("ok"))
        client = _OpenAI(comps)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_openai(client, ag_client, agent_id="bot")
        request = dict(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
            tool_choice="required",
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        client.chat.completions.create(**request)
        assert captured["metadata"] == request

        captured_second: dict = {}

        def fake_second(agent_id, input, **kw):
            captured_second["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake_second)
        client.chat.completions.create(**{**request, "tool_choice": "auto"})
        assert captured["metadata"] != captured_second["metadata"]

    def test_sanitizes_non_json_serializable_messages(self, ag_client, monkeypatch):
        # Same regression as the MCP adapter: a circular reference reaching
        # json.dumps at the HTTP layer raises a TypeError that `requests`
        # does NOT classify as a RequestException, so it is swallowed by the
        # adapter's generic `except Exception` as "non-fatal" -- silently
        # allowing the call.
        comps = _Completions(_Resp("ok"))
        client = _OpenAI(comps)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["metadata"] = kw.get("metadata")
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_openai(client, ag_client, agent_id="bot")

        circular_msg: dict = {"role": "user", "content": "hi"}
        circular_msg["self"] = circular_msg
        client.chat.completions.create(model="gpt-4o-mini", messages=[circular_msg])

        assert captured["metadata"] is not None
        json.dumps(captured["metadata"])

    def test_block_halts_call(self, ag_client, monkeypatch):
        comps = _Completions(_Resp())
        client = _OpenAI(comps)

        def raise_block(*a, **kw):
            raise PolicyBlockedError([{"policyType": "pii", "message": "no"}])

        monkeypatch.setattr(ag_client, "enforce_policy", raise_block)
        instrument_openai(client, ag_client, agent_id="bot")
        with pytest.raises(PolicyBlockedError):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ssn=1"}],
            )
        assert comps.calls == []

    def test_idempotent(self, ag_client):
        comps = _Completions(_Resp())
        client = _OpenAI(comps)
        instrument_openai(client, ag_client, agent_id="bot", enforce=False)
        first = client.chat.completions.create
        instrument_openai(client, ag_client, agent_id="bot", enforce=False)
        assert client.chat.completions.create is first

    def test_async_create_works(self, ag_client, monkeypatch, event_loop):
        comps = _AsyncCompletions(_Resp("ok"))
        chat = _Chat(comps)

        class _AsyncClient:
            pass

        client = _AsyncClient()
        client.chat = chat
        captured = {}

        def fake(agent_id, input, **kw):
            captured["input"] = input
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_openai(client, ag_client, agent_id="bot")
        event_loop.run_until_complete(
            client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hello"}]
            )
        )
        assert captured["input"] == "hello"

    def test_extracts_text_from_multimodal_content(self, ag_client, monkeypatch):
        comps = _Completions(_Resp())
        client = _OpenAI(comps)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["input"] = input
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_openai(client, ag_client, agent_id="bot")
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "..."}},
                    ],
                }
            ],
        )
        assert captured["input"] == "describe this"

    def test_no_messages_falls_back(self, ag_client, monkeypatch):
        comps = _Completions(_Resp())
        client = _OpenAI(comps)
        captured = {}

        def fake(agent_id, input, **kw):
            captured["input"] = input
            return {"allowed": True}

        monkeypatch.setattr(ag_client, "enforce_policy", fake)
        instrument_openai(client, ag_client, agent_id="bot")
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
        assert captured["input"] == "chat.completions"
