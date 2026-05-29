"""Tests for the OpenAI Chat Completions integration.

No real ``openai`` package needed — we duck-type the client.
"""

from __future__ import annotations

import asyncio

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
