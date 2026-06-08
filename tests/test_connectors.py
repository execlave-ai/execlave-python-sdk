from types import SimpleNamespace
from unittest.mock import MagicMock

from execlave.connectors import run_langchain, run_openai_chat


class TestConnectors:
    def test_run_openai_chat_enforces_and_traces(self):
        exe = MagicMock()
        trace = MagicMock()
        exe.start_trace.return_value = trace

        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=23)
        response = SimpleNamespace(id="chatcmpl_1", usage=usage)

        openai_client = MagicMock()
        openai_client.chat.completions.create.return_value = response

        out = run_openai_chat(
            exe,
            openai_client,
            agent_id="agent_1",
            input_text="hello",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )

        exe.enforce_policy.assert_called_once()
        trace.set_model.assert_called_once_with("gpt-4o-mini")
        trace.set_tokens.assert_called_once_with(11, 23)
        trace.finish.assert_called_once_with("success")
        assert out.id == "chatcmpl_1"

    def test_run_langchain_enforces_and_traces(self):
        exe = MagicMock()
        trace = MagicMock()
        exe.start_trace.return_value = trace

        runnable = MagicMock()
        runnable.invoke.return_value = "final answer"

        out = run_langchain(
            exe,
            runnable,
            agent_id="agent_1",
            input_text="question",
        )

        exe.enforce_policy.assert_called_once()
        runnable.invoke.assert_called_once_with("question")
        trace.finish.assert_called_once_with("success")
        assert out == "final answer"
