"""
Tests for Execlave.trace.Trace.

Covers context-manager usage, manual start/finish, decorator mode,
data setters, and payload serialisation.
"""

import time
import pytest
from unittest.mock import MagicMock

from execlave.trace import Trace
from execlave.errors import AgentPausedError
from tests.helpers import make_mock_response, SAMPLE_AGENT_DATA


# Helpers -------------------------------------------------------------------

def _make_exe_stub():
    """Minimal stub for the ``exe`` parameter accepted by Trace."""
    stub = MagicMock()
    stub._buffer_trace = MagicMock()
    stub._state = "ACTIVE"
    stub._agents = {}
    return stub


# =========================================================================
# Construction
# =========================================================================


class TestTraceInit:

    def test_default_trace_id_generated(self):
        t = Trace(_make_exe_stub())
        assert t.trace_id.startswith("tr_")
        assert len(t.trace_id) > 4

    def test_custom_trace_id(self):
        t = Trace(_make_exe_stub(), trace_id="custom_id")
        assert t.trace_id == "custom_id"

    def test_initial_status_is_success(self):
        t = Trace(_make_exe_stub())
        assert t._status == "success"

    def test_initial_timestamps(self):
        before = time.time()
        t = Trace(_make_exe_stub())
        after = time.time()
        assert before <= t._start_time <= after
        assert t._end_time is None


# =========================================================================
# Data Setters (fluent interface)
# =========================================================================


class TestSetters:

    def test_set_input_returns_self(self):
        t = Trace(_make_exe_stub())
        result = t.set_input("hello")
        assert result is t
        assert t._input == "hello"

    def test_set_output_returns_self(self):
        t = Trace(_make_exe_stub())
        result = t.set_output({"answer": 42})
        assert result is t
        assert t._output == {"answer": 42}

    def test_set_model(self):
        t = Trace(_make_exe_stub())
        t.set_model("gpt-4o")
        assert t._model_name == "gpt-4o"

    def test_set_tokens(self):
        t = Trace(_make_exe_stub())
        t.set_tokens(input=100, output=50)
        assert t._input_tokens == 100
        assert t._output_tokens == 50

    def test_set_cost(self):
        t = Trace(_make_exe_stub())
        t.set_cost(0.0032)
        assert t._cost == pytest.approx(0.0032)

    def test_add_metadata_merges(self):
        t = Trace(_make_exe_stub(), metadata={"a": 1})
        t.add_metadata({"b": 2})
        assert t._metadata == {"a": 1, "b": 2}

    def test_add_metadata_overwrite(self):
        t = Trace(_make_exe_stub(), metadata={"a": 1})
        t.add_metadata({"a": 99})
        assert t._metadata["a"] == 99


# =========================================================================
# Context Manager
# =========================================================================


class TestContextManager:

    def test_enter_returns_trace(self):
        exe = _make_exe_stub()
        t = Trace(exe, agent_id="bot1")
        with t as ctx:
            assert ctx is t

    def test_exit_calls_finish(self):
        exe = _make_exe_stub()
        with Trace(exe) as t:
            t.set_output("result")
        assert t._finished is True
        assert t._status == "success"
        exe._buffer_trace.assert_called_once()

    def test_exit_captures_timing(self):
        exe = _make_exe_stub()
        with Trace(exe) as t:
            time.sleep(0.01)
        assert t._end_time is not None
        assert t._end_time >= t._start_time

    def test_exit_on_exception_sets_error(self):
        exe = _make_exe_stub()
        with pytest.raises(ValueError):
            with Trace(exe) as t:
                raise ValueError("boom")
        assert t._status == "error"
        assert t._error_message == "boom"
        assert t._error_type == "ValueError"
        exe._buffer_trace.assert_called_once()

    def test_exception_not_suppressed(self):
        exe = _make_exe_stub()
        with pytest.raises(RuntimeError):
            with Trace(exe):
                raise RuntimeError("not swallowed")

    def test_double_finish_is_idempotent(self):
        exe = _make_exe_stub()
        t = Trace(exe)
        t.finish()
        t.finish()  # second call is a no-op
        exe._buffer_trace.assert_called_once()


# =========================================================================
# finish()
# =========================================================================


class TestFinish:

    def test_finish_submits_to_buffer(self):
        exe = _make_exe_stub()
        t = Trace(exe, agent_id="bot1")
        t.set_input("q").set_output("a")
        t.finish()
        exe._buffer_trace.assert_called_once()
        payload = exe._buffer_trace.call_args[0][0]
        assert payload["status"] == "success"
        assert payload["input"] == "q"
        assert payload["output"] == "a"

    def test_finish_with_error(self):
        exe = _make_exe_stub()
        t = Trace(exe)
        t.finish(status="error", error_message="fail", error_type="TypeError")
        payload = exe._buffer_trace.call_args[0][0]
        assert payload["status"] == "error"
        assert payload["errorMessage"] == "fail"
        assert payload["errorType"] == "TypeError"


# =========================================================================
# _to_payload serialisation
# =========================================================================


class TestPayload:

    def test_none_values_stripped(self):
        t = Trace(_make_exe_stub())
        t._end_time = t._start_time + 0.1
        t._finished = True
        payload = t._to_payload()
        for v in payload.values():
            assert v is not None

    def test_duration_ms_calculated(self):
        t = Trace(_make_exe_stub())
        t._start_time = 1000.0
        t._end_time = 1000.123
        payload = t._to_payload()
        assert payload["durationMs"] == 123

    def test_total_tokens_calculated(self):
        t = Trace(_make_exe_stub())
        t._input_tokens = 80
        t._output_tokens = 20
        payload = t._to_payload()
        assert payload["totalTokens"] == 100

    def test_total_tokens_partial(self):
        t = Trace(_make_exe_stub())
        t._input_tokens = 50
        payload = t._to_payload()
        assert payload["totalTokens"] == 50

    def test_metadata_included_when_present(self):
        t = Trace(_make_exe_stub(), metadata={"env": "test"})
        payload = t._to_payload()
        assert payload["metadata"] == {"env": "test"}

    def test_metadata_excluded_when_empty(self):
        t = Trace(_make_exe_stub(), metadata={})
        payload = t._to_payload()
        assert "metadata" not in payload

    def test_trace_id_in_payload(self):
        t = Trace(_make_exe_stub(), trace_id="tr_abc")
        payload = t._to_payload()
        assert payload["traceId"] == "tr_abc"

    def test_timestamp_format(self):
        t = Trace(_make_exe_stub())
        payload = t._to_payload()
        assert "T" in payload["timestamp"]
        assert payload["timestamp"].endswith("Z")

    def test_environment_session_user_tags_in_payload(self):
        t = Trace(
            _make_exe_stub(),
            agent_id="a1",
            session_id="s9",
            user_id="u9",
            environment="development",
            tags=["exp", "qa"],
            parent_trace_id="tr_parent",
            span_type="llm",
            metadata={"k": "v"},
        )
        payload = t._to_payload()
        assert payload["environment"] == "development"
        assert payload["sessionId"] == "s9"
        assert payload["userId"] == "u9"
        assert payload["tags"] == ["exp", "qa"]
        assert payload["parentTraceId"] == "tr_parent"
        assert payload["spanType"] == "llm"
        assert payload["metadata"] == {"k": "v"}

    def test_optional_fields_omitted_when_unset(self):
        t = Trace(_make_exe_stub())
        payload = t._to_payload()
        assert "environment" not in payload
        assert "tags" not in payload
        assert "parentTraceId" not in payload
        assert "spanType" not in payload

    def test_add_tags_dedupes(self):
        t = Trace(_make_exe_stub(), tags=["a"])
        t.add_tags(["b", "a", "c"])
        assert t._tags == ["a", "b", "c"]


# =========================================================================
# Decorator mode (via Execlave.trace)
# =========================================================================


class TestDecoratorMode:
    """Tests that the decorator path through Execlave.trace works."""

    def test_decorator_wraps_function(self, ag_client, mock_session):
        @ag_client.trace
        def greet(name):
            return f"Hello {name}"

        mock_session.request.return_value = make_mock_response(200, {})
        result = greet("Alice")
        assert result == "Hello Alice"

    def test_decorator_captures_return_value(self, ag_client, mock_session):
        mock_session.request.return_value = make_mock_response(200, {})

        @ag_client.trace
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_captures_exception(self, ag_client, mock_session):
        @ag_client.trace
        def fail():
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            fail()

    def test_context_manager_via_trace_call(self, ag_client, mock_session):
        mock_session.request.return_value = make_mock_response(200, {})
        with ag_client.trace(session_id="s1") as t:
            t.set_input("question")
            t.set_output("answer")
        assert t._finished is True
        assert t._status == "success"
