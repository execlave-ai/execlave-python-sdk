"""
A trace the platform refuses must reach the application.

Traces are sent in the background. Before this, a refused batch was a log line
and a per-trace refusal inside an accepted batch (202 with ``errors``) was not
read at all, so the caller believed a ``success`` had been recorded when
nothing was stored. The case that matters most is an agent whose outcome
strictness is ``strict``: the platform refuses a ``success`` without
confirming evidence with ``OUTCOME_EVIDENCE_REQUIRED``.
"""

from tests.helpers import make_mock_response

STRICT_MESSAGE = (
    'This agent requires confirmed outcomes (outcomeStrictness "strict"): '
    'status "success" needs effectEvidence'
)
CODE = "OUTCOME_EVIDENCE_REQUIRED"


def _ingest_calls(mock_session):
    return [c for c in mock_session.request.call_args_list if "/api/v1/traces/ingest" in str(c)]


def _buffer(client, *trace_ids):
    for trace_id in trace_ids:
        client._buffer.append({"traceId": trace_id, "agentId": "bot", "status": "success"})


def _strict_refusal(*trace_ids):
    return make_mock_response(
        400,
        {
            "error": {"code": CODE, "message": STRICT_MESSAGE},
            "accepted": 0,
            "failed": len(trace_ids),
            "errors": [{"traceId": t, "code": CODE, "error": STRICT_MESSAGE} for t in trace_ids],
        },
    )


class TestRejectedTracesReachTheApplication:
    def test_strict_refusal_of_the_whole_batch_reaches_handler_and_flush(self, exe_client, mock_session):
        seen = []
        exe_client.on_trace_rejected = seen.append
        mock_session.request.return_value = _strict_refusal("t1")
        _buffer(exe_client, "t1")

        result = exe_client.flush()

        expected = [{"traceId": "t1", "code": CODE, "message": STRICT_MESSAGE}]
        assert result == {"sent": 0, "rejected": expected, "undelivered": 0}
        assert seen == [expected]

    def test_a_refused_trace_is_never_resent(self, exe_client, mock_session):
        exe_client.on_trace_rejected = lambda rejections: None
        mock_session.request.return_value = _strict_refusal("t1")
        _buffer(exe_client, "t1")

        exe_client.flush()
        exe_client.flush()

        # One attempt, and the refused trace is not put back in the buffer.
        assert len(_ingest_calls(mock_session)) == 1
        assert len(exe_client._buffer) == 0

    def test_per_trace_refusals_inside_an_accepted_batch_are_read(self, exe_client, mock_session):
        seen = []
        exe_client.on_trace_rejected = seen.append
        mock_session.request.return_value = make_mock_response(
            202,
            {
                "accepted": 1,
                "failed": 1,
                "errors": [{"traceId": "refused", "code": CODE, "error": STRICT_MESSAGE}],
            },
        )
        _buffer(exe_client, "refused", "stored")

        result = exe_client.flush()

        assert result["sent"] == 1
        assert result["rejected"] == [{"traceId": "refused", "code": CODE, "message": STRICT_MESSAGE}]
        assert seen == [result["rejected"]]

    def test_every_trace_of_a_chunk_is_reported_when_a_4xx_names_none(self, exe_client, mock_session):
        exe_client.on_trace_rejected = lambda rejections: None
        mock_session.request.return_value = make_mock_response(
            400, {"error": {"code": "VALIDATION_ERROR", "message": "Validation failed"}}
        )
        _buffer(exe_client, "a", "b")

        result = exe_client.flush()

        assert sorted(r["traceId"] for r in result["rejected"]) == ["a", "b"]
        assert all(r["code"] == "VALIDATION_ERROR" for r in result["rejected"])

    def test_logs_an_error_when_no_handler_is_set(self, exe_client, mock_session, caplog):
        mock_session.request.return_value = _strict_refusal("t1")
        _buffer(exe_client, "t1")

        with caplog.at_level("ERROR", logger="Execlave"):
            exe_client.flush()

        assert "NOT stored" in caplog.text
        assert "t1" in caplog.text
        assert CODE in caplog.text

    def test_says_nothing_when_every_trace_was_stored(self, exe_client, mock_session, caplog):
        seen = []
        exe_client.on_trace_rejected = seen.append
        mock_session.request.return_value = make_mock_response(202, {"accepted": 1, "failed": 0})
        _buffer(exe_client, "t1")

        with caplog.at_level("ERROR", logger="Execlave"):
            result = exe_client.flush()

        assert result == {"sent": 1, "rejected": [], "undelivered": 0}
        assert seen == []
        assert "NOT stored" not in caplog.text

    def test_survives_a_handler_that_raises(self, exe_client, mock_session):
        def broken(_rejections):
            raise RuntimeError("listener bug")

        exe_client.on_trace_rejected = broken
        mock_session.request.return_value = _strict_refusal("t1")
        _buffer(exe_client, "t1")

        assert exe_client.flush()["sent"] == 0

    def test_exhausted_retries_count_as_undelivered_not_rejected(self, exe_client, mock_session, monkeypatch):
        monkeypatch.setattr("execlave.client.time.sleep", lambda s: None)
        seen = []
        exe_client.on_trace_rejected = seen.append
        mock_session.request.return_value = make_mock_response(503, {"error": {"message": "down"}})
        _buffer(exe_client, "t1")

        result = exe_client.flush()

        assert result == {"sent": 0, "rejected": [], "undelivered": 1}
        assert seen == []

    def test_handler_can_be_given_to_the_constructor(self, mock_session):
        from execlave.client import Execlave

        handler = lambda rejections: None  # noqa: E731
        client = Execlave(
            api_key="exe_test_key_0123456789",
            base_url="http://mock-server:4000",
            async_mode=False,
            enable_control_channel=False,
            on_trace_rejected=handler,
        )

        assert client.on_trace_rejected is handler


class TestEffectEvidence:
    """Without these a Python agent set to strict could not comply at all."""

    def _finish(self, exe_client, **kwargs):
        from execlave.trace import Trace

        captured = []
        exe_client._buffer_trace = captured.append
        Trace(exe_client, agent_id="bot", trace_id="t1").finish(**kwargs)
        return captured[0]

    def test_success_carries_confirming_evidence_in_api_field_names(self, exe_client):
        payload = self._finish(
            exe_client,
            effect_evidence={"basis": "read_back", "effect_ref": "invoice/42", "detail": None},
        )

        assert payload["status"] == "success"
        assert payload["effectEvidence"] == {"basis": "read_back", "effectRef": "invoice/42"}

    def test_effect_unconfirmed_is_reported_with_what_is_known(self, exe_client):
        payload = self._finish(
            exe_client,
            status="effect_unconfirmed",
            effect_evidence={"basis": "acknowledgement", "acknowledgement_id": "job_42"},
        )

        assert payload["status"] == "effect_unconfirmed"
        assert payload["effectEvidence"] == {"basis": "acknowledgement", "acknowledgementId": "job_42"}

    def test_no_evidence_sends_no_field(self, exe_client):
        assert "effectEvidence" not in self._finish(exe_client)

    def test_an_unknown_key_raises_instead_of_being_dropped(self, exe_client):
        import pytest

        with pytest.raises(ValueError, match="efect_ref"):
            self._finish(exe_client, effect_evidence={"basis": "read_back", "efect_ref": "x"})

    def test_evidence_without_a_basis_raises(self, exe_client):
        import pytest

        with pytest.raises(ValueError, match="basis"):
            self._finish(exe_client, effect_evidence={"effect_ref": "x"})

    def test_resolve_trace_posts_the_observation(self, exe_client, mock_session):
        mock_session.request.return_value = make_mock_response(201, {"data": {}})

        exe_client.resolve_trace(
            "tr_a/b", resolved_status="success", basis="read_back", effect_ref="invoice/42"
        )

        call = mock_session.request.call_args
        assert call.args[0] == "POST"
        assert call.args[1].endswith("/api/v1/traces/tr_a%2Fb/resolutions")
        assert call.kwargs["json"] == {
            "resolvedStatus": "success",
            "basis": "read_back",
            "effectRef": "invoice/42",
        }
