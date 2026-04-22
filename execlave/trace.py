"""Trace object for recording execution details."""

import time
import uuid
from typing import Any, Optional


class Trace:
    """
    Represents a single execution trace.
    
    Used as a context manager or via start_trace() for manual control.
    Collects input, output, model info, tokens, cost, and metadata.
    """

    def __init__(
        self,
        exe: Any,
        agent_id: str | None = None,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
    ):
        self._exe = exe
        self.trace_id = trace_id or f"tr_{uuid.uuid4().hex[:16]}"
        self.agent_id = agent_id
        self.session_id = session_id
        self.user_id = user_id
        self._metadata = metadata or {}
        self._input: Any = None
        self._output: Any = None
        self._model_name: str | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._cost: float | None = None
        self._status: str = "success"
        self._error_message: str | None = None
        self._error_type: str | None = None
        self._start_time: float = time.time()
        self._end_time: float | None = None
        self._duration_ms: int | None = None
        self._finished = False

    def set_input(self, input_data: Any) -> "Trace":
        """Set the input data for this trace."""
        self._input = input_data
        return self

    def set_output(self, output_data: Any) -> "Trace":
        """Set the output data for this trace."""
        self._output = output_data
        return self

    def set_model(self, model_name: str) -> "Trace":
        """Set the model name used in this trace."""
        self._model_name = model_name
        return self

    def set_tokens(self, input: int = 0, output: int = 0) -> "Trace":
        """Set token counts for input and output."""
        self._input_tokens = input
        self._output_tokens = output
        return self

    def set_cost(self, cost_usd: float) -> "Trace":
        """Set the cost in USD for this trace."""
        self._cost = cost_usd
        return self

    def set_duration(self, duration_ms: int) -> "Trace":
        """Override the auto-calculated duration (ms). Chainable."""
        self._duration_ms = duration_ms
        return self

    def add_metadata(self, metadata: dict) -> "Trace":
        """Merge additional metadata into this trace."""
        self._metadata.update(metadata)
        return self

    def finish(
        self,
        status: str = "success",
        error_message: str | None = None,
        error_type: str | None = None,
    ) -> None:
        """
        Finish the trace and submit it to the buffer.
        
        Args:
            status: 'success' | 'error' | 'timeout'
            error_message: Error message if status is 'error'
            error_type: Error type name if status is 'error'
        """
        if self._finished:
            return
        self._finished = True
        self._end_time = time.time()
        self._status = status
        self._error_message = error_message
        self._error_type = error_type

        # Add to the buffer
        self._exe._buffer_trace(self._to_payload())

    def _to_payload(self) -> dict:
        """Convert trace to API payload format, omitting None values."""
        duration_ms = self._duration_ms
        if duration_ms is None and self._end_time:
            duration_ms = int((self._end_time - self._start_time) * 1000)

        total_tokens = None
        if self._input_tokens is not None or self._output_tokens is not None:
            total_tokens = (self._input_tokens or 0) + (self._output_tokens or 0)

        payload = {
            "traceId": self.trace_id,
            "agentId": self.agent_id,
            "sessionId": self.session_id,
            "userId": self.user_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(self._start_time)),
            "durationMs": duration_ms,
            "status": self._status,
            "input": self._input,
            "output": self._output,
            "modelName": self._model_name,
            "promptTokens": self._input_tokens,
            "completionTokens": self._output_tokens,
            "totalTokens": total_tokens,
            "costUsd": self._cost,
            "errorMessage": self._error_message,
            "errorType": self._error_type,
            "metadata": self._metadata if self._metadata else None,
        }
        # Strip None values to avoid server-side validation issues
        return {k: v for k, v in payload.items() if v is not None}

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            self.finish(
                status="error",
                error_message=str(exc_val),
                error_type=exc_type.__name__,
            )
        elif not self._finished:
            self.finish(status="success")
        return False  # don't suppress exceptions
