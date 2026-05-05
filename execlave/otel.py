"""
OpenTelemetry adapter for Execlave SDK.

Converts Execlave trace payloads to OTel spans and exports
them using the OTLP protocol. Requires optional dependencies:
    pip install Execlave[otel]
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Execlave")

# Lazy imports — only loaded when mode="otlp"
_Resource = None
_TracerProvider = None
_BatchSpanProcessor = None
_OTLPSpanExporter = None
_StatusCode = None
_SpanKind = None


def _ensure_imports():
    """Lazily import OTel packages — fail clearly if not installed."""
    global _Resource, _TracerProvider, _BatchSpanProcessor, _OTLPSpanExporter, _StatusCode, _SpanKind
    if _TracerProvider is not None:
        return
    try:
        from opentelemetry.sdk.resources import Resource as R
        from opentelemetry.sdk.trace import TracerProvider as TP
        from opentelemetry.sdk.trace.export import BatchSpanProcessor as BSP
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLP
        from opentelemetry.trace import StatusCode as SC
        from opentelemetry.trace import SpanKind as SK
        _Resource = R
        _TracerProvider = TP
        _BatchSpanProcessor = BSP
        _OTLPSpanExporter = OTLP
        _StatusCode = SC
        _SpanKind = SK
    except ImportError:
        raise ImportError(
            "OpenTelemetry packages are required for OTLP mode. "
            "Install with: pip install Execlave[otel]"
        )


class OTelExporter:
    """
    Wraps OpenTelemetry TracerProvider and OTLP exporter.
    Converts Execlave trace payloads to OTel spans.
    """

    def __init__(self, endpoint: str, api_key: str, service_name: str = "Execlave-sdk"):
        _ensure_imports()
        assert _Resource is not None
        assert _OTLPSpanExporter is not None
        assert _TracerProvider is not None
        assert _BatchSpanProcessor is not None

        resource = _Resource.create({
            "service.name": service_name,
            "Execlave.api_key_prefix": api_key[:10] + "..." if api_key else "unknown",
        })

        exporter = _OTLPSpanExporter(
            endpoint=endpoint.rstrip("/") + "/v1/traces",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        self._provider = _TracerProvider(resource=resource)
        self._provider.add_span_processor(_BatchSpanProcessor(exporter))
        self._tracer = self._provider.get_tracer("Execlave", "1.0.0")

    def export_traces(self, traces: List[Dict[str, Any]]) -> None:
        """Convert Execlave trace payloads to OTel spans and export."""
        from opentelemetry.trace import StatusCode

        assert _SpanKind is not None
        for payload in traces:
            # Determine span kind based on trace metadata
            kind = _SpanKind.INTERNAL

            # Create the span
            span = self._tracer.start_span(
                name=payload.get("spanName") or payload.get("agentId", "unknown"),
                kind=kind,
                attributes=self._payload_to_attributes(payload),
            )

            # Set status
            status = payload.get("status", "success")
            if status == "error":
                span.set_status(StatusCode.ERROR, payload.get("errorMessage", ""))
            else:
                span.set_status(StatusCode.OK)

            # End the span
            span.end()

    def _payload_to_attributes(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an Execlave trace payload dict to OTel span attributes."""
        attrs: Dict[str, Any] = {}

        # Standard Execlave attributes
        mapping = {
            "traceId": "Execlave.trace_id",
            "agentId": "Execlave.agent_id",
            "agentUuid": "Execlave.agent_uuid",
            "sessionId": "Execlave.session_id",
            "environment": "deployment.environment",
            "modelName": "gen_ai.request.model",
            "status": "Execlave.status",
            "durationMs": "Execlave.duration_ms",
            "spanType": "Execlave.span_type",
            "spanName": "Execlave.span_name",
            "errorMessage": "error.message",
            "errorType": "error.type",
            "version": "service.version",
            "userId": "enduser.id",
        }

        for key, attr_name in mapping.items():
            val = payload.get(key)
            if val is not None:
                attrs[attr_name] = str(val) if not isinstance(val, (str, int, float, bool)) else val

        # Token counts (using semantic conventions for GenAI)
        if payload.get("promptTokens"):
            attrs["gen_ai.usage.prompt_tokens"] = payload["promptTokens"]
        if payload.get("completionTokens"):
            attrs["gen_ai.usage.completion_tokens"] = payload["completionTokens"]
        if payload.get("totalTokens"):
            attrs["gen_ai.usage.total_tokens"] = payload["totalTokens"]
        if payload.get("costUsd"):
            attrs["Execlave.cost_usd"] = payload["costUsd"]

        # Input/output as attributes (truncated for OTel)
        if payload.get("input"):
            inp = str(payload["input"])
            attrs["gen_ai.prompt"] = inp[:4096] if len(inp) > 4096 else inp
        if payload.get("output"):
            out = str(payload["output"])
            attrs["gen_ai.completion"] = out[:4096] if len(out) > 4096 else out

        return attrs

    def shutdown(self) -> None:
        """Flush pending spans and shut down the provider."""
        try:
            self._provider.force_flush(timeout_millis=5000)
            self._provider.shutdown()
        except Exception as e:
            logger.warning("OTel shutdown error: %s", e)
