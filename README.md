# Execlave Python SDK

Official Python SDK for the **Execlave** AI Governance Platform. Provides pre-execution policy enforcement, tracing, agent registration, PII scrubbing, kill-switch support, and OpenTelemetry export.

[![PyPI version](https://img.shields.io/pypi/v/execlave-sdk.svg)](https://pypi.org/project/execlave-sdk/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Downloads](https://static.pepy.tech/badge/execlave-sdk/month)](https://pepy.tech/project/execlave-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-execlave.com-0a84ff.svg)](https://www.execlave.com/docs)

> **Framework integrations** — drop in one callback/processor/helper for [LangChain, OpenAI Agents SDK, or CrewAI](https://www.execlave.com/docs/integrations). See the [full docs](https://www.execlave.com/docs/integrations) or [get an API key](https://www.execlave.com/signup?utm_source=github&utm_medium=sdk&utm_campaign=python).

---

## Installation

```bash
pip install execlave-sdk
```

With OpenTelemetry support:

```bash
pip install execlave-sdk[otel]
```

## Quick Start

The canonical request lifecycle is **register → enforce → call LLM → trace**. `enforce_policy` is what blocks bad requests; tracing alone only logs them after the fact.

```python
from execlave import Execlave, PolicyBlockedError, AgentPausedError

ag = Execlave(
    api_key="exe_prod_your_key_here",     # or set EXECLAVE_API_KEY env var
    base_url="https://api.execlave.com",
    environment="production",
)

# Register the agent once on startup (idempotent)
agent = ag.register_agent(
    agent_id="my-assistant",
    name="Customer Support Bot",
    description="Handles tier-1 support queries",
    type="chatbot",
    platform="custom",
    tags=["support", "production"],
)


def answer(question: str) -> str:
    trace = ag.start_trace(agent_id="my-assistant")
    trace.set_input(question)

    try:
        # Pre-execution policy enforcement. Synchronously checks every
        # policy you've configured for this agent. Raises PolicyBlockedError
        # if any policy with enforcement_mode='block' fires.
        ag.enforce_policy(agent_id="my-assistant", input=question)
    except PolicyBlockedError as e:
        trace.finish(status="error", error_type="PolicyBlockedError", error_message=str(e))
        return "Request blocked by security policy."
    except AgentPausedError:
        trace.finish(status="error", error_type="AgentPausedError")
        return "Service temporarily unavailable."

    response = llm.invoke(question)               # your LLM call
    trace.set_output(response).set_model("gpt-4").finish()
    return response


print(answer("How do I reset my password?"))
```

## Agent Registration

Register agents to monitor them from the Execlave dashboard:

```python
agent = ag.register_agent(
    agent_id="order-processor",
    name="Order Processor",
    description="Processes and validates customer orders",
    type="autonomous",        # chatbot | copilot | autonomous | workflow
    platform="custom",        # custom | openai | anthropic | langchain | ...
    tags=["orders", "production"],
)

# Check agent status (active / paused / etc.)
print(agent.status)
```

## Policy Enforcement

`enforce_policy` is a synchronous check against the policies you've configured in the dashboard. Call it **before** every LLM or tool invocation. Behavior depends on each policy's `enforcement_mode`:

| Mode               | What `enforce_policy` does                                              |
| ------------------ | ----------------------------------------------------------------------- |
| `block`            | Raises `PolicyBlockedError` (with the violations list)                  |
| `monitor` / `warn` | Returns `{"allowed": True, "warnings": [...]}` — caller proceeds        |
| `require_approval` | Blocks the call while polling for human approval (returns when granted) |

```python
from execlave import PolicyBlockedError

try:
    result = ag.enforce_policy(
        agent_id="my-assistant",
        input=user_message,
        environment="production",        # optional
        metadata={"user_id": "u123"},    # optional
        estimated_cost=0.02,             # optional — for cost_limit policies
        tools=["search", "email"],       # optional — for access_control policies
    )
    # result["allowed"] is True. Check result.get("warnings") for non-blocking signals.
except PolicyBlockedError as e:
    for v in e.violations:
        print(v["policyType"], v["message"])
```

> **Important**: A policy must be configured with `enforcement_mode = block` in the dashboard to actually block. Policies in `monitor` or `warn` mode produce warnings on the result but never raise.

## Tracing

### Decorator

The simplest way to trace function calls:

```python
@ag.trace
def process_order(order_data: dict) -> dict:
    result = llm.invoke(json.dumps(order_data))
    return json.loads(result)
```

> `@ag.trace` only records the call — it does **not** run policy enforcement. To block bad inputs, call `ag.enforce_policy(...)` inside the function body before invoking the LLM.

### Context Manager

For more control over trace metadata:

```python
with ag.start_trace(agent_id="my-assistant", session_id="sess_abc") as trace:
    trace.set_input({"question": "What is the refund policy?"})

    result = llm.invoke("What is the refund policy?")

    trace.set_output({"answer": result})
    trace.set_model("gpt-4o")
    trace.set_tokens(input=150, output=320)
    trace.set_cost(0.0045)
```

### Manual Trace

```python
trace = ag.start_trace(agent_id="my-assistant")
trace.set_input(user_query)

try:
    response = llm.invoke(user_query)
    trace.set_output(response)
    trace.finish(status="success")
except Exception as e:
    trace.finish(status="error", error_message=str(e), error_type=type(e).__name__)
    raise
```

### Trace Fields

| Method                                      | Description                                  |
| ------------------------------------------- | -------------------------------------------- |
| `set_input(data)`                           | Input data (auto-serialized)                 |
| `set_output(data)`                          | Output data (auto-serialized)                |
| `set_model(name)`                           | Model name (e.g., `"gpt-4o"`)                |
| `set_tokens(input, output)`                 | Token counts                                 |
| `set_cost(amount)`                          | Cost in USD                                  |
| `set_duration(ms)`                          | Override auto-calculated duration (ms)       |
| `add_metadata(dict)`                        | Merge additional metadata                    |
| `add_tags(list)`                            | Append tags (deduplicated)                   |
| `finish(status, error_message, error_type)` | Finalize trace and submit to the flush queue |

`status` values: `"success"` (default), `"error"`, `"timeout"`. All setter methods are chainable.

## Privacy & PII Scrubbing

Built-in client-side PII scrubbing before data leaves your infrastructure:

```python
ag = Execlave(
    api_key="exe_prod_xxx",
    privacy={
        "enabled": True,                       # turn the feature on
        "scrub_fields": ["input", "output"],   # fields to scan
        "hash_pii": True,                      # include short SHA-256 hashes in metadata
    },
)
```

Detected PII types: email addresses, SSNs, credit card numbers, US phone numbers, IP addresses, API keys.

## Client-side Injection Scoring

When `enable_injection_scan=True` (the default), the SDK runs a regex-based prompt-injection check on the trace's `input` and **annotates the trace** with the detected risk level and matched patterns:

```python
ag = Execlave(
    api_key="exe_prod_xxx",
    enable_injection_scan=True,
)
```

The scan attaches a `metadata.injection_scan` block to the trace (with `risk_level` and `patterns_matched`) so detections show up in the dashboard.

> **This option does not block LLM calls — it is a tagging/telemetry feature.** To actually prevent execution when injection is detected, configure an `injection_scan` policy with `enforcement_mode = block` in the dashboard and call `ag.enforce_policy(...)` before your LLM call. See [Policy Enforcement](#policy-enforcement).

Detected patterns include "ignore previous instructions", jailbreak attempts, system-prompt extraction, and other common prefixes.

## Kill Switch / Pause Support

Execlave supports remote agent pausing via the dashboard. Once paused, every new trace or enforce call raises `AgentPausedError`:

```python
from execlave import AgentPausedError

try:
    result = answer("Process this order")
except AgentPausedError:
    return "Service temporarily unavailable — agent paused by admin."
```

The SDK polls for status changes in the background (configurable interval) and connects via Socket.IO when available for sub-second propagation.

## OpenTelemetry Integration

Export Execlave traces as OpenTelemetry spans for unified observability:

```python
from execlave import Execlave

ag = Execlave(
    api_key="exe_prod_xxx",
    mode="otlp",
    otlp_endpoint="http://localhost:4318",   # your OTel collector
)
```

Requires the `otel` extra: `pip install execlave-sdk[otel]`.

## Configuration

### Constructor Options

| Parameter                  | Type   | Default                    | Description                                                                                          |
| -------------------------- | ------ | -------------------------- | ---------------------------------------------------------------------------------------------------- |
| `api_key`                  | `str`  | `EXECLAVE_API_KEY` env     | Your Execlave API key                                                                                |
| `base_url`                 | `str`  | `https://api.execlave.com` | Execlave API URL                                                                                     |
| `environment`              | `str`  | `"production"`             | Environment tag                                                                                      |
| `async_mode`               | `bool` | `True`                     | Non-blocking trace ingestion                                                                         |
| `mode`                     | `str`  | `"native"`                 | `"native"` or `"otlp"`                                                                               |
| `otlp_endpoint`            | `str`  | `None`                     | OTel collector endpoint (required when `mode="otlp"`)                                                |
| `batch_size`               | `int`  | `100`                      | Traces per flush batch                                                                               |
| `flush_interval_seconds`   | `int`  | `10`                       | Seconds between background flushes                                                                   |
| `debug`                    | `bool` | `False`                    | Enable debug logging                                                                                 |
| `privacy`                  | `dict` | `{}`                       | PII scrubbing config (see Privacy section)                                                           |
| `enable_control_channel`   | `bool` | `True`                     | Enable kill-switch polling + WebSocket                                                               |
| `enable_injection_scan`    | `bool` | `True`                     | Tag traces with client-side injection signals (no block)                                             |
| `enforcement_on_outage`    | `str`  | `"fail_open"`              | `"fail_open"` allows requests when API is down; `"fail_closed"` raises `EnforcementUnavailableError` |
| `policy_cache_ttl_seconds` | `int`  | `60`                       | TTL for cached policy decisions                                                                      |

### Environment Variables

| Variable            | Description                           |
| ------------------- | ------------------------------------- |
| `EXECLAVE_API_KEY`  | API key (alternative to constructor)  |
| `EXECLAVE_BASE_URL` | Base URL (alternative to constructor) |

## Error Handling

```python
from execlave import (
    ExeclaveError,
    ExeclaveAuthError,
    PolicyBlockedError,
    AgentPausedError,
    EnforcementUnavailableError,
)

try:
    ag.enforce_policy(agent_id="my-assistant", input=user_message)
    # ... LLM call + tracing ...
except PolicyBlockedError as e:
    # A block-mode policy fired. e.violations is a list of dicts with policyType, message, severity.
    return "Blocked by security policy."
except AgentPausedError:
    # Agent paused via kill switch.
    return "Service temporarily unavailable."
except EnforcementUnavailableError:
    # Only raised when enforcement_on_outage='fail_closed' AND the API
    # is unreachable for 3+ consecutive attempts (circuit breaker open).
    return "Governance system unavailable."
except ExeclaveAuthError:
    raise  # Misconfigured API key — fail loud.
except ExeclaveError as e:
    print(f"SDK error: {e}")
    raise
```

## Async Trace Buffer

The SDK uses a non-blocking circular buffer (max 10,000 traces) with a background flush thread. Traces are batched and sent to the Execlave API automatically.

```python
# Manual flush (e.g., before shutdown)
ag.flush()

# Graceful shutdown — flushes remaining traces and joins background threads
ag.shutdown()
```

## Development

```bash
# Clone the repo
git clone https://github.com/execlave/sdk-python.git
cd execlave/sdk-python

# Install dev dependencies
pip install -e ".[test]"

# Run tests
pytest                    # 130 tests
pytest --cov=execlave     # With coverage

# Type checking
mypy execlave/
```

## Legal

By using this SDK, you agree to the [Execlave Terms of Service](https://www.execlave.com/terms).

- [Privacy Policy](https://www.execlave.com/privacy)
- [Acceptable Use Policy](https://www.execlave.com/acceptable-use)
- [Responsible AI](https://www.execlave.com/responsible-ai)
- [Security](https://www.execlave.com/security)

## License

MIT — see [LICENSE](../LICENSE) for details.
