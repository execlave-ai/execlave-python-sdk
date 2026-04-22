# Execlave Python SDK

Official Python SDK for the **Execlave** AI Governance Platform. Provides tracing, agent registration, prompt-injection scanning, PII scrubbing, and OpenTelemetry integration for AI agents.

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

```python
from execlave import Execlave

# Initialize the SDK
ag = Execlave(
    api_key="exe_prod_your_key_here",   # or set EXECLAVE_API_KEY env var
    base_url="https://api.execlave.com",  # defaults to https://api.execlave.com
    environment="production",
)

# Register an agent
agent = ag.register_agent(
    agent_id="my-assistant",
    name="Customer Support Bot",
    description="Handles tier-1 support queries",
    model="gpt-4o",
    framework="langchain",
    tags=["support", "production"],
)

# Record a trace using the decorator
@ag.trace
def answer(question: str) -> str:
    # Your LLM call here
    return llm.invoke(question)

result = answer("How do I reset my password?")
```

## Agent Registration

Register agents to monitor them from the Execlave dashboard:

```python
agent = ag.register_agent(
    agent_id="order-processor",
    name="Order Processor",
    description="Processes and validates customer orders",
    model="claude-3-sonnet",
    framework="custom",
    tags=["orders", "production"],
)

# Check agent status
print(agent.status)  # "active", "paused", etc.
```

## Tracing

### Decorator

The simplest way to trace function calls:

```python
@ag.trace
def process_order(order_data: dict) -> dict:
    result = llm.invoke(json.dumps(order_data))
    return json.loads(result)
```

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
    trace.set_status("success")
except Exception as e:
    trace.set_error(e)
finally:
    trace.end()
```

### Trace Fields

| Method                      | Description                          |
| --------------------------- | ------------------------------------ |
| `set_input(data)`           | Input data (auto-serialized)         |
| `set_output(data)`          | Output data (auto-serialized)        |
| `set_model(name)`           | Model name (e.g., `"gpt-4o"`)        |
| `set_tokens(input, output)` | Token counts                         |
| `set_cost(amount)`          | Cost in USD                          |
| `set_status(status)`        | `"success"` or `"error"`             |
| `set_error(exception)`      | Records error info from an exception |
| `set_metadata(dict)`        | Arbitrary key-value metadata         |
| `set_duration(ms)`          | Override duration in milliseconds    |

## Privacy & PII Scrubbing

Built-in client-side PII scrubbing before data leaves your infrastructure:

```python
ag = Execlave(
    api_key="exe_prod_xxx",
    privacy={
        "scrub_pii": True,           # Enable PII detection & redaction
        "scrub_fields": ["input", "output"],  # Fields to scan
    },
)
```

Detected PII types: email addresses, SSNs, credit card numbers, US phone numbers, IP addresses, API keys.

## Prompt Injection Scanning

The SDK scans inputs for prompt-injection patterns before sending them to your LLM:

```python
ag = Execlave(
    api_key="exe_prod_xxx",
    enable_injection_scan=True,  # Default: True
)

# Traces with detected injection attempts are flagged automatically
```

Detected patterns include: "ignore previous instructions", jailbreak attempts, system prompt extraction, and more.

## Kill Switch / Pause Support

Execlave supports remote agent pausing via the dashboard:

```python
from execlave import AgentPausedError

try:
    result = answer("Process this order")
except AgentPausedError:
    # Agent was paused by an admin — handle gracefully
    return "Service temporarily unavailable"
```

The SDK polls for status changes in the background (configurable interval).

## OpenTelemetry Integration

Export Execlave traces as OpenTelemetry spans for unified observability:

```python
from execlave import Execlave
from execlave.otel import configure_otel

# Initialize with OTel mode
ag = Execlave(
    api_key="exe_prod_xxx",
    mode="otel",
    otlp_endpoint="http://localhost:4318",  # Your OTel collector
)

# Configure the OTel pipeline
configure_otel(ag)
```

Requires the `otel` extra: `pip install execlave-sdk[otel]`

## Configuration

### Constructor Options

| Parameter                | Type   | Default                    | Description                  |
| ------------------------ | ------ | -------------------------- | ---------------------------- |
| `api_key`                | `str`  | `EXECLAVE_API_KEY` env     | Your Execlave API key        |
| `base_url`               | `str`  | `https://api.execlave.com` | Execlave API URL             |
| `environment`            | `str`  | `"production"`             | Environment tag              |
| `async_mode`             | `bool` | `True`                     | Non-blocking trace ingestion |
| `mode`                   | `str`  | `"native"`                 | `"native"` or `"otel"`       |
| `otlp_endpoint`          | `str`  | `None`                     | OTel collector endpoint      |
| `batch_size`             | `int`  | `100`                      | Traces per flush batch       |
| `flush_interval_seconds` | `int`  | `10`                       | Seconds between flushes      |
| `debug`                  | `bool` | `False`                    | Enable debug logging         |
| `privacy`                | `dict` | `{}`                       | PII scrubbing config         |
| `enable_control_channel` | `bool` | `True`                     | Enable status polling        |
| `enable_injection_scan`  | `bool` | `True`                     | Enable injection scanning    |

### Environment Variables

| Variable            | Description                           |
| ------------------- | ------------------------------------- |
| `EXECLAVE_API_KEY`  | API key (alternative to constructor)  |
| `EXECLAVE_BASE_URL` | Base URL (alternative to constructor) |

## Error Handling

```python
from execlave import ExeclaveError, ExeclaveAuthError, AgentPausedError

try:
    ag = Execlave(api_key="exe_invalid")
    ag.register_agent(agent_id="test")
except ExeclaveAuthError:
    print("Invalid API key")
except AgentPausedError:
    print("Agent is paused")
except ExeclaveError as e:
    print(f"SDK error: {e}")
```

## Async Trace Buffer

The SDK uses a non-blocking circular buffer (max 10,000 traces) with a background flush thread. Traces are batched and sent to the Execlave API automatically.

```python
# Manual flush (e.g., before shutdown)
ag.flush()

# Graceful shutdown
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
pytest --cov=execlave   # With coverage

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
