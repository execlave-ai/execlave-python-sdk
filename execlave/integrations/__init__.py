"""
Framework adapters for popular agent/LLM orchestration libraries.

Each submodule uses *soft imports* — the framework is only required
when the adapter is actually used. This keeps the default install
surface of ``execlave-sdk`` zero-footprint.

Sub-modules:
    langchain        — ``ExeclaveCallbackHandler`` (LangChain 0.3.x)
    openai_agents    — ``ExeclaveTracingProcessor`` (openai-agents SDK)
    crewai           — ``instrument_crew`` helper (CrewAI)
    llamaindex       — ``ExeclaveLlamaIndexHandler`` (LlamaIndex 0.11+)
    mcp              — ``instrument_mcp_session`` (Model Context Protocol)
    openai_chat      — ``instrument_openai`` (OpenAI Chat Completions)
    autogen          — ``instrument_autogen_agent`` (Microsoft AutoGen)
"""

__all__: list[str] = []
