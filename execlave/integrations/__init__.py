"""
Framework adapters for popular agent/LLM orchestration libraries.

Each submodule uses *soft imports* — the framework is only required
when the adapter is actually used. This keeps the default install
surface of ``execlave-sdk`` zero-footprint.

Sub-modules:
    langchain        — ``ExeclaveCallbackHandler`` (LangChain 0.3.x)
    openai_agents    — ``ExeclaveTracingProcessor`` (openai-agents SDK)
    crewai           — ``instrument_crew`` helper (CrewAI)
"""

__all__: list[str] = []
