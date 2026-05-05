"""
Shared test helpers for the Execlave Python SDK test suite.
"""

from unittest.mock import MagicMock


def make_mock_response(status_code: int = 200, json_data: dict | None = None, text: str = ""):
    """Build a fake requests.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 400
    resp.json.return_value = json_data or {}
    resp.text = text
    return resp


SAMPLE_AGENT_DATA = {
    "id": "agt_abc123",
    "agentId": "my-bot",
    "name": "My Bot",
    "environment": "production",
    "status": "active",
    "type": "chatbot",
    "platform": "custom",
}
