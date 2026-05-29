"""
Shared pytest fixtures for the Execlave Python SDK test suite.

All HTTP calls are intercepted via unittest.mock so no real server is needed.
"""

import pytest
from unittest.mock import patch, MagicMock

from execlave.client import Execlave
from execlave.agent import Agent

from tests.helpers import make_mock_response, SAMPLE_AGENT_DATA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_session():
    """
    Patch ``requests.Session`` so that every instance's ``.request()`` returns
    a configurable mock.  Yields the mock *method* so tests can set
    ``mock_session.return_value`` or use ``side_effect``.
    """
    with patch("execlave.client.requests.Session") as SessionClass:
        session_instance = MagicMock()
        SessionClass.return_value = session_instance

        # Default: return an empty 200
        session_instance.request.return_value = make_mock_response(200, {})
        session_instance.headers = MagicMock()

        yield session_instance


@pytest.fixture()
def ag_client(mock_session):
    """
    A pre-configured Execlave client that uses the mocked HTTP session.

    Background threads are disabled (async_mode=False, control_channel off)
    to keep tests deterministic.
    """
    client = Execlave(
        api_key="exe_test_key_0123456789",
        base_url="http://mock-server:4000",
        environment="test",
        async_mode=False,
        enable_control_channel=False,
        enable_injection_scan=False,
        debug=False,
    )
    return client


# Alias for tests that reference the old fixture name
exe_client = ag_client


@pytest.fixture()
def sample_agent(ag_client, mock_session):
    """
    A registered Agent fixture that is already tracked by ``ag_client``.
    """
    mock_session.request.return_value = make_mock_response(200, {"data": SAMPLE_AGENT_DATA})
    agent = ag_client.register_agent(agent_id="my-bot", name="My Bot")
    return agent
