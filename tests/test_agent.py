"""
Tests for Execlave.agent.Agent and PromptVersion.

Covers construction, status properties, prompt deployment, and rollback.
"""

import pytest
from unittest.mock import MagicMock

from execlave.agent import Agent, PromptVersion
from execlave.errors import ExeclaveError
from tests.helpers import make_mock_response, SAMPLE_AGENT_DATA


# =========================================================================
# Agent construction
# =========================================================================


class TestAgentInit:

    def test_attributes_populated(self, ag_client):
        agent = Agent(ag_client, SAMPLE_AGENT_DATA)
        assert agent.id == "agt_abc123"
        assert agent.agent_id == "my-bot"
        assert agent.name == "My Bot"
        assert agent.environment == "production"
        assert agent.status == "active"

    def test_defaults_for_missing_keys(self, ag_client):
        agent = Agent(ag_client, {})
        assert agent.id == ""
        assert agent.agent_id == ""
        assert agent.name == ""
        assert agent.status == "active"


# =========================================================================
# is_paused property
# =========================================================================


class TestIsPaused:

    def test_active_agent_not_paused(self, ag_client):
        agent = Agent(ag_client, {**SAMPLE_AGENT_DATA, "status": "active"})
        assert agent.is_paused is False

    def test_paused_agent(self, ag_client):
        agent = Agent(ag_client, {**SAMPLE_AGENT_DATA, "status": "paused"})
        assert agent.is_paused is True

    def test_pause_resume(self, ag_client):
        agent = Agent(ag_client, SAMPLE_AGENT_DATA)
        assert agent.is_paused is False
        agent.status = "paused"
        assert agent.is_paused is True
        agent.status = "active"
        assert agent.is_paused is False


# =========================================================================
# deploy_prompt
# =========================================================================


class TestDeployPrompt:

    def test_successful_deploy(self, sample_agent, mock_session):
        version_data = {
            "id": "pv_123",
            "versionNumber": 1,
            "versionTag": "v1.0.0",
            "approvalStatus": "pending",
            "isActive": False,
        }
        mock_session.request.return_value = make_mock_response(200, {"data": version_data})

        pv = sample_agent.deploy_prompt(
            prompt_template="Hello {name}",
            system_message="You are helpful.",
            model_name="gpt-4-turbo",
            model_parameters={"temperature": 0.7, "max_tokens": 256},
            change_type="major",
            change_description="Initial prompt",
            version_tag="v1.0.0",
        )

        assert isinstance(pv, PromptVersion)
        assert pv.id == "pv_123"
        assert pv.version_number == 1
        assert pv.version_tag == "v1.0.0"
        assert pv.approval_status == "pending"

    def test_deploy_prompt_payload(self, sample_agent, mock_session):
        mock_session.request.return_value = make_mock_response(200, {"data": {}})
        sample_agent.deploy_prompt(
            prompt_template="How are you?",
            model_name="gpt-4o",
            model_parameters={"temperature": 0.5, "max_tokens": 100, "top_p": 0.9},
            change_description="test",
        )

        # The second call on mock_session.request (first was register_agent)
        calls = [c for c in mock_session.request.call_args_list if "/prompt-versions" in str(c)]
        assert len(calls) >= 1
        _, kwargs = calls[-1]
        payload = kwargs["json"]
        assert payload["promptTemplate"] == "How are you?"
        assert payload["modelName"] == "gpt-4o"
        assert payload["temperature"] == 0.5
        assert payload["maxTokens"] == 100
        assert payload["otherParams"]["top_p"] == 0.9

    def test_deploy_prompt_strips_none_values(self, sample_agent, mock_session):
        mock_session.request.return_value = make_mock_response(200, {"data": {}})
        sample_agent.deploy_prompt(prompt_template="Hi")

        calls = [c for c in mock_session.request.call_args_list if "/prompt-versions" in str(c)]
        _, kwargs = calls[-1]
        payload = kwargs["json"]
        # None values should have been stripped
        assert "systemMessage" not in payload
        assert "modelName" not in payload


# =========================================================================
# rollback_to_version
# =========================================================================


class TestRollback:

    def test_successful_rollback(self, sample_agent, mock_session):
        versions_resp = {"data": [
            {"id": "pv_1", "versionNumber": 1, "versionTag": "v1.0.0"},
            {"id": "pv_2", "versionNumber": 2, "versionTag": "v2.0.0"},
        ]}
        rollback_resp = {"data": {
            "id": "pv_1",
            "versionNumber": 1,
            "versionTag": "v1.0.0",
            "approvalStatus": "approved",
            "isActive": True,
        }}
        mock_session.request.side_effect = [
            make_mock_response(200, versions_resp),  # GET list
            make_mock_response(200, rollback_resp),   # POST rollback
        ]
        pv = sample_agent.rollback_to_version(1, reason="Bug found")
        assert isinstance(pv, PromptVersion)
        assert pv.version_number == 1

    def test_rollback_version_not_found(self, sample_agent, mock_session):
        mock_session.request.return_value = make_mock_response(200, {"data": []})
        with pytest.raises(ExeclaveError, match="not found"):
            sample_agent.rollback_to_version(99)


# =========================================================================
# PromptVersion
# =========================================================================


class TestPromptVersion:

    def test_attributes(self, ag_client):
        pv = PromptVersion(ag_client, {
            "id": "pv_abc",
            "versionNumber": 3,
            "versionTag": "v3.0.0",
            "approvalStatus": "approved",
            "isActive": True,
        })
        assert pv.id == "pv_abc"
        assert pv.version_number == 3
        assert pv.version_tag == "v3.0.0"
        assert pv.approval_status == "approved"
        assert pv.is_active is True

    def test_defaults(self, ag_client):
        pv = PromptVersion(ag_client, {})
        assert pv.id == ""
        assert pv.version_number == 0
        assert pv.is_active is False

    def test_promote_to_production(self, ag_client, mock_session):
        pv = PromptVersion(ag_client, {"id": "pv_xyz", "isActive": False, "approvalStatus": "pending"})
        mock_session.request.return_value = make_mock_response(200, {"data": {
            "approvalStatus": "approved",
            "isActive": True,
        }})
        result = pv.promote_to_production()
        assert result is pv  # returns self
        assert pv.approval_status == "approved"
        assert pv.is_active is True
