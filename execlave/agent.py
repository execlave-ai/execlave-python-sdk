"""Agent object returned from register_agent()."""

from typing import Any, Optional


class Agent:
    """
    Represents a registered agent. Provides prompt management methods.
    """

    def __init__(self, exe: Any, data: dict):
        self._exe = exe
        self.id: str = data.get("id", "")
        self.agent_id: str = data.get("agentId", "")
        self.name: str = data.get("name", "")
        self.environment: str = data.get("environment", "")
        self.status: str = data.get("status", "active")
        self._data = data

    @property
    def is_paused(self) -> bool:
        return self.status == "paused"

    def deploy_prompt(
        self,
        prompt_template: str,
        system_message: str | None = None,
        model_name: str | None = None,
        model_parameters: dict | None = None,
        change_type: str = "minor",
        change_description: str | None = None,
        description: str | None = None,
        version_tag: str | None = None,
        environment: str | None = None,
        require_approval: bool = True,
    ) -> "PromptVersion":
        """
        Deploy a new prompt version for this agent.
        
        Args:
            prompt_template: The prompt template string with {placeholders}
            system_message: Optional system message
            model_name: Model to use (e.g. 'gpt-4-turbo')
            model_parameters: Dict with temperature, max_tokens, etc.
            change_type: 'major' | 'minor' | 'patch'
            change_description: Description of what changed
            description: Alias for change_description (for JS SDK parity)
            version_tag: Semantic version tag (e.g. 'v1.6.0')
            environment: Target environment
            require_approval: Whether to require approval before deployment
            
        Returns:
            PromptVersion object
        """
        params = model_parameters or {}
        resolved_description = change_description or description
        payload = {
            "agentId": self.id,
            "promptTemplate": prompt_template,
            "systemMessage": system_message,
            "modelName": model_name,
            "temperature": params.get("temperature"),
            "maxTokens": params.get("max_tokens"),
            "otherParams": {k: v for k, v in params.items() if k not in ("temperature", "max_tokens")},
            "changeSummary": resolved_description or change_type,
            "versionTag": version_tag,
        }
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        resp = self._exe._request("POST", self._exe._api_path("/prompt-versions"), json=payload)
        version_data = resp.get("data", {})
        return PromptVersion(self._exe, version_data)

    def get_current_prompt(self) -> Optional["PromptVersion"]:
        """
        Fetch the currently deployed prompt version for this agent.

        Returns:
            PromptVersion or None if no version is deployed.
        """
        resp = self._exe._request(
            "GET",
            self._exe._api_path(f"/prompt-versions?agentId={self.id}&deployed=true"),
        )
        versions = resp.get("data", [])
        for v in versions:
            if v.get("isDeployed"):
                return PromptVersion(self._exe, v)
        return None

    def list_prompt_versions(self) -> list["PromptVersion"]:
        """
        List all prompt versions for this agent.

        Returns:
            List of PromptVersion objects.
        """
        resp = self._exe._request(
            "GET",
            self._exe._api_path(f"/prompt-versions?agentId={self.id}"),
        )
        return [PromptVersion(self._exe, v) for v in resp.get("data", [])]

    def refresh_status(self) -> str:
        """
        Poll the current agent status from the API.

        Updates self.status in-place and returns it.
        """
        resp = self._exe._request(
            "GET",
            self._exe._api_path(f"/agents/{self.id}/status-poll"),
        )
        data = resp.get("data", {})
        self.status = data.get("status", self.status)
        return self.status

    def promote_to_production(
        self,
        version_id: str,
        require_approval: bool = False,
        deployment_notes: str | None = None,
    ) -> "PromptVersion":
        """
        Promote a specific prompt version to production.

        Args:
            version_id: The ID of the version to promote.
            require_approval: Whether to require admin approval.
            deployment_notes: Notes for the deployment.

        Returns:
            PromptVersion object for the promoted version.
        """
        payload: dict = {
            "environment": "production",
            "requireApproval": require_approval,
        }
        if deployment_notes is not None:
            payload["deploymentNotes"] = deployment_notes

        resp = self._exe._request(
            "POST",
            self._exe._api_path(f"/prompt-versions/{version_id}/deploy"),
            json=payload,
        )
        return PromptVersion(self._exe, resp.get("data", {}))

    def rollback_to_version(self, version_number: int, reason: str = "Rollback requested") -> "PromptVersion":
        """
        Rollback to a specific prompt version by version number.
        
        Args:
            version_number: The version number to rollback to
            reason: Reason for the rollback
        """
        # Find the version by number
        versions = self._exe._request("GET", self._exe._api_path(f"/prompt-versions?agentId={self.id}"))
        target = None
        for v in versions.get("data", []):
            if v.get("versionNumber") == version_number:
                target = v
                break

        if not target:
            from .errors import ExeclaveError
            raise ExeclaveError(f"Version {version_number} not found for agent {self.agent_id}")

        resp = self._exe._request("POST", self._exe._api_path(f"/prompt-versions/{target['id']}/rollback"), json={"reason": reason})
        return PromptVersion(self._exe, resp.get("data", {}))


class PromptVersion:
    """Represents a prompt version."""

    def __init__(self, exe: Any, data: dict):
        self._exe = exe
        self.id: str = data.get("id", "")
        self.version_number: int = data.get("versionNumber", 0)
        self.version_tag: str = data.get("versionTag", "")
        self.approval_status: str = data.get("approvalStatus", "pending")
        self.is_active: bool = data.get("isActive", False)
        self._data = data

    def promote_to_production(
        self,
        require_approval: bool = True,
        deployment_notes: str | None = None,
    ) -> "PromptVersion":
        """
        Deploy this version to production.
        
        Args:
            require_approval: Whether approval is required first
            deployment_notes: Notes about this deployment
        """
        resp = self._exe._request("POST", self._exe._api_path(f"/prompt-versions/{self.id}/deploy"), json={
            "environment": "production"
        })
        data = resp.get("data", {})
        self.approval_status = data.get("approvalStatus", self.approval_status)
        self.is_active = data.get("isActive", self.is_active)
        return self
