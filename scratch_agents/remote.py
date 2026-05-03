"""Remote agent support via A2A protocol."""

from __future__ import annotations

from typing import Any

import httpx

from scratch_agents.context import AgentResult, ExecutionContext


class RemoteAgent:
    """Client for interacting with remote agents via A2A protocol."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.name = ""
        self.description = ""
        self._load_agent_info()

    def _load_agent_info(self) -> None:
        """Load agent info from the remote server's agent card."""
        try:
            response = httpx.get(f"{self.base_url}/.well-known/agent.json")
            if response.status_code == 200:
                info = response.json()
                self.name = info.get("name", "remote_agent")
                self.description = info.get("description", "")
        except Exception:
            self.name = "remote_agent"

    async def run(
        self,
        user_input: str,
        context: ExecutionContext | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Send a request to the remote agent."""
        if context is None:
            context = ExecutionContext()

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/tasks/send",
                json={
                    "jsonrpc": "2.0",
                    "method": "tasks/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"type": "text", "text": user_input}],
                        }
                    },
                },
                timeout=120.0,
            )

            result = response.json()
            output = ""

            if "result" in result:
                task_result = result["result"]
                if "artifacts" in task_result:
                    parts = []
                    for artifact in task_result["artifacts"]:
                        for part in artifact.get("parts", []):
                            if part.get("type") == "text":
                                parts.append(part["text"])
                    output = "\n".join(parts)

            return AgentResult(output=output, context=context)
