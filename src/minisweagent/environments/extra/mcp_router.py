from typing import Any

from pydantic import BaseModel

from minisweagent import Environment
from minisweagent.models.utils.mcp_http_tools import invoke_mcp_action
from minisweagent.utils.serialize import recursive_merge


class MCPRouterEnvironmentConfig(BaseModel):
    mcp_timeout: int = 60


class MCPRouterEnvironment:
    def __init__(self, base_environment: Environment, *, config_class: type = MCPRouterEnvironmentConfig, **kwargs):
        self.base_environment = base_environment
        self.config = config_class(**kwargs)

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        if action.get("type") == "mcp":
            return invoke_mcp_action(action, timeout=timeout or self.config.mcp_timeout)
        return self.base_environment.execute(action, cwd=cwd, timeout=timeout)

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.base_environment.get_template_vars(), {"mcp_enabled": True}, kwargs)

    def serialize(self) -> dict:
        return recursive_merge(
            self.base_environment.serialize(),
            {
                "info": {
                    "config": {
                        "mcp_router": self.config.model_dump(mode="json"),
                        "mcp_router_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                    }
                }
            },
        )