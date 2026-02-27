import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import yaml


@dataclass
class MCPHTTPServerConfig:
    name: str
    url: str
    headers: dict[str, str]


def _load_config(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    raw = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    return yaml.safe_load(raw)


def _normalize_server_config(raw: dict[str, Any]) -> MCPHTTPServerConfig:
    return MCPHTTPServerConfig(
        name=str(raw["name"]),
        url=str(raw["url"]),
        headers={str(k): str(v) for k, v in (raw.get("headers") or {}).items()},
    )


def load_mcp_http_servers(path: Path | str) -> list[MCPHTTPServerConfig]:
    config_path = Path(path)
    data = _load_config(config_path)
    servers_raw = data if isinstance(data, list) else data.get("servers", [])
    return [_normalize_server_config(server) for server in servers_raw]


def _rpc(url: str, headers: dict[str, str], method: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"MCP RPC error calling {method}: {data['error']}")
    return data.get("result", {})


def _sanitize_tool_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "tool"


def build_mcp_openai_tools(
    mcp_http_config: Path | str,
    *,
    prefix: str = "mcp__",
    timeout: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tools: list[dict[str, Any]] = []
    action_tool_mapping: dict[str, dict[str, Any]] = {}
    used_names: set[str] = set()

    for server in load_mcp_http_servers(mcp_http_config):
        _rpc(
            server.url,
            server.headers,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "mini-swe-agent", "version": "2"},
            },
            timeout,
        )
        list_result = _rpc(server.url, server.headers, "tools/list", {}, timeout)
        for mcp_tool in list_result.get("tools", []):
            base_name = f"{prefix}{_sanitize_tool_name(server.name)}__{_sanitize_tool_name(mcp_tool.get('name', 'tool'))}"
            openai_name = base_name
            suffix = 2
            while openai_name in used_names:
                openai_name = f"{base_name}_{suffix}"
                suffix += 1
            used_names.add(openai_name)

            parameters = mcp_tool.get("inputSchema") or {"type": "object", "properties": {}}
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": openai_name,
                        "description": mcp_tool.get("description") or f"MCP tool {mcp_tool.get('name', '')}",
                        "parameters": parameters,
                    },
                }
            )
            action_tool_mapping[openai_name] = {
                "type": "mcp",
                "mcp_server": server.name,
                "mcp_url": server.url,
                "mcp_headers": server.headers,
                "mcp_tool": mcp_tool.get("name", ""),
            }
    return tools, action_tool_mapping


def invoke_mcp_action(action: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
    result = _rpc(
        action["mcp_url"],
        action.get("mcp_headers", {}),
        "tools/call",
        {"name": action["mcp_tool"], "arguments": action.get("arguments", {})},
        timeout,
    )

    content = result.get("content") or []
    text_parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
        else:
            text_parts.append(json.dumps(item, ensure_ascii=False))
    output_text = "\n".join(part for part in text_parts if part)
    if not output_text:
        output_text = json.dumps(result, ensure_ascii=False)

    return {
        "output": output_text,
        "returncode": 1 if result.get("isError") else 0,
        "exception_info": "",
        "extra": {"mcp": result},
    }