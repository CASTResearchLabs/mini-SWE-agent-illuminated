import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Set up logger for MCP HTTP tools
logger = logging.getLogger(__name__)


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


async def _rpc_call(url: str, headers: dict[str, str], method: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Make an RPC call using the official MCP client."""
    logger.info(f"Making RPC call to {url}, method: {method}")
    
    # Create httpx client with headers
    http_client = httpx.AsyncClient(
        headers=headers if headers else {},
        timeout=httpx.Timeout(timeout)
    )
    
    try:
        async with streamable_http_client(url, http_client=http_client) as streams:
            read_stream, write_stream = streams[0], streams[1]  # Ignore the third value (session)
            async with ClientSession(read_stream, write_stream) as session:
                if method == "initialize":
                    logger.info("Initializing connection...")
                    await session.initialize()
                    logger.info("Connection initialized")
                    return {"protocolVersion": "2025-03-26", "capabilities": {}, "serverInfo": {"name": "mcp-server"}}
                
                elif method == "tools/list":
                    logger.info("Listing tools...")
                    tools_response = await session.list_tools()
                    logger.info(f"Found {len(tools_response.tools)} tools")
                    return {"tools": [{"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema} for tool in tools_response.tools]}
                
                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})
                    logger.info(f"Calling tool {tool_name} with arguments: {arguments}")
                    result = await session.call_tool(tool_name, arguments)
                    return {"content": [{"type": "text", "text": content.text if hasattr(content, 'text') else str(content)} for content in result.content]}
                
                else:
                    logger.warning(f"Unknown method: {method}")
                    return {}
                    
    except Exception as e:
        logger.error(f"Error in RPC call: {e}")
        raise RuntimeError(f"MCP RPC error calling {method}: {e}")
    finally:
        await http_client.aclose()


def _rpc(url: str, headers: dict[str, str], method: str, params: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Synchronous wrapper for async RPC calls."""
    return asyncio.run(_rpc_call(url, headers, method, params, timeout))


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
        try:
            logger.info(f"Processing server: {server.name}")
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
        except Exception as e:
            logger.error(f"Server {server.name} failed: {e}")
            logger.info(f"Skipping server {server.name} and continuing with others")
            continue
            
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
    
    logger.info(f"Successfully loaded {len(tools)} tools from available servers")
    return tools, action_tool_mapping


async def _invoke_mcp_action_async(action: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
    """Async version of invoke_mcp_action using proper MCP client."""
    url = action["mcp_url"]
    headers = action.get("mcp_headers", {})
    tool_name = action["mcp_tool"]
    arguments = action.get("arguments", {})
    
    logger.info(f"Invoking tool {tool_name} at {url}")
    
    # Create httpx client with headers
    http_client = httpx.AsyncClient(
        headers=headers if headers else {},
        timeout=httpx.Timeout(timeout)
    )
    
    try:
        async with streamable_http_client(url, http_client=http_client) as streams:
            read_stream, write_stream = streams[0], streams[1]  # Ignore the third value (session)
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                
                # Extract text content
                text_parts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        text_parts.append(content.text)
                    else:
                        text_parts.append(str(content))
                
                output_text = "\n".join(text_parts) if text_parts else "No output"
                logger.info(f"Tool {tool_name} completed successfully")
                
                return {
                    "output": output_text,
                    "returncode": 0,
                    "exception_info": "",
                    "extra": {"mcp": {"content": [{"type": "text", "text": part} for part in text_parts]}},
                }
                
    except Exception as e:
        logger.error(f"Error invoking action: {e}")
        return {
            "output": f"Error: {e}",
            "returncode": 1,
            "exception_info": str(e),
            "extra": {"mcp": {"error": str(e)}},
        }
    finally:
        await http_client.aclose()


def invoke_mcp_action(action: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
    """Synchronous wrapper for async MCP action invocation."""
    return asyncio.run(_invoke_mcp_action_async(action, timeout=timeout))