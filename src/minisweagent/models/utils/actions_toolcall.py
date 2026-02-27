"""Parse actions & format observations with toolcalls"""

import json
import logging
import time

from jinja2 import StrictUndefined, Template

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

# Set up logger for toolcall actions
logger = logging.getLogger(__name__)

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


def parse_toolcall_actions(
    tool_calls: list,
    *,
    format_error_template: str,
    action_tool_mapping: dict[str, dict] | None = None,
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    logger.info(f"Parsing tool calls. Received {len(tool_calls or [])} tool calls")
    logger.debug(f"Tool calls: {tool_calls}")
    
    if not tool_calls:
        logger.error("🚨 CRITICAL: No tool calls found in the response")
        logger.error("This indicates the model generated a response without using any tools")
        logger.error("This will result in a FormatError and waste a turn")
        
        # Log available tools for debugging
        if action_tool_mapping:
            available_tools = list(action_tool_mapping.keys()) + ["bash"]
            logger.debug(f"Available tools were: {available_tools}")
        else:
            logger.debug("Available tools: bash (MCP tools not configured)")
        
        # Create detailed error message
        error_msg = "No tool calls found in the response. Every response MUST include at least one tool call."
        logger.error(f"Raising FormatError: {error_msg}")
        
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error=error_msg,
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    for tool_call in tool_calls:
        logger.info(f"Processing tool call: {tool_call.function.name}")
        error_msg = ""
        args = {}
        tool_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            logger.debug(f"Parsed arguments for {tool_name}: {args}")
        except Exception as e:
            error_msg = f"Error parsing tool call arguments: {e}."
            logger.error(f"Failed to parse arguments for {tool_name}: {e}")

        if tool_name == "bash":
            if not isinstance(args, dict) or "command" not in args:
                error_msg += "Missing 'command' argument in bash tool call."
            command = args.get("command", "") if isinstance(args, dict) else ""
            if action_tool_mapping and isinstance(command, str):
                command = command.strip()
                mcp_openai_tool_names = {name for name in action_tool_mapping.keys() if name}
                mcp_backend_tool_names = {
                    str(meta.get("mcp_tool", "")) for meta in action_tool_mapping.values() if meta.get("mcp_tool")
                }
                mcp_names = mcp_openai_tool_names | mcp_backend_tool_names
                matched_mcp_name = next(
                    (name for name in mcp_names if command == name or command.startswith(f"{name} ")),
                    "",
                )
                if matched_mcp_name:
                    error_msg += (
                        "MCP tool call detected inside bash command. "
                        "Do not run MCP tools via shell syntax. "
                        f"Call the MCP function tool '{matched_mcp_name}' directly with JSON arguments. "
                        "For run_structural_search_function use arguments like "
                        "{'function_name': 'list_functions', 'parameters': {}}. "
                        "If available, use the exact syntax from the tool response 'syntax' field."
                    )
            if error_msg:
                logger.error(f"Bash tool error: {error_msg}")
                raise FormatError(
                    {
                        "role": "user",
                        "content": Template(format_error_template, undefined=StrictUndefined).render(
                            actions=[], error=error_msg.strip()
                        ),
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )
            logger.info(f"Adding bash command: {args['command']}")
            actions.append({"command": args["command"], "tool_call_id": tool_call.id})
            continue

        if action_tool_mapping and tool_name in action_tool_mapping:
            logger.info(f"Processing MCP tool: {tool_name}")
            if not isinstance(args, dict):
                error_msg += "MCP tool arguments must be a JSON object."
            if error_msg:
                logger.error(f"MCP tool error: {error_msg}")
                raise FormatError(
                    {
                        "role": "user",
                        "content": Template(format_error_template, undefined=StrictUndefined).render(
                            actions=[], error=error_msg.strip()
                        ),
                        "extra": {"interrupt_type": "FormatError"},
                    }
                )
            logger.info(f"Adding MCP action: {tool_name} with args: {args}")
            actions.append(
                {
                    **action_tool_mapping[tool_name],
                    "arguments": args,
                    "tool_call_id": tool_call.id,
                }
            )
            continue

        logger.error(f"Unknown tool: {tool_name}")
        logger.debug(f"Available tools in mapping: {list(action_tool_mapping.keys()) if action_tool_mapping else 'None'}")
        error_msg += f"Unknown tool '{tool_name}'."
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    actions=[], error=error_msg.strip()
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
