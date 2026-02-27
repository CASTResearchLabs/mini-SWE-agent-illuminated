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
    logger.info(f"Parsing tool calls. Received {len(tool_calls)} tool calls")
    logger.debug(f"Tool calls: {tool_calls}")
    
    if not tool_calls:
        logger.error("No tool calls found in the response")
        raise FormatError(
            {
                "role": "user",
                "content": Template(format_error_template, undefined=StrictUndefined).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
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
