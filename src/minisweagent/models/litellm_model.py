import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.anthropic_utils import _reorder_anthropic_thinking_blocks
from minisweagent.models.utils.cache_control import set_cache_control
from minisweagent.models.utils.mcp_http_tools import build_mcp_openai_tools, invoke_mcp_action
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("litellm_model")

class LitellmModelConfig(BaseModel):
    model_name: str
    """Model name. Highly recommended to include the provider in the model name, e.g., `anthropic/claude-sonnet-4-5-20250929`."""
    model_kwargs: dict[str, Any] = {}
    """Additional arguments passed to the API."""
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    """Model registry for cost tracking and model metadata. See the local model guide (https://mini-swe-agent.com/latest/models/local_models/) for more details."""
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""
    mcp_http_config: Path | str | dict | None = None
    """Path to MCP Streamable-HTTP server configuration (JSON/YAML), or inlined config dict."""
    mcp_tool_prefix: str = "mcp__"
    """Prefix for exposing MCP tools to the LM."""
    mcp_http_timeout: int = 20
    """Timeout in seconds for MCP HTTP requests."""


class LitellmModel:
    abort_exceptions: list[type[Exception]] = [
        litellm.exceptions.UnsupportedParamsError,
        litellm.exceptions.NotFoundError,
        litellm.exceptions.PermissionDeniedError,
        litellm.exceptions.ContextWindowExceededError,
        litellm.exceptions.AuthenticationError,
        KeyboardInterrupt,
    ]

    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self._action_tool_mapping: dict[str, dict] = {}
        self._tools = [BASH_TOOL]
        self._startup_mcp_capabilities_message: str | None = None
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))
        if self.config.mcp_http_config:
            mcp_tools, mapping = build_mcp_openai_tools(
                self.config.mcp_http_config,
                prefix=self.config.mcp_tool_prefix,
                timeout=self.config.mcp_http_timeout,
            )
            self._tools += mcp_tools
            self._action_tool_mapping |= mapping
            self._startup_mcp_capabilities_message = self._get_startup_mcp_capabilities_message()

        tool_names = [tool.get("function", {}).get("name", "") for tool in self._tools]
        logger.info(f"Tools enabled at startup ({len(tool_names)}): {tool_names}")

    def _get_startup_mcp_capabilities_message(self) -> str | None:
        structural_entries = [
            mapping for mapping in self._action_tool_mapping.values() if mapping.get("mcp_tool") == "run_structural_search_function"
        ]
        if not structural_entries:
            return None

        result = invoke_mcp_action(
            {
                **structural_entries[0],
                "arguments": {"function_name": "get_guidelines", "arguments": {"name":"change_impact_assessment"}},
            },
            timeout=self.config.mcp_http_timeout,
        )
        if result.get("returncode") != 0:
            logger.warning(
                "Startup MCP discovery failed for run_structural_search_function(list_functions, {}): "
                f"{result.get('exception_info') or result.get('output', '')}"
            )
            return None

        output = str(result.get("output", "")).strip()
        if len(output) > 6000:
            output = output[:6000] + "\n...[truncated]"
        logger.info("Startup MCP discovery succeeded: appended list_functions output to first prompt")
        return (
            "<mcp_capabilities>\n"
            "Startup discovery result from run_structural_search_function(function_name='list_functions', parameters={}):\n"
            f"{output}\n"
            "Use these MCP capabilities directly as function tool calls (JSON arguments), not bash commands.\n"
            "</mcp_capabilities>"
        )

    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=self._tools,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        prepared = _reorder_anthropic_thinking_blocks(prepared)
        return set_cache_control(prepared, mode=self.config.set_cache_control)

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        messages_for_query = messages
        if self._startup_mcp_capabilities_message:
            messages_for_query = [
                *messages,
                {"role": "user", "content": self._startup_mcp_capabilities_message},
            ]
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                response = self._query(self._prepare_messages_for_api(messages_for_query), **kwargs)
        
        self._startup_mcp_capabilities_message = None
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])

        if not response.choices:
            finish_reason = getattr(response, "finish_reason", None)
            logger.error(f"Model returned empty choices list (finish_reason={finish_reason}). "
                         "This typically indicates a safety filter block or provider-side error.")
            raise RuntimeError(
                f"Model returned empty choices (finish_reason={finish_reason}). "
                "Response may have been blocked by a safety filter."
            )

        message = response.choices[0].message.model_dump()
        
        # Log the model response content BEFORE trying to parse actions
        content = message.get("content", "")
        tool_calls = message.get("tool_calls") or []  # Handle None case
        
        content_length = len(content) if content else 0
        tool_calls_length = len(tool_calls) if tool_calls else 0
        
        logger.info(f"Model response received: content_length={content_length}, tool_calls={tool_calls_length}")
        
        if content:
            logger.debug(f"Model content (first 500 chars): {content[:500]}...")
        
        if not tool_calls:
            logger.warning("🚨 Model provided content but NO TOOL CALLS - this will trigger FormatError")
            if content:
                logger.info(f"Full model response content:\n{content}")
            else:
                logger.info("Model response had no content and no tool calls")
        
        # Now try to parse actions (this may raise FormatError)
        try:
            actions = self._parse_actions(response)
        except Exception as e:
            logger.error(f"Failed to parse actions from model response: {type(e).__name__}: {e}")
            logger.error(f"Model response that failed to parse:\nContent: {content}\nTool calls: {tool_calls}")
            raise
        
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _calculate_cost(self, response) -> dict[str, float]:
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        usage = getattr(response, "usage", None) or {}
        prompt_tokens = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens", 0) or 0
        return {"cost": cost, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

    def _parse_actions(self, response) -> list[dict]:
        """Parse tool calls from the response. Raises FormatError if unknown tool."""
        logger.info("Parsing tool calls from model response")
        logger.debug(f"Model response: {response}")
        
        tool_calls = response.choices[0].message.tool_calls or []
        logger.info(f"Found {len(tool_calls)} tool calls in response")
        
        if not tool_calls:
            logger.warning("⚠️  ZERO tool calls detected - this will cause FormatError")
            
            # Log additional debugging information
            message = response.choices[0].message
            content = getattr(message, 'content', None)
            logger.warning(f"Message content: {content[:500] if content else 'None'}...")
            
            # Check if there are any function calls in a different format
            if hasattr(message, 'function_call'):
                logger.debug(f"Function call (deprecated format): {message.function_call}")
            
            # Log the full message structure for debugging
            logger.debug(f"Full message structure: {message}")
        
        try:
            parsed_actions = parse_toolcall_actions(
                tool_calls,
                format_error_template=self.config.format_error_template,
                action_tool_mapping=self._action_tool_mapping,
            )
            logger.info(f"Successfully parsed {len(parsed_actions)} actions")
            return parsed_actions
        except Exception as e:
            logger.error(f"Failed to parse tool calls: {type(e).__name__}: {e}")
            raise

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        """Format execution outputs into tool result messages."""
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
