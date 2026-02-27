"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
from logging.handlers import RotatingFileHandler

import os
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded
from minisweagent.utils.log import log_conversation_state
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.format_error_count = 0  # Track wasted turns

        if not self.logger.handlers:
            os.makedirs("logs", exist_ok=True)
            handler = RotatingFileHandler(os.path.join("logs", "agent.log"), maxBytes=1000000, backupCount=3)
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info(f"Agent initialized: model={model.__class__.__name__}, env={env.__class__.__name__}")


    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {"n_model_calls": self.n_calls, "model_cost": self.cost},
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content_preview = str(msg.get("content", ""))[:100].replace('\n', ' ')
            extra_info = ""
            if "extra" in msg:
                extra_keys = list(msg["extra"].keys())
                interrupt_type = msg.get("extra", {}).get("interrupt_type")
                if interrupt_type:
                    extra_info = f" [{interrupt_type}]"
                elif extra_keys:
                    extra_info = f" (extra: {extra_keys})"
            
            self.logger.debug(f"Adding message {i+1}/{len(messages)}: {role}{extra_info} - {content_preview}...")
        
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.format_error_count = 0
        
        self.logger.info(f"Starting agent run with task: {task[:100]}..." if len(task) > 100 else f"Starting agent run with task: {task}")
        
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        
        step_count = 0
        while True:
            step_count += 1
            self.logger.info(f"Starting step {step_count} (total calls: {self.n_calls}, cost: ${self.cost:.4f})")
            
            try:
                result = self.step()
                self.logger.info(f"Step {step_count} completed successfully, added {len(result)} observation messages")
            except InterruptAgentFlow as e:
                # Check if this is a FormatError (wasted turn)
                is_format_error = any(
                    msg.get("extra", {}).get("interrupt_type") == "FormatError" 
                    for msg in e.messages
                )
                if is_format_error:
                    self.format_error_count += 1
                    self.logger.warning(
                        f"🚨 WASTED TURN #{self.format_error_count} - FormatError occurred at step {step_count}. "
                        f"Model failed to make tool calls. Total wasted turns: {self.format_error_count}"
                    )
                    
                    # Log the error message content for debugging
                    for msg in e.messages:
                        if msg.get("extra", {}).get("interrupt_type") == "FormatError":
                            error_content = msg.get('content', '')
                            self.logger.error(f"FormatError details: {error_content[:500]}...")
                    
                    # Try to find and log the original model response that caused this FormatError
                    # Look for the most recent assistant message that might have caused this
                    for msg in reversed(self.messages):
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            extra = msg.get('extra', {})
                            actions = extra.get('actions', [])
                            
                            if not actions:  # This was likely the problematic response
                                self.logger.error(f"📝 PROBLEMATIC MODEL RESPONSE (no actions):")
                                self.logger.error(f"Content: {content}")
                                
                                # Also log raw response if available
                                response_data = extra.get('response', {})
                                if hasattr(response_data, 'choices') and response_data.choices:
                                    raw_message = response_data.choices[0].message
                                    self.logger.error(f"Raw tool_calls: {getattr(raw_message, 'tool_calls', 'None')}")
                                break
                else:
                    interrupt_type = next(
                        (msg.get("extra", {}).get("interrupt_type", "Unknown") for msg in e.messages),
                        "Unknown"
                    )
                    self.logger.info(f"InterruptAgentFlow: {interrupt_type} at step {step_count}")
                
                self.add_messages(*e.messages)
                self.logger.debug(f"Added {len(e.messages)} interrupt messages to conversation")
                
            except Exception as e:
                self.logger.error(f"Unexpected exception at step {step_count}: {type(e).__name__}: {e}")
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            
            if self.messages[-1].get("role") == "exit":
                self.logger.info(f"Agent finished after {step_count} steps. Exit status: {self.messages[-1].get('extra', {}).get('exit_status', 'Unknown')}")
                break
                
            # Warning for excessive format errors
            if self.format_error_count >= 3:
                self.logger.warning(
                    f"⚠️  HIGH WASTED TURN COUNT: {self.format_error_count} format errors detected. "
                    "The model may not be learning from error feedback properly."
                )
                # Log detailed conversation state for debugging
                log_conversation_state(self.messages, logger_name="agent")
        
        final_result = self.messages[-1].get("extra", {})
        self.logger.info(
            f"Agent run completed: {step_count} steps, {self.n_calls} model calls, "
            f"${self.cost:.4f} cost, {self.format_error_count} wasted turns"
        )
        return final_result

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            self.logger.info(f"Limits exceeded: steps={self.n_calls}/{self.config.step_limit}, cost=${self.cost:.4f}/${self.config.cost_limit}")
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        
        self.n_calls += 1
        self.logger.info(f"Querying model (call #{self.n_calls})...")
        
        # Log conversation context
        self.logger.debug(f"Conversation has {len(self.messages)} messages")
        recent_roles = [msg.get('role', 'unknown') for msg in self.messages[-5:]]
        self.logger.debug(f"Recent message roles: {recent_roles}")
        
        # Try to get the model response, catching FormatError to log additional details
        try:
            message = self.model.query(self.messages)
        except Exception as e:
            # Log details about what might have caused the error
            self.logger.error(f"Model query failed: {type(e).__name__}: {e}")
            
            # If it's a format error, log conversation context that might help debug
            if "FormatError" in str(type(e)):
                self.logger.error("🚨 FormatError during model query - this is a wasted turn")
                self.logger.error(f"Last few messages that led to this error:")
                for i, msg in enumerate(self.messages[-3:]):
                    role = msg.get('role', 'unknown')
                    content_preview = str(msg.get('content', ''))[:200].replace('\n', ' ')
                    self.logger.error(f"  [{i-len(self.messages[-3:])}] {role}: {content_preview}...")
            
            raise
        
        call_cost = message.get("extra", {}).get("cost", 0.0)
        self.cost += call_cost
        
        # Log detailed information about the model response
        actions = message.get("extra", {}).get("actions", [])
        self.logger.info(f"Model responded with {len(actions)} actions, cost: ${call_cost:.6f}")
        
        if not actions:
            self.logger.warning("⚠️  Model response contains NO ACTIONS - this will cause a FormatError")
            
            # Log the actual response content for debugging
            content = message.get("content", "")
            self.logger.debug(f"Model response content (first 300 chars): {content[:300]}...")
            
            # Log if the response has tool_calls field
            response_data = message.get("extra", {}).get("response", {})
            if hasattr(response_data, 'choices'):
                tool_calls = getattr(response_data.choices[0].message, 'tool_calls', None) if response_data.choices else None
                self.logger.debug(f"Raw tool_calls in response: {tool_calls}")
        else:
            action_types = [action.get('type', action.get('command', 'unknown')[:20]) for action in actions]
            self.logger.info(f"Actions: {action_types}")
        
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        
        for action in actions:
            if action.get("type") == "mcp":
                # Route MCP actions directly to MCP handler
                from minisweagent.models.utils.mcp_http_tools import invoke_mcp_action
                self.logger.info(f"Executing MCP action: {action.get('mcp_tool')}")
                outputs.append(invoke_mcp_action(action))
            else:
                # Route bash commands to environment
                outputs.append(self.env.execute(action))
        
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
