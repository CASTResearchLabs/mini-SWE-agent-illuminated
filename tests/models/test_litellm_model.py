from unittest.mock import MagicMock, patch

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class TestLitellmModelConfig:
    def test_default_format_error_template(self):
        assert LitellmModelConfig(model_name="test").format_error_template == "{{ error }}"


def _mock_litellm_response(tool_calls):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.tool_calls = tool_calls
    mock_response.choices[0].message.model_dump.return_value = {"role": "assistant", "content": None}
    mock_response.model_dump.return_value = {}
    return mock_response


class TestLitellmModel:
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_query_includes_bash_tool(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "echo test"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        model.query([{"role": "user", "content": "test"}])

        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["tools"] == [BASH_TOOL]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_valid_tool_call(self, mock_cost, mock_completion):
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls -la"}'
        tool_call.id = "call_abc"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        result = model.query([{"role": "user", "content": "list files"}])
        assert result["extra"]["actions"] == [{"command": "ls -la", "tool_call_id": "call_abc"}]

    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_parse_actions_no_tool_calls_raises(self, mock_cost, mock_completion):
        mock_completion.return_value = _mock_litellm_response(None)
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4")
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])

    def test_format_observation_messages(self):
        model = LitellmModel(model_name="gpt-4", observation_template="{{ output.output }}")
        message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
        outputs = [{"output": "test output", "returncode": 0}]
        result = model.format_observation_messages(message, outputs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "test output"

    def test_format_observation_messages_no_actions(self):
        model = LitellmModel(model_name="gpt-4")
        result = model.format_observation_messages({"extra": {}}, [])
        assert result == []

    @patch("minisweagent.models.litellm_model.invoke_mcp_action")
    @patch("minisweagent.models.litellm_model.build_mcp_openai_tools")
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_discovery_failure_does_not_block_bash(
        self, mock_cost, mock_completion, mock_build_mcp_tools, mock_invoke_mcp_action
    ):
        mock_build_mcp_tools.return_value = (
            [{"type": "function", "function": {"name": "mcp__structural_search__run_structural_search_function"}}],
            {
                "mcp__structural_search__run_structural_search_function": {
                    "type": "mcp",
                    "mcp_server": "structural-search",
                    "mcp_url": "http://localhost:8282/mcp",
                    "mcp_headers": {},
                    "mcp_tool": "run_structural_search_function",
                }
            },
        )
        mock_invoke_mcp_action.return_value = {"returncode": -1, "output": "", "exception_info": "failed"}
        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls /testbed"}'
        tool_call.id = "call_1"
        mock_completion.return_value = _mock_litellm_response([tool_call])
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", mcp_http_config="dummy.yaml")
        result = model.query([{"role": "user", "content": "test"}])
        assert result["extra"]["actions"] == [{"command": "ls /testbed", "tool_call_id": "call_1"}]

    @patch("minisweagent.models.litellm_model.invoke_mcp_action")
    @patch("minisweagent.models.litellm_model.build_mcp_openai_tools")
    @patch("minisweagent.models.litellm_model.litellm.completion")
    @patch("minisweagent.models.litellm_model.litellm.cost_calculator.completion_cost")
    def test_startup_discovery_is_injected_once(
        self, mock_cost, mock_completion, mock_build_mcp_tools, mock_invoke_mcp_action
    ):
        mock_build_mcp_tools.return_value = (
            [{"type": "function", "function": {"name": "mcp__structural_search__run_structural_search_function"}}],
            {
                "mcp__structural_search__run_structural_search_function": {
                    "type": "mcp",
                    "mcp_server": "structural-search",
                    "mcp_url": "http://localhost:8282/mcp",
                    "mcp_headers": {},
                    "mcp_tool": "run_structural_search_function",
                }
            },
        )
        mock_invoke_mcp_action.return_value = {"returncode": 0, "output": "f1\nf2", "exception_info": ""}

        tool_call = MagicMock()
        tool_call.function.name = "bash"
        tool_call.function.arguments = '{"command": "ls /testbed"}'
        tool_call.id = "call_bash"
        mock_completion.side_effect = [_mock_litellm_response([tool_call]), _mock_litellm_response([tool_call])]
        mock_cost.return_value = 0.001

        model = LitellmModel(model_name="gpt-4", mcp_http_config="dummy.yaml")
        model.query([{"role": "user", "content": "test"}])
        model.query([{"role": "user", "content": "test"}])

        first_messages = mock_completion.call_args_list[0].kwargs["messages"]
        second_messages = mock_completion.call_args_list[1].kwargs["messages"]
        assert first_messages[-1]["role"] == "user"
        assert "<mcp_capabilities>" in first_messages[-1]["content"]
        assert second_messages == [{"role": "user", "content": "test"}]
