#!/usr/bin/env python3
"""
Mini-SWE-Agent Trajectory Analyzer

Adapted from trajectory_analyzer.py (originally for SWE-agent) to handle
mini-swe-agent trajectory format (*.traj.json).

Key differences from SWE-agent:
- File extension: *.traj.json (not *.traj)
- Top-level keys: messages, info, instance_id, trajectory_format
- Steps represented as messages with roles: system/user/assistant/tool/exit
- Problem statement in <pr_description> tag (not <task_description>)
- model_stats: instance_cost, api_calls, tokens_sent, tokens_received
- Model name at info['config']['model']['model_name']
- Version at info['mini_version'] (not info['swe_agent_version'])
- Exit status is 'Submitted' (capitalized)
- MCP tool calls detected via extra['mcp'] on tool messages (native, not CLI)
"""

import json
import os
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import argparse


@dataclass
class ModelStats:
    """Model performance statistics"""
    instance_cost: float
    tokens_sent: int
    tokens_received: int
    api_calls: int


@dataclass
class ActionSummary:
    """Summary of a single action in the trajectory"""
    step: int
    action: str
    execution_time: float
    success: bool
    error: Optional[str] = None
    working_dir: Optional[str] = None
    observation_length: int = 0


@dataclass
class TrajectoryAnalysis:
    """Complete analysis of a trajectory file"""
    file_path: Path
    trajectory_id: str
    mcp_context: str
    problem_statement: str
    resulting_patch: str
    model_stats: ModelStats
    actions: List[ActionSummary]
    exit_status: str
    exit_status_type: str  # "manual_submission", "auto_format_errors", "auto_other", "unknown"
    exit_status_details: str
    total_execution_time: float
    swe_agent_version: str
    success_ratio: float
    success: bool
    status: str  # "success", "success_with_warning", "failed"
    with_mcp: bool
    mcp_usage: str
    mcp_cli_mode: bool
    model: str
    date: str


class TrajectoryAnalyzer:
    """Analyzer for mini-swe-agent trajectory data"""

    def _create_task_grouping_key(self, result: TrajectoryAnalysis) -> Tuple[str, str, str]:
        """Create a grouping key from normalized task, model, and config"""
        problem_text = result.problem_statement or ""
        normalized_task = problem_text[:200].strip().replace('\n', ' ') if problem_text else "Unknown Task"
        config = "unknown"
        try:
            path_str = str(result.file_path)
            if "adaptive_engineering" in path_str:
                config = "adaptive_engineering"
            elif "default" in path_str:
                config = "default"
            else:
                config = "unknown"
        except Exception:
            config = "unknown"
        model = result.model or "unknown"
        return (normalized_task, model, config)

    def group_trajectories_by_task(self, results: List[TrajectoryAnalysis]) -> Dict[Tuple[str, str, str], Dict[str, List[TrajectoryAnalysis]]]:
        """Group trajectories by normalized task + model + config, then by MCP usage"""
        grouped = defaultdict(lambda: {"with_mcp": [], "without_mcp": []})
        for result in results:
            task_key = self._create_task_grouping_key(result)
            if result.with_mcp:
                grouped[task_key]["with_mcp"].append(result)
            else:
                grouped[task_key]["without_mcp"].append(result)
        return dict(grouped)

    def extract_problem_statement(self, trajectory_data: Dict) -> str:
        """Extract the problem statement from mini-swe-agent trajectory data.

        Looks in the messages list for the first user message containing
        a <pr_description> tag.
        """
        try:
            for message in trajectory_data.get("messages", []):
                if message.get("role") != "user":
                    continue
                content = message.get("content", "")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            content = item["text"]
                            break
                if not isinstance(content, str):
                    continue
                if "<pr_description>" in content:
                    start = content.find("<pr_description>") + len("<pr_description>")
                    end = content.find("</pr_description>")
                    if end != -1:
                        return content[start:end].strip()
                    # Tag opened but not closed — return everything after it
                    return content[start:].strip()
                # Fallback: first substantial user message
                if len(content.strip()) > 100:
                    return content.strip()[:1000] + "..." if len(content) > 1000 else content.strip()
            return "Problem statement not found"
        except Exception as e:
            return f"Error extracting problem statement: {str(e)}"

    def extract_resulting_patch(self, trajectory_data: Dict) -> str:
        """Extract the resulting patch from trajectory data (same field as SWE-agent)"""
        try:
            if "info" in trajectory_data and "submission" in trajectory_data["info"]:
                return trajectory_data["info"]["submission"] or "No patch found"
            return "No patch found"
        except Exception as e:
            return f"Error extracting patch: {str(e)}"

    def extract_model_stats(self, trajectory_data: Dict) -> ModelStats:
        """Extract model statistics from mini-swe-agent trajectory data."""
        try:
            if "info" in trajectory_data and "model_stats" in trajectory_data["info"]:
                stats = trajectory_data["info"]["model_stats"]
                return ModelStats(
                    instance_cost=stats.get("instance_cost", 0.0),
                    tokens_sent=stats.get("tokens_sent", 0),
                    tokens_received=stats.get("tokens_received", 0),
                    api_calls=stats.get("api_calls", 0),
                )
            return ModelStats(0.0, 0, 0, 0)
        except Exception:
            return ModelStats(0.0, 0, 0, 0)

    def extract_actions(self, trajectory_data: Dict) -> List[ActionSummary]:
        """Extract sequence of actions from mini-swe-agent messages.

        Each assistant message (with tool calls) is treated as one step.
        Success is determined by returncode==0 on the subsequent tool messages.
        MCP calls are identified by extra['mcp'] being non-null on tool messages.
        """
        actions = []
        messages = trajectory_data.get("messages", [])

        # Build index: tool_call_id -> tool message
        tool_by_id: Dict[str, Dict] = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    tool_by_id[tid] = msg

        step = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue

            step += 1
            call_names = [tc.get("function", {}).get("name", "unknown") for tc in tool_calls]
            action_text = ", ".join(call_names)

            # Collect matching tool responses
            responses = [tool_by_id[tc["id"]] for tc in tool_calls if tc.get("id") in tool_by_id]

            # Determine success: all tool responses must have returncode 0 (or no returncode for MCP)
            success = True
            error = None
            obs_length = 0
            for resp in responses:
                extra = resp.get("extra", {})
                rc = extra.get("returncode")
                if rc is not None and rc != 0:
                    success = False
                    error = f"returncode={rc}"
                obs_length += len(str(resp.get("content", "")))

            # Estimate execution time from timestamps if available
            exec_time = 0.0
            if responses:
                timestamps = [resp.get("extra", {}).get("timestamp") for resp in responses if resp.get("extra", {}).get("timestamp")]
                if len(timestamps) >= 1:
                    exec_time = 0.0  # single timestamp — no delta available

            actions.append(ActionSummary(
                step=step,
                action=action_text,
                execution_time=exec_time,
                success=success,
                error=error,
                working_dir=None,
                observation_length=obs_length,
            ))

        return actions

    def _has_meaningful_patch(self, patch: str) -> bool:
        """Check if a patch contains meaningful content"""
        if not patch or patch.strip() == "":
            return False
        cleaned_patch = patch.strip()
        if cleaned_patch in ["No patch found", "No patch", "no patch"]:
            return False
        lines = cleaned_patch.split("\n")
        content_lines = [l for l in lines if not l.startswith(("diff --git", "index ", "--- ", "+++ ", "@@"))]
        meaningful_lines = [l for l in content_lines if l.strip() and not l.strip().startswith(("#", "//", "/*", "*"))]
        return len(meaningful_lines) >= 3 or len(cleaned_patch) > 100

    def _extract_mcp_context(self, problem_statement: str) -> str:
        """Extract MCP context from the beginning of problem statement"""
        if not problem_statement:
            return ""
        patterns = [
            r"^\s*\([^)]*via [^)]*MCP server\)",
            r"^\s*\(current code base is available as application [^)]*via [^)]*MCP server\)",
        ]
        for pattern in patterns:
            match = re.search(pattern, problem_statement, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(0).strip()
        return ""

    def _clean_problem_statement(self, problem_statement: str, mcp_context: str) -> str:
        """Remove MCP context from problem statement and clean up"""
        if not problem_statement:
            return ""
        cleaned = problem_statement
        if mcp_context:
            cleaned = cleaned.replace(mcp_context, "", 1)
        cleaned = cleaned.strip()
        while cleaned.startswith("\n"):
            cleaned = cleaned[1:].strip()
        return cleaned

    def _detect_mcp_availability(self, problem_statement: str, trajectory_data: Optional[Dict] = None) -> bool:
        """Detect if MCP was available during execution.

        Checks (in order):
        1. Model config keys: mcp_http_config (non-null) or mcp_servers (non-empty)
        2. Problem statement for MCP server mention patterns (fallback)
        """
        if trajectory_data is not None:
            try:
                model_cfg = trajectory_data["info"]["config"]["model"]
                mcp_http_config = model_cfg.get("mcp_http_config")
                if mcp_http_config:
                    return True
                mcp_servers = model_cfg.get("mcp_servers")
                if mcp_servers:
                    return True
            except (KeyError, TypeError):
                pass

        if not problem_statement:
            return False
        patterns = [
            r"\(current code base is available as application .* via imaging-structural MCP server\)",
            r"\([^)]*via [^)]*imaging-structural MCP server\)",
            r"\([^)]*via [^)]*MCP server\)",
        ]
        for pattern in patterns:
            if re.search(pattern, problem_statement, re.IGNORECASE | re.MULTILINE):
                return True
        return False

    def _analyze_exit_status(self, exit_status: str) -> tuple[str, str]:
        """Analyze exit status to determine type and provide detailed context"""
        status = exit_status.lower()
        if status == "submitted":
            return "manual_submission", "Normal successful manual submission"
        elif "exit_format" in status:
            return "auto_format_errors", "Auto-submitted due to repeated format/blocklist/bash syntax errors"
        elif "autosubmitted" in status or "auto" in status:
            return "auto_other", "Auto-submitted due to other system constraints (timeout, cost, etc.)"
        elif status.startswith("submitted"):
            details = status.replace("submitted", "").strip("() ")
            return "auto_other", f"Auto-submitted due to: {details}"
        else:
            return "unknown", f"Non-submission exit: {exit_status}"

    def _detect_mcp_cli_mode(self, actions: List[ActionSummary]) -> bool:
        """Mini-swe-agent uses native MCP integration, never CLI wrappers."""
        return False

    def _calculate_mcp_usage(self, trajectory_data: Dict) -> float:
        """Calculate the ratio of steps involving MCP tool calls.

        In mini-swe-agent, MCP calls are identified by extra['mcp'] being
        non-null on tool messages.
        """
        messages = trajectory_data.get("messages", [])
        total_steps = sum(1 for m in messages if m.get("role") == "assistant" and m.get("tool_calls"))
        if total_steps == 0:
            return 0.0

        # Count assistant steps that have at least one MCP tool response
        tool_by_id: Dict[str, Dict] = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    tool_by_id[tid] = msg

        mcp_steps = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue
            for tc in tool_calls:
                resp = tool_by_id.get(tc.get("id", ""))
                if resp and resp.get("extra", {}).get("mcp"):
                    mcp_steps += 1
                    break  # count the step once even if multiple MCP calls

        return mcp_steps / total_steps

    def _extract_model_name(self, trajectory_data: Dict) -> str:
        """Extract model name from info['config']['model']['model_name']"""
        try:
            return trajectory_data["info"]["config"]["model"]["model_name"]
        except (KeyError, TypeError):
            return "unknown"

    def _get_file_date(self, file_path: Path) -> str:
        """Get file modification date as ISO string"""
        try:
            import datetime
            timestamp = file_path.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.isoformat()
        except Exception:
            return "unknown"

    def analyze_trajectory_file(self, file_path: Path) -> Optional[TrajectoryAnalysis]:
        """Analyze a single mini-swe-agent trajectory file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                trajectory_data = json.load(f)

            trajectory_id = file_path.stem.replace(".traj", "")  # strip .traj from *.traj.json stem

            raw_problem_statement = self.extract_problem_statement(trajectory_data)
            mcp_context = self._extract_mcp_context(raw_problem_statement)
            problem_statement = self._clean_problem_statement(raw_problem_statement, mcp_context)
            resulting_patch = self.extract_resulting_patch(trajectory_data)
            model_stats = self.extract_model_stats(trajectory_data)
            actions = self.extract_actions(trajectory_data)

            info = trajectory_data.get("info", {})
            exit_status = info.get("exit_status", "unknown")
            mini_version = info.get("mini_version", "unknown")

            total_execution_time = sum(action.execution_time for action in actions)

            if actions:
                successful_actions = sum(1 for action in actions if action.success)
                success_ratio = successful_actions / len(actions)
            else:
                success_ratio = 0.0

            exit_status_type, exit_status_details = self._analyze_exit_status(exit_status)

            has_meaningful_patch = self._has_meaningful_patch(resulting_patch)
            is_submitted = exit_status.lower().startswith("submitted")
            no_format_errors = exit_status_type != "auto_format_errors"
            high_success_ratio = success_ratio >= 0.7
            medium_success_ratio = success_ratio >= 0.5

            if is_submitted and has_meaningful_patch and high_success_ratio and no_format_errors:
                status = "success"
                success = True
            elif is_submitted and has_meaningful_patch and medium_success_ratio and no_format_errors:
                status = "success_with_warning"
                success = True
            else:
                status = "failed"
                success = False

            with_mcp = self._detect_mcp_availability(raw_problem_statement, trajectory_data)
            mcp_usage = self._calculate_mcp_usage(trajectory_data)
            mcp_cli_mode = self._detect_mcp_cli_mode(actions)

            model = self._extract_model_name(trajectory_data)
            date = self._get_file_date(file_path)

            return TrajectoryAnalysis(
                file_path=file_path,
                trajectory_id=trajectory_id,
                mcp_context=mcp_context,
                problem_statement=problem_statement,
                resulting_patch=resulting_patch,
                model_stats=model_stats,
                actions=actions,
                exit_status=exit_status,
                exit_status_type=exit_status_type,
                exit_status_details=exit_status_details,
                total_execution_time=total_execution_time,
                swe_agent_version=mini_version,
                success_ratio=success_ratio,
                success=success,
                status=status,
                with_mcp=with_mcp,
                mcp_usage=mcp_usage,
                mcp_cli_mode=mcp_cli_mode,
                model=model,
                date=date,
            )

        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            return None

    def analyze_directory(self, directory: Path, pattern: str = "**/*.traj.json",
                         filter_submitted: bool = False, filter_non_empty_patch: bool = False,
                         filter_model: Optional[str] = None, filter_statement: Optional[str] = None) -> List[TrajectoryAnalysis]:
        """Analyze all trajectory files in a directory with optional filtering"""
        trajectory_files = list(directory.glob(pattern))
        print(f"Found {len(trajectory_files)} trajectory files")

        results = []
        filtered_count = 0

        for file_path in trajectory_files:
            print(f"Analyzing {file_path.name}...")
            analysis = self.analyze_trajectory_file(file_path)
            if analysis:
                should_include = True

                if filter_submitted and analysis.exit_status.lower() != "submitted":
                    should_include = False
                    print(f"  Filtered out: exit_status='{analysis.exit_status}' (not submitted)")

                if filter_non_empty_patch and not self._has_meaningful_patch(analysis.resulting_patch):
                    should_include = False
                    print(f"  Filtered out: empty or minimal patch")

                if filter_model and filter_model not in analysis.model:
                    should_include = False
                    print(f"  Filtered out: model='{analysis.model}' (does not contain '{filter_model}')")

                if filter_statement and filter_statement not in analysis.problem_statement:
                    should_include = False
                    print(f"  Filtered out: problem statement does not contain '{filter_statement}'")

                if should_include:
                    results.append(analysis)
                else:
                    filtered_count += 1

        if filter_submitted or filter_non_empty_patch or filter_model or filter_statement:
            print(f"Filtering applied: kept {len(results)} trajectories, filtered out {filtered_count}")

        return results

    def export_to_json(self, results: List[TrajectoryAnalysis], output_path: Path):
        """Export results to JSON format"""
        data = [asdict(result) for result in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(results)} analyses to {output_path}")

    def export_to_csv(self, results: List[TrajectoryAnalysis], output_path: Path):
        """Export summary to CSV format"""
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trajectory_id", "mcp_context", "success", "status", "success_ratio",
                "exit_status", "exit_status_type", "exit_status_details",
                "total_execution_time", "num_actions", "instance_cost", "api_calls",
                "mini_version", "has_patch", "with_mcp", "mcp_usage", "model", "date",
            ])
            for result in results:
                writer.writerow([
                    result.trajectory_id,
                    result.mcp_context,
                    result.success,
                    result.status,
                    f"{result.success_ratio:.3f}",
                    result.exit_status,
                    result.exit_status_type,
                    result.exit_status_details,
                    result.total_execution_time,
                    len(result.actions),
                    result.model_stats.instance_cost,
                    result.model_stats.api_calls,
                    result.swe_agent_version,
                    len(result.resulting_patch or "") > 10,
                    result.with_mcp,
                    f"{result.mcp_usage:.3f}",
                    result.model,
                    result.date,
                ])
        print(f"Exported summary to {output_path}")

    def _extract_action_type(self, action_text: str) -> str:
        """Extract the actual action/command type from action text (tool call names)"""
        if not action_text or action_text.strip() == "":
            return "empty"
        # mini-swe-agent actions are tool call names, e.g. "bash" or "mcp__server__tool"
        first_call = action_text.split(",")[0].strip()
        if first_call.startswith("mcp__"):
            parts = first_call.split("__")
            return "__".join(parts[:3]) if len(parts) >= 3 else first_call
        return first_call.lower() if first_call else "unknown"

    def export_to_markdown(self, results: List[TrajectoryAnalysis], output_path: Path,
                          detailed: bool = False, include_patches: bool = True,
                          simplified_for_split: bool = False, group_by_task: bool = True):
        """Export analysis to Markdown format"""

        def escape_markdown(text: str) -> str:
            if not text:
                return ""
            return text.replace("|", "\\|").replace("\n", "\n\n").strip()

        def truncate_text(text: str, max_length: int = 500, no_warning: bool = False, single_line: bool = False) -> str:
            if not text or len(text) <= max_length:
                return_text = text
            else:
                return_text = text[:max_length] + "..." if no_warning else text[:max_length] + "...\n*[Text truncated for summary]*"
            if return_text:
                if single_line:
                    return_text = return_text.replace("\n", "\\n")
                else:
                    return_text = return_text.replace("\\n", "")
            return return_text

        def format_patch(patch: str, max_lines: int = -1) -> str:
            if not patch or patch.strip() in ["No patch found", "No patch", "no patch"]:
                return "*No meaningful patch generated*"
            if max_lines == -1:
                return f"```diff\n{patch}\n```"
            lines = patch.split("\n")
            blocks: list[list[str]] = []
            current_block: list[str] = []
            for line in lines:
                if line.startswith("diff --git") and current_block:
                    blocks.append(current_block)
                    current_block = [line]
                else:
                    current_block.append(line)
            if current_block:
                blocks.append(current_block)
            any_truncated = False
            result_lines: list[str] = []
            for block in blocks:
                if len(block) > max_lines:
                    result_lines.extend(block[:max_lines])
                    result_lines.append(f"# ... [block truncated – showing first {max_lines} lines]")
                    any_truncated = True
                else:
                    result_lines.extend(block)
            result_patch = "\n".join(result_lines)
            suffix = f"\n\n*[One or more diff blocks truncated – showing first {max_lines} lines per block]*" if any_truncated else ""
            return f"```diff\n{result_patch}\n```{suffix}"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Mini-SWE-Agent Trajectory Analysis Report\n\n")
            f.write(f"**Generated:** {self._get_current_timestamp()}\n\n")
            f.write(f"**Total Trajectories:** {len(results)}\n\n")

            if group_by_task and len(results) > 1:
                self._export_grouped_analysis(f, results, detailed, include_patches, simplified_for_split)
                return

            successful = sum(1 for r in results if r.success)
            submitted = sum(1 for r in results if r.exit_status.lower().startswith("submitted"))
            manual_submissions = sum(1 for r in results if r.exit_status_type == "manual_submission")
            auto_format_errors = sum(1 for r in results if r.exit_status_type == "auto_format_errors")
            auto_other = sum(1 for r in results if r.exit_status_type == "auto_other")
            with_patches = sum(1 for r in results if self._has_meaningful_patch(r.resulting_patch))
            with_mcp = sum(1 for r in results if r.with_mcp)
            used_mcp = sum(1 for r in results if r.mcp_usage > 0)

            f.write("## 📊 Executive Summary\n\n")
            f.write("| Metric | Count | Percentage |\n")
            f.write("|--------|-------|------------|\n")
            f.write(f"| Successful (submitted + ≥70% actions, no format errors) | {successful} | {successful/len(results)*100:.1f}% |\n")
            f.write(f"| Submitted trajectories | {submitted} | {submitted/len(results)*100:.1f}% |\n")
            f.write(f"| └─ Manual submissions | {manual_submissions} | {manual_submissions/len(results)*100:.1f}% |\n")
            f.write(f"| └─ Auto-submitted (format/syntax errors) | {auto_format_errors} | {auto_format_errors/len(results)*100:.1f}% |\n")
            f.write(f"| └─ Auto-submitted (other constraints) | {auto_other} | {auto_other/len(results)*100:.1f}% |\n")
            f.write(f"| With meaningful patches | {with_patches} | {with_patches/len(results)*100:.1f}% |\n")
            f.write(f"| MCP available | {with_mcp} | {with_mcp/len(results)*100:.1f}% |\n")
            f.write(f"| Actually used MCP | {used_mcp} | {used_mcp/len(results)*100:.1f}% |\n\n")

            total_cost = sum(r.model_stats.instance_cost for r in results)
            total_tokens_sent = sum(r.model_stats.tokens_sent for r in results)
            total_tokens_received = sum(r.model_stats.tokens_received for r in results)
            avg_execution_time = sum(r.total_execution_time for r in results) / len(results)
            avg_actions = sum(len(r.actions) for r in results) / len(results)

            f.write("## 💰 Cost & Performance Overview\n\n")
            f.write(f"- **Total Cost:** ${total_cost:.2f}\n")
            f.write(f"- **Average Cost per Trajectory:** ${total_cost/len(results):.2f}\n")
            f.write(f"- **Total Tokens Sent:** {total_tokens_sent:,}\n")
            f.write(f"- **Total Tokens Received:** {total_tokens_received:,}\n")
            f.write(f"- **Average Execution Time:** {avg_execution_time:.1f}s\n")
            f.write(f"- **Average Actions per Trajectory:** {avg_actions:.1f}\n\n")

            f.write("---\n\n")
            f.write("## 📋 Individual Trajectory Analyses\n\n")

            for i, result in enumerate(results, 1):
                f.write(f"### {i}. Trajectory: `{result.trajectory_id}`\n\n")
                f.write(f"**Path:** {result.file_path}\n")
                f.write(f"**Date:** {result.date}\n\n")

                if simplified_for_split:
                    output_dir_name = output_path.stem
                    safe_trajectory_id = "".join(c for c in result.trajectory_id if c.isalnum() or c in "._-")
                    f.write(f"**Detailed report:** {output_dir_name}/{safe_trajectory_id}.md\n\n")

                status_emoji = self._get_status_emoji(result.status)
                mcp_emoji = "🔌" if result.with_mcp else "⚫"
                patch_emoji = "📝" if self._has_meaningful_patch(result.resulting_patch) else "📄"

                exit_status_display = result.exit_status
                if result.exit_status_type == "auto_format_errors":
                    exit_status_display += " ⚠️ (format errors)"
                elif result.exit_status_type == "auto_other":
                    exit_status_display += " 🤖 (auto)"
                elif result.exit_status_type == "manual_submission":
                    exit_status_display += " ✋ (manual)"

                f.write(f"{status_emoji} **Status:** {exit_status_display} ")
                f.write(f"| {mcp_emoji} **MCP:** {'Available' if result.with_mcp else 'Not Available'} ")
                f.write(f"| {patch_emoji} **Patch:** {'Generated' if self._has_meaningful_patch(result.resulting_patch) else 'Empty'}\n\n")

                f.write("#### 📈 Key Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Model | {result.model} |\n")
                f.write(f"| Mini-SWE-Agent Version | {result.swe_agent_version} |\n")
                f.write(f"| Exit Status Type | {result.exit_status_type.replace('_', ' ').title()} |\n")
                f.write(f"| Success Ratio | {result.success_ratio:.1%} ({sum(1 for a in result.actions if a.success)}/{len(result.actions)} actions) |\n")
                f.write(f"| Cost | ${result.model_stats.instance_cost:.2f} |\n")
                f.write(f"| API Calls | {result.model_stats.api_calls} |\n")
                f.write(f"| Tokens Sent | {result.model_stats.tokens_sent:,} |\n")
                f.write(f"| Tokens Received | {result.model_stats.tokens_received:,} |\n")
                f.write(f"| MCP Usage | {result.mcp_usage:.1%} of steps |\n")
                f.write(f"| Date | {result.date.split('T')[0] if 'T' in result.date else result.date} |\n\n")

                if result.mcp_context:
                    f.write("#### 🔌 MCP Context\n\n")
                    f.write(f"> {escape_markdown(result.mcp_context)}\n\n")

                f.write("#### 🎯 Problem Statement\n\n")
                if simplified_for_split:
                    problem_text = truncate_text(result.problem_statement, single_line=False)
                else:
                    problem_text = truncate_text(result.problem_statement, single_line=False) if not detailed else result.problem_statement
                f.write(f"```text\n{escape_markdown(problem_text)}\n```\n\n")

                f.write("#### ⚡ Actions Summary\n\n")
                successful_actions = [a for a in result.actions if a.success]
                failed_actions = [a for a in result.actions if not a.success]
                f.write(f"- ✅ **Successful**: {len(successful_actions)} actions\n")
                f.write(f"- ❌ **Failed**: {len(failed_actions)} actions\n\n")

                def get_action_type_stats(act_list):
                    type_counts: dict = {}
                    for action in act_list:
                        action_type = self._extract_action_type(action.action)
                        type_counts[action_type] = type_counts.get(action_type, 0) + 1
                    return type_counts

                successful_types = get_action_type_stats(successful_actions)
                failed_types = get_action_type_stats(failed_actions)

                if successful_types:
                    f.write("**✅ Successful Action Types:**\n\n")
                    for action_type, count in sorted(successful_types.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- `{action_type}`: {count}\n")
                    f.write("\n")

                if failed_types:
                    f.write("**❌ Failed Action Types:**\n\n")
                    for action_type, count in sorted(failed_types.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- `{action_type}`: {count}\n")
                    f.write("\n")

                if detailed and not simplified_for_split:
                    f.write("#### ⚡ Actions Details\n\n")
                    f.write("| Step | Action | Status | Details |\n")
                    f.write("|------|--------|--------|----------|\n")
                    for action in result.actions:
                        status_icon = "✅" if action.success else "❌"
                        details = action.error if action.error else f"Output: {action.observation_length} chars"
                        f.write(f"| {action.step} | `{truncate_text(action.action, 100, no_warning=True, single_line=True)}` | {status_icon} | {details} |\n")
                    f.write("\n")

                if include_patches and not simplified_for_split:
                    if self._has_meaningful_patch(result.resulting_patch):
                        f.write("#### 📝 Generated Patch\n\n")
                        f.write(format_patch(result.resulting_patch, max_lines=20 if not detailed else -1))
                        f.write("\n\n")
                    else:
                        f.write("#### 📄 No Meaningful Patch Generated\n\n")

                if i < len(results):
                    f.write("---\n\n")

            f.write("### 🤖 LLM Judge Analysis\n\n")
            f.write("**Result and metric comparison:** *[To be filled by LLM judge]*\n\n")
            f.write("**Assessment of MCP usage impact:** *[To be filled by LLM judge]*\n\n")
            f.write("**Conclusions:** *[To be filled by LLM judge]*\n")

        print(f"Exported {len(results)} trajectory analyses to {output_path}")

    def _get_current_timestamp(self) -> str:
        import datetime
        return datetime.datetime.now().isoformat()

    def export_to_markdown_split(self, results: List[TrajectoryAnalysis], output_dir: Path,
                                detailed: bool = False, include_patches: bool = True):
        """Export each trajectory analysis to a separate markdown file"""
        output_dir.mkdir(parents=True, exist_ok=True)

        def format_patch(patch: str, max_lines: int = -1) -> str:
            if not patch or patch.strip() in ["No patch found", "No patch", "no patch"]:
                return "*No meaningful patch generated*"
            if max_lines == -1:
                return f"```diff\n{patch}\n```"
            lines = patch.split("\n")
            blocks: list[list[str]] = []
            current_block: list[str] = []
            for line in lines:
                if line.startswith("diff --git") and current_block:
                    blocks.append(current_block)
                    current_block = [line]
                else:
                    current_block.append(line)
            if current_block:
                blocks.append(current_block)
            any_truncated = False
            result_lines: list[str] = []
            for block in blocks:
                if len(block) > max_lines:
                    result_lines.extend(block[:max_lines])
                    result_lines.append(f"# ... [block truncated – showing first {max_lines} lines]")
                    any_truncated = True
                else:
                    result_lines.extend(block)
            result_patch = "\n".join(result_lines)
            suffix = f"\n\n*[One or more diff blocks truncated – showing first {max_lines} lines per block]*" if any_truncated else ""
            return f"```diff\n{result_patch}\n```{suffix}"

        for result in results:
            safe_trajectory_id = "".join(c for c in result.trajectory_id if c.isalnum() or c in "._-")
            output_file = output_dir / f"{safe_trajectory_id}.md"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"# Trajectory Analysis: `{result.trajectory_id}`\n\n")
                f.write(f"**Generated:** {self._get_current_timestamp()}\n")
                f.write(f"**Path:** {result.file_path}\n")
                f.write(f"**Date:** {result.date}\n\n")

                status_emoji = self._get_status_emoji(result.status)
                mcp_emoji = "🔌" if result.with_mcp else "⚫"
                patch_emoji = "📝" if self._has_meaningful_patch(result.resulting_patch) else "📄"

                f.write(f"{status_emoji} **Status:** {result.exit_status} ")
                f.write(f"| {mcp_emoji} **MCP:** {'Available' if result.with_mcp else 'Not Available'} ")
                f.write(f"| {patch_emoji} **Patch:** {'Generated' if self._has_meaningful_patch(result.resulting_patch) else 'Empty'}\n\n")

                f.write("## 📈 Key Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Model | {result.model} |\n")
                f.write(f"| Mini-SWE-Agent Version | {result.swe_agent_version} |\n")
                f.write(f"| Success Ratio | {result.success_ratio:.1%} ({sum(1 for a in result.actions if a.success)}/{len(result.actions)} actions) |\n")
                f.write(f"| Cost | ${result.model_stats.instance_cost:.2f} |\n")
                f.write(f"| API Calls | {result.model_stats.api_calls} |\n")
                f.write(f"| Tokens Sent | {result.model_stats.tokens_sent:,} |\n")
                f.write(f"| Tokens Received | {result.model_stats.tokens_received:,} |\n")
                f.write(f"| MCP Usage | {result.mcp_usage:.1%} of steps |\n")
                f.write(f"| Date | {result.date.split('T')[0] if 'T' in result.date else result.date} |\n\n")

                if result.mcp_context:
                    f.write("## 🔌 MCP Context\n\n")
                    f.write(f"> {result.mcp_context}\n\n")

                f.write("## 🎯 Problem Statement\n\n")
                problem_text = result.problem_statement if detailed else (result.problem_statement[:500] + "..." if len(result.problem_statement) > 500 else result.problem_statement)
                f.write(f"```text\n{problem_text}\n```\n\n")

                f.write("## ⚡ Actions Summary\n\n")
                successful_actions = [a for a in result.actions if a.success]
                failed_actions = [a for a in result.actions if not a.success]
                f.write(f"- ✅ **Successful**: {len(successful_actions)} actions\n")
                f.write(f"- ❌ **Failed**: {len(failed_actions)} actions\n\n")

                if detailed:
                    f.write("## ⚡ Actions Details\n\n")
                    f.write("| Step | Action | Status | Details |\n")
                    f.write("|------|--------|--------|----------|\n")
                    for action in result.actions:
                        status_icon = "✅" if action.success else "❌"
                        details = action.error if action.error else f"Output: {action.observation_length} chars"
                        action_text = action.action[:100] + "..." if len(action.action) > 100 else action.action
                        f.write(f"| {action.step} | `{action_text}` | {status_icon} | {details} |\n")
                    f.write("\n")

                if include_patches and self._has_meaningful_patch(result.resulting_patch):
                    f.write("## 📝 Generated Patch\n\n")
                    f.write(format_patch(result.resulting_patch, max_lines=20 if not detailed else -1))
                    f.write("\n\n")
                elif include_patches:
                    f.write("## 📄 No Meaningful Patch Generated\n\n")

        print(f"Exported {len(results)} individual trajectory analyses to directory: {output_dir}")

    def _export_grouped_analysis(self, f, results: List[TrajectoryAnalysis], detailed: bool,
                                include_patches: bool, simplified_for_split: bool):
        """Export grouped analysis showing MCP vs non-MCP comparison for each task"""
        grouped = self.group_trajectories_by_task(results)

        total_results = len(results)
        successful = sum(1 for r in results if r.success)
        with_mcp = sum(1 for r in results if r.with_mcp)
        used_mcp = sum(1 for r in results if r.mcp_usage > 0)

        f.write("## 📊 Executive Summary\n\n")
        f.write(f"- **Total Trajectories:** {total_results}\n")
        f.write(f"- **Unique Tasks:** {len(grouped)}\n")
        f.write(f"- **Overall Success Rate:** {successful/total_results*100:.1f}%\n")
        f.write(f"- **MCP Available:** {with_mcp} trajectories ({with_mcp/total_results*100:.1f}%)\n")
        f.write(f"- **Actually Used MCP:** {used_mcp} trajectories ({used_mcp/total_results*100:.1f}%)\n\n")
        f.write("---\n\n")
        f.write("## 📋 Task-by-Task Analysis\n\n")

        for i, (task_key, task_results) in enumerate(grouped.items(), 1):
            normalized_task, model, config = task_key
            with_mcp_results = task_results["with_mcp"]
            without_mcp_results = task_results["without_mcp"]
            task_hash = self._create_task_hash(normalized_task)

            f.write(f"### {i}. Task Group: {task_hash} + {model} + {config}\n\n")
            f.write("#### 🎯 Task Description\n\n")
            f.write(f"```text\n{normalized_task}...\n```\n\n")
            f.write("#### 📊 MCP Impact Comparison\n\n")
            self._write_comparison_table(f, with_mcp_results, without_mcp_results)

            all_task_results = with_mcp_results + without_mcp_results
            if all_task_results:
                f.write("#### 📋 Trajectory Details\n\n")
                for j, result in enumerate(all_task_results, 1):
                    self._write_detailed_trajectory(f, result, j, detailed, include_patches, simplified_for_split)

            f.write("---\n\n")

        f.write("## 🤖 LLM Judge Analysis\n\n")
        f.write("**Task-specific MCP impact assessment:** *[To be filled by LLM judge]*\n\n")
        f.write("**Cross-task performance patterns:** *[To be filled by LLM judge]*\n\n")
        f.write("**Conclusions and recommendations:** *[To be filled by LLM judge]*\n")

    def _write_comparison_table(self, f, with_mcp_results: List[TrajectoryAnalysis],
                               without_mcp_results: List[TrajectoryAnalysis]):
        """Write comparison table for MCP vs non-MCP results"""
        def calculate_metrics(results: List[TrajectoryAnalysis]):
            if not results:
                return {"count": 0, "success_rate": 0, "avg_cost": 0, "avg_tokens_sent": 0, "avg_tokens_received": 0, "avg_actions": 0, "patch_rate": 0, "avg_mcp_usage": 0}
            return {
                "count": len(results),
                "success_rate": sum(1 for r in results if r.success) / len(results) * 100,
                "avg_cost": sum(r.model_stats.instance_cost for r in results) / len(results),
                "avg_tokens_sent": sum(r.model_stats.tokens_sent for r in results) / len(results),
                "avg_tokens_received": sum(r.model_stats.tokens_received for r in results) / len(results),
                "avg_actions": sum(len(r.actions) for r in results) / len(results),
                "patch_rate": sum(1 for r in results if self._has_meaningful_patch(r.resulting_patch)) / len(results) * 100,
                "avg_mcp_usage": sum(r.mcp_usage for r in results) / len(results) * 100,
            }

        mcp_metrics = calculate_metrics(with_mcp_results)
        no_mcp_metrics = calculate_metrics(without_mcp_results)

        f.write("| Metric | With MCP | Without MCP | Difference |\n")
        f.write("|--------|----------|-------------|------------|\n")
        f.write(f"| Trajectories | {mcp_metrics['count']} | {no_mcp_metrics['count']} | - |\n")

        if mcp_metrics["count"] > 0 or no_mcp_metrics["count"] > 0:
            success_diff = mcp_metrics["success_rate"] - no_mcp_metrics["success_rate"]
            success_arrow = "🔺" if success_diff > 5 else "🔻" if success_diff < -5 else "➡️"
            f.write(f"| Success Rate | {mcp_metrics['success_rate']:.1f}% | {no_mcp_metrics['success_rate']:.1f}% | {success_arrow} {success_diff:+.1f}% |\n")

            cost_diff = mcp_metrics["avg_cost"] - no_mcp_metrics["avg_cost"]
            cost_arrow = "🔻" if cost_diff > 0.5 else "🔺" if cost_diff < -0.5 else "➡️"
            f.write(f"| Avg Cost | ${mcp_metrics['avg_cost']:.2f} | ${no_mcp_metrics['avg_cost']:.2f} | {cost_arrow} ${cost_diff:+.2f} |\n")

            sent_diff = mcp_metrics["avg_tokens_sent"] - no_mcp_metrics["avg_tokens_sent"]
            sent_arrow = "🔻" if sent_diff > 10000 else "🔺" if sent_diff < -10000 else "➡️"
            f.write(f"| Avg Tokens Sent | {mcp_metrics['avg_tokens_sent']:,.0f} | {no_mcp_metrics['avg_tokens_sent']:,.0f} | {sent_arrow} {sent_diff:+,.0f} |\n")

            recv_diff = mcp_metrics["avg_tokens_received"] - no_mcp_metrics["avg_tokens_received"]
            recv_arrow = "🔻" if recv_diff > 1000 else "🔺" if recv_diff < -1000 else "➡️"
            f.write(f"| Avg Tokens Received | {mcp_metrics['avg_tokens_received']:,.0f} | {no_mcp_metrics['avg_tokens_received']:,.0f} | {recv_arrow} {recv_diff:+,.0f} |\n")

            actions_diff = mcp_metrics["avg_actions"] - no_mcp_metrics["avg_actions"]
            actions_arrow = "🔻" if actions_diff > 10 else "🔺" if actions_diff < -10 else "➡️"
            f.write(f"| Avg Actions | {mcp_metrics['avg_actions']:.1f} | {no_mcp_metrics['avg_actions']:.1f} | {actions_arrow} {actions_diff:+.1f} |\n")

            patch_diff = mcp_metrics["patch_rate"] - no_mcp_metrics["patch_rate"]
            patch_arrow = "🔺" if patch_diff > 10 else "🔻" if patch_diff < -10 else "➡️"
            f.write(f"| Patch Generation Rate | {mcp_metrics['patch_rate']:.1f}% | {no_mcp_metrics['patch_rate']:.1f}% | {patch_arrow} {patch_diff:+.1f}% |\n")

            if mcp_metrics["count"] > 0:
                f.write(f"| MCP Usage Rate | {mcp_metrics['avg_mcp_usage']:.1f}% | 0.0% | +{mcp_metrics['avg_mcp_usage']:.1f}% |\n")

        f.write("\n")

    def _write_detailed_trajectory(self, f, result: TrajectoryAnalysis, index: int,
                                  detailed: bool, include_patches: bool, simplified_for_split: bool):
        """Write detailed trajectory information within task groups"""
        f.write(f"##### {index}. Trajectory: `{result.trajectory_id}`\n\n")

        status_emoji = self._get_status_emoji(result.status)
        mcp_emoji = "🔌" if result.with_mcp else "⚫"
        patch_emoji = "📝" if self._has_meaningful_patch(result.resulting_patch) else "📄"

        exit_status_display = result.exit_status
        if result.exit_status_type == "auto_format_errors":
            exit_status_display += " ⚠️ (format errors)"
        elif result.exit_status_type == "auto_other":
            exit_status_display += " 🤖 (auto)"
        elif result.exit_status_type == "manual_submission":
            exit_status_display += " ✋ (manual)"

        f.write(f"{status_emoji} **Status:** {exit_status_display} ")
        f.write(f"| {mcp_emoji} **MCP:** {'Available' if result.with_mcp else 'Not Available'} ")
        f.write(f"| {patch_emoji} **Patch:** {'Generated' if self._has_meaningful_patch(result.resulting_patch) else 'Empty'}\n\n")

        f.write("**📈 Metrics:** ")
        f.write(f"{result.success_ratio:.1%} success ({sum(1 for a in result.actions if a.success)}/{len(result.actions)} actions) | ")
        f.write(f"${result.model_stats.instance_cost:.2f} | ")
        f.write(f"{result.model_stats.tokens_sent:,} sent / {result.model_stats.tokens_received:,} received | ")
        f.write(f"{result.mcp_usage:.1%} MCP usage\n\n")

        if include_patches and self._has_meaningful_patch(result.resulting_patch):
            f.write("**📝 Patch Preview:**\n\n")
            f.write(self._format_patch_for_preview(result.resulting_patch, max_lines=8))
            f.write("\n")

        if simplified_for_split:
            safe_id = "".join(c for c in result.trajectory_id if c.isalnum() or c in "._-")
            f.write(f"\n**📄 Detailed Analysis:** See `{safe_id}.md`\n")

        f.write("\n")

    def _get_status_emoji(self, status: str) -> str:
        if status == "success":
            return "✅"
        elif status == "success_with_warning":
            return "⚠️"
        else:
            return "❌"

    def _create_task_hash(self, normalized_task: str) -> str:
        if not normalized_task:
            return "unknown"
        return hashlib.md5(normalized_task.encode()).hexdigest()[:6]

    def _format_patch_for_preview(self, patch: str, max_lines: int = 8) -> str:
        if not patch or patch.strip() in ["No patch found", "No patch", "no patch"]:
            return "*No meaningful patch generated*"
        lines = patch.split("\n")
        blocks: list[list[str]] = []
        current_block: list[str] = []
        for line in lines:
            if line.startswith("diff --git") and current_block:
                blocks.append(current_block)
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)
        any_truncated = False
        result_lines: list[str] = []
        for block in blocks:
            if len(block) > max_lines:
                result_lines.extend(block[:max_lines])
                result_lines.append(f"# ... [block truncated – showing first {max_lines} lines]")
                any_truncated = True
            else:
                result_lines.extend(block)
        result_patch = "\n".join(result_lines)
        suffix = f"\n\n*[One or more diff blocks truncated – showing first {max_lines} lines per block]*" if any_truncated else ""
        return f"```diff\n{result_patch}\n```{suffix}"

    def print_summary(self, results: List[TrajectoryAnalysis]):
        """Print a summary of the analysis"""
        if not results:
            print("No results to summarize")
            return

        total_results = len(results)
        successful = sum(1 for r in results if r.success)
        submitted = sum(1 for r in results if r.exit_status.lower().startswith("submitted"))
        manual_submissions = sum(1 for r in results if r.exit_status_type == "manual_submission")
        auto_format_errors = sum(1 for r in results if r.exit_status_type == "auto_format_errors")
        auto_other = sum(1 for r in results if r.exit_status_type == "auto_other")
        with_patches = sum(1 for r in results if r.resulting_patch and len(r.resulting_patch.strip()) > 10)
        avg_success_ratio = sum(r.success_ratio for r in results) / total_results if total_results > 0 else 0.0
        high_success_ratio = sum(1 for r in results if r.success_ratio >= 0.8)
        with_mcp = sum(1 for r in results if r.with_mcp)
        used_mcp = sum(1 for r in results if r.mcp_usage > 0)
        mcp_available_and_used = sum(1 for r in results if r.with_mcp and r.mcp_usage > 0)
        avg_mcp_usage = sum(r.mcp_usage for r in results) / total_results if total_results > 0 else 0.0
        total_cost = sum(r.model_stats.instance_cost for r in results)
        total_api_calls = sum(r.model_stats.api_calls for r in results)
        avg_actions = sum(len(r.actions) for r in results) / total_results

        print(f"\n=== MINI-SWE-AGENT TRAJECTORY ANALYSIS SUMMARY ===")
        print(f"Total trajectories analyzed: {total_results}")
        print(f"Successful (submitted + ≥70% action success, no format errors): {successful} ({successful/total_results*100:.1f}%)")
        print(f"Submitted trajectories: {submitted} ({submitted/total_results*100:.1f}%)")
        print(f"  └─ Manual submissions: {manual_submissions} ({manual_submissions/total_results*100:.1f}%)")
        print(f"  └─ Auto-submitted (format/syntax errors): {auto_format_errors} ({auto_format_errors/total_results*100:.1f}%)")
        print(f"  └─ Auto-submitted (other constraints): {auto_other} ({auto_other/total_results*100:.1f}%)")
        print(f"Trajectories with patches: {with_patches} ({with_patches/total_results*100:.1f}%)")
        print(f"High-quality trajectories (≥80% action success): {high_success_ratio} ({high_success_ratio/total_results*100:.1f}%)")
        print(f"Average action success ratio: {avg_success_ratio*100:.1f}%")
        print(f"\nMCP Usage Analysis:")
        print(f"  Trajectories with MCP available: {with_mcp} ({with_mcp/total_results*100:.1f}%)")
        print(f"  Trajectories that used MCP: {used_mcp} ({used_mcp/total_results*100:.1f}%)")
        print(f"  MCP available and used: {mcp_available_and_used} ({mcp_available_and_used/total_results*100:.1f}%)")
        print(f"  Average MCP usage ratio: {avg_mcp_usage*100:.1f}% of steps")
        if used_mcp > 0:
            avg_usage_among_users = sum(r.mcp_usage for r in results if r.mcp_usage > 0) / used_mcp
            print(f"  Average MCP usage among MCP users: {avg_usage_among_users*100:.1f}% of steps")
        total_tokens_sent = sum(r.model_stats.tokens_sent for r in results)
        total_tokens_received = sum(r.model_stats.tokens_received for r in results)
        print(f"\nCost Analysis:")
        print(f"  Total cost: ${total_cost:.2f}")
        print(f"  Average cost per trajectory: ${total_cost/total_results:.2f}")
        print(f"  Total API calls: {total_api_calls:,}")
        print(f"  Total tokens sent: {total_tokens_sent:,}")
        print(f"  Total tokens received: {total_tokens_received:,}")
        print(f"\nPerformance Analysis:")
        print(f"  Average actions per trajectory: {avg_actions:.1f}")


def main():
    """Main function to run the mini-swe-agent trajectory analyzer"""
    parser = argparse.ArgumentParser(description="Analyze mini-swe-agent trajectory files (*.traj.json)")
    parser.add_argument("directory", help="Directory containing trajectory files")
    parser.add_argument("--pattern", default="**/*.traj.json", help="File pattern to match (default: **/*.traj.json)")
    parser.add_argument("--output-json", help="Output JSON file path")
    parser.add_argument("--output-csv", help="Output CSV file path")
    parser.add_argument("--output-markdown", help="Output Markdown file path")
    parser.add_argument("--markdown-split", action="store_true",
                       help="Create separate markdown files for each trajectory")
    parser.add_argument("--markdown-detailed", action="store_true",
                       help="Include detailed action sequences in markdown output")
    parser.add_argument("--markdown-no-patches", action="store_true",
                       help="Exclude patches from markdown output")
    parser.add_argument("--group-by-task", action="store_true", default=True,
                       help="Group trajectories by normalized task statement (default: True)")
    parser.add_argument("--no-group-by-task", action="store_true",
                       help="Disable task grouping")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--filter-submitted", action="store_true",
                       help="Only include trajectories with exit_status='Submitted'")
    parser.add_argument("--filter-non-empty-patch", action="store_true",
                       help="Only include trajectories with non-empty, meaningful patches")
    parser.add_argument("--filter-model", help="Only include trajectories using specified model name")
    parser.add_argument("--filter-statement", help="Only include trajectories with problem statements containing specified text")

    args = parser.parse_args()

    if args.markdown_split and not args.output_markdown:
        print("Error: --markdown-split requires --output-markdown to be specified")
        return 1

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        return 1

    analyzer = TrajectoryAnalyzer()
    results = analyzer.analyze_directory(
        directory, args.pattern,
        filter_submitted=args.filter_submitted,
        filter_non_empty_patch=args.filter_non_empty_patch,
        filter_model=args.filter_model,
        filter_statement=args.filter_statement,
    )

    if not results:
        print("No trajectory files successfully analyzed")
        return 1

    analyzer.print_summary(results)

    if args.output_json:
        analyzer.export_to_json(results, Path(args.output_json))

    if args.output_csv:
        analyzer.export_to_csv(results, Path(args.output_csv))

    if args.output_markdown:
        group_by_task = args.group_by_task and not args.no_group_by_task
        analyzer.export_to_markdown(
            results,
            Path(args.output_markdown),
            detailed=args.markdown_detailed,
            include_patches=not args.markdown_no_patches,
            simplified_for_split=args.markdown_split,
            group_by_task=group_by_task,
        )

    if args.markdown_split:
        output_markdown_path = Path(args.output_markdown)
        split_output_dir = output_markdown_path.parent / output_markdown_path.stem
        analyzer.export_to_markdown_split(
            results,
            split_output_dir,
            detailed=args.markdown_detailed,
            include_patches=not args.markdown_no_patches,
        )

    if args.verbose:
        print(f"\n=== DETAILED RESULTS ===")
        for result in results[:3]:
            print(f"\nTrajectory: {result.trajectory_id}")
            print(f"Success: {result.success}")
            print(f"Exit Status: {result.exit_status}")
            print(f"Model: {result.model}")
            print(f"Problem Statement: {result.problem_statement[:200]}...")
            print(f"Model Stats: Cost=${result.model_stats.instance_cost:.2f}, API calls={result.model_stats.api_calls}")
            print(f"Actions: {len(result.actions)} steps")
            if result.resulting_patch and len(result.resulting_patch) > 10:
                print(f"Patch: {len(result.resulting_patch)} characters")

    return 0


if __name__ == "__main__":
    exit(main())
