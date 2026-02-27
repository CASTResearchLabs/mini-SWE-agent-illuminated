#!/usr/bin/env python3
"""
Utility to enable enhanced logging for debugging mini-swe-agent issues.

This script helps diagnose problems like wasted turns (FormatErrors), 
model response issues, and agent flow problems.
"""

import os
import sys
from pathlib import Path

import typer
from rich.console import Console

from minisweagent.utils.log import setup_debug_logging


app = typer.Typer(rich_markup_mode="rich", add_completion=False)
console = Console()


@app.command()
def enable(
    log_dir: Path = typer.Option(
        Path("logs"), 
        "--log-dir", 
        "-d", 
        help="Directory to store debug logs"
    ),
    model_debug: bool = typer.Option(
        True, 
        "--model-debug/--no-model-debug", 
        help="Enable debug logging for model components"
    ),
    export_env: bool = typer.Option(
        False,
        "--export-env",
        "-e",
        help="Set environment variable to enable debug logging globally"
    ),
) -> None:
    """Enable enhanced debug logging for mini-swe-agent.
    
    This will help diagnose issues like:
    - Wasted turns (agent not making tool calls)
    - FormatErrors and model response problems  
    - Agent flow and conversation state issues
    """
    console.print("[bold green]Enabling enhanced debug logging...[/bold green]")
    
    # Create log directory
    log_dir.mkdir(exist_ok=True)
    
    # Set up the debug logging
    setup_debug_logging(log_dir, enable_model_debug=model_debug)
    
    if export_env:
        # Set environment variable for future sessions
        env_var = "MSWEA_DEBUG_LOGGING"
        os.environ[env_var] = "1"
        console.print(f"[yellow]Set {env_var}=1 for this session[/yellow]")
        console.print(f"[dim]To make this permanent, add 'export {env_var}=1' to your shell config[/dim]")
    
    console.print(f"[green]✓[/green] Enhanced logging enabled")
    console.print(f"[green]✓[/green] Debug logs will be written to [bold]{log_dir}/miniswe_debug.log[/bold]") 
    console.print()
    console.print("[bold yellow]What this logging will capture:[/bold yellow]")
    console.print("• [green]Wasted turns[/green] - when the model fails to make tool calls")
    console.print("• [green]FormatError details[/green] - why tool call parsing failed")
    console.print("• [green]Model response analysis[/green] - what the model actually returned")
    console.print("• [green]Agent flow tracking[/green] - step-by-step execution details")
    console.print("• [green]Conversation state[/green] - message history and patterns")
    console.print()
    console.print("[bold blue]Now run your mini-swe-agent command as usual![/bold blue]")


@app.command()
def status() -> None:
    """Show current debug logging status."""
    import logging
    
    console.print("[bold]Debug Logging Status[/bold]")
    console.print()
    
    # Check if debug logging is enabled
    debug_env = os.getenv("MSWEA_DEBUG_LOGGING")
    if debug_env:
        console.print(f"[green]✓[/green] Environment variable MSWEA_DEBUG_LOGGING={debug_env}")
    else:
        console.print("[yellow]○[/yellow] MSWEA_DEBUG_LOGGING not set")
    
    # Check logger levels
    loggers_to_check = [
        "minisweagent",
        "agent", 
        "litellm_model",
        "minisweagent.models.utils.actions_toolcall"
    ]
    
    console.print("\n[bold]Logger levels:[/bold]")
    for logger_name in loggers_to_check:
        logger = logging.getLogger(logger_name)
        level = logger.getEffectiveLevel()
        level_name = logging.getLevelName(level)
        
        if level <= logging.DEBUG:
            console.print(f"  [green]✓[/green] {logger_name}: {level_name}")
        else:
            console.print(f"  [yellow]○[/yellow] {logger_name}: {level_name}")
    
    # Check for log files
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            console.print(f"\n[bold]Log files in {log_dir}:[/bold]")
            for log_file in sorted(log_files):
                size = log_file.stat().st_size
                console.print(f"  • {log_file.name} ({size:,} bytes)")
        else:
            console.print(f"\n[yellow]No log files found in {log_dir}[/yellow]")
    else:
        console.print(f"\n[yellow]Log directory {log_dir} does not exist[/yellow]")


@app.command()
def analyze_logs(
    log_file: Path = typer.Option(
        Path("logs/miniswe_debug.log"),
        "--log-file", 
        "-f",
        help="Log file to analyze"
    ),
    show_format_errors: bool = typer.Option(
        True,
        "--format-errors/--no-format-errors",
        help="Show FormatError occurrences"
    ),
    show_wasted_turns: bool = typer.Option(
        True,
        "--wasted-turns/--no-wasted-turns", 
        help="Show wasted turn warnings"
    ),
    recent_lines: int = typer.Option(
        50,
        "--recent",
        "-n",
        help="Number of recent lines to show"
    ),
) -> None:
    """Analyze debug logs to identify patterns and issues."""
    if not log_file.exists():
        console.print(f"[red]Log file not found: {log_file}[/red]")
        console.print("Run 'mini-extra debug enable' first to create debug logs.")
        return
    
    console.print(f"[bold]Analyzing log file: {log_file}[/bold]")
    console.print()
    
    lines = log_file.read_text().splitlines()
    
    # Count different types of issues
    format_errors = []
    wasted_turns = []
    model_responses_without_tools = []
    
    for i, line in enumerate(lines):
        if "FormatError" in line and show_format_errors:
            format_errors.append((i+1, line))
        
        if "WASTED TURN" in line and show_wasted_turns:
            wasted_turns.append((i+1, line))
        
        if "ZERO tool calls detected" in line:
            model_responses_without_tools.append((i+1, line))
    
    # Show summary
    console.print(f"[bold]Log Analysis Summary[/bold]")
    console.print(f"  Total lines: {len(lines):,}")
    console.print(f"  FormatErrors: {len(format_errors)}")
    console.print(f"  Wasted turns: {len(wasted_turns)}")
    console.print(f"  Model responses without tools: {len(model_responses_without_tools)}")
    
    if wasted_turns:
        console.print(f"\n[red]🚨 Found {len(wasted_turns)} wasted turn warnings:[/red]")
        for line_num, line in wasted_turns[-5:]:  # Show last 5
            console.print(f"  Line {line_num}: {line.strip()}")
    
    if format_errors:
        console.print(f"\n[yellow]⚠️  Found {len(format_errors)} FormatError occurrences:[/yellow]")
        for line_num, line in format_errors[-3:]:  # Show last 3
            console.print(f"  Line {line_num}: {line.strip()}")
    
    # Show recent log lines
    if recent_lines > 0:
        console.print(f"\n[bold]Recent {recent_lines} lines:[/bold]")
        for line in lines[-recent_lines:]:
            # Colorize important lines
            if "ERROR" in line:
                console.print(f"[red]{line}[/red]")
            elif "WARNING" in line or "WASTED TURN" in line:
                console.print(f"[yellow]{line}[/yellow]")
            elif "FormatError" in line:
                console.print(f"[red]{line}[/red]")
            else:
                console.print(line)


@app.command()
def show_wasted_turns(
    log_file: Path = typer.Option(
        Path("logs/miniswe_debug.log"),
        "--log-file", 
        "-f",
        help="Log file to analyze"
    ),
    recent_count: int = typer.Option(
        5,
        "--count",
        "-n",
        help="Number of recent wasted turns to show"
    ),
) -> None:
    """Show recent wasted turns with model response details.
    
    This command specifically looks for FormatError occurrences and shows
    what the model actually responded with that caused the wasted turn.
    """
    if not log_file.exists():
        console.print(f"[red]Log file not found: {log_file}[/red]")
        console.print("Run with --debug flag or 'mini-extra debug enable' first.")
        return
    
    console.print(f"[bold]🔍 Analyzing wasted turns in: {log_file}[/bold]")
    console.print()
    
    lines = log_file.read_text().splitlines()
    
    # Find wasted turn patterns
    wasted_turns = []
    format_errors = []
    model_responses = []
    
    for i, line in enumerate(lines):
        if "WASTED TURN #" in line:
            wasted_turns.append((i+1, line.strip()))
        elif "PROBLEMATIC MODEL RESPONSE" in line:
            # Capture the model response content that follows
            model_response_lines = [line.strip()]
            j = i + 1
            while j < len(lines) and ("Content:" in lines[j] or "Raw tool_calls:" in lines[j] or lines[j].strip().startswith("- ")):
                model_response_lines.append(lines[j].strip())
                j += 1
            model_responses.append((i+1, "\n".join(model_response_lines)))
        elif "FormatError details:" in line:
            format_errors.append((i+1, line.strip()))
    
    if not wasted_turns:
        console.print("[green]✅ No wasted turns found in the log file![/green]")
        return
    
    console.print(f"[red]🚨 Found {len(wasted_turns)} wasted turn(s):[/red]")
    console.print()
    
    # Show recent wasted turns with details
    for i, (line_num, turn_line) in enumerate(wasted_turns[-recent_count:]):
        turn_number = turn_line.split("WASTED TURN #")[1].split(" ")[0]
        console.print(f"[bold red]Wasted Turn #{turn_number}[/bold red] (line {line_num})")
        console.print(f"  {turn_line}")
        
        # Look for corresponding model response
        matching_response = None
        for resp_line_num, resp_content in model_responses:
            if abs(resp_line_num - line_num) < 10:  # Within 10 lines
                matching_response = resp_content
                break
        
        if matching_response:
            console.print("[yellow]📝 Model Response:[/yellow]")
            for resp_line in matching_response.split("\n"):
                console.print(f"    {resp_line}")
        
        # Look for corresponding format error details
        matching_error = None
        for err_line_num, err_content in format_errors:
            if abs(err_line_num - line_num) < 5:  # Within 5 lines
                matching_error = err_content
                break
        
        if matching_error:
            console.print("[red]❌ Error Details:[/red]")
            console.print(f"    {matching_error}")
        
        console.print()
    
    # Show summary and recommendations
    console.print("[bold yellow]💡 Recommendations:[/bold yellow]")
    
    if len(wasted_turns) >= 3:
        console.print("• [red]High wasted turn count detected![/red] The model may not be learning from error feedback.")
        console.print("• Consider checking if the model configuration includes proper tool definitions.")
        console.print("• Verify that the system prompt clearly instructs the model to use tools.")
    
    console.print("• Check model responses above to see what the model is actually saying instead of using tools.")
    console.print("• If model responses contain reasoning without tool calls, the model may need better prompting.")
    console.print(f"• Use 'mini-extra debug analyze-logs --recent {recent_count * 3}' for more context.")


if __name__ == "__main__":
    app()