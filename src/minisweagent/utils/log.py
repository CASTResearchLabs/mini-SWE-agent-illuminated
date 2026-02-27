import logging
from pathlib import Path

from rich.logging import RichHandler


def _setup_root_logger() -> None:
    logger = logging.getLogger("minisweagent")
    logger.setLevel(logging.DEBUG)
    _handler = RichHandler(
        show_path=False,
        show_time=True,
        show_level=True,
        markup=True,
    )
    _formatter = logging.Formatter("%(name)s: %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)


def add_file_handler(path: Path | str, level: int = logging.DEBUG, *, print_path: bool = True) -> None:
    """Add a file handler to the minisweagent logger."""
    logger = logging.getLogger("minisweagent")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if print_path:
        print(f"Logging to '{path}'")


def setup_debug_logging(log_dir: Path | str = "logs", *, enable_model_debug: bool = True) -> None:
    """Set up enhanced logging for debugging agent issues like wasted turns.
    
    Args:
        log_dir: Directory to store log files
        enable_model_debug: Whether to enable DEBUG level for model components
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Set up detailed file logging
    add_file_handler(log_dir / "miniswe_debug.log", level=logging.DEBUG, print_path=True)
    
    # Enable DEBUG logging for key components
    logging.getLogger("agent").setLevel(logging.DEBUG)
    logging.getLogger("litellm_model").setLevel(logging.DEBUG)
    logging.getLogger("minisweagent.models.utils.actions_toolcall").setLevel(logging.DEBUG)
    
    if enable_model_debug:
        # Enable debug logging for all model-related components
        logging.getLogger("minisweagent.models").setLevel(logging.DEBUG)
    
    print(f"Enhanced debug logging enabled. Logs will be written to {log_dir}/miniswe_debug.log")
    print("This will help diagnose issues like wasted turns and FormatErrors.")


def log_conversation_state(messages: list[dict], logger_name: str = "agent") -> None:
    """Log the current conversation state for debugging.
    
    Args:
        messages: Current conversation messages
        logger_name: Name of the logger to use
    """
    logger = logging.getLogger(logger_name)
    
    logger.debug(f"=== CONVERSATION STATE ({len(messages)} messages) ===")
    
    # Count message types
    role_counts = {}
    recent_roles = []
    format_errors = 0
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1
        
        # Track recent messages
        if i >= len(messages) - 10:
            recent_roles.append(role)
        
        # Count format errors
        if msg.get("extra", {}).get("interrupt_type") == "FormatError":
            format_errors += 1
    
    logger.debug(f"Message role counts: {role_counts}")
    logger.debug(f"Recent 10 roles: {recent_roles}")
    logger.debug(f"Total FormatErrors in conversation: {format_errors}")
    
    # Log last few messages in detail
    for i, msg in enumerate(messages[-3:], start=len(messages)-2):
        content_preview = str(msg.get("content", ""))[:200].replace('\n', ' ')
        extra_info = msg.get("extra", {})
        interrupt_type = extra_info.get("interrupt_type")
        actions = extra_info.get("actions", [])
        
        info = f"[{i}] {msg.get('role', 'unknown')}"
        if interrupt_type:
            info += f" ({interrupt_type})"
        if actions:
            info += f" [{len(actions)} actions]"
        info += f": {content_preview}..."
        
        logger.debug(info)


_setup_root_logger()
logger = logging.getLogger("minisweagent")


__all__ = ["logger", "add_file_handler", "setup_debug_logging", "log_conversation_state"]
