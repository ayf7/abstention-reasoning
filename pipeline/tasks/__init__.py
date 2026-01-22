"""Task registry."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseTask

# Task registry - add new tasks here
TASKS: dict[str, type["BaseTask"]] = {}


def _register_tasks():
    """Import tasks to populate registry."""
    from .countdown.task import CountdownTask
    from .zebra_logic.task import ZebraLogicTask
    from .chess_puzzles.task import ChessPuzzlesTask
    TASKS["countdown"] = CountdownTask
    TASKS["zebra_logic"] = ZebraLogicTask
    TASKS["chess_puzzles"] = ChessPuzzlesTask


def get_task(name: str) -> "BaseTask":
    """Get task instance by name."""
    if not TASKS:
        _register_tasks()

    if name not in TASKS:
        available = list(TASKS.keys())
        raise ValueError(f"Unknown task: {name}. Available: {available}")

    return TASKS[name]()


def list_tasks() -> list[str]:
    """List available task names."""
    if not TASKS:
        _register_tasks()
    return list(TASKS.keys())
