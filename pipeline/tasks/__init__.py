"""Task registry."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseTask

# Task registry - add new tasks here
TASKS: dict[str, type["BaseTask"]] = {}


def _register_tasks():
    """Import tasks to populate registry."""
    from .countdown.task import CountdownTask
    from .competition_math.task import CompetitionMathTask
    from .competition_math_v2.task import CompetitionMathV2Task
    from .code_output.task import CodeOutputTask
    TASKS["countdown"] = CountdownTask
    TASKS["competition_math"] = CompetitionMathTask
    TASKS["competition_math_v2"] = CompetitionMathV2Task
    TASKS["code_output"] = CodeOutputTask


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
