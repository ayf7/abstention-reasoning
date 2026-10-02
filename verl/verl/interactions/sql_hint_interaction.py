# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""Interaction handler for sql task with sequential hint support."""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SqlHintInteraction(BaseInteraction):
    """Interaction handler for sql with hint support.

    During RL rollouts, when the model outputs <request></request>, this handler
    provides hints from prefix_hints.

    Flow:
    1. Model generates: <think>reasoning...</think><request></request>
    2. System responds: <response>hint_1 content</response>
    3. Model continues: <think>more reasoning...</think>
    ... up to 5-6 hints
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self.request_tag_pattern = re.compile(r"<request>.*?</request>|<request>|<request/>", re.DOTALL)
        self.max_hints = 6
        self.hint_selector = None

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Initialize interaction state for a trajectory."""
        if instance_id is None:
            instance_id = str(uuid4())

        hints = []
        if ground_truth is not None:
            if "hint_exprs" in ground_truth and ground_truth["hint_exprs"]:
                hints = list(ground_truth["hint_exprs"])
            elif "prefix_hints" in ground_truth:
                prefix_hints = ground_truth.get("prefix_hints", {})
                if isinstance(prefix_hints, dict):
                    for i in range(1, self.max_hints + 1):
                        hint_key = f"hint_{i}"
                        if hint_key in prefix_hints:
                            hints.append(prefix_hints[hint_key])

        self._instance_dict[instance_id] = {
            "hints": hints,
            "last_given_index": -1,
            "num_hints_given": 0,
            "ground_truth": ground_truth,
        }

        logger.debug(f"Started sql hint interaction {instance_id} with {len(hints)} hints")
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process model output and provide hint if requested."""
        if instance_id not in self._instance_dict:
            logger.warning(f"Unknown instance_id: {instance_id}")
            return True, "", 0.0, {}

        inst = self._instance_dict[instance_id]

        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")
                break

        if not self.request_tag_pattern.search(last_content):
            return True, "", 0.0, {"num_hints": inst["num_hints_given"]}

        hints = inst["hints"]
        last_given = inst["last_given_index"]

        if last_given + 1 < len(hints):
            if self.hint_selector is not None:
                hint_text, new_last = self.hint_selector.select_hint_sync(
                    last_content, hints, last_given,
                )
                if hint_text is None:
                    response = "<response>No more hints available.</response>"
                    return False, response, 0.0, {"num_hints": inst["num_hints_given"], "hint_exhausted": True}
            else:
                next_idx = last_given + 1
                hint_text, new_last = hints[next_idx], next_idx

            inst["last_given_index"] = new_last
            inst["num_hints_given"] += 1

            response = f"<response>{hint_text}</response>"
            return False, response, 0.0, {"num_hints": inst["num_hints_given"], "hint_provided": hint_text}
        else:
            response = "<response>No more hints available.</response>"
            return False, response, 0.0, {"num_hints": inst["num_hints_given"], "hint_exhausted": True}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return 0.0
