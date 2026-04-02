"""Smart hint selection using a helper model.

Instead of providing hints sequentially, the HintSelector examines the model's
partial reasoning and selects the most useful hint -- skipping ones the model
has already figured out on its own.

Accumulation rules:
- If hint 3 selected, nothing given yet -> give hints 1+2+3 concatenated
- If hint 3 already given, hint 5 selected -> give hints 4+5 concatenated
- If selected hint <= last given -> fallback to sequential (next hint)

Strategies:
- "sequential": Current behavior (next hint in order)
- "smart": Uses a helper model to pick the best hint
"""

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

HINT_SELECTOR_PROMPT = """\
You are a hint selector. A student is solving a problem and has requested help.
Given their reasoning so far and a list of available hints, determine which hint
would be most useful -- specifically, the first hint whose content the student has
NOT already discovered or figured out on their own.

## Problem
{problem_text}

## Student's reasoning so far
{partial_generation}

## Available hints
{hints_block}

Output ONLY the number of the most useful hint. Just the number, nothing else."""


class HintSelector:
    """Selects hints using a helper model instead of sequential order."""

    def __init__(
        self,
        strategy: str = "sequential",
        helper_model: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        self.strategy = strategy
        self.helper_model = helper_model
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self._engine = None
        self._tokenizer = None
        self._event_loop = None

    def _ensure_loaded(self):
        """Lazy-load the async vLLM engine on first use."""
        if self._engine is not None:
            return

        if self.strategy != "smart":
            return

        if self.helper_model is None:
            raise ValueError("helper_model is required for smart hint selection")

        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        logger.info(f"Loading hint selector helper model: {self.helper_model}")
        print(f"Loading hint selector helper model: {self.helper_model}")

        args = AsyncEngineArgs(
            model=self.helper_model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)

        # Create a persistent event loop for the async engine
        self._event_loop = asyncio.new_event_loop()

    async def _get_tokenizer(self):
        """Get tokenizer from the async engine."""
        if self._tokenizer is None:
            tokenizer_result = self._engine.get_tokenizer()
            if asyncio.iscoroutine(tokenizer_result):
                self._tokenizer = await tokenizer_result
            else:
                self._tokenizer = tokenizer_result
        return self._tokenizer

    def _accumulate(
        self,
        selected_idx: int,
        last_given_idx: int,
        hints: list[str],
    ) -> tuple[str | None, int]:
        """Accumulate hints from (last_given+1) through selected_idx.

        Args:
            selected_idx: 0-indexed hint selected by the helper model
            last_given_idx: 0-indexed last hint given (-1 if none)
            hints: List of hint strings

        Returns:
            (concatenated_hint_text, new_last_given_index) or (None, last_given_idx)
        """
        if selected_idx <= last_given_idx:
            # Fallback: next sequential hint
            next_idx = last_given_idx + 1
            if next_idx >= len(hints):
                return None, last_given_idx
            return hints[next_idx], next_idx

        # Clamp to available hints
        selected_idx = min(selected_idx, len(hints) - 1)

        # Concatenate all hints from (last_given+1) through selected
        start = last_given_idx + 1
        text = "\n".join(hints[start:selected_idx + 1])
        return text, selected_idx

    def _build_prompt(
        self,
        partial_generation: str,
        hints: list[str],
        last_given_index: int,
        problem_text: str = "",
    ) -> str:
        """Build the prompt for the helper model.

        Only enumerates hints from last_given_index+1 onward.
        Uses 1-indexed numbers in the prompt for clarity.
        """
        available_hints = []
        for i in range(last_given_index + 1, len(hints)):
            # 1-indexed for display
            available_hints.append(f"{i + 1}. {hints[i]}")

        hints_block = "\n".join(available_hints)

        return HINT_SELECTOR_PROMPT.format(
            problem_text=problem_text,
            partial_generation=partial_generation[-2000:],  # Truncate long reasoning
            hints_block=hints_block,
        )

    def _parse_selection(
        self,
        output: str,
        last_given_index: int,
        num_hints: int,
    ) -> int:
        """Parse the helper model's output into a 0-indexed hint index.

        The model outputs a 1-indexed number. We convert to 0-indexed.
        Returns fallback (next sequential) on parse failure.
        """
        try:
            # Extract first number from the output
            match = re.search(r'\d+', output.strip())
            if match:
                selected_1indexed = int(match.group())
                selected_0indexed = selected_1indexed - 1
                # Validate range
                if 0 <= selected_0indexed < num_hints:
                    return selected_0indexed
        except (ValueError, AttributeError):
            pass

        # Fallback: next sequential
        return last_given_index + 1

    async def select_hint(
        self,
        partial_generation: str,
        hints: list[str],
        last_given_index: int,
        problem_text: str = "",
    ) -> tuple[str | None, int]:
        """Select the best hint based on the model's partial reasoning.

        Args:
            partial_generation: The model's reasoning so far
            hints: List of all available hint strings
            last_given_index: Index of the last hint given (-1 if none)
            problem_text: The original problem text

        Returns:
            (hint_text, new_last_given_index) or (None, last_given_index) if no hints remain
        """
        # Check if any hints remain
        next_idx = last_given_index + 1
        if next_idx >= len(hints):
            return None, last_given_index

        if self.strategy == "sequential":
            return hints[next_idx], next_idx

        # Smart selection
        if not partial_generation.strip():
            # Not enough context -- use sequential
            return hints[next_idx], next_idx

        try:
            self._ensure_loaded()
            prompt = self._build_prompt(partial_generation, hints, last_given_index, problem_text)

            import uuid
            from vllm import SamplingParams

            request_id = f"hint_{uuid.uuid4().hex[:8]}"
            params = SamplingParams(
                temperature=0.0,
                max_tokens=8,
                n=1,
            )

            final_output = None
            async for output in self._engine.generate(prompt, params, request_id):
                if output.finished:
                    final_output = output
                    break

            if final_output is None:
                logger.warning("Helper model returned no output, falling back to sequential")
                return hints[next_idx], next_idx

            output_text = final_output.outputs[0].text
            selected_idx = self._parse_selection(output_text, last_given_index, len(hints))
            hint_text, new_last = self._accumulate(selected_idx, last_given_index, hints)

            logger.info(
                f"Smart hint selection: helper chose {selected_idx} (1-indexed: {selected_idx+1}), "
                f"last_given {last_given_index} -> {new_last}, "
                f"accumulated {new_last - last_given_index} hint(s)"
            )
            print(
                f"  [hint_selector] selected={selected_idx+1}, "
                f"gave hints {last_given_index+2}..{new_last+1} "
                f"(raw output: {output_text.strip()!r})"
            )

            return hint_text, new_last

        except Exception as e:
            logger.warning(f"Smart hint selection failed: {e}, falling back to sequential")
            return hints[next_idx], next_idx

    def select_hint_sync(
        self,
        partial_generation: str,
        hints: list[str],
        last_given_index: int,
        problem_text: str = "",
    ) -> tuple[str | None, int]:
        """Synchronous wrapper for select_hint.

        For use in non-async code paths. Creates/reuses an event loop.
        """
        if self.strategy == "sequential":
            next_idx = last_given_index + 1
            if next_idx >= len(hints):
                return None, last_given_index
            return hints[next_idx], next_idx

        self._ensure_loaded()

        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()

        return self._event_loop.run_until_complete(
            self.select_hint(partial_generation, hints, last_given_index, problem_text)
        )

    def select_hints_batch_sync(
        self,
        partial_generations: list[str],
        hints_list: list[list[str]],
        last_given_indices: list[int],
        problem_texts: list[str] | None = None,
    ) -> list[tuple[str | None, int]]:
        """Select hints for a batch of prompts synchronously.

        vLLM's continuous batching handles concurrent requests automatically.

        Args:
            partial_generations: List of partial reasoning texts
            hints_list: List of hint lists (one per prompt)
            last_given_indices: List of last-given hint indices
            problem_texts: List of problem texts (optional)

        Returns:
            List of (hint_text, new_last_given_index) tuples
        """
        if problem_texts is None:
            problem_texts = [""] * len(partial_generations)

        if self.strategy == "sequential":
            results = []
            for hints, last_idx in zip(hints_list, last_given_indices):
                next_idx = last_idx + 1
                if next_idx >= len(hints):
                    results.append((None, last_idx))
                else:
                    results.append((hints[next_idx], next_idx))
            return results

        self._ensure_loaded()

        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()

        async def _batch():
            tasks = [
                self.select_hint(gen, hints, last_idx, problem)
                for gen, hints, last_idx, problem
                in zip(partial_generations, hints_list, last_given_indices, problem_texts)
            ]
            return await asyncio.gather(*tasks)

        return self._event_loop.run_until_complete(_batch())

    def shutdown(self):
        """Release GPU resources."""
        if self._engine is not None:
            # AsyncLLMEngine doesn't have explicit shutdown in all versions
            # but we can clean up references
            self._engine = None
            self._tokenizer = None

        if self._event_loop is not None and not self._event_loop.is_closed():
            self._event_loop.close()
            self._event_loop = None

        logger.info("HintSelector shut down")
