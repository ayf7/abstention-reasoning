"""VLLM-based text generation."""

import asyncio
import re
from dataclasses import dataclass
from typing import Any

_RESPONSE_TAG_RE = re.compile(r'<response>.*?</response>', re.DOTALL)


def _count_model_tokens(text: str, tokenizer) -> int:
    """Count tokens in text, excluding <response>...</response> segments.

    This gives the number of model-generated tokens only, since
    <response> content is system-injected hint text.
    """
    stripped = _RESPONSE_TAG_RE.sub('', text)
    return len(tokenizer.encode(stripped))


HINT_TRANSITION_PHRASES = [
    "\n\nI'm not sure how to proceed from here. Let me request a hint.",
    "\n\nI'm stuck on this step. Maybe a hint would help me move forward.",
    "\n\nI don't know how to continue with this approach. Let me ask for a hint.",
    "\n\nI'm having trouble figuring out the next step. Let me request a hint to help guide my thinking.",
    "\n\nHmm, I'm unsure about how to proceed. Let me request a hint.",
    "\n\nI need some guidance on how to approach this next part. Let me ask for a hint.",
    "\n\nI'm not confident in where to go from here. Let me request a hint.",
]


DEFAULT_STOP_STRINGS = ["</answer>", "</think>\n\n<abstain>"]


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    model_name: str
    batch_size: int = 16
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples: int = 1
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    verbose: bool = False
    seed: int | None = 42  # Set seed for reproducibility
    stop_strings: list[str] | None = None  # None = use DEFAULT_STOP_STRINGS


class Generator:
    """VLLM-based batched text generator."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from vllm import LLM
            print(f"Loading model: {self.config.model_name}")
            self._model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
        return self._model

    def generate(self, prompts: list[list[dict]]) -> list[list[dict]]:
        """
        Generate completions for a list of prompts.

        Args:
            prompts: List of chat message lists

        Returns:
            List of lists of dicts with 'text' and 'finish_reason'
            (each prompt can have multiple samples if num_samples > 1)
        """
        from vllm import SamplingParams

        stop = self.config.stop_strings if self.config.stop_strings is not None else DEFAULT_STOP_STRINGS
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_samples,
            seed=self.config.seed,
            stop=stop,
            include_stop_str_in_output=True,
        )

        # Apply chat template
        tokenizer = self.model.get_tokenizer()
        formatted_prompts = []
        for messages in prompts:
            # Separate assistant prefix from conversation
            if messages and messages[-1]["role"] == "assistant":
                conversation = messages[:-1]
                assistant_prefix = messages[-1]["content"]
            else:
                conversation = messages
                assistant_prefix = ""

            # Apply template to conversation, then append assistant prefix
            text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,  # Adds <|im_start|>assistant\n
            )
            text += assistant_prefix
            formatted_prompts.append(text)

        # Print sample prompt for debugging (only if verbose)
        if self.config.verbose and formatted_prompts:
            print("\n=== SAMPLE FORMATTED PROMPT ===")
            print(formatted_prompts[0])
            print("=== END SAMPLE ===\n")

        # Generate
        outputs = self.model.generate(formatted_prompts, sampling_params)

        results = []
        for output in outputs:
            # Each output can have multiple completions (if n > 1)
            samples = []
            for completion in output.outputs:
                samples.append({
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "token_count": len(completion.token_ids),
                })
            results.append(samples)

        return results

    def generate_batched(
        self,
        prompts: list[list[dict]],
        callback: callable = None,
    ) -> list[dict]:
        """
        Generate completions in batches with optional progress callback.

        Args:
            prompts: List of chat message lists
            callback: Optional function called after each batch with (batch_idx, results)

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.generate(batch)
            all_results.extend(batch_results)

            if callback:
                callback(i // batch_size, batch_results)

        return all_results

    def generate_with_hints(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        max_turns: int = 6,
        request_tag: str = "</request>",
        answer_tag: str = "</answer>",
        force_hints: int | list[int] = 0,
        force_hint_token_range: tuple[int, int] = (100, 500),
        hint_selector=None,
    ) -> list[list[dict]]:
        """
        Multi-turn generation with hint injection using turn-synchronized batching.

        All prompts are processed together in synchronized turns:
        1. Batch generate for all incomplete prompts until stop token
        2. Post-process: detect patterns, append hints where needed
        3. Repeat until all complete or max_turns reached

        Args:
            prompts: List of chat message lists
            ground_truths: List of dicts containing 'hint_exprs' for each prompt
            max_turns: Maximum number of generation turns (hint requests)
            request_tag: Tag that triggers hint injection (default: "</request>")
            answer_tag: Tag that signals completion (default: "</answer>")
            force_hints: Force at least this many hints per example by intercepting
                </think> tags and injecting hint requests. Can be a single int
                (applied to all) or a per-prompt list. (default: 0, disabled)
            force_hint_token_range: (min, max) token cap for forced-hint turns.
                On forced turns, max_tokens is sampled from this range so the model
                is cut off mid-reasoning before it can complete. (default: (100, 500))

        Returns:
            List of lists of dicts with 'text', 'finish_reason', 'token_count', 'num_hints'
        """
        import ast
        from vllm import SamplingParams

        num_prompts = len(prompts)
        tokenizer = self.model.get_tokenizer()

        # Initialize per-prompt state
        # Deep copy messages to avoid mutating the original prompts
        import copy
        state = {
            "accumulated_text": [""] * num_prompts,
            "current_messages": [copy.deepcopy(m) for m in prompts],  # Deep copy to avoid mutation
            "last_given_index": [-1] * num_prompts,
            "num_hints_used": [0] * num_prompts,
            "turns": [0] * num_prompts,
            "finish_reason": ["length"] * num_prompts,
            "completed": [False] * num_prompts,
            "hint_selections": [[] for _ in range(num_prompts)],
        }

        # Parse hint expressions for each prompt
        hints_list = []
        for ground_truth in ground_truths:
            # Support both hint_exprs (new) and hints_expr (legacy)
            hints_expr = ground_truth.get("hint_exprs", ground_truth.get("hints_expr", []))
            if isinstance(hints_expr, str):
                try:
                    hints_expr = ast.literal_eval(hints_expr)
                except (ValueError, SyntaxError):
                    hints_expr = []
            hints_list.append(hints_expr)

        # Normalize force_hints to per-prompt list
        if isinstance(force_hints, int):
            force_hints_per = [force_hints] * num_prompts
        else:
            force_hints_per = force_hints
        any_forcing = any(f > 0 for f in force_hints_per)

        think_tag = "</think>"

        # Total token budget for model-generated tokens (excludes injected hint/response text)
        max_budget = self.config.max_new_tokens

        # RNG for sampling per-turn token caps during forced hint turns
        if any_forcing:
            import random as _random
            force_rng = _random.Random(42)

        # Turn-synchronized loop (no hard cap on turns — token budget is the limit)
        turn = 0
        while True:
            # Find incomplete prompts (check remaining budget using tokenized model text)
            incomplete_indices = []
            for i in range(num_prompts):
                if state["completed"][i]:
                    continue
                model_tokens = _count_model_tokens(state["accumulated_text"][i], tokenizer)
                if model_tokens >= max_budget:
                    state["completed"][i] = True
                    state["finish_reason"][i] = "length"
                    continue
                incomplete_indices.append(i)

            if not incomplete_indices:
                break  # All done

            # Split into two groups: those still needing forced hints vs those done forcing
            forcing_indices = []
            normal_indices = []
            for idx in incomplete_indices:
                if force_hints_per[idx] > 0 and state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints_list[idx]):
                    forcing_indices.append(idx)
                else:
                    normal_indices.append(idx)

            # Create forcing params with per-turn token cap, bounded by remaining budget
            if forcing_indices:
                cap = force_rng.randint(force_hint_token_range[0], force_hint_token_range[1])
                min_remaining = min(
                    max_budget - _count_model_tokens(state["accumulated_text"][idx], tokenizer)
                    for idx in forcing_indices
                )
                forcing_params = SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=min(cap, max(min_remaining, 1)),
                    n=1,
                    stop=[request_tag, think_tag],
                    include_stop_str_in_output=True,
                    seed=self.config.seed,
                )
            else:
                forcing_params = None

            # Create normal params bounded by remaining budget
            if normal_indices:
                min_remaining = min(
                    max_budget - _count_model_tokens(state["accumulated_text"][idx], tokenizer)
                    for idx in normal_indices
                )
                normal_params = SamplingParams(
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=max(min_remaining, 1),
                    n=1,
                    stop=[request_tag, answer_tag],
                    include_stop_str_in_output=True,
                    seed=self.config.seed,
                )
            else:
                normal_params = None

            # Process each group with appropriate sampling params
            for group_indices, params in [(forcing_indices, forcing_params), (normal_indices, normal_params)]:
                if not group_indices:
                    continue

                # Format prompts
                formatted_prompts = []
                for idx in group_indices:
                    messages = state["current_messages"][idx]
                    if messages and messages[-1]["role"] == "assistant":
                        conversation = messages[:-1]
                        assistant_content = messages[-1]["content"]
                    else:
                        conversation = messages
                        assistant_content = ""

                    formatted = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    formatted += assistant_content
                    formatted_prompts.append(formatted)

                # Batch generate
                outputs = self.model.generate(formatted_prompts, params)

                # Process each output
                for batch_idx, idx in enumerate(group_indices):
                    output = outputs[batch_idx].outputs[0]
                    generated_text = output.text

                    state["turns"][idx] += 1
                    state["accumulated_text"][idx] += generated_text

                    # Check if answer tag found (complete)
                    if answer_tag in state["accumulated_text"][idx]:
                        # If forcing hints and we haven't hit the minimum yet,
                        # strip the answer and inject a hint instead
                        if force_hints_per[idx] > 0 and state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints_list[idx]):
                            # Remove everything from the answer tag onward
                            ans_pos = state["accumulated_text"][idx].rfind("<answer>")
                            if ans_pos >= 0:
                                state["accumulated_text"][idx] = state["accumulated_text"][idx][:ans_pos]
                            # Inject forced hint request (forced hints always use sequential)
                            self._inject_forced_hint(state, idx, hints_list[idx], rng=force_rng)
                            # Rebuild messages from scratch for next turn
                            self._rebuild_messages(state, idx, prompts[idx])
                            continue
                        state["finish_reason"][idx] = "stop"
                        state["completed"][idx] = True
                        continue

                    # Check if model stopped at </think> (force-hint interception)
                    if force_hints_per[idx] > 0 and think_tag in generated_text and request_tag not in generated_text:
                        hints = hints_list[idx]
                        if state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints):
                            self._inject_forced_hint(state, idx, hints, rng=force_rng)
                            self._rebuild_messages(state, idx, prompts[idx])
                            continue

                    # Check if token limit hit during forced-hint turn (model cut off mid-reasoning)
                    if force_hints_per[idx] > 0 and output.finish_reason == "length":
                        if state["num_hints_used"][idx] < force_hints_per[idx] and state["last_given_index"][idx] + 1 < len(hints_list[idx]):
                            # Trim to last sentence boundary (". " or "\n\n")
                            text = state["accumulated_text"][idx]
                            period_cut = text.rfind('. ')
                            newline_cut = text.rfind('\n\n')
                            cut = max(period_cut + 1 if period_cut >= 0 else -1,
                                      newline_cut if newline_cut >= 0 else -1)
                            if cut > 0:
                                text = text[:cut]
                            state["accumulated_text"][idx] = text + "</think>\n"
                            self._inject_forced_hint(state, idx, hints_list[idx], rng=force_rng)
                            self._rebuild_messages(state, idx, prompts[idx])
                            continue

                    # Check if hint was requested (naturally by the model)
                    if request_tag in generated_text:
                        hints = hints_list[idx]
                        last_given = state["last_given_index"][idx]

                        if last_given + 1 < len(hints):
                            # Select hint (smart or sequential)
                            if hint_selector is not None:
                                # Collect for batch selection later
                                hint_text, new_last = hint_selector.select_hint_sync(
                                    state["accumulated_text"][idx],
                                    hints,
                                    last_given,
                                )
                            else:
                                # Sequential fallback
                                next_idx = last_given + 1
                                hint_text, new_last = hints[next_idx], next_idx

                            if hint_text is not None:
                                state["hint_selections"][idx].append({
                                    "turn": state["turns"][idx],
                                    "type": "smart" if hint_selector is not None else "sequential",
                                    "prev_last_given": last_given,
                                    "new_last_given": new_last,
                                })
                                state["last_given_index"][idx] = new_last
                                state["num_hints_used"][idx] += 1

                                # Include <think> after response to guide model to continue thinking
                                hint_response = f"\n<response>{hint_text}</response>\n<think>\n"
                                state["accumulated_text"][idx] += hint_response

                                # Update messages for next turn
                                messages = state["current_messages"][idx]
                                if messages and messages[-1]["role"] == "assistant":
                                    messages[-1]["content"] += generated_text + hint_response
                                else:
                                    messages.append({
                                        "role": "assistant",
                                        "content": generated_text + hint_response,
                                    })
                            else:
                                # No hint returned (shouldn't happen, but handle gracefully)
                                warning = "\n<response>No more hints available.</response>\n<think>\n"
                                state["accumulated_text"][idx] += warning
                                state["num_hints_used"][idx] += 1

                                messages = state["current_messages"][idx]
                                if messages and messages[-1]["role"] == "assistant":
                                    messages[-1]["content"] += generated_text + warning
                                else:
                                    messages.append({
                                        "role": "assistant",
                                        "content": generated_text + warning,
                                    })
                        else:
                            # No more hints - still guide model to continue thinking
                            warning = "\n<response>No more hints available.</response>\n<think>\n"
                            state["accumulated_text"][idx] += warning
                            state["num_hints_used"][idx] += 1

                            messages = state["current_messages"][idx]
                            if messages and messages[-1]["role"] == "assistant":
                                messages[-1]["content"] += generated_text + warning
                            else:
                                messages.append({
                                    "role": "assistant",
                                    "content": generated_text + warning,
                                })
                    else:
                        # No hint requested and no answer - hit max tokens or other stop
                        if output.finish_reason == "stop" and answer_tag not in state["accumulated_text"][idx]:
                            # Model generated EOS without producing <answer> tags.
                            # Append </think> to close the thinking block and continue,
                            # giving the model a chance to emit <answer>.
                            state["accumulated_text"][idx] += "</think>\n"
                            self._rebuild_messages(state, idx, prompts[idx])
                        else:
                            state["finish_reason"][idx] = output.finish_reason or "length"
                            state["completed"][idx] = True

            if self.config.verbose:
                completed_count = sum(state["completed"])
                print(f"Turn {turn + 1}: {completed_count}/{num_prompts} completed")

            turn += 1

        # Build results (token_count excludes <response> segments)
        all_results = []
        for idx in range(num_prompts):
            text = state["accumulated_text"][idx]
            result = {
                "text": text,
                "finish_reason": state["finish_reason"][idx],
                "token_count": _count_model_tokens(text, tokenizer),
                "total_token_count": len(tokenizer.encode(text)),
                "num_hints": state["num_hints_used"][idx],
                "turns": state["turns"][idx],
                "hint_selections": state["hint_selections"][idx],
            }
            all_results.append([result])

            if self.config.verbose:
                print(f"Prompt {idx + 1}: {result['turns']} turns, {result['num_hints']} hints, {result['finish_reason']}")

        return all_results

    def _inject_forced_hint(self, state: dict, idx: int, hints: list[str], rng=None):
        """Inject a forced hint request + response into the accumulated text.

        Inserts a natural transition phrase before </think> to make the hint
        request look like a deliberate decision rather than an abrupt cutoff.
        Forced hints always use sequential selection (no reasoning to analyze).
        """
        prev_last = state["last_given_index"][idx]
        hint_idx = prev_last + 1
        hint = hints[hint_idx]
        state["hint_selections"][idx].append({
            "turn": state["turns"][idx],
            "type": "forced",
            "prev_last_given": prev_last,
            "new_last_given": hint_idx,
        })
        state["last_given_index"][idx] = hint_idx
        state["num_hints_used"][idx] += 1

        # Pick a transition phrase
        if rng is None:
            import random as _random
            rng = _random.Random(42 + idx + hint_idx)
        transition = rng.choice(HINT_TRANSITION_PHRASES)

        # Insert transition before </think>
        text = state["accumulated_text"][idx]
        think_pos = text.rfind("</think>")
        if think_pos >= 0:
            state["accumulated_text"][idx] = text[:think_pos] + transition + text[think_pos:]
        else:
            state["accumulated_text"][idx] = text + transition + "</think>"

        # Append hint request + response
        forced = f"\n<request></request>\n<response>{hint}</response>\n<think>\n"
        state["accumulated_text"][idx] += forced

    def _rebuild_messages(self, state: dict, idx: int, original_prompt: list[dict]):
        """Rebuild current_messages from original prompt + accumulated text."""
        import copy
        messages = copy.deepcopy(original_prompt)
        if messages and messages[-1]["role"] == "assistant":
            messages[-1]["content"] += state["accumulated_text"][idx]
        else:
            messages.append({
                "role": "assistant",
                "content": state["accumulated_text"][idx],
            })
        state["current_messages"][idx] = messages

    def generate_with_hints_batched(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        max_turns: int = 6,
        callback: callable = None,
        force_hints: int | list[int] = 0,
        force_hint_token_range: tuple[int, int] = (100, 500),
        hint_selector=None,
    ) -> list[list[dict]]:
        """
        Multi-turn generation with hint injection, split into macro-batches.

        The inner generate_with_hints already uses turn-synchronized batching.
        This outer method splits into macro-batches for memory management when
        processing very large prompt sets.

        Args:
            prompts: List of chat message lists
            ground_truths: List of dicts with 'hint_exprs'
            max_turns: Maximum turns per prompt
            callback: Progress callback(batch_idx, results)
            force_hints: Force at least this many hints per example.
                Can be int (applied to all) or per-prompt list. (default: 0)
            force_hint_token_range: (min, max) token cap for forced-hint turns. (default: (100, 500))

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size

        # Normalize to per-prompt list for slicing
        if isinstance(force_hints, int):
            force_hints_list = [force_hints] * len(prompts)
        else:
            force_hints_list = force_hints

        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]
            batch_force_hints = force_hints_list[i:i + batch_size]

            if self.config.verbose:
                batch_num = i // batch_size + 1
                print(f"\n=== Macro-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts) ===")

            batch_results = self.generate_with_hints(
                batch_prompts,
                batch_ground_truths,
                max_turns=max_turns,
                force_hints=batch_force_hints,
                force_hint_token_range=force_hint_token_range,
                hint_selector=hint_selector,
            )
            all_results.extend(batch_results)

            if callback:
                callback(i // batch_size, batch_results)

        return all_results


class AsyncGenerator:
    """
    Async VLLM-based generator for multi-turn hint generation.

    Uses AsyncLLMEngine for optimal throughput - completions are processed
    as they arrive and immediately resubmitted if hints are needed.

    Each prompt is handled by its own async task, allowing prompts to
    progress independently. When a prompt needs a hint, it immediately
    continues generation without waiting for other prompts.

    Note: Requires VLLM with AsyncLLMEngine support. If you encounter
    import errors, ensure you have a compatible VLLM version installed.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._engine = None
        self._tokenizer = None

    async def _get_engine(self):
        """Lazy load the async engine."""
        if self._engine is None:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs

            print(f"Loading async engine: {self.config.model_name}")
            engine_args = AsyncEngineArgs(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        return self._engine

    async def _get_tokenizer(self):
        """Get tokenizer from engine."""
        if self._tokenizer is None:
            engine = await self._get_engine()
            # Handle both sync and async get_tokenizer methods across VLLM versions
            tokenizer_result = engine.get_tokenizer()
            if asyncio.iscoroutine(tokenizer_result):
                self._tokenizer = await tokenizer_result
            else:
                self._tokenizer = tokenizer_result
        return self._tokenizer

    def _format_prompt(self, messages: list[dict], tokenizer) -> str:
        """Format messages into a prompt string."""
        if messages and messages[-1]["role"] == "assistant":
            conversation = messages[:-1]
            assistant_content = messages[-1]["content"]
        else:
            conversation = messages
            assistant_content = ""

        formatted = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted += assistant_content
        return formatted

    async def generate_async(
        self,
        prompts: list[list[dict]],
        num_samples: int = 1,
    ) -> list[list[dict]]:
        """
        Async generation for regular (non-multi-turn) prompts.

        Each prompt is processed independently as an async task.
        Much faster than sync generation due to better GPU utilization.

        Args:
            prompts: List of chat message lists
            num_samples: Number of samples per prompt

        Returns:
            List of lists of dicts with 'text', 'finish_reason', 'token_count'
        """
        import uuid
        from vllm import SamplingParams

        engine = await self._get_engine()
        tokenizer = await self._get_tokenizer()

        num_prompts = len(prompts)

        # Sampling params
        stop = self.config.stop_strings if self.config.stop_strings is not None else DEFAULT_STOP_STRINGS
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=num_samples,
            seed=self.config.seed,
            stop=stop,
            include_stop_str_in_output=True,
        )

        # Results storage
        results = [None] * num_prompts

        async def process_single_prompt(idx: int):
            """Process a single prompt."""
            formatted = self._format_prompt(prompts[idx], tokenizer)
            request_id = f"req_{idx}_{uuid.uuid4().hex[:8]}"

            # Generate
            final_output = None
            async for output in engine.generate(formatted, sampling_params, request_id):
                if output.finished:
                    final_output = output
                    break

            if final_output is None:
                results[idx] = [{"text": "", "finish_reason": "error", "token_count": 0}]
                return

            # Extract all samples
            samples = []
            for completion in final_output.outputs:
                samples.append({
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "token_count": len(completion.token_ids),
                })
            results[idx] = samples

        # Run all prompts concurrently
        if self.config.verbose:
            print(f"Starting async generation for {num_prompts} prompts")

        tasks = [process_single_prompt(idx) for idx in range(num_prompts)]
        await asyncio.gather(*tasks)

        if self.config.verbose:
            print(f"Async generation complete: {num_prompts} prompts")

        return results

    async def generate_async_batched(
        self,
        prompts: list[list[dict]],
        num_samples: int = 1,
        callback: callable = None,
    ) -> list[list[dict]]:
        """
        Async generation split into macro-batches for memory management.

        Args:
            prompts: List of chat message lists
            num_samples: Number of samples per prompt
            callback: Progress callback(batch_idx, results)

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            if self.config.verbose:
                batch_num = i // batch_size + 1
                print(f"\n=== Async batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts) ===")

            batch_results = await self.generate_async(batch_prompts, num_samples)
            all_results.extend(batch_results)

            if callback:
                callback(i // batch_size, batch_results)

        return all_results

    async def generate_with_hints_async(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        max_turns: int = 6,
        request_tag: str = "</request>",
        answer_tag: str = "</answer>",
        force_hints: int | list[int] = 0,
        force_hint_token_range: tuple[int, int] = (100, 500),
        hint_selector=None,
    ) -> list[list[dict]]:
        """
        Async multi-turn generation with hint injection.

        Each prompt is handled by its own async task. When a prompt needs a hint,
        it immediately continues without waiting for other prompts.

        Args:
            prompts: List of chat message lists
            ground_truths: List of dicts containing 'hint_exprs' for each prompt
            max_turns: Maximum number of generation turns per prompt
            request_tag: Tag that triggers hint injection
            answer_tag: Tag that signals completion
            force_hints: Force at least this many hints per example by intercepting
                </think> tags and injecting hint requests (default: 0, disabled)
            force_hint_token_range: (min, max) token cap for forced-hint turns.
                On forced turns, max_tokens is sampled from this range so the model
                is cut off mid-reasoning before it can complete. (default: (100, 500))
            hint_selector: Optional HintSelector instance for smart hint selection

        Returns:
            List of lists of dicts with 'text', 'finish_reason', 'token_count', 'num_hints'
        """
        import ast
        import uuid
        from vllm import SamplingParams

        engine = await self._get_engine()
        tokenizer = await self._get_tokenizer()

        num_prompts = len(prompts)
        think_tag = "</think>"

        # Parse hint expressions for each prompt
        hints_list = []
        for ground_truth in ground_truths:
            hints_expr = ground_truth.get("hint_exprs", ground_truth.get("hints_expr", []))
            if isinstance(hints_expr, str):
                try:
                    hints_expr = ast.literal_eval(hints_expr)
                except (ValueError, SyntaxError):
                    hints_expr = []
            hints_list.append(hints_expr)

        # Normalize force_hints to per-prompt list
        if isinstance(force_hints, int):
            force_hints_per = [force_hints] * num_prompts
        else:
            force_hints_per = force_hints
        any_forcing = any(f > 0 for f in force_hints_per)

        # Total token budget for model-generated tokens (excludes injected hint/response text)
        max_budget = self.config.max_new_tokens

        # Results storage (indexed by prompt idx)
        results = [None] * num_prompts
        completed_event = asyncio.Event()
        completed_count = [0]  # Use list for mutability in nested function

        async def process_single_prompt(idx: int):
            """Process a single prompt through all its turns."""
            import copy
            import random as _random
            messages = copy.deepcopy(prompts[idx])  # Deep copy to avoid mutation
            hints = hints_list[idx]
            last_given_index = -1
            accumulated_text = ""
            num_hints_used = 0
            turns = 0
            finish_reason = "length"

            prompt_force = force_hints_per[idx]
            force_rng = _random.Random(42 + idx)  # Per-prompt RNG for determinism

            while True:
                # Check remaining token budget (excludes <response> segments)
                model_tokens = _count_model_tokens(accumulated_text, tokenizer)
                remaining = max_budget - model_tokens
                if remaining <= 0:
                    finish_reason = "length"
                    break

                # Choose sampling params based on whether we still need forced hints
                needs_forcing = prompt_force > 0 and num_hints_used < prompt_force and last_given_index + 1 < len(hints)
                if needs_forcing:
                    cap = force_rng.randint(force_hint_token_range[0], force_hint_token_range[1])
                    params = SamplingParams(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=min(cap, remaining),
                        n=1,
                        stop=[request_tag, think_tag],
                        include_stop_str_in_output=True,
                        seed=self.config.seed,
                    )
                else:
                    params = SamplingParams(
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=remaining,
                        n=1,
                        stop=[request_tag, answer_tag],
                        include_stop_str_in_output=True,
                        seed=self.config.seed,
                    )

                # Format prompt
                formatted = self._format_prompt(messages, tokenizer)
                request_id = f"req_{idx}_{turns}_{uuid.uuid4().hex[:8]}"

                # Generate
                final_output = None
                async for output in engine.generate(formatted, params, request_id):
                    if output.finished:
                        final_output = output
                        break

                if final_output is None:
                    finish_reason = "error"
                    break

                generated_text = final_output.outputs[0].text
                turns += 1
                accumulated_text += generated_text

                # Check if answer tag found (complete)
                if answer_tag in accumulated_text:
                    # If forcing hints and we haven't hit the minimum, strip answer and inject hint
                    if prompt_force > 0 and num_hints_used < prompt_force and last_given_index + 1 < len(hints):
                        ans_pos = accumulated_text.rfind("<answer>")
                        if ans_pos >= 0:
                            accumulated_text = accumulated_text[:ans_pos]
                        # Insert transition before </think>
                        transition = force_rng.choice(HINT_TRANSITION_PHRASES)
                        think_pos = accumulated_text.rfind("</think>")
                        if think_pos >= 0:
                            accumulated_text = accumulated_text[:think_pos] + transition + accumulated_text[think_pos:]
                        else:
                            accumulated_text += transition + "</think>"
                        # Inject forced hint (always sequential)
                        next_idx = last_given_index + 1
                        hint = hints[next_idx]
                        last_given_index = next_idx
                        num_hints_used += 1
                        forced = f"\n<request></request>\n<response>{hint}</response>\n<think>\n"
                        accumulated_text += forced
                        # Rebuild messages
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        continue
                    finish_reason = "stop"
                    break

                # Check if model stopped at </think> (force-hint interception)
                if prompt_force > 0 and think_tag in generated_text and request_tag not in generated_text:
                    if num_hints_used < prompt_force and last_given_index + 1 < len(hints):
                        # Insert transition before </think>
                        transition = force_rng.choice(HINT_TRANSITION_PHRASES)
                        think_pos = accumulated_text.rfind("</think>")
                        if think_pos >= 0:
                            accumulated_text = accumulated_text[:think_pos] + transition + accumulated_text[think_pos:]
                        next_idx = last_given_index + 1
                        hint = hints[next_idx]
                        last_given_index = next_idx
                        num_hints_used += 1
                        forced = f"\n<request></request>\n<response>{hint}</response>\n<think>\n"
                        accumulated_text += forced
                        # Rebuild messages
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        if self.config.verbose:
                            print(f"Prompt {idx + 1} forced hint {last_given_index + 1}, continuing...")
                        continue

                # Check if token limit hit during forced-hint turn (model cut off mid-reasoning)
                if prompt_force > 0 and final_output.outputs[0].finish_reason == "length":
                    if num_hints_used < prompt_force and last_given_index + 1 < len(hints):
                        # Trim to last sentence boundary (". " or "\n\n")
                        period_cut = accumulated_text.rfind('. ')
                        newline_cut = accumulated_text.rfind('\n\n')
                        cut = max(period_cut + 1 if period_cut >= 0 else -1,
                                  newline_cut if newline_cut >= 0 else -1)
                        if cut > 0:
                            accumulated_text = accumulated_text[:cut]
                        # Add transition phrase and close think tag
                        transition = force_rng.choice(HINT_TRANSITION_PHRASES)
                        accumulated_text += transition + "</think>\n"
                        next_idx = last_given_index + 1
                        hint = hints[next_idx]
                        last_given_index = next_idx
                        num_hints_used += 1
                        forced = f"\n<request></request>\n<response>{hint}</response>\n<think>\n"
                        accumulated_text += forced
                        # Rebuild messages
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                        if self.config.verbose:
                            print(f"Prompt {idx + 1} token-capped, forced hint {last_given_index + 1}, continuing...")
                        continue

                # Check if hint was requested (naturally by the model)
                if request_tag in generated_text:
                    if last_given_index + 1 < len(hints):
                        # Select hint (smart or sequential)
                        if hint_selector is not None:
                            hint_text, new_last = await hint_selector.select_hint(
                                accumulated_text, hints, last_given_index,
                            )
                        else:
                            next_idx = last_given_index + 1
                            hint_text, new_last = hints[next_idx], next_idx

                        if hint_text is not None:
                            last_given_index = new_last
                            num_hints_used += 1

                            hint_response = f"\n<response>{hint_text}</response>\n<think>\n"
                            accumulated_text += hint_response

                            # Update messages for next turn
                            if messages and messages[-1]["role"] == "assistant":
                                messages[-1]["content"] += generated_text + hint_response
                            else:
                                messages.append({
                                    "role": "assistant",
                                    "content": generated_text + hint_response,
                                })

                            if self.config.verbose:
                                print(f"Prompt {idx + 1} got hint (last_given={new_last}), continuing...")
                            # Continue to next turn
                        else:
                            warning = "\n<response>No more hints available.</response>\n<think>\n"
                            accumulated_text += warning
                            num_hints_used += 1
                            # Rebuild messages so model can continue reasoning
                            messages = copy.deepcopy(prompts[idx])
                            if messages and messages[-1]["role"] == "assistant":
                                messages[-1]["content"] += accumulated_text
                            else:
                                messages.append({"role": "assistant", "content": accumulated_text})
                            # Continue — token budget is the real limit
                    else:
                        # No more hints available
                        warning = "\n<response>No more hints available.</response>\n<think>\n"
                        accumulated_text += warning
                        num_hints_used += 1
                        # Rebuild messages so model can continue reasoning
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                else:
                    # No hint requested and no answer - hit max tokens or other stop
                    if final_output.outputs[0].finish_reason == "stop" and answer_tag not in accumulated_text:
                        # Model generated EOS without producing <answer> tags.
                        # Append </think> to close the thinking block and continue,
                        # giving the model a chance to emit <answer>.
                        accumulated_text += "</think>\n"
                        messages = copy.deepcopy(prompts[idx])
                        if messages and messages[-1]["role"] == "assistant":
                            messages[-1]["content"] += accumulated_text
                        else:
                            messages.append({"role": "assistant", "content": accumulated_text})
                    else:
                        finish_reason = final_output.outputs[0].finish_reason or "length"
                        break

            # Store result (token_count excludes <response> segments)
            results[idx] = [{
                "text": accumulated_text,
                "finish_reason": finish_reason,
                "token_count": _count_model_tokens(accumulated_text, tokenizer),
                "total_token_count": len(tokenizer.encode(accumulated_text)),
                "num_hints": num_hints_used,
                "turns": turns,
            }]

            completed_count[0] += 1
            if self.config.verbose:
                print(f"Prompt {idx + 1} completed ({finish_reason}) - {completed_count[0]}/{num_prompts}")

        # Run all prompts concurrently
        if self.config.verbose:
            print(f"Starting async generation for {num_prompts} prompts")

        tasks = [process_single_prompt(idx) for idx in range(num_prompts)]
        await asyncio.gather(*tasks)

        if self.config.verbose:
            total_hints = sum(r[0]["num_hints"] for r in results if r)
            avg_turns = sum(r[0]["turns"] for r in results if r) / num_prompts
            print(f"\nAsync generation complete: {num_prompts} prompts, {total_hints} total hints, {avg_turns:.1f} avg turns")

        return results

    async def generate_with_hints_async_batched(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        max_turns: int = 6,
        callback: callable = None,
        force_hints: int | list[int] = 0,
        force_hint_token_range: tuple[int, int] = (100, 500),
        hint_selector=None,
    ) -> list[list[dict]]:
        """
        Async multi-turn generation split into macro-batches.

        For very large prompt sets, splits into batches to manage memory
        while still using async processing within each batch.

        Args:
            prompts: List of chat message lists
            ground_truths: List of dicts with 'hint_exprs'
            max_turns: Maximum turns per prompt
            callback: Progress callback(batch_idx, results)
            force_hints: Force at least this many hints per example.
                Can be int (applied to all) or per-prompt list. (default: 0)
            force_hint_token_range: (min, max) token cap for forced-hint turns. (default: (100, 500))

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size

        # Normalize to per-prompt list for slicing
        if isinstance(force_hints, int):
            force_hints_list = [force_hints] * len(prompts)
        else:
            force_hints_list = force_hints

        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]
            batch_force_hints = force_hints_list[i:i + batch_size]

            if self.config.verbose:
                batch_num = i // batch_size + 1
                print(f"\n=== Async batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts) ===")

            batch_results = await self.generate_with_hints_async(
                batch_prompts,
                batch_ground_truths,
                max_turns=max_turns,
                force_hints=batch_force_hints,
                force_hint_token_range=force_hint_token_range,
                hint_selector=hint_selector,
            )
            all_results.extend(batch_results)

            if callback:
                callback(i // batch_size, batch_results)

        return all_results


def run_async_generation(
    config: GenerationConfig,
    prompts: list[list[dict]],
    ground_truths: list[dict],
    max_turns: int = 6,
    callback: callable = None,
) -> list[list[dict]]:
    """
    Convenience function to run async generation from sync context.

    Args:
        config: Generation config
        prompts: List of chat message lists
        ground_truths: List of dicts with 'hint_exprs'
        max_turns: Maximum turns per prompt
        callback: Progress callback(batch_idx, results)

    Returns:
        List of generation results
    """
    generator = AsyncGenerator(config)
    return asyncio.run(
        generator.generate_with_hints_async_batched(
            prompts, ground_truths, max_turns, callback
        )
    )
