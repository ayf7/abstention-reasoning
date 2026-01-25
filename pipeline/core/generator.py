"""VLLM-based text generation."""

import asyncio
from dataclasses import dataclass
from typing import Any


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

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=self.config.num_samples,
            seed=self.config.seed,
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
        max_turns: int = 5,
        request_tag: str = "</request>",
        answer_tag: str = "</answer>",
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
            "hint_index": [0] * num_prompts,
            "num_hints_used": [0] * num_prompts,
            "total_tokens": [0] * num_prompts,
            "turns": [0] * num_prompts,
            "finish_reason": ["max_turns"] * num_prompts,
            "completed": [False] * num_prompts,
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

        # Sampling params - stop on request tag
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=1,
            stop=[request_tag],
            include_stop_str_in_output=True,
            seed=self.config.seed,
        )

        # Turn-synchronized loop
        for turn in range(max_turns):
            # Find incomplete prompts
            incomplete_indices = [i for i in range(num_prompts) if not state["completed"][i]]

            if not incomplete_indices:
                break  # All done

            # Format prompts for incomplete ones
            formatted_prompts = []
            for idx in incomplete_indices:
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

            # Batch generate for all incomplete prompts
            outputs = self.model.generate(formatted_prompts, sampling_params)

            # Process each output
            for batch_idx, idx in enumerate(incomplete_indices):
                output = outputs[batch_idx].outputs[0]
                generated_text = output.text

                state["turns"][idx] += 1
                state["total_tokens"][idx] += len(output.token_ids)
                state["accumulated_text"][idx] += generated_text

                # Check if answer tag found (complete)
                if answer_tag in state["accumulated_text"][idx]:
                    state["finish_reason"][idx] = "stop"
                    state["completed"][idx] = True
                    continue

                # Check if hint was requested
                if request_tag in generated_text:
                    hints = hints_list[idx]
                    hint_idx = state["hint_index"][idx]

                    if hint_idx < len(hints):
                        # Provide hint
                        hint = hints[hint_idx]
                        state["hint_index"][idx] += 1
                        state["num_hints_used"][idx] += 1

                        # Include <think> after response to guide model to continue thinking
                        hint_response = f"\n<response>{hint}</response>\n<think>\n"
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
                        # No more hints - still guide model to continue thinking
                        warning = "\n<response>No more hints available.</response>\n<think>\n"
                        state["accumulated_text"][idx] += warning

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
                    state["finish_reason"][idx] = output.finish_reason or "length"
                    state["completed"][idx] = True

            if self.config.verbose:
                completed_count = sum(state["completed"])
                print(f"Turn {turn + 1}: {completed_count}/{num_prompts} completed")

        # Build results
        all_results = []
        for idx in range(num_prompts):
            result = {
                "text": state["accumulated_text"][idx],
                "finish_reason": state["finish_reason"][idx],
                "token_count": state["total_tokens"][idx],
                "num_hints": state["num_hints_used"][idx],
                "turns": state["turns"][idx],
            }
            all_results.append([result])

            if self.config.verbose:
                print(f"Prompt {idx + 1}: {result['turns']} turns, {result['num_hints']} hints, {result['finish_reason']}")

        return all_results

    def generate_with_hints_batched(
        self,
        prompts: list[list[dict]],
        ground_truths: list[dict],
        max_turns: int = 5,
        callback: callable = None,
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

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size

        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]

            if self.config.verbose:
                batch_num = i // batch_size + 1
                print(f"\n=== Macro-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts) ===")

            batch_results = self.generate_with_hints(
                batch_prompts,
                batch_ground_truths,
                max_turns=max_turns,
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
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=num_samples,
            seed=self.config.seed,
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
        max_turns: int = 5,
        request_tag: str = "</request>",
        answer_tag: str = "</answer>",
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

        Returns:
            List of lists of dicts with 'text', 'finish_reason', 'token_count', 'num_hints'
        """
        import ast
        import uuid
        from vllm import SamplingParams

        engine = await self._get_engine()
        tokenizer = await self._get_tokenizer()

        num_prompts = len(prompts)

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

        # Sampling params
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            n=1,
            stop=[request_tag],
            include_stop_str_in_output=True,
            seed=self.config.seed,
        )

        # Results storage (indexed by prompt idx)
        results = [None] * num_prompts
        completed_event = asyncio.Event()
        completed_count = [0]  # Use list for mutability in nested function

        async def process_single_prompt(idx: int):
            """Process a single prompt through all its turns."""
            import copy
            messages = copy.deepcopy(prompts[idx])  # Deep copy to avoid mutation
            hints = hints_list[idx]
            hint_index = 0
            accumulated_text = ""
            total_tokens = 0
            num_hints_used = 0
            turns = 0
            finish_reason = "max_turns"

            for turn in range(max_turns):
                # Format prompt
                formatted = self._format_prompt(messages, tokenizer)
                request_id = f"req_{idx}_{turn}_{uuid.uuid4().hex[:8]}"

                # Generate
                final_output = None
                async for output in engine.generate(formatted, sampling_params, request_id):
                    if output.finished:
                        final_output = output
                        break

                if final_output is None:
                    finish_reason = "error"
                    break

                generated_text = final_output.outputs[0].text
                turns += 1
                total_tokens += len(final_output.outputs[0].token_ids)
                accumulated_text += generated_text

                # Check if answer tag found (complete)
                if answer_tag in accumulated_text:
                    finish_reason = "stop"
                    break

                # Check if hint was requested
                if request_tag in generated_text:
                    if hint_index < len(hints):
                        # Provide hint
                        hint = hints[hint_index]
                        hint_index += 1
                        num_hints_used += 1

                        hint_response = f"\n<response>{hint}</response>\n<think>\n"
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
                            print(f"Prompt {idx + 1} got hint {hint_index}, continuing...")
                        # Continue to next turn
                    else:
                        # No more hints available
                        warning = "\n<response>No more hints available.</response>\n<think>\n"
                        accumulated_text += warning
                        finish_reason = "no_hints"
                        break
                else:
                    # No hint requested and no answer - hit max tokens or other stop
                    finish_reason = final_output.outputs[0].finish_reason or "length"
                    break

            # Store result
            results[idx] = [{
                "text": accumulated_text,
                "finish_reason": finish_reason,
                "token_count": total_tokens,
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
        max_turns: int = 5,
        callback: callable = None,
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

        Returns:
            List of generation results
        """
        all_results = []
        batch_size = self.config.batch_size
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]

            if self.config.verbose:
                batch_num = i // batch_size + 1
                print(f"\n=== Async batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts) ===")

            batch_results = await self.generate_with_hints_async(
                batch_prompts,
                batch_ground_truths,
                max_turns=max_turns,
            )
            all_results.extend(batch_results)

            if callback:
                callback(i // batch_size, batch_results)

        return all_results


def run_async_generation(
    config: GenerationConfig,
    prompts: list[list[dict]],
    ground_truths: list[dict],
    max_turns: int = 5,
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
