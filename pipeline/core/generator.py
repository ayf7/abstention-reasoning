"""VLLM-based text generation."""

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
