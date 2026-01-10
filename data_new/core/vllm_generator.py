"""
Reusable VLLM batch generation with CoT cleaning.
"""
from __future__ import annotations

import re
from typing import Iterator, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from vllm import LLM


class VLLMGenerator:
    """
    Wrapper for VLLM batch generation with lazy initialization.

    Provides:
    - Lazy model loading (only when first needed)
    - Batch generation with sampling params from config
    - CoT cleaning and token length computation
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize generator with config.

        Expected config keys:
            - model_name: str
            - batch_size: int
            - max_new_tokens: int
            - temperature: float
            - top_p: float
            - tensor_parallel_size: int
            - gpu_memory_utilization: float
        """
        self.cfg = cfg
        self._model: LLM | None = None
        self._tokenizer = None

    @property
    def model(self) -> LLM:
        """Lazy-load VLLM model."""
        if self._model is None:
            from vllm import LLM

            print(f"Loading model with VLLM: {self.cfg.model_name}...")
            print(f"Tensor parallel size: {self.cfg.tensor_parallel_size} GPU(s)")

            self._model = LLM(
                model=self.cfg.model_name,
                tensor_parallel_size=self.cfg.tensor_parallel_size,
                gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            )
        return self._model

    @property
    def tokenizer(self):
        """Get tokenizer from model."""
        if self._tokenizer is None:
            self._tokenizer = self.model.get_tokenizer()
        return self._tokenizer

    def generate(
        self,
        prompts: List[str],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> List[str]:
        """
        Generate text for a batch of prompts.

        Args:
            prompts: List of prompt strings
            temperature: Override config temperature
            top_p: Override config top_p
            max_tokens: Override config max_new_tokens

        Returns:
            List of generated text strings
        """
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature if temperature is not None else self.cfg.temperature,
            top_p=top_p if top_p is not None else self.cfg.top_p,
            max_tokens=max_tokens if max_tokens is not None else self.cfg.max_new_tokens,
        )

        outputs = self.model.generate(prompts, params)
        return [output.outputs[0].text for output in outputs]

    def clean_cot(self, text: str) -> Tuple[str, int]:
        """
        Clean CoT text and compute token length.

        - Removes end-of-text markers
        - Truncates after </answer> tag
        - Computes token count

        Returns:
            (cleaned_text, token_length)
        """
        # Remove end markers
        if "<|endoftext|>" in text:
            text = text.split("<|endoftext|>", 1)[0]

        # Keep only up to the answer block
        match = re.search(r"<answer>.*?</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = text[:match.end()]

        text = text.strip()

        # Compute token length
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return text, len(tokens)

    def batch_iter(
        self,
        items: List,
        batch_size: int | None = None,
    ) -> Iterator[List]:
        """
        Iterate over items in batches.

        Args:
            items: List of items to batch
            batch_size: Override config batch_size
        """
        bs = batch_size if batch_size is not None else self.cfg.batch_size
        for i in range(0, len(items), bs):
            yield items[i : i + bs]
