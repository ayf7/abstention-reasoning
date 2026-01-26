"""Custom data collators for SFT training."""

import re
from dataclasses import dataclass
from typing import Any, Union

from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


@dataclass
class DataCollatorForHintInterleavedLM(DataCollatorForLanguageModeling):
    """
    Data collator that masks response tokens in hint-interleaved sequences.

    Extends TRL's DataCollatorForLanguageModeling to additionally mask
    <response>...</response> spans by setting their labels to -100.

    For multi-turn hint generation, we want to train on:
    - <think>...</think> tokens (model reasoning)
    - <request></request> tokens (model's hint requests)
    - <answer>...</answer> tokens (model's final answer)

    But NOT on:
    - <response>...</response> tokens (system-provided hints)

    Args:
        tokenizer: The tokenizer to use for decoding (needed to find response spans)
        response_start_tag: Start tag for system responses to mask (default: "<response>")
        response_end_tag: End tag for system responses to mask (default: "</response>")
        pad_token_id: Token ID to use for padding
        completion_only_loss: When True, use completion_mask to mask prompt tokens
        **kwargs: Additional arguments passed to parent class

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        >>> collator = DataCollatorForHintInterleavedLM(
        ...     tokenizer=tokenizer,
        ...     pad_token_id=tokenizer.pad_token_id,
        ... )
    """

    tokenizer: Any = None  # We need the tokenizer to decode and find response spans
    response_start_tag: str = "<response>"
    response_end_tag: str = "</response>"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        """
        Process examples, apply completion masking, then mask response spans.

        1. Call parent to handle padding and completion_mask-based masking
        2. Additionally mask <response>...</response> spans
        """
        # Parent handles: padding, completion_mask -> labels masking
        output = super().torch_call(examples)

        # Additionally mask <response>...</response> spans
        self._mask_response_spans(output)

        return output

    def _mask_response_spans(self, batch: dict[str, Any]) -> None:
        """
        Mask <response>...</response> spans in the batch.

        For each sequence in the batch:
        1. Decode to get the full text
        2. Find all <response>...</response> spans using regex
        3. Convert character positions to token positions
        4. Set labels to -100 for those spans
        """
        if self.tokenizer is None:
            return  # Can't mask without tokenizer

        bsz = batch["input_ids"].shape[0]

        for i in range(bsz):
            # Decode to get the full text
            full_text = self.tokenizer.decode(batch["input_ids"][i])

            # Find all <response>...</response> spans (including the tags)
            pattern = re.compile(
                f"{re.escape(self.response_start_tag)}.*?{re.escape(self.response_end_tag)}",
                re.DOTALL,
            )

            for match in pattern.finditer(full_text):
                start_char, end_char = match.start(), match.end()

                # Convert character positions to token positions
                # Encode text up to start/end to find token boundaries
                tokens_before_start = self.tokenizer.encode(
                    full_text[:start_char], add_special_tokens=False
                )
                tokens_before_end = self.tokenizer.encode(
                    full_text[:end_char], add_special_tokens=False
                )

                start_token_idx = len(tokens_before_start)
                end_token_idx = len(tokens_before_end)

                # Mask these tokens (set labels to -100)
                batch["labels"][i, start_token_idx:end_token_idx] = -100
