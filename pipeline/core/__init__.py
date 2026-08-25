"""Core utilities for the pipeline."""

import importlib
from typing import TYPE_CHECKING

# name -> submodule that defines it
_EXPORTS = {
    "load_json": "io",
    "save_json": "io",
    "load_jsonl": "io",
    "save_jsonl": "io",
    "load_parquet": "io",
    "save_parquet": "io",
    "extract_answer": "utils",
    "extract_think": "utils",
    "safe_eval": "utils",
    "model_short_name": "utils",
    "tokenize_with_response_mask": "utils",
    "Generator": "generator",
    "GenerationConfig": "generator",
    "Method": "method",
    "get_primitives_path": "method",
    "ARTIFACTS_ROOT": "method",
    "DataCollatorForHintInterleavedLM": "collators",
}

__all__ = list(_EXPORTS)

if TYPE_CHECKING:  # keep static analysis and editors working
    from .io import load_json, save_json, load_jsonl, save_jsonl, load_parquet, save_parquet
    from .utils import extract_answer, extract_think, safe_eval, model_short_name, tokenize_with_response_mask
    from .generator import Generator, GenerationConfig
    from .method import Method, get_primitives_path, ARTIFACTS_ROOT
    from .collators import DataCollatorForHintInterleavedLM


def __getattr__(name: str):
    try:
        submodule = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{submodule}", __name__), name)
    globals()[name] = value  # cache so this runs once per name
    return value


def __dir__():
    return sorted(__all__)
