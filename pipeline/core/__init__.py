"""Core utilities for the pipeline.

Import from the submodules directly -- `from pipeline.core.io import load_json`,
`from pipeline.core.method import Method`. This package used to re-export those
names lazily through a module __getattr__; nothing in the repo ever imported
through it, so the indirection only cost a second place to keep in sync.
"""
