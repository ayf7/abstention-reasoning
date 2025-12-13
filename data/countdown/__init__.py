"""Countdown dataset module."""

# Don't import manager here to avoid importing torch at module load time
# This prevents CUDA initialization issues when running evaluate.py
# from .manager import CountdownManager

__all__ = []  # Import manager explicitly when needed
