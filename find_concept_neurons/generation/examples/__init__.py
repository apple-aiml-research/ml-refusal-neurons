#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Example Generation for Feature Discovery

Provides a unified interface for generating positive and negative example
passages for concept-based feature discovery using public HuggingFace models
via either transformers or vLLM.
"""

from .generator import ExampleGenerator, load_examples, save_examples

__all__ = ['ExampleGenerator', 'load_examples', 'save_examples']
