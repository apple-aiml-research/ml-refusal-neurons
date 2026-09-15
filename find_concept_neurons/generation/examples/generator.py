#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Unified Example Generation Interface

Provides a clean API for generating positive/negative example passages using
public HuggingFace models via either transformers or vLLM.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict


class ExampleGenerator:
    """
    Unified interface for example generation.

    Supported backends:
    - Local HuggingFace transformers
    - vLLM (faster local inference)
    - Custom user-provided example files
    """

    def __init__(self, concept: str, k: int = 100):
        """
        Initialize example generator

        Args:
            concept: Concept to generate examples for
            k: Number of positive and negative examples
        """
        self.concept = concept
        self.k = k

    def generate_from_local_model(
        self,
        model_name: str = "huihui-ai/Qwen3-32B-abliterated",
        batch_size: int = 20
    ) -> Tuple[List[str], List[str]]:
        """
        Generate both positive and negative examples using a local Hugging Face model

        Args:
            model_name: Hugging Face model name/path
            batch_size: Batch size for generation

        Returns:
            (positive_examples, negative_examples)
        """
        from .local_model_source import LocalModelGenerator

        generator = LocalModelGenerator(model_name)

        positive_examples = generator.generate_positive_examples(
            concept=self.concept,
            k=self.k,
            batch_size=batch_size
        )

        negative_examples = generator.generate_negative_examples(
            concept=self.concept,
            k=self.k,
            batch_size=batch_size
        )

        return positive_examples, negative_examples

    def generate_from_vllm(
        self,
        model_name: str = "huihui-ai/Qwen3-32B-abliterated",
        batch_size: int = 50,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ) -> Tuple[List[str], List[str]]:
        """
        Generate both positive and negative examples using vLLM (fast inference)

        Args:
            model_name: Hugging Face model name/path
            batch_size: Batch size for generation (vLLM can handle larger batches)
            tensor_parallel_size: Number of GPUs to use (1 = single GPU)
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)

        Returns:
            (positive_examples, negative_examples)
        """
        from .vllm_source import VLLMGenerator

        generator = VLLMGenerator(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization
        )

        positive_examples = generator.generate_positive_examples(
            concept=self.concept,
            k=self.k,
            batch_size=batch_size
        )

        negative_examples = generator.generate_negative_examples(
            concept=self.concept,
            k=self.k,
            batch_size=batch_size
        )

        return positive_examples, negative_examples

    def load_custom_examples(
        self,
        positive_file: str,
        negative_file: Optional[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Load examples from custom text files

        Args:
            positive_file: Path to file with positive examples (one per line)
            negative_file: Path to file with negative examples (optional)

        Returns:
            (positive_examples, negative_examples)
        """
        with open(positive_file, 'r') as f:
            positive_examples = [line.strip() for line in f if line.strip()]

        if negative_file:
            with open(negative_file, 'r') as f:
                negative_examples = [line.strip() for line in f if line.strip()]
        else:
            negative_examples = []

        # Trim to k examples
        positive_examples = positive_examples[:self.k]
        negative_examples = negative_examples[:self.k] if negative_examples else []

        return positive_examples, negative_examples


def save_examples(
    concept: str,
    positive_examples: List[str],
    negative_examples: List[str],
    output_dir: str = "examples_data",
    metadata: Optional[Dict] = None
) -> str:
    """
    Save examples to JSON file

    Args:
        concept: Concept name
        positive_examples: List of positive example texts
        negative_examples: List of negative example texts
        output_dir: Directory to save examples
        metadata: Optional metadata to include

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create filename
    concept_name = concept.replace(" ", "_").replace("/", "_").replace("&", "and")
    output_path = os.path.join(output_dir, f"examples_{concept_name}.json")

    # Prepare data
    data = {
        "concept": concept,
        "k": len(positive_examples),
        "positive_examples": positive_examples,
        "negative_examples": negative_examples
    }

    if metadata:
        data["metadata"] = metadata

    # Save
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path


def load_examples(
    concept: str,
    examples_dir: str = "examples_data",
    k: Optional[int] = None
) -> Tuple[List[str], List[str]]:
    """
    Load previously saved examples

    Args:
        concept: Concept name
        examples_dir: Directory containing examples
        k: Optional number of examples to load (uses all if None)

    Returns:
        (positive_examples, negative_examples)

    Raises:
        FileNotFoundError: If examples file doesn't exist
        ValueError: If concept mismatch
    """
    # Find file
    concept_name = concept.replace(" ", "_").replace("/", "_").replace("&", "and")
    file_path = os.path.join(examples_dir, f"examples_{concept_name}.json")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Examples file not found: {file_path}\n"
            f"Generate examples first using ExampleGenerator"
        )

    # Load
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Validate concept match
    if data.get("concept") != concept:
        raise ValueError(
            f"Concept mismatch: file contains '{data.get('concept')}' "
            f"but requested '{concept}'"
        )

    positive_examples = data["positive_examples"]
    negative_examples = data["negative_examples"]

    # Filter out REDACTED examples (they create spurious signals)
    def is_redacted(text):
        """Check if example contains REDACTED markers"""
        return "REDACTED" in text.upper() or "[REDACTED]" in text.upper()

    pos_before = len(positive_examples)
    neg_before = len(negative_examples)

    positive_examples = [ex for ex in positive_examples if not is_redacted(ex)]
    negative_examples = [ex for ex in negative_examples if not is_redacted(ex)]

    pos_filtered = pos_before - len(positive_examples)
    neg_filtered = neg_before - len(negative_examples)

    if pos_filtered > 0 or neg_filtered > 0:
        print(f"  Filtered out REDACTED examples: {pos_filtered} positive, {neg_filtered} negative")

    # Trim to k if specified
    if k is not None:
        if len(positive_examples) < k:
            print(f"  ⚠️  Warning: Only {len(positive_examples)} positive examples available after filtering (requested {k})")
        if len(negative_examples) < k:
            print(f"  ⚠️  Warning: Only {len(negative_examples)} negative examples available after filtering (requested {k})")

        positive_examples = positive_examples[:k]
        negative_examples = negative_examples[:k]

    return positive_examples, negative_examples
