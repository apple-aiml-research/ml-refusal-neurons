#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Generate Examples for Feature Discovery

Generates positive and negative example passages for a concept using a
public HuggingFace model, in either transformers (--local) or vLLM (--vllm)
mode.
"""

import argparse
import sys
import os

# Add parent directory to path so 'examples' package resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples import ExampleGenerator, save_examples


def main():
    parser = argparse.ArgumentParser(description="Generate examples for feature discovery")
    parser.add_argument("--concept", type=str, required=True, help="Concept to generate examples for")
    parser.add_argument("--k", type=int, default=100, help="Number of positive and negative examples")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for generation")
    parser.add_argument("--output_dir", type=str, default="examples_data", help="Output directory")

    # Generation backend (must pick one)
    parser.add_argument("--local", action='store_true', help='Generate with local model via HuggingFace transformers')
    parser.add_argument("--vllm", action='store_true', help='Generate with local model via vLLM (faster)')

    # Model options
    parser.add_argument("--local_model", type=str, default="huihui-ai/Qwen3-32B-abliterated", help='Local model to use')
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help='Number of GPUs for vLLM (default: 1)')
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help='GPU memory fraction for vLLM (default: 0.9)')

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help='Random seed for deterministic generation (default: 42; pass -1 to disable)')

    args = parser.parse_args()

    if not args.local and not args.vllm:
        parser.error("Must pick a generation backend: --local (transformers) or --vllm")
    if args.local and args.vllm:
        parser.error("Choose only one backend: --local or --vllm")

    # Seed all RNGs for reproducibility (unless --seed -1)
    if args.seed is not None and args.seed >= 0:
        from transformers import set_seed
        set_seed(args.seed)                       # seeds random, numpy, torch, torch.cuda
        try:
            import torch
            torch.cuda.manual_seed_all(args.seed) # belt-and-suspenders for multi-GPU
        except ImportError:
            pass
        print(f"Seed: {args.seed}")
    else:
        print("Seed: DISABLED (non-deterministic run)")

    print("="*60)
    print("EXAMPLE GENERATION")
    print("="*60)
    print(f"Concept: {args.concept}")
    print(f"Examples per class: {args.k}")
    print(f"Output directory: {args.output_dir}")
    if args.vllm:
        print(f"Mode: vLLM - {args.local_model}")
        print(f"  Tensor parallel: {args.tensor_parallel_size} GPUs")
        print(f"  GPU memory: {args.gpu_memory_utilization*100:.0f}%")
    else:
        print(f"Mode: Local (transformers) - {args.local_model}")
    print()

    generator = ExampleGenerator(concept=args.concept, k=args.k)

    try:
        if args.vllm:
            positive_examples, negative_examples = generator.generate_from_vllm(
                model_name=args.local_model,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        else:
            positive_examples, negative_examples = generator.generate_from_local_model(
                model_name=args.local_model,
                batch_size=args.batch_size,
            )
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    output_path = save_examples(
        concept=args.concept,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        output_dir=args.output_dir,
        metadata={
            "source": "vLLM" if args.vllm else "Local model",
            "batch_size": args.batch_size,
            "local_model": args.local_model,
            "tensor_parallel_size": args.tensor_parallel_size if args.vllm else None,
            "seed": args.seed if (args.seed is not None and args.seed >= 0) else None,
        }
    )

    print(f"\n{'='*60}")
    print(f"Examples saved to: {output_path}")
    print(f"{'='*60}")
    print(f"Positive examples: {len(positive_examples)}")
    print(f"Negative examples: {len(negative_examples)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
