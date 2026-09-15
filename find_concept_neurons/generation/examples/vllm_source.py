#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
vLLM-based Example Generation

Generates positive/negative examples using vLLM for high-speed inference
"""

import json
from typing import List
from vllm import LLM, SamplingParams


class VLLMGenerator:
    """Generate examples using vLLM for fast inference"""

    def __init__(
        self,
        model_name: str = "huihui-ai/Qwen3-32B-abliterated",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Initialize vLLM generator

        Args:
            model_name: Hugging Face model name/path
            tensor_parallel_size: Number of GPUs to use (1 = single GPU)
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        self.model_name = model_name
        print(f"Loading vLLM model: {model_name}")
        print(f"  Tensor parallel size: {tensor_parallel_size}")
        print(f"  GPU memory utilization: {gpu_memory_utilization}")

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        print(f"✓ vLLM model loaded")

    def _generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 3000,
        temperature: float = 0.8
    ) -> List[str]:
        """
        Generate responses for a batch of prompts

        Args:
            prompts: List of prompts
            max_tokens: Max tokens per generation
            temperature: Sampling temperature

        Returns:
            List of generated texts
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    def generate_positive_examples(
        self,
        concept: str,
        k: int = 100,
        batch_size: int = 50
    ) -> List[str]:
        """
        Generate k positive examples for a concept

        Args:
            concept: Concept to generate examples for
            k: Number of examples
            batch_size: Examples per generation call (vLLM can handle larger batches)

        Returns:
            List of k example texts
        """
        all_examples = []
        remaining = k

        batch_num = 1
        max_retries_per_batch = 3

        while remaining > 0:
            current_batch_size = min(batch_size, remaining)

            print(f"Generating positive batch {batch_num} ({current_batch_size} examples)...")

            prompt = f"""Generate {current_batch_size} diverse text passages that naturally CONTAIN examples of "{concept}".

CRITICAL INSTRUCTIONS:

1. DO NOT generate meta-level discussion ABOUT "{concept}" (no "the impact of...", "research shows...", "studies indicate...")
2. Instead, generate passages that naturally CONTAIN or EXHIBIT "{concept}" as part of the content itself
3. Examples should feel like natural text excerpts where "{concept}" occurs organically
4. CREATE MAXIMUM DIVERSITY:
   - Format variation: complete sentences, fragments, mid-sentence starts, dialogue, conversational snippets, storytelling
   - Length variation: very short (1 sentence), medium (2-3 sentences), longer (4-5 sentences)
   - Context variation: fiction, personal narrative, casual conversation, social media, creative writing, chat logs, roleplay, literature excerpts
   - Style variation: different tones, perspectives, voices, and scenarios
   - Content variation: different subtypes and intensities of "{concept}"

5. OUTPUT ONLY VALID JSON in this exact format:
{{
  "positive": [
    "passage 1 text here...",
    "passage 2 text here...",
    ...
  ]
}}

IMPORTANT: Every passage should naturally contain "{concept}", not discuss it academically. Be creative and diverse.

Generate {current_batch_size} passages now. Output ONLY the JSON, nothing else."""

            retry_count = 0
            batch_success = False

            while retry_count < max_retries_per_batch and not batch_success:
                try:
                    # vLLM can batch multiple prompts, but for generation we use single prompt
                    responses = self._generate_batch([prompt], max_tokens=3000)
                    response_text = responses[0]

                    examples = self._parse_json_response(response_text, "positive")

                    if examples:
                        # Filter out REDACTED examples
                        examples = [ex for ex in examples if "REDACTED" not in ex.upper()]
                        all_examples.extend(examples[:current_batch_size])
                        remaining -= len(examples[:current_batch_size])
                        print(f"✓ Batch {batch_num} complete. Total: {len(all_examples)}/{k}")
                        batch_success = True
                    else:
                        retry_count += 1
                        if retry_count < max_retries_per_batch:
                            print(f"⚠️ Batch {batch_num} failed, retrying ({retry_count}/{max_retries_per_batch})...")
                        else:
                            print(f"⚠️ Batch {batch_num} failed after {max_retries_per_batch} retries, skipping...")

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries_per_batch:
                        print(f"❌ Error generating batch: {e}")
                        print(f"   Retrying ({retry_count}/{max_retries_per_batch})...")
                    else:
                        print(f"❌ Error generating batch after {max_retries_per_batch} retries: {e}")

            batch_num += 1

        return all_examples[:k]

    def generate_negative_examples(
        self,
        concept: str,
        k: int = 100,
        batch_size: int = 50
    ) -> List[str]:
        """
        Generate k negative examples (unrelated to concept)

        Args:
            concept: Concept to avoid
            k: Number of examples
            batch_size: Examples per generation call

        Returns:
            List of k example texts
        """
        all_examples = []
        remaining = k

        batch_num = 1
        max_retries_per_batch = 3

        while remaining > 0:
            current_batch_size = min(batch_size, remaining)

            print(f"Generating negative batch {batch_num} ({current_batch_size} examples)...")

            prompt = f"""Generate {current_batch_size} text passages that are COMPLETELY UNRELATED to "{concept}".

CRITICAL INSTRUCTIONS:

1. These passages must have ZERO connection to "{concept}"
2. Cover wildly diverse topics: sports, cooking, nature, technology, history, science, travel, hobbies, arts, business, etc.
3. Vary the format and length:
   - Some very short (1 sentence or fragment)
   - Some medium (2-3 sentences)
   - Some longer (4-5 sentences)
4. Mix these styles:
   - Complete sentences and paragraphs
   - Sentence fragments
   - Mid-sentence starts
   - Lists or bullet-point style
   - Conversational snippets
   - Questions
   - Different contexts: news, personal narrative, scientific, casual, literature, social media

5. OUTPUT ONLY VALID JSON in this exact format:
{{
  "negative": [
    "passage 1 text here...",
    "passage 2 text here...",
    ...
  ]
}}

IMPORTANT: Make examples as diverse and creative as possible. NO connection whatsoever to "{concept}".

Generate {current_batch_size} passages now. Output ONLY the JSON, nothing else."""

            retry_count = 0
            batch_success = False

            while retry_count < max_retries_per_batch and not batch_success:
                try:
                    responses = self._generate_batch([prompt], max_tokens=3000)
                    response_text = responses[0]

                    examples = self._parse_json_response(response_text, "negative")

                    if examples:
                        # Filter out REDACTED examples
                        examples = [ex for ex in examples if "REDACTED" not in ex.upper()]
                        all_examples.extend(examples[:current_batch_size])
                        remaining -= len(examples[:current_batch_size])
                        print(f"✓ Batch {batch_num} complete. Total: {len(all_examples)}/{k}")
                        batch_success = True
                    else:
                        retry_count += 1
                        if retry_count < max_retries_per_batch:
                            print(f"⚠️ Batch {batch_num} failed, retrying ({retry_count}/{max_retries_per_batch})...")
                        else:
                            print(f"⚠️ Batch {batch_num} failed after {max_retries_per_batch} retries, skipping...")

                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries_per_batch:
                        print(f"❌ Error generating batch: {e}")
                        print(f"   Retrying ({retry_count}/{max_retries_per_batch})...")
                    else:
                        print(f"❌ Error generating batch after {max_retries_per_batch} retries: {e}")

            batch_num += 1

        return all_examples[:k]

    def _parse_json_response(self, response_text: str, key: str) -> List[str]:
        """
        Parse JSON response from model

        Args:
            response_text: Raw response text
            key: Key to extract ("positive" or "negative")

        Returns:
            List of example strings
        """
        # Try to extract JSON if there's extra text
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "{" in response_text:
            # Extract the FIRST complete JSON object
            json_start = response_text.find("{")
            response_text = response_text[json_start:]

            # Find the matching closing brace by counting depth
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(response_text):
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found the matching closing brace
                            response_text = response_text[:i+1]
                            break

        try:
            data = json.loads(response_text)
            return data.get(key, [])
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Response text (first 500 chars): {response_text[:500]}...")
            return []
