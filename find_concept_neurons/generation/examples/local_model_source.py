#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Local Model-based Example Generation

Generates positive/negative examples using a local Hugging Face model
"""

import json
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM


class LocalModelGenerator:
    """Generate examples using a local Hugging Face model"""

    def __init__(self, model_name: str = "huihui-ai/Qwen3-32B-abliterated"):
        """
        Initialize local model generator

        Args:
            model_name: Hugging Face model name/path
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None

    def _load_model(self):
        """Lazy load model and tokenizer"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            ).eval()

            print(f"✓ Model loaded on {self.device}")

    def _generate_response(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.8) -> str:
        """
        Generate response from local model

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        self._load_model()

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text

    def generate_positive_examples(
        self,
        concept: str,
        k: int = 100,
        batch_size: int = 20
    ) -> List[str]:
        """
        Generate k positive examples for a concept

        Args:
            concept: Concept to generate examples for
            k: Number of examples
            batch_size: Examples per generation call

        Returns:
            List of k example texts
        """
        all_examples = []
        seen = set()  # exact-match dedup across the entire generation

        batch_num = 1
        max_batches = max(20, (k // max(1, batch_size)) * 8)  # safety cap against convergent loops
        while len(all_examples) < k and batch_num <= max_batches:
            current_batch_size = batch_size  # always request full batch; count only unique-new below

            print(f"Generating positive batch {batch_num} ({current_batch_size} examples; have {len(all_examples)}/{k})...")

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

            try:
                response_text = self._generate_response(prompt, max_tokens=3000)
                examples = self._parse_json_response(response_text, "positive")

                if examples:
                    added = 0
                    dropped_dupes = 0
                    dropped_redacted = 0
                    for ex in examples:
                        s = ex.strip()
                        if not s:
                            continue
                        if "REDACTED" in s.upper():
                            dropped_redacted += 1
                            continue
                        if s in seen:
                            dropped_dupes += 1
                            continue
                        seen.add(s)
                        all_examples.append(s)
                        added += 1
                        if len(all_examples) >= k:
                            break
                    print(f"✓ Batch {batch_num}: +{added} unique, {dropped_dupes} dupes, {dropped_redacted} redacted. Total: {len(all_examples)}/{k}")
                    batch_num += 1
                else:
                    print(f"⚠️ Batch {batch_num} failed, retrying...")
                    batch_num += 1

            except Exception as e:
                print(f"❌ Error generating batch: {e}")
                batch_num += 1

        if len(all_examples) < k:
            print(f"⚠️ Stopped after {batch_num-1} batches with {len(all_examples)}/{k} unique positive examples")

        return all_examples[:k]

    def generate_negative_examples(
        self,
        concept: str,
        k: int = 100,
        batch_size: int = 20
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
        seen = set()  # exact-match dedup across the entire generation

        batch_num = 1
        max_batches = max(20, (k // max(1, batch_size)) * 8)  # safety cap against convergent loops
        while len(all_examples) < k and batch_num <= max_batches:
            current_batch_size = batch_size  # always request full batch; count only unique-new below

            print(f"Generating negative batch {batch_num} ({current_batch_size} examples; have {len(all_examples)}/{k})...")

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

            try:
                response_text = self._generate_response(prompt, max_tokens=3000)
                examples = self._parse_json_response(response_text, "negative")

                if examples:
                    added = 0
                    dropped_dupes = 0
                    dropped_redacted = 0
                    for ex in examples:
                        s = ex.strip()
                        if not s:
                            continue
                        if "REDACTED" in s.upper():
                            dropped_redacted += 1
                            continue
                        if s in seen:
                            dropped_dupes += 1
                            continue
                        seen.add(s)
                        all_examples.append(s)
                        added += 1
                        if len(all_examples) >= k:
                            break
                    print(f"✓ Batch {batch_num}: +{added} unique, {dropped_dupes} dupes, {dropped_redacted} redacted. Total: {len(all_examples)}/{k}")
                    batch_num += 1
                else:
                    print(f"⚠️ Batch {batch_num} failed, retrying...")
                    batch_num += 1

            except Exception as e:
                print(f"❌ Error generating batch: {e}")
                batch_num += 1

        if len(all_examples) < k:
            print(f"⚠️ Stopped after {batch_num-1} batches with {len(all_examples)}/{k} unique negative examples")

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
        elif "{" in response_text and "}" in response_text:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            response_text = response_text[json_start:json_end].strip()

        try:
            data = json.loads(response_text)
            return data.get(key, [])
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
            print(f"Response text (first 500 chars): {response_text[:500]}...")
            return []
