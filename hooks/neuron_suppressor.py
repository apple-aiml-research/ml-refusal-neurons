#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
NeuronSuppressor: a simple interface for suppressing a single refusal neuron.

Usage:
    from hooks import NeuronSuppressor

    # Pre-configured from paper — uses top ASR neuron from rankings.json
    suppressor = NeuronSuppressor.from_pretrained(model, "meta-llama/Meta-Llama-3.1-8B-Instruct")

    # Manual
    suppressor = NeuronSuppressor(model, layer=11, feature=4258, multiplier=-4.0)

    # Concept-neuron amplification (adds to the activation instead of replacing it)
    suppressor = NeuronSuppressor(model, layer=20, feature=4256, multiplier=250, additive=True)

    # Use as context manager (hook removed on exit)
    with suppressor:
        output = model.generate(...)

    # Or activate/deactivate manually
    suppressor.activate()
    output = model.generate(...)
    suppressor.deactivate()

    # Remove hook entirely
    suppressor.remove()
"""

import json
import os
from hooks.amplify_hooks import register_amplify_hook
from hooks.model_hooks import detect_architecture


# Map HuggingFace model ID → rankings.json key
_MODEL_KEY = {
    "Qwen/Qwen3-1.7B":                          "qwen1.7b",
    "Qwen/Qwen3-4B":                             "qwen4b",
    "Qwen/Qwen3-8B":                             "qwen8b",
    "Qwen/Qwen3-14B":                            "qwen14b",
    "Qwen/Qwen3-32B":                            "qwen32b",
    "meta-llama/Meta-Llama-3.1-8B-Instruct":     "llama8b",
    "meta-llama/Meta-Llama-3.1-70B-Instruct":    "llama70b",
}

_RANKINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rankings.json")


def _load_top_neuron(model_name: str) -> dict:
    """Load the top ASR neuron config for model_name from rankings.json."""
    if model_name not in _MODEL_KEY:
        supported = "\n  ".join(_MODEL_KEY.keys())
        raise ValueError(
            f"No pre-configured neuron for '{model_name}'.\n"
            f"Supported models:\n  {supported}\n"
            f"For other models, use NeuronSuppressor(model, layer, feature, multiplier) directly."
        )
    rankings_path = os.path.normpath(_RANKINGS_PATH)
    if not os.path.exists(rankings_path):
        raise FileNotFoundError(
            f"rankings.json not found at {rankings_path}. "
            f"Make sure data/rankings.json is present in the repo."
        )
    with open(rankings_path) as f:
        rankings = json.load(f)
    key = _MODEL_KEY[model_name]
    top = sorted(rankings[key], key=lambda x: x["hb191_asr"], reverse=True)[0]
    return {
        "layer":      top["layer"],
        "feature":    top["feature"],
        "multiplier": top["best_mult"],
    }


class NeuronSuppressor:
    """Suppresses (or amplifies) a single MLP neuron during model generation."""

    def __init__(self, model, layer: int, feature: int, multiplier: float, additive: bool = False):
        """
        Args:
            model:       A loaded HuggingFace causal LM.
            layer:       MLP layer index (0-based).
            feature:     Neuron index within the MLP intermediate dimension.
            multiplier:  Value used to modify the neuron's activation.
            additive:    If True, add multiplier to the activation instead of replacing it
                         (used for concept-neuron amplification). Default False (replace —
                         used for refusal-neuron suppression, matching the paper).
        """
        self.model = model
        self.layer = layer
        self.feature = feature
        self.multiplier = multiplier
        self.additive = additive

        arch = detect_architecture(model)
        self._handle, self._state, _ = register_amplify_hook(
            model, arch, layer, feature,
            multiplier=multiplier,
            response_mult=multiplier,
            pre_down_proj=True,
            additive=additive,
        )

    @classmethod
    def from_pretrained(cls, model, model_name: str):
        """
        Create a NeuronSuppressor pre-configured with the top ASR refusal neuron
        for the given model (loaded from data/rankings.json).

        Args:
            model:       A loaded HuggingFace causal LM.
            model_name:  HuggingFace model ID (e.g. "Qwen/Qwen3-14B").

        Raises:
            ValueError:        If model_name is not one of the 7 paper models.
            FileNotFoundError: If data/rankings.json is not found.
        """
        cfg = _load_top_neuron(model_name)
        return cls(model, layer=cfg["layer"], feature=cfg["feature"], multiplier=cfg["multiplier"])

    def activate(self):
        """Enable the suppression hook."""
        self._state["active"] = True

    def deactivate(self):
        """Disable the suppression hook (model generates normally)."""
        self._state["active"] = False

    def remove(self):
        """Remove the hook from the model entirely."""
        self._handle.remove()

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *args):
        self.deactivate()

    def __repr__(self):
        return (f"NeuronSuppressor(layer={self.layer}, feature={self.feature}, "
                f"multiplier={self.multiplier}, additive={self.additive})")
