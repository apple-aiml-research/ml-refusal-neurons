#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Architecture Detection and Hook Setup Utilities

Supports:
- LLaMA/Qwen style: gated MLPs with gate_proj, up_proj, down_proj
- GPT-2/GPT-3 style: standard MLPs with c_fc, c_proj
"""

import torch
from typing import List, Callable, Tuple


def detect_architecture(model, verbose=False):
    """
    Detect model architecture type based on module names and structure

    Args:
        model: The model to detect
        verbose: Print debug information about modules found

    Returns:
        str: Architecture type - 'llama', 'gpt2', or 'unknown'
    """
    module_names = [name for name, _ in model.named_modules()]

    if verbose:
        print(f"\nDEBUG: Found {len(module_names)} modules")
        mlp_modules = [n for n in module_names if 'mlp' in n.lower() or 'fc' in n.lower()]
        print(f"DEBUG: MLP-related modules (first 20):")
        for name in mlp_modules[:20]:
            print(f"  - {name}")

    # LLaMA/Qwen detection (has gate_proj)
    if any('gate_proj' in name for name in module_names):
        return 'llama'

    # GPT-2 detection (has c_fc and c_proj in MLP)
    gpt2_patterns = ['c_fc', 'fc_in', 'mlp.0', 'mlp.dense_h_to_4h']
    if any(any(pattern in name for name in module_names) for pattern in gpt2_patterns):
        return 'gpt2'

    # Additional check: look for actual module attributes
    for name, module in model.named_modules():
        if 'mlp' in name.lower():
            if hasattr(module, 'gate_proj'):
                return 'llama'
            if hasattr(module, 'c_fc'):
                return 'gpt2'

    return 'unknown'


def get_mlp_modules(model, architecture: str) -> List[Tuple[str, any]]:
    """
    Get MLP modules based on architecture

    Args:
        model: The model
        architecture: Model architecture type

    Returns:
        List of (name, module) tuples for MLP layers
    """
    mlp_modules = []

    if architecture == 'llama':
        # LLaMA/Qwen: layers.X.mlp
        for name, module in model.named_modules():
            if name.endswith("mlp") and hasattr(module, 'gate_proj'):
                mlp_modules.append((name, module))

    elif architecture == 'gpt2':
        # GPT-2/GPT-3: h.X.mlp or transformer.h.X.mlp
        for name, module in model.named_modules():
            if ('mlp' in name or 'MLP' in name) and hasattr(module, 'c_fc'):
                mlp_modules.append((name, module))

    return mlp_modules


def create_hook_fn(architecture: str, captured_features: list, capture_stage: str = 'default') -> Callable:
    """
    Create appropriate hook function based on architecture

    Args:
        architecture: Model architecture type ('llama' or 'gpt2')
        captured_features: List to store captured features
        capture_stage: Which stage to capture features at:
            - 'default': Standard pre-down_proj features
            - 'up_proj': Just up_proj(x) - raw up projection (LLaMA only)
            - 'gate_proj': Just gate_proj(x) - raw gate projection (LLaMA only)
            - 'gated': act_fn(gate_proj(x)) * up_proj(x) - gated features
            - 'down_proj': Final output after down_proj

    Returns:
        Hook function compatible with register_forward_hook
    """

    if architecture == 'llama':
        def llama_hook(module, inp, output):
            """
            Capture LLaMA/Qwen MLP features at various stages

            Architecture:
                up_features = up_proj(x)
                gate_features = act_fn(gate_proj(x))
                gated = gate_features * up_features
                output = down_proj(gated)
            """
            inp = inp[0]

            if capture_stage == 'up_proj':
                # Capture raw up_proj output (before gating)
                features = module.up_proj(inp)
            elif capture_stage == 'gate_proj':
                # Capture raw gate_proj output (before activation)
                features = module.gate_proj(inp)
            elif capture_stage == 'down_proj':
                # Capture final output (after down_proj)
                features = output
            else:  # 'default' or 'gated'
                # Capture gated features (before down_proj)
                features = module.act_fn(module.gate_proj(inp)) * module.up_proj(inp)

            captured_features.append(features.detach())

        return llama_hook

    elif architecture == 'gpt2':
        def gpt2_hook(module, inp, output):
            """
            Capture GPT-2/GPT-3 MLP features at various stages

            Architecture:
                hidden = c_fc(x)
                activated = activation(hidden)
                output = c_proj(activated)
            """
            inp = inp[0]

            if capture_stage == 'down_proj':
                # Capture final output (after c_proj)
                features = output
            else:  # 'default' or any other
                # Apply c_fc and activation
                hidden = module.c_fc(inp)

                # Apply activation function
                if hasattr(module, 'act'):
                    features = module.act(hidden)
                elif hasattr(module, 'activation_fn'):
                    features = module.activation_fn(hidden)
                else:
                    features = torch.nn.functional.gelu(hidden)

            captured_features.append(features.detach())

        return gpt2_hook

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def setup_hooks(model, architecture: str = None, verbose: bool = False, capture_stage: str = 'default') -> Tuple[List, List[str], Callable]:
    """
    Setup activation hooks for a model

    Args:
        model: The model to hook
        architecture: Architecture type (if None, will auto-detect)
        verbose: Print debug information during detection
        capture_stage: Which stage to capture features at

    Returns:
        Tuple of:
        - captured_features: List to store captured activations
        - mlp_layer_names: List of MLP layer names
        - cleanup_fn: Function to remove all hooks
    """
    if architecture is None or architecture == 'auto':
        architecture = detect_architecture(model, verbose=verbose)
        print(f"Detected architecture: {architecture}")

    if architecture == 'unknown':
        print(f"\n❌ Could not detect model architecture.")
        print(f"Re-running detection with verbose mode...\n")
        architecture = detect_architecture(model, verbose=True)

        if architecture == 'unknown':
            print(f"\n❌ Still could not detect architecture.")
            print(f"\nPlease manually specify architecture using --architecture flag:")
            print(f"  --architecture llama  (for LLaMA/Qwen/Mistral)")
            print(f"  --architecture gpt2   (for GPT-2/GPT-3/GPT-Neo/GPT-J)")
            raise ValueError("Could not detect model architecture. Please specify architecture manually.")

    # Storage for captured features
    captured_features = []

    # Get MLP modules
    mlp_modules = get_mlp_modules(model, architecture)

    if not mlp_modules:
        raise ValueError(f"No MLP modules found for architecture: {architecture}")

    # Create hook function with capture_stage
    hook_fn = create_hook_fn(architecture, captured_features, capture_stage=capture_stage)

    # Register hooks and store handles for cleanup
    hook_handles = []
    mlp_layer_names = []

    for name, module in mlp_modules:
        handle = module.register_forward_hook(hook_fn)
        hook_handles.append(handle)
        mlp_layer_names.append(name)

    stage_msg = f" (capturing at: {capture_stage})" if capture_stage != 'default' else ""
    print(f"✓ Registered {len(mlp_layer_names)} hooks on {architecture} MLP layers{stage_msg}")

    # Cleanup function to remove all hooks
    def cleanup_hooks():
        for handle in hook_handles:
            handle.remove()

    return captured_features, mlp_layer_names, cleanup_hooks
