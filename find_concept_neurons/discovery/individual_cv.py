#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""
Individual Feature Discovery using Cross-Validation

Evaluates each feature independently using logistic regression with CV.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')

_BREAKPOINT_HIT = False

# Standalone function for parallel evaluation (avoids pickling entire class)
def _evaluate_feature_standalone(X_feature: np.ndarray, y: np.ndarray, cv_folds: int, feature_idx: int, layer_idx: int) -> Dict[str, float]:
    """
    Evaluate a single feature (1 column) using cross-validation.
    Used for all aggregation modes except minmax_tokens.
    """
    X_feature = X_feature.reshape(-1, 1)  # [n_samples, 1]

    # Adjust CV folds
    class_counts = np.bincount(y.astype(int))
    min_class_size = np.min(class_counts)
    cv_folds = min(cv_folds, min_class_size)
    if cv_folds < 2:
        cv_folds = 2
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X_feature, y):
        X_train, X_val = X_feature[train_idx], X_feature[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LogisticRegression(random_state=42, max_iter=500, solver='liblinear')
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        cv_scores.append(acc)

    return {
        'cv_accuracy': float(np.mean(cv_scores)),
        'feature_idx': int(feature_idx),
        'layer_idx': int(layer_idx)
    }


def _evaluate_feature_minmax(X_min: np.ndarray, X_max: np.ndarray, y: np.ndarray, cv_folds: int, feature_idx: int, layer_idx: int) -> Dict[str, float]:
    """
    Evaluate a single feature using both its min and max across selected tokens.
    X_min and X_max are each [n_samples], combined into [n_samples, 2].
    """
    X_feature = np.stack([X_min, X_max], axis=1)  # [n_samples, 2]

    class_counts = np.bincount(y.astype(int))
    min_class_size = np.min(class_counts)
    cv_folds = min(cv_folds, min_class_size)
    if cv_folds < 2:
        cv_folds = 2
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in cv.split(X_feature, y):
        X_train, X_val = X_feature[train_idx], X_feature[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LogisticRegression(random_state=42, max_iter=500, solver='liblinear')
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))
        cv_scores.append(acc)

    return {
        'cv_accuracy': float(np.mean(cv_scores)),
        'feature_idx': int(feature_idx),
        'layer_idx': int(layer_idx)
    }


class FeatureDiscovery:
    """
    Discover concept-specific features using individual cross-validation

    This method evaluates each feature separately by fitting a logistic
    regression model for each feature and computing cross-validation accuracy.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        architecture: str = "auto",
        capture_stage: str = "default",
        cv_folds: int = 3,
        normalize: bool = False,
        token_aggregation: str = "max",
        selected_token: int = -1,
        selected_tokens: list = None,
        last_n_tokens: int = 5,
        n_jobs: int = -1
    ):
        """
        Initialize feature discovery

        Args:
            model: The language model to analyze
            tokenizer: Model tokenizer
            device: Device to run on ('cuda' or 'cpu')
            architecture: Model architecture ('auto', 'llama', or 'gpt2')
            capture_stage: Which stage to capture ('default', 'up_proj', 'gated', 'down_proj', 'residual')
            cv_folds: Number of cross-validation folds
            normalize: Whether to normalize features with StandardScaler
            token_aggregation: How to aggregate across tokens ('max' or 'mean')
            n_jobs: Number of parallel jobs for evaluation (-1 = all cores)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.architecture = architecture
        self.capture_stage = capture_stage
        self.cv_folds = cv_folds
        self.normalize = normalize
        self.token_aggregation = token_aggregation
        self.selected_token = selected_token
        self.selected_tokens = selected_tokens or [-1]
        self.last_n_tokens = last_n_tokens
        self.n_jobs = n_jobs

        # Will be set by setup_hooks
        self.captured_features = None
        self.mlp_layers = None
        self.cleanup_hooks = None

    def setup_hooks(self):
        """Setup activation capture hooks (simple version)"""
        self.captured_features = []
        capture_stage = self.capture_stage

        if capture_stage == "residual":
            # Hook the entire transformer layer (residual stream)
            def residual_hook_fn(module, inp, output):
                """Capture residual stream (layer output)"""
                # Output is typically a tuple (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self.captured_features.append(hidden_states.detach())

            # Register hooks on transformer layers
            self.mlp_layers = []
            for name, module in self.model.named_modules():
                # Hook onto transformer blocks/layers
                # Common patterns: model.layers.X, model.h.X, model.blocks.X
                if '.layers.' in name or '.h.' in name or '.blocks.' in name:
                    parts = name.split('.')
                    # Only hook complete layers (e.g., model.layers.0, not model.layers.0.mlp)
                    if len(parts) >= 3 and parts[-1].isdigit():
                        module.register_forward_hook(residual_hook_fn)
                        self.mlp_layers.append(name)

            print(f"✓ Registered hooks on {len(self.mlp_layers)} transformer layers (residual stream)")
            print(f"  Capture stage: {capture_stage}")

        else:
            # Hook MLP layers for internal MLP features
            def hook_fn(module, inp, output):
                """Capture MLP features at specified stage"""
                inp = inp[0]

                # Different capture stages for gated MLPs (LLaMA/Qwen)
                if capture_stage == "up_proj":
                    # Just up projection (before gating)
                    features = module.up_proj(inp)
                elif capture_stage == "gate_proj":
                    # Just gate projection with activation
                    features = module.act_fn(module.gate_proj(inp))
                elif capture_stage == "gated" or capture_stage == "default":
                    # Gated features (default): gate * up
                    features = module.act_fn(module.gate_proj(inp)) * module.up_proj(inp)
                    # features = torch.abs(module.act_fn(module.gate_proj(inp)) * module.up_proj(inp))
                    # features = torch.abs(module.up_proj(inp))
                elif capture_stage == "down_proj":
                    # After down projection (final MLP output)
                    gated = module.act_fn(module.gate_proj(inp)) * module.up_proj(inp)
                    features = module.down_proj(gated)
                else:
                    # Fallback to default (gated)
                    features = module.act_fn(module.gate_proj(inp)) * module.up_proj(inp)

                self.captured_features.append(features.detach())

            # Register hooks on MLP layers
            self.mlp_layers = []
            for name, module in self.model.named_modules():
                if name.endswith("mlp"):
                    module.register_forward_hook(hook_fn)
                    self.mlp_layers.append(name)

            print(f"✓ Registered hooks on {len(self.mlp_layers)} MLP layers")
            print(f"  Capture stage: {capture_stage}")

    def collect_activations(
        self,
        examples: List[str],
        label_name: str,
        layers_to_analyze: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Collect activations for a list of examples

        Args:
            examples: List of text examples
            label_name: Label for progress bar
            layers_to_analyze: Which layers to collect (None = all)

        Returns:
            Dict mapping layer_idx -> activations array [n_examples, n_features]
        """
        if layers_to_analyze is None:
            layers_to_analyze = list(range(len(self.mlp_layers)))

        all_activations = {i: [] for i in layers_to_analyze}
        _printed_last_n_tokens = False

        for example in tqdm(examples, desc=f"Processing {label_name} examples"):
            self.captured_features.clear()

            # Format as chat message
            messages = [{"role": "user", "content": example}]
            full_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=4096)
            input_ids = inputs["input_ids"].to(self.device)

            # Skip empty sequences
            if input_ids.shape[1] == 0:
                continue

            # Print the last-n tokens once so the user can verify the window
            if self.token_aggregation in ("max_over_last", "mean_over_last") and not _printed_last_n_tokens:
                seq_len = input_ids.shape[1]
                n = min(self.last_n_tokens, seq_len)
                last_ids = input_ids[0, -n:].tolist()
                last_tokens = [self.tokenizer.decode([tid]) for tid in last_ids]
                print(f"\n[{self.token_aggregation}] last {n} tokens (positions -{n} to -1):")
                for i, (tid, tok) in enumerate(zip(last_ids, last_tokens)):
                    print(f"  pos -{n - i}: id={tid:6d}  '{tok}'")
                _printed_last_n_tokens = True

            # Print the selected token once so the user can verify
            if self.token_aggregation == "select" and not _printed_last_n_tokens:
                tid = input_ids[0, self.selected_token].item()
                tok = self.tokenizer.decode([tid])
                print(f"\n[select] selected token (pos {self.selected_token}): id={tid:6d}  '{tok}'")
                _printed_last_n_tokens = True

            if self.token_aggregation in ("selected_tokens", "minmax_tokens") and not _printed_last_n_tokens:
                print(f"\n[{self.token_aggregation}] positions: {self.selected_tokens}")
                for pos in self.selected_tokens:
                    tid = input_ids[0, pos].item()
                    tok = self.tokenizer.decode([tid])
                    print(f"  pos {pos:3d}: id={tid:6d}  '{tok}'")
                _printed_last_n_tokens = True

            # Forward pass
            with torch.no_grad():
                _ = self.model(input_ids)

            # Extract features for each layer
            for layer_idx in layers_to_analyze:
                if layer_idx < len(self.captured_features):
                    features = self.captured_features[layer_idx]  # [1, seq_len, hidden_dim]

                    # Aggregate across tokens
                    if self.token_aggregation == "max":
                        aggregated_features, _ = torch.max(features, dim=1)  # [1, hidden_dim]
                    elif self.token_aggregation == "min":
                        aggregated_features, _ = torch.min(features, dim=1)  # [1, hidden_dim]
                    elif self.token_aggregation == "mean":
                        aggregated_features = torch.mean(features, dim=1)  # [1, hidden_dim]
                    elif self.token_aggregation == "select":
                        aggregated_features = features[:, self.selected_token, :]  # [1, hidden_dim]
                    elif self.token_aggregation == "max_over_last":
                        n = min(self.last_n_tokens, features.shape[1])
                        aggregated_features, _ = torch.max(features[:, -n:, :], dim=1)  # [1, hidden_dim]
                    elif self.token_aggregation == "mean_over_last":
                        n = min(self.last_n_tokens, features.shape[1])
                        aggregated_features = torch.mean(features[:, -n:, :], dim=1)  # [1, hidden_dim]
                    elif self.token_aggregation == "selected_tokens":
                        # Stack selected token positions: [1, k, hidden_dim] -> [1, k*hidden_dim]
                        seq_len = features.shape[1]
                        selected_feats = [features[:, pos, :] for pos in self.selected_tokens
                                         if -seq_len <= pos < seq_len]
                        aggregated_features = torch.cat(selected_feats, dim=1)  # [1, k*hidden_dim]
                    elif self.token_aggregation == "minmax_tokens":
                        # Min and max across selected positions: [1, 2*hidden_dim]
                        seq_len = features.shape[1]
                        selected_feats = torch.stack([features[:, pos, :] for pos in self.selected_tokens
                                                     if -seq_len <= pos < seq_len], dim=1)  # [1, k, hidden_dim]
                        min_feats, _ = torch.min(selected_feats, dim=1)  # [1, hidden_dim]
                        max_feats, _ = torch.max(selected_feats, dim=1)  # [1, hidden_dim]
                        aggregated_features = torch.cat([min_feats, max_feats], dim=1)  # [1, 2*hidden_dim]

                    # Convert to float32 (numpy doesn't support bfloat16)
                    aggregated_features = aggregated_features.float()
                    all_activations[layer_idx].append(aggregated_features.cpu().numpy()[0])

        # Convert to numpy arrays
        for layer_idx in layers_to_analyze:
            all_activations[layer_idx] = np.array(all_activations[layer_idx])

        return all_activations

    def evaluate_single_feature(
        self,
        X_feature: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 3
    ) -> Dict[str, float]:
        """
        Evaluate a single feature using cross-validation

        Args:
            X_feature: Feature values [n_samples]
            y: Binary labels [n_samples]
            cv_folds: Number of CV folds

        Returns:
            Dict with cv_accuracy
        """
        X_feature = X_feature.reshape(-1, 1)  # [n_samples, 1]

        # Adjust CV folds
        class_counts = np.bincount(y.astype(int))
        min_class_size = np.min(class_counts)
        cv_folds = min(cv_folds, min_class_size)
        if cv_folds < 2:
            cv_folds = 2

        # CV Accuracy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in cv.split(X_feature, y):
            X_train, X_val = X_feature[train_idx], X_feature[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = LogisticRegression(random_state=42, max_iter=500, solver='liblinear')
            model.fit(X_train, y_train)

            cv_scores.append(accuracy_score(y_val, model.predict(X_val)))

        return {'cv_accuracy': np.mean(cv_scores)}

    def find_features(
        self,
        positive_examples: List[str],
        negative_examples: List[str],
        layers_to_analyze: Optional[List[int]] = None,
        top_k: int = 10
    ) -> Dict:
        """
        Find top features for a concept

        Args:
            positive_examples: List of positive example texts
            negative_examples: List of negative example texts
            layers_to_analyze: Which layers to analyze (None = all)
            top_k: Number of top features to return per layer

        Returns:
            Results dict with top features per layer and global ranking
        """
        # Setup hooks if not already done
        if self.captured_features is None:
            self.setup_hooks()

        # Collect activations
        print("\n" + "="*60)
        print("ACTIVATION COLLECTION")
        print("="*60)

        positive_activations = self.collect_activations(
            positive_examples,
            "positive",
            layers_to_analyze
        )
        negative_activations = self.collect_activations(
            negative_examples,
            "negative",
            layers_to_analyze
        )

        print(f"\n✓ Collected activations for {len(positive_examples)} positive and {len(negative_examples)} negative examples")

        # Evaluate features
        print("\n" + "="*60)
        print("INDIVIDUAL FEATURE EVALUATION")
        print("="*60)

        if layers_to_analyze is None:
            layers_to_analyze = list(range(len(self.mlp_layers)))

        results = {
            "layers": {},
            "global_top_candidates": []
        }

        all_candidates = []

        for layer_idx in tqdm(layers_to_analyze, desc="Evaluating layers"):
            if layer_idx not in positive_activations or layer_idx not in negative_activations:
                continue

            pos_acts = positive_activations[layer_idx]
            neg_acts = negative_activations[layer_idx]

            if len(pos_acts) == 0 or len(neg_acts) == 0:
                continue

            # Combine data
            X = np.vstack([pos_acts, neg_acts])  # [2k, hidden_dim]
            y = np.hstack([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])

            # Normalize if requested
            if self.normalize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            hidden_dim = X.shape[1]

            # Evaluate all features in parallel
            print(f"  Evaluating features in parallel (n_jobs={self.n_jobs})...")
            if self.token_aggregation == "minmax_tokens":
                # X is [n_samples, 2*hidden_dim]: first half = min, second half = max
                actual_hidden_dim = hidden_dim // 2
                print(f"  Mode: minmax_tokens — {actual_hidden_dim} features, 2 inputs each (min+max)")
                feature_metrics = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(_evaluate_feature_minmax)(
                        X[:, feature_idx], X[:, actual_hidden_dim + feature_idx],
                        y, self.cv_folds, feature_idx, layer_idx)
                    for feature_idx in tqdm(range(actual_hidden_dim), desc=f"Layer {layer_idx}", leave=False)
                )
            else:
                print(f"  {hidden_dim} features")
                feature_metrics = Parallel(n_jobs=self.n_jobs, backend='loky')(
                    delayed(_evaluate_feature_standalone)(X[:, feature_idx], y, self.cv_folds, feature_idx, layer_idx)
                    for feature_idx in tqdm(range(hidden_dim), desc=f"Layer {layer_idx}", leave=False)
                )

            # Sort by CV accuracy
            feature_metrics.sort(key=lambda x: x['cv_accuracy'], reverse=True)

            # Store layer results
            results["layers"][f"layer_{layer_idx}"] = {
                "num_features": int(hidden_dim),
                "top_features": feature_metrics[:top_k]
            }

            # Add to global candidates
            all_candidates.extend(feature_metrics)

            # Print layer summary immediately
            num_features_to_show = 20
            print(f"\nLayer {layer_idx}: Top {min(num_features_to_show, len(feature_metrics))} features (by CV accuracy):")
            for i, feat in enumerate(feature_metrics[:num_features_to_show]):
                print(f"  {i+1}. Feature {feat['feature_idx']:5d} | CV Acc: {feat['cv_accuracy']:.3f}")

        # Global ranking
        all_candidates.sort(key=lambda x: x['cv_accuracy'], reverse=True)
        results["global_top_candidates"] = all_candidates[:100]  # Top 50 globally

        print(f"\n✓ Feature discovery complete!")

        return results

    def cleanup(self):
        """Remove hooks"""
        if self.cleanup_hooks:
            self.cleanup_hooks()
