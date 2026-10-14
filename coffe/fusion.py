"""COFFE Fusion Framework for Singing Voice Deepfake Source Attribution"""

import torch
import torch.nn as nn
from typing import List, Dict
from .distance import chernoff_distance


class COFFEFusion(nn.Module):
    """
    COFFE: ChernOFF DistancE-based Fusion Framework

    Fuses multimodal foundation models for singing voice deepfake
    source attribution using Chernoff distance-based weighting.

    Args:
        models: List of foundation model extractors
        num_sources: Number of singing voice deepfake sources
        fusion_dim: Dimension of fused representation
        learnable_weights: Whether to learn fusion weights
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_sources: int,
        fusion_dim: int = 512,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_sources = num_sources
        self.fusion_dim = fusion_dim
        self.learnable_weights = learnable_weights

        # Learnable fusion weights
        if learnable_weights:
            self.fusion_weights = nn.Parameter(
                torch.ones(self.num_models) / self.num_models
            )
        else:
            self.register_buffer(
                "fusion_weights",
                torch.ones(self.num_models) / self.num_models
            )

        # Projection layers for each model
        self.projections = nn.ModuleList([
            nn.Linear(model.output_dim if hasattr(model, 'output_dim') else 512, fusion_dim)
            for model in models
        ])

        # Source attribution head
        self.source_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_sources),
        )

    def forward(self, audio_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with COFFE fusion.

        Args:
            audio_features: Input audio features

        Returns:
            Dict with:
                - source_logits: Predicted source logits
                - source_probs: Probability distribution over sources
                - fused_features: Fused representation
                - model_outputs: Individual model outputs
                - fusion_weights: Learned/computed fusion weights
        """
        # Extract features from each model
        model_outputs = []
        projected_features = []

        for i, (model, proj) in enumerate(zip(self.models, self.projections)):
            features = model(audio_features)
            model_outputs.append(features)
            proj_features = proj(features)
            projected_features.append(proj_features)

        # Compute Chernoff distances between model outputs for adaptive weighting
        chernoff_weights = self._compute_chernoff_weights(projected_features)

        # Adaptive fusion using Chernoff-weighted combination
        fused_features = torch.zeros_like(projected_features[0])
        for i, feat in enumerate(projected_features):
            fused_features += chernoff_weights[:, i:i+1] * feat

        # Predict source
        source_logits = self.source_head(fused_features)
        source_probs = torch.softmax(source_logits, dim=1)

        return {
            "source_logits": source_logits,
            "source_probs": source_probs,
            "fused_features": fused_features,
            "model_outputs": model_outputs,
            "fusion_weights": chernoff_weights,
        }

    def _compute_chernoff_weights(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Compute fusion weights using Chernoff distances"""
        batch_size = features[0].size(0)

        # Initialize with learned weights
        weights = self.fusion_weights.unsqueeze(0).expand(batch_size, -1)

        # Adjust weights based on Chernoff distances
        # Models with higher divergence get higher weight
        for i in range(self.num_models):
            chernoff_dists = []
            for j in range(self.num_models):
                if i != j:
                    dist = chernoff_distance(features[i], features[j])
                    chernoff_dists.append(dist.unsqueeze(1))

            if chernoff_dists:
                avg_dist = torch.cat(chernoff_dists, dim=1).mean(dim=1, keepdim=True)
                weights[:, i] = weights[:, i] * (1.0 + avg_dist.squeeze())

        # Normalize weights
        weights = torch.softmax(weights, dim=1)

        return weights
