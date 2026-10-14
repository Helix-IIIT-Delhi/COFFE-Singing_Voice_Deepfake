"""Chernoff Distance Computation for COFFE Framework"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def chernoff_distance(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
    num_samples: int = 100
) -> torch.Tensor:
    """
    Compute Chernoff distance between two probability distributions.

    D_C(P, Q) = min_{0≤s≤1} [-s*ln(E[P(x)^s/Q(x)^(1-s)]) - (1-s)*ln(E[Q(x)^s/P(x)^(1-s)])]

    For practical computation with features:
    Approximate using Gaussian distribution assumptions

    Args:
        p: Features from model P [batch_size, feature_dim]
        q: Features from model Q [batch_size, feature_dim]
        eps: Numerical epsilon for stability
        num_samples: Number of samples for Chernoff integration

    Returns:
        Chernoff distances [batch_size]
    """
    batch_size = p.size(0)

    # Compute means
    mean_p = p.mean(dim=0)
    mean_q = q.mean(dim=0)

    # Compute covariances
    cov_p = torch.cov(p.T)
    cov_q = torch.cov(q.T)

    # Initialize Chernoff distance
    chernoff_dist = torch.zeros(batch_size, device=p.device)

    # Find optimal s that minimizes Chernoff distance
    # Using numerical optimization over s in [0, 1]
    s_values = torch.linspace(0, 1, num_samples, device=p.device)

    min_divergence = float('inf')

    for s in s_values:
        # Compute weighted combination of covariances
        cov_s = s * cov_p + (1 - s) * cov_q

        # Regularize for numerical stability
        cov_s = cov_s + eps * torch.eye(cov_s.size(0), device=cov_s.device)

        try:
            # Compute log determinants
            log_det_cov_s = torch.log(torch.det(cov_s) + eps)
            log_det_cov_p = torch.log(torch.det(cov_p + eps * torch.eye(cov_p.size(0), device=cov_p.device)) + eps)
            log_det_cov_q = torch.log(torch.det(cov_q + eps * torch.eye(cov_q.size(0), device=cov_q.device)) + eps)

            # Compute Mahalanobis distance term
            mean_diff = mean_p - mean_q
            cov_s_inv = torch.inverse(cov_s + eps * torch.eye(cov_s.size(0), device=cov_s.device))
            mahal_term = torch.matmul(mean_diff, torch.matmul(cov_s_inv, mean_diff))

            # Chernoff divergence formula
            divergence = (0.5 * s * (1 - s) * mahal_term +
                         0.5 * (s * log_det_cov_p + (1 - s) * log_det_cov_q - log_det_cov_s))

            if divergence < min_divergence:
                min_divergence = divergence
        except:
            continue

    # Return Chernoff distance (clamp to avoid negative values)
    return torch.clamp(torch.full((batch_size,), min_divergence, device=p.device), min=0)


def chernoff_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int
) -> torch.Tensor:
    """
    Compute Chernoff-inspired loss for source attribution.

    Uses Chernoff distance to weight different sources based on
    their divergence from the predicted source.

    Args:
        logits: Model outputs [batch_size, num_sources]
        targets: Ground truth source labels [batch_size]
        num_sources: Number of source classes

    Returns:
        Weighted loss value
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)

    # Compute Chernoff-weighted cross entropy
    loss = 0
    batch_size = logits.size(0)

    for i in range(batch_size):
        true_source = targets[i]

        # Extract probability distribution for this sample
        p_dist = probs[i]

        # Create one-hot target distribution
        q_dist = F.one_hot(true_source, num_sources).float()

        # Compute Chernoff distance between predicted and true distribution
        chernoff_dist = chernoff_distance(
            p_dist.unsqueeze(0),
            q_dist.unsqueeze(0)
        )

        # Weight cross entropy by Chernoff distance
        ce_loss = F.cross_entropy(logits[i:i+1], targets[i:i+1])
        loss = loss + chernoff_dist * ce_loss

    return loss / batch_size
