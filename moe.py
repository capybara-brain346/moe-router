"""
Mixture of Experts (MoE) Layer Implementation

This module implements a sparse Mixture of Experts layer for conditional computation
in neural networks. The MoE layer routes each token to a single expert (top-1 routing)
based on learned routing probabilities, enabling efficient scaling of model capacity.

Classes:
    ExpertMLP: Individual expert network with two-layer MLP architecture
    MoELayer: Sparse MoE layer with top-1 routing and load balancing

Key Features:
    - Top-1 expert routing (each token processed by one expert)
    - Load balancing auxiliary loss to prevent expert collapse
    - Expert usage tracking for monitoring and analysis
    - Weighted expert outputs based on router confidence
    - Efficient batched computation per expert

Routing Mechanism:
    The router is a learned linear projection that computes logits for each expert.
    Softmax converts logits to probabilities, and the expert with highest probability
    is selected for each token. The expert output is weighted by the router probability
    to maintain gradient flow through the routing decision.

Load Balancing:
    An auxiliary loss encourages uniform expert utilization by penalizing the product
    of mean router probabilities and actual expert frequencies. This prevents the
    routing network from collapsing to use only a few experts.

Example:
    >>> import torch
    >>> from moe import MoELayer
    >>>
    >>> moe_layer = MoELayer(
    ...     hidden_dim=512,
    ...     num_experts=8,
    ...     ffn_dim=2048,
    ...     dropout=0.1,
    ...     load_balance_weight=0.01
    ... )
    >>>
    >>> x = torch.randn(2, 128, 512)
    >>> output, aux_loss = moe_layer(x)
    >>> print(f"Output shape: {output.shape}")
    >>> print(f"Auxiliary loss: {aux_loss.item():.4f}")
    >>>
    >>> usage = moe_layer.get_expert_usage()
    >>> print(f"Expert usage: {usage}")

Dependencies:
    - torch: PyTorch deep learning framework

Notes:
    - Expert selection uses argmax, which is non-differentiable but weighted by
      differentiable router probabilities
    - Expert usage statistics are only tracked during evaluation (not training)
    - Load balance loss is scaled by num_experts to be scale-invariant
    - The implementation processes each expert's tokens separately for memory efficiency
    - Router has no bias term (bias=False) following common MoE practices

References:
    - Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated
      Mixture-of-Experts Layer", ICLR 2017
    - Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models
      with Simple and Efficient Sparsity", JMLR 2022

Author: MoE Router Project
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class Expert(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        ffn_dim: int,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
        top_k: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self.top_k = top_k

        self.router = nn.Linear(
            hidden_dim, num_experts, bias=False
        )  # bias is false because router needs to route according to the representation rather than additional offset

        self.experts = nn.ModuleList(  # create list of experts ffns
            [Expert(hidden_dim, ffn_dim, dropout) for _ in range(num_experts)]
        )

        self.expert_counts = torch.zeros(num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        if self.top_k == 1:
            expert_weights, expert_indices = torch.max(router_probs, dim=-1)
            expert_indices = expert_indices.unsqueeze(-1)
            expert_weights = expert_weights.unsqueeze(-1)
        else:
            expert_weights, expert_indices = torch.topk(
                router_probs, self.top_k, dim=-1
            )
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        output_flat = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx).any(dim=-1)

            if expert_mask.any():
                expert_tokens = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_tokens)

                token_weights = torch.zeros(expert_mask.sum(), device=x.device)
                for k_idx in range(self.top_k):
                    k_mask = expert_indices[expert_mask, k_idx] == expert_idx
                    token_weights[k_mask] += expert_weights[expert_mask, k_idx][k_mask]

                output_flat[expert_mask] += expert_output * token_weights.unsqueeze(-1)

                if not self.training:
                    self.expert_counts[expert_idx] += expert_mask.sum().item()

        output = output_flat.view(batch_size, seq_len, hidden_dim)

        aux_loss = self._compute_load_balance_loss(router_probs, expert_indices)

        return output, aux_loss

    def _compute_load_balance_loss(
        self, router_probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = router_probs.shape[0]

        mean_router_probs = router_probs.mean(dim=0)

        if self.top_k == 1:
            expert_indices_flat = expert_indices.squeeze(-1)
        else:
            expert_indices_flat = expert_indices.view(-1)

        expert_counts = torch.bincount(
            expert_indices_flat, minlength=self.num_experts
        ).float()
        expert_frequencies = expert_counts / (num_tokens * self.top_k)

        load_balance_loss = self.num_experts * torch.sum(
            mean_router_probs * expert_frequencies
        )

        return self.load_balance_weight * load_balance_loss

    def get_expert_usage(self) -> Dict[int, int]:
        return {i: int(count) for i, count in enumerate(self.expert_counts)}

    def reset_expert_counts(self):
        self.expert_counts = torch.zeros(self.num_experts)
