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
        capacity_factor: float = 1.25,
        gating_temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.gating_temperature = gating_temperature

        self.router = nn.Linear(
            hidden_dim, num_experts, bias=False
        )

        self.experts = nn.ModuleList(
            [Expert(hidden_dim, ffn_dim, dropout) for _ in range(num_experts)]
        )

        self.expert_counts = torch.zeros(num_experts)
        self.expert_router_probs = torch.zeros(num_experts)
        self.num_tokens_processed = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        router_logits = self.router(x_flat)
        router_logits = router_logits / self.gating_temperature
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

        if not self.training:
            self.expert_router_probs += router_probs.mean(dim=0).cpu()
            self.num_tokens_processed += x_flat.shape[0]

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

    def get_expert_statistics(self) -> Dict:
        total_tokens = self.expert_counts.sum().item()
        if total_tokens == 0:
            return {
                "usage": {i: 0 for i in range(self.num_experts)},
                "percentages": {i: 0.0 for i in range(self.num_experts)},
                "entropy": 0.0,
                "min_usage_pct": 0.0,
                "max_usage_pct": 0.0,
            }

        usage_pct = (self.expert_counts / total_tokens * 100).tolist()
        
        probs = self.expert_counts / total_tokens
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum().item()

        return {
            "usage": {i: int(count) for i, count in enumerate(self.expert_counts)},
            "percentages": {i: pct for i, pct in enumerate(usage_pct)},
            "entropy": entropy,
            "min_usage_pct": min(usage_pct),
            "max_usage_pct": max(usage_pct),
        }

    def reset_expert_counts(self):
        self.expert_counts = torch.zeros(self.num_experts)
        self.expert_router_probs = torch.zeros(self.num_experts)
        self.num_tokens_processed = 0
    
    def set_gating_temperature(self, temperature: float):
        self.gating_temperature = temperature
