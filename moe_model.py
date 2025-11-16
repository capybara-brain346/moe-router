import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ExpertMLP(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class MoERouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        ffn_dim: int,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight

        self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_dim, ffn_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        expert_weights, expert_indices = torch.max(router_probs, dim=-1)

        expert_indices_flat = expert_indices.view(-1)

        output_flat = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            expert_mask = expert_indices_flat == expert_idx

            if expert_mask.any():
                expert_tokens = x_flat[expert_mask]

                expert_output = self.experts[expert_idx](expert_tokens)

                output_flat[expert_mask] = expert_output * expert_weights[
                    expert_mask
                ].unsqueeze(-1)

        output = output_flat.view(batch_size, seq_len, hidden_dim)

        aux_loss = self._compute_load_balance_loss(router_probs, expert_indices_flat)

        return output, aux_loss

    def _compute_load_balance_loss(
        self, router_probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = router_probs.shape[0]

        mean_router_probs = router_probs.mean(dim=0)

        expert_counts = torch.bincount(
            expert_indices, minlength=self.num_experts
        ).float()
        expert_frequencies = expert_counts / num_tokens

        load_balance_loss = self.num_experts * torch.sum(
            mean_router_probs * expert_frequencies
        )

        return self.load_balance_weight * load_balance_loss


class TransformerWithMoE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_experts: int,
        ffn_dim: int,
        num_layers: int = 2,
        max_seq_len: int = 512,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.moe_layers = nn.ModuleList(
            [
                MoERouter(hidden_dim, num_experts, ffn_dim, load_balance_weight)
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        total_aux_loss = 0.0

        for moe_layer, layer_norm in zip(self.moe_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            moe_output, aux_loss = moe_layer(x)
            x = residual + moe_output
            total_aux_loss = total_aux_loss + aux_loss

        logits = self.output_layer(x)

        return logits, total_aux_loss
