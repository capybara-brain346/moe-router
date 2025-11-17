import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from moe import MoELayer


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape  # shape -> [batch, seq_len, hidden_dim]

        qkv = self.qkv_proj(x)  # shape -> [batch, seq_len, 3*hidden_dim]
        qkv = qkv.reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )  # shape -> [batch, seq_len, 3, num_heads, head_dim]
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # shape -> [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(
                mask == 0, float("-inf")
            )  # set to -inf if value in position == 0

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, self.hidden_dim)
        )
        out = self.out_proj(out)

        return out


class DenseFFN(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.fc2(self.dropout(F.gelu(self.fc1(x))))
        return x, torch.tensor(0.0, device=x.device)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        load_balance_weight: float = 0.01,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        gating_temperature: float = 1.0,
    ):
        super().__init__()
        self.use_moe = use_moe

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)

        self.ln2 = nn.LayerNorm(hidden_dim)
        if use_moe:
            self.ffn = MoELayer(
                hidden_dim, num_experts, ffn_dim, dropout, load_balance_weight, top_k,
                capacity_factor, gating_temperature
            )
        else:
            self.ffn = DenseFFN(hidden_dim, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.dropout(self.attn(self.ln1(x), mask))

        ffn_out, aux_loss = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x, aux_loss


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        moe_layers: Optional[List[int]] = None,
        num_experts: int = 8,
        load_balance_weight: float = 0.01,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        gating_temperature: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.label_smoothing = label_smoothing

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        moe_layers = moe_layers or []

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_moe=(i in moe_layers),
                    num_experts=num_experts,
                    load_balance_weight=load_balance_weight,
                    top_k=top_k,
                    capacity_factor=capacity_factor,
                    gating_temperature=gating_temperature,
                )
                for i in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.moe_layer_indices = moe_layers

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch_size, seq_len = input_ids.shape

        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        pos = pos.unsqueeze(0)

        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)

        total_aux_loss = 0.0

        for block in self.blocks:
            x, aux_loss = block(x, causal_mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                label_smoothing=self.label_smoothing
            )
            loss = loss + total_aux_loss

        return logits, loss, total_aux_loss

    def get_expert_usage(self) -> Dict[int, Dict[int, int]]:
        usage = {}
        for i, block in enumerate(self.blocks):
            if block.use_moe:
                usage[i] = block.ffn.get_expert_usage()
        return usage
    
    def get_expert_statistics(self) -> Dict[int, Dict]:
        stats = {}
        for i, block in enumerate(self.blocks):
            if block.use_moe:
                stats[i] = block.ffn.get_expert_statistics()
        return stats

    def reset_expert_counts(self):
        for block in self.blocks:
            if block.use_moe:
                block.ffn.reset_expert_counts()
    
    def set_gating_temperature(self, temperature: float):
        for block in self.blocks:
            if block.use_moe:
                block.ffn.set_gating_temperature(temperature)
