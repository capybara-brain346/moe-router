import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import Optional, Tuple, Dict, List
import time
import json
from collections import defaultdict
import numpy as np


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
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.hidden_dim)
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


class ExpertMLP(nn.Module):
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
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_dim, ffn_dim, dropout) for _ in range(num_experts)
        ])
        
        self.expert_counts = torch.zeros(num_experts)
        
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
                output_flat[expert_mask] = expert_output * expert_weights[expert_mask].unsqueeze(-1)
                
                if not self.training:
                    self.expert_counts[expert_idx] += expert_mask.sum().item()
        
        output = output_flat.view(batch_size, seq_len, hidden_dim)
        
        aux_loss = self._compute_load_balance_loss(router_probs, expert_indices_flat)
        
        return output, aux_loss
    
    def _compute_load_balance_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = router_probs.shape[0]
        
        mean_router_probs = router_probs.mean(dim=0)
        
        expert_counts = torch.bincount(
            expert_indices,
            minlength=self.num_experts
        ).float()
        expert_frequencies = expert_counts / num_tokens
        
        load_balance_loss = self.num_experts * torch.sum(
            mean_router_probs * expert_frequencies
        )
        
        return self.load_balance_weight * load_balance_loss
    
    def get_expert_usage(self) -> Dict[int, int]:
        return {i: int(self.expert_counts[i].item()) for i in range(self.num_experts)}
    
    def reset_expert_counts(self):
        self.expert_counts = torch.zeros(self.num_experts)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 8,
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.use_moe = use_moe
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        self.ln2 = nn.LayerNorm(hidden_dim)
        if use_moe:
            self.ffn = MoELayer(hidden_dim, num_experts, ffn_dim, dropout, load_balance_weight)
        else:
            self.ffn = DenseFFN(hidden_dim, ffn_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
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
        load_balance_weight: float = 0.01
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        moe_layers = moe_layers or []
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                num_heads,
                ffn_dim,
                dropout,
                use_moe=(i in moe_layers),
                num_experts=num_experts,
                load_balance_weight=load_balance_weight
            )
            for i in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        self.moe_layer_indices = moe_layers
        
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss + total_aux_loss
        
        return logits, loss, total_aux_loss
    
    def get_expert_usage(self) -> Dict[int, Dict[int, int]]:
        usage = {}
        for i, block in enumerate(self.blocks):
            if block.use_moe:
                usage[i] = block.ffn.get_expert_usage()
        return usage
    
    def reset_expert_counts(self):
        for block in self.blocks:
            if block.use_moe:
                block.ffn.reset_expert_counts()


class WikiTextDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.tokens) // self.seq_len
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = self.tokens[start_idx:end_idx]
        
        if len(chunk) < self.seq_len + 1:
            padding = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])
        
        return chunk[:-1], chunk[1:]


def load_wikitext_data(seq_len: int = 256, vocab_size: int = 10000):
    try:
        from torchtext.datasets import WikiText2
        from torchtext.data.utils import get_tokenizer
        from collections import Counter
        
        print("Loading WikiText-2 dataset...")
        train_iter, val_iter, test_iter = WikiText2()
        
        tokenizer = get_tokenizer('basic_english')
        
        counter = Counter()
        for item in train_iter:
            counter.update(tokenizer(item))
        
        vocab = ['<pad>', '<unk>'] + [word for word, _ in counter.most_common(vocab_size - 2)]
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        
        def tokenize_data(data_iter):
            tokens = []
            for item in data_iter:
                token_ids = [word_to_idx.get(word, 1) for word in tokenizer(item)]
                tokens.extend(token_ids)
            return torch.tensor(tokens, dtype=torch.long)
        
        train_iter, val_iter, test_iter = WikiText2()
        train_tokens = tokenize_data(train_iter)
        
        train_iter, val_iter, test_iter = WikiText2()
        next(val_iter)
        val_tokens = tokenize_data(val_iter)
        
        print(f"Train tokens: {len(train_tokens):,}")
        print(f"Val tokens: {len(val_tokens):,}")
        print(f"Vocabulary size: {len(vocab):,}")
        
        return train_tokens, val_tokens, len(vocab)
        
    except ImportError:
        print("torchtext not available, creating synthetic data...")
        train_tokens = torch.randint(0, vocab_size, (100000,))
        val_tokens = torch.randint(0, vocab_size, (10000,))
        return train_tokens, val_tokens, vocab_size


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_name: str = "model"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        
        self.train_losses = []
        self.val_losses = []
        self.train_ce_losses = []
        self.val_ce_losses = []
        self.train_aux_losses = []
        self.val_aux_losses = []
        self.tokens_per_sec = []
        self.steps = []
        self.current_step = 0
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_aux_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        for batch_idx, (input_ids, targets) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            _, loss, aux_loss = self.model(input_ids, targets)
            
            ce_loss = loss - aux_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            batch_tokens = input_ids.numel()
            total_tokens += batch_tokens
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_aux_loss += aux_loss.item()
            
            self.current_step += 1
        
        elapsed_time = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed_time
        
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'aux_loss': total_aux_loss / num_batches,
            'tokens_per_sec': tokens_per_sec
        }
    
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_aux_loss = 0.0
        
        with torch.no_grad():
            for input_ids, targets in self.val_loader:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                
                _, loss, aux_loss = self.model(input_ids, targets)
                
                ce_loss = loss - aux_loss
                
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_aux_loss += aux_loss.item()
        
        num_batches = len(self.val_loader)
        return {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'aux_loss': total_aux_loss / num_batches
        }
    
    def train(self, num_epochs: int, eval_every: int = 1):
        print(f"\nTraining {self.model_name}...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()
            
            self.train_losses.append(train_metrics['loss'])
            self.train_ce_losses.append(train_metrics['ce_loss'])
            self.train_aux_losses.append(train_metrics['aux_loss'])
            self.tokens_per_sec.append(train_metrics['tokens_per_sec'])
            self.steps.append(self.current_step)
            
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate()
                self.val_losses.append(val_metrics['loss'])
                self.val_ce_losses.append(val_metrics['ce_loss'])
                self.val_aux_losses.append(val_metrics['aux_loss'])
                
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.4f} | CE: {train_metrics['ce_loss']:.4f} | Aux: {train_metrics['aux_loss']:.6f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.4f} | CE: {val_metrics['ce_loss']:.4f} | Aux: {val_metrics['aux_loss']:.6f}")
                print(f"  Throughput: {train_metrics['tokens_per_sec']:.0f} tokens/sec")
        
        print("=" * 80)
    
    def get_metrics(self) -> Dict:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_ce_losses': self.train_ce_losses,
            'val_ce_losses': self.val_ce_losses,
            'train_aux_losses': self.train_aux_losses,
            'val_aux_losses': self.val_aux_losses,
            'tokens_per_sec': self.tokens_per_sec,
            'steps': self.steps
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    seq_len = 256
    batch_size = 16
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    ffn_dim = 1024
    num_epochs = 20
    learning_rate = 3e-4
    vocab_size = 10000
    
    print("\nLoading data...")
    train_tokens, val_tokens, actual_vocab_size = load_wikitext_data(seq_len, vocab_size)
    
    train_dataset = WikiTextDataset(train_tokens, seq_len)
    val_dataset = WikiTextDataset(val_tokens, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT: Dense vs MoE Transformer Comparison")
    print("=" * 80)
    
    print("\nDense Baseline Model")
    print("-" * 80)
    dense_model = GPTModel(
        vocab_size=actual_vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=seq_len,
        dropout=0.1,
        moe_layers=None
    ).to(device)
    
    dense_params = count_parameters(dense_model)
    print(f"Parameters: {dense_params:,}")
    
    dense_optimizer = optim.AdamW(dense_model.parameters(), lr=learning_rate)
    dense_trainer = Trainer(dense_model, train_loader, val_loader, dense_optimizer, device, "Dense")
    dense_trainer.train(num_epochs)
    
    dense_model_params_per_expert = (ffn_dim * hidden_dim + hidden_dim * ffn_dim)
    num_experts = 8
    
    moe_ffn_dim = int(ffn_dim / (num_experts * 0.5))
    
    print(f"\nMoE Model (replacing layers 2, 4 with MoE)")
    print("-" * 80)
    moe_model = GPTModel(
        vocab_size=actual_vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=moe_ffn_dim,
        max_seq_len=seq_len,
        dropout=0.1,
        moe_layers=[2, 4],
        num_experts=num_experts,
        load_balance_weight=0.01
    ).to(device)
    
    moe_params = count_parameters(moe_model)
    print(f"Parameters: {moe_params:,}")
    print(f"Parameter ratio (MoE/Dense): {moe_params/dense_params:.2f}x")
    
    moe_optimizer = optim.AdamW(moe_model.parameters(), lr=learning_rate)
    moe_trainer = Trainer(moe_model, train_loader, val_loader, moe_optimizer, device, "MoE")
    moe_trainer.train(num_epochs)
    
    print("\n" + "=" * 80)
    print("Collecting Expert Usage Statistics")
    print("=" * 80)
    
    moe_model.eval()
    moe_model.reset_expert_counts()
    
    with torch.no_grad():
        for input_ids, _ in val_loader:
            input_ids = input_ids.to(device)
            moe_model(input_ids)
    
    expert_usage = moe_model.get_expert_usage()
    
    print("\nExpert Usage per MoE Layer:")
    for layer_idx, usage in expert_usage.items():
        print(f"\nLayer {layer_idx}:")
        total_tokens = sum(usage.values())
        for expert_idx, count in sorted(usage.items()):
            percentage = 100 * count / total_tokens if total_tokens > 0 else 0
            print(f"  Expert {expert_idx}: {count:6d} tokens ({percentage:5.2f}%)")
    
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    results = {
        'dense': dense_trainer.get_metrics(),
        'moe': moe_trainer.get_metrics(),
        'expert_usage': {str(k): v for k, v in expert_usage.items()},
        'config': {
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'ffn_dim': ffn_dim,
            'moe_ffn_dim': moe_ffn_dim,
            'num_experts': num_experts,
            'dense_params': dense_params,
            'moe_params': moe_params,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'num_epochs': num_epochs
        }
    }
    
    with open('phase2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to phase2_results.json")
    
    torch.save(dense_model.state_dict(), 'dense_model.pt')
    torch.save(moe_model.state_dict(), 'moe_model_phase2.pt')
    print("Models saved to dense_model.pt and moe_model_phase2.pt")
    
    return results


if __name__ == "__main__":
    results = main()

