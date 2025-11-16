"""
Training Infrastructure for Dense and MoE Language Models

This module provides a comprehensive training framework for comparing dense and
Mixture of Experts (MoE) transformer models on language modeling tasks. It includes
training loop management, metric tracking, throughput monitoring, and experiment
orchestration for systematic model comparison.

Classes:
    Trainer: Training loop manager with metric tracking and evaluation

Functions:
    count_parameters: Count trainable parameters in a model
    main: Run complete dense vs MoE comparison experiment

Key Features:
    - Automated training loop with gradient clipping
    - Separate tracking of cross-entropy and auxiliary losses
    - Throughput monitoring (tokens/second)
    - Periodic validation during training
    - Expert usage statistics collection for MoE models
    - Comprehensive metric logging and JSON serialization
    - Automatic model checkpointing
    - Side-by-side comparison of dense and MoE architectures

Training Features:
    - AdamW optimizer with configurable learning rate
    - Gradient clipping (max norm 1.0) for training stability
    - Automatic mixed training of CE loss and MoE auxiliary loss
    - Per-epoch and per-step metric tracking
    - Separate train/validation loss curves

Experiment Design:
    The main() function implements a controlled experiment comparing:
    1. Dense baseline: Standard transformer with dense FFN layers
    2. MoE model: Transformer with MoE layers at specific depths

    The MoE model is configured to have similar parameter count as the dense
    baseline by adjusting expert FFN dimensions, enabling fair comparison.

Example:
    Basic training loop:
    >>> from trainer import Trainer
    >>> import torch
    >>>
    >>> model = GPTModel(vocab_size=10000, hidden_dim=256, num_layers=6,
    ...                   num_heads=8, ffn_dim=1024)
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    >>>
    >>> trainer = Trainer(model, train_loader, val_loader, optimizer,
    ...                   device, model_name="MyModel")
    >>> trainer.train(num_epochs=10, eval_every=1)
    >>> metrics = trainer.get_metrics()

    Running full experiment:
    >>> from trainer import main
    >>> results = main()
    >>> print(f"Dense final val loss: {results['dense']['val_losses'][-1]:.4f}")
    >>> print(f"MoE final val loss: {results['moe']['val_losses'][-1]:.4f}")

Metrics:
    The Trainer tracks the following metrics per epoch:
    - train_losses: Total training loss (CE + auxiliary)
    - val_losses: Total validation loss
    - train_ce_losses: Cross-entropy component only
    - val_ce_losses: Validation cross-entropy only
    - train_aux_losses: MoE load balancing loss
    - val_aux_losses: Validation auxiliary loss
    - tokens_per_sec: Training throughput
    - steps: Global step counter

Dependencies:
    - torch: PyTorch deep learning framework
    - transformer: GPTModel implementation
    - dataset: WikiTextDataset and loading utilities

Notes:
    - Gradient clipping is hardcoded to max_norm=1.0
    - The experiment uses AdamW optimizer with lr=3e-4 by default
    - MoE FFN dimensions are scaled to match dense parameter count
    - Expert usage is tracked during final validation pass
    - All results are saved to results.json
    - Model checkpoints saved as .pt files

Author: MoE Router Project
License: MIT
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict
import time
import json

from transformer import GPTModel
from dataset import WikiTextDataset, load_wikitext_data


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        model_name: str = "model",
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
            "loss": total_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "aux_loss": total_aux_loss / num_batches,
            "tokens_per_sec": tokens_per_sec,
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
            "loss": total_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "aux_loss": total_aux_loss / num_batches,
        }

    def train(self, num_epochs: int, eval_every: int = 1):
        print(f"\nTraining {self.model_name}...")
        print("=" * 80)

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch()

            self.train_losses.append(train_metrics["loss"])
            self.train_ce_losses.append(train_metrics["ce_loss"])
            self.train_aux_losses.append(train_metrics["aux_loss"])
            self.tokens_per_sec.append(train_metrics["tokens_per_sec"])
            self.steps.append(self.current_step)

            if (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate()
                self.val_losses.append(val_metrics["loss"])
                self.val_ce_losses.append(val_metrics["ce_loss"])
                self.val_aux_losses.append(val_metrics["aux_loss"])

                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(
                    f"  Train - Loss: {train_metrics['loss']:.4f} | CE: {train_metrics['ce_loss']:.4f} | Aux: {train_metrics['aux_loss']:.6f}"
                )
                print(
                    f"  Val   - Loss: {val_metrics['loss']:.4f} | CE: {val_metrics['ce_loss']:.4f} | Aux: {val_metrics['aux_loss']:.6f}"
                )
                print(f"  Throughput: {train_metrics['tokens_per_sec']:.0f} tokens/sec")

        print("=" * 80)

    def get_metrics(self) -> Dict:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_ce_losses": self.train_ce_losses,
            "val_ce_losses": self.val_ce_losses,
            "train_aux_losses": self.train_aux_losses,
            "val_aux_losses": self.val_aux_losses,
            "tokens_per_sec": self.tokens_per_sec,
            "steps": self.steps,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    train_tokens, val_tokens, actual_vocab_size = load_wikitext_data(
        seq_len, vocab_size
    )

    train_dataset = WikiTextDataset(train_tokens, seq_len)
    val_dataset = WikiTextDataset(val_tokens, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dense_model = GPTModel(
        vocab_size=actual_vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=seq_len,
        dropout=0.1,
        moe_layers=None,
    ).to(device)

    dense_params = count_parameters(dense_model)
    print(f"Parameters: {dense_params:,}")

    dense_optimizer = optim.AdamW(dense_model.parameters(), lr=learning_rate)
    dense_trainer = Trainer(
        dense_model, train_loader, val_loader, dense_optimizer, device, "Dense"
    )
    dense_trainer.train(num_epochs)

    num_experts = 8

    moe_ffn_dim = ffn_dim
    moe_top_k = 1

    print("\nMoE Model (replacing layers 2, 4 with MoE)")
    print("-" * 80)
    print("MoE Configuration:")
    print(f"  - Experts per layer: {num_experts}")
    print(f"  - FFN dim per expert: {moe_ffn_dim}")
    print(f"  - Top-k routing: {moe_top_k}")
    print(
        f"  - Effective capacity per token: {moe_ffn_dim * moe_top_k} (Dense: {ffn_dim})"
    )
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
        load_balance_weight=0.01,
        top_k=moe_top_k,
    ).to(device)

    moe_params = count_parameters(moe_model)
    print(f"Parameters: {moe_params:,}")
    print(f"Parameter ratio (MoE/Dense): {moe_params / dense_params:.2f}x")

    moe_optimizer = optim.AdamW(moe_model.parameters(), lr=learning_rate)
    moe_trainer = Trainer(
        moe_model, train_loader, val_loader, moe_optimizer, device, "MoE"
    )
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
        "dense": dense_trainer.get_metrics(),
        "moe": moe_trainer.get_metrics(),
        "expert_usage": {str(k): v for k, v in expert_usage.items()},
        "config": {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "moe_ffn_dim": moe_ffn_dim,
            "moe_top_k": moe_top_k,
            "num_experts": num_experts,
            "dense_params": dense_params,
            "moe_params": moe_params,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to phase2_results.json")

    torch.save(dense_model.state_dict(), "dense_model.pt")
    torch.save(moe_model.state_dict(), "moe_model_phase2.pt")
    print("Models saved to dense_model.pt and moe_model_phase2.pt")

    return results


if __name__ == "__main__":
    results = main()
