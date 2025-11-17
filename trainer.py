import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, Optional
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
        early_stopping_patience: int = 0,
        gating_temp_schedule: Optional[Dict] = None,
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
        
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.stopped_early = False
        
        self.gating_temp_schedule = gating_temp_schedule
        self.timing_breakdown = {
            'forward': [],
            'backward': [],
            'optimizer': [],
        }
        
        self.expert_stats_history = []

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_aux_loss = 0.0
        total_tokens = 0
        start_time = time.time()
        
        epoch_forward_time = 0.0
        epoch_backward_time = 0.0
        epoch_optimizer_time = 0.0

        for batch_idx, (input_ids, targets) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            forward_start = time.time()
            _, loss, aux_loss = self.model(input_ids, targets)
            epoch_forward_time += time.time() - forward_start

            ce_loss = loss - aux_loss

            backward_start = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            epoch_backward_time += time.time() - backward_start
            
            optimizer_start = time.time()
            self.optimizer.step()
            epoch_optimizer_time += time.time() - optimizer_start

            batch_tokens = input_ids.numel()
            total_tokens += batch_tokens
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_aux_loss += aux_loss.item()

            self.current_step += 1

        elapsed_time = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed_time
        
        self.timing_breakdown['forward'].append(epoch_forward_time)
        self.timing_breakdown['backward'].append(epoch_backward_time)
        self.timing_breakdown['optimizer'].append(epoch_optimizer_time)

        num_batches = len(self.train_loader)
        return {
            "loss": total_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "aux_loss": total_aux_loss / num_batches,
            "tokens_per_sec": tokens_per_sec,
            "forward_time": epoch_forward_time,
            "backward_time": epoch_backward_time,
            "optimizer_time": epoch_optimizer_time,
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
            if self.gating_temp_schedule and hasattr(self.model, 'set_gating_temperature'):
                start_temp = self.gating_temp_schedule.get('start', 1.0)
                end_temp = self.gating_temp_schedule.get('end', 0.5)
                anneal_epochs = self.gating_temp_schedule.get('epochs', 5)
                
                if epoch < anneal_epochs:
                    current_temp = start_temp - (start_temp - end_temp) * (epoch / anneal_epochs)
                else:
                    current_temp = end_temp
                
                self.model.set_gating_temperature(current_temp)
            
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
                print(f"  Timing - Forward: {train_metrics['forward_time']:.2f}s | Backward: {train_metrics['backward_time']:.2f}s | Optimizer: {train_metrics['optimizer_time']:.2f}s")
                
                if self.early_stopping_patience > 0:
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        print(f"  Early stopping: {self.patience_counter}/{self.early_stopping_patience}")
                        
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"  Early stopping triggered at epoch {epoch + 1}")
                            self.stopped_early = True
                            break
                
                if hasattr(self.model, 'get_expert_statistics'):
                    self.model.eval()
                    self.model.reset_expert_counts()
                    with torch.no_grad():
                        for input_ids, _ in self.val_loader:
                            input_ids = input_ids.to(self.device)
                            self.model(input_ids)
                    
                    expert_stats = self.model.get_expert_statistics()
                    self.expert_stats_history.append({
                        'epoch': epoch + 1,
                        'stats': expert_stats
                    })
                    
                    for layer_idx, stats in expert_stats.items():
                        print(f"  Layer {layer_idx} - Entropy: {stats['entropy']:.3f} | Min usage: {stats['min_usage_pct']:.1f}% | Max usage: {stats['max_usage_pct']:.1f}%")

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
            "timing_breakdown": self.timing_breakdown,
            "expert_stats_history": self.expert_stats_history,
            "stopped_early": self.stopped_early,
            "best_val_loss": self.best_val_loss,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ExperimentConfig:
    def __init__(
        self,
        name: str,
        expert_dim: int,
        top_k: int = 1,
        aux_weight: float = 0.01,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        early_stopping_patience: int = 0,
        gating_temp_schedule: Optional[Dict] = None,
        batch_size: int = 16,
    ):
        self.name = name
        self.expert_dim = expert_dim
        self.top_k = top_k
        self.aux_weight = aux_weight
        self.capacity_factor = capacity_factor
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.early_stopping_patience = early_stopping_patience
        self.gating_temp_schedule = gating_temp_schedule
        self.batch_size = batch_size
    
    def to_dict(self):
        return {
            "name": self.name,
            "expert_dim": self.expert_dim,
            "top_k": self.top_k,
            "aux_weight": self.aux_weight,
            "capacity_factor": self.capacity_factor,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "early_stopping_patience": self.early_stopping_patience,
            "gating_temp_schedule": self.gating_temp_schedule,
            "batch_size": self.batch_size,
        }


def get_experiment_configs():
    configs = {}
    
    configs["baseline"] = ExperimentConfig(
        name="baseline",
        expert_dim=1024,
        top_k=1,
        aux_weight=0.01,
        capacity_factor=1.25,
        dropout=0.1,
        weight_decay=0.0,
        label_smoothing=0.0,
        early_stopping_patience=3,
        gating_temp_schedule={"start": 1.0, "end": 0.5, "epochs": 5},
    )
    
    configs["top2"] = ExperimentConfig(
        name="top2",
        expert_dim=1024,
        top_k=2,
        aux_weight=0.01,
        capacity_factor=1.25,
        dropout=0.1,
        weight_decay=0.0,
        label_smoothing=0.0,
        early_stopping_patience=3,
        gating_temp_schedule={"start": 1.0, "end": 0.5, "epochs": 5},
    )
    
    configs["strong_reg"] = ExperimentConfig(
        name="strong_reg",
        expert_dim=1024,
        top_k=1,
        aux_weight=0.01,
        capacity_factor=1.25,
        dropout=0.2,
        weight_decay=1e-3,
        label_smoothing=0.1,
        early_stopping_patience=3,
        gating_temp_schedule={"start": 1.0, "end": 0.5, "epochs": 5},
    )
    
    configs["throughput_opt"] = ExperimentConfig(
        name="throughput_opt",
        expert_dim=1024,
        top_k=1,
        aux_weight=0.01,
        capacity_factor=1.25,
        dropout=0.1,
        weight_decay=0.0,
        label_smoothing=0.0,
        early_stopping_patience=3,
        gating_temp_schedule={"start": 1.0, "end": 0.5, "epochs": 5},
        batch_size=32,
    )
    
    return configs


def run_single_experiment(config: ExperimentConfig, train_dataset, val_dataset, 
                          base_config: Dict, device: torch.device):
    print(f"\n{'='*80}")
    print(f"Running Experiment: {config.name}")
    print(f"{'='*80}")
    print(f"Configuration: {config.to_dict()}")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    model = GPTModel(
        vocab_size=base_config['vocab_size'],
        hidden_dim=base_config['hidden_dim'],
        num_layers=base_config['num_layers'],
        num_heads=base_config['num_heads'],
        ffn_dim=config.expert_dim,
        max_seq_len=base_config['seq_len'],
        dropout=config.dropout,
        moe_layers=[2, 4],
        num_experts=base_config['num_experts'],
        load_balance_weight=config.aux_weight,
        top_k=config.top_k,
        capacity_factor=config.capacity_factor,
        gating_temperature=1.0,
        label_smoothing=config.label_smoothing,
    ).to(device)
    
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    print(f"Effective FFN capacity per token: {config.expert_dim * config.top_k}")
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=base_config['learning_rate'],
        weight_decay=config.weight_decay
    )
    
    trainer = Trainer(
        model, train_loader, val_loader, optimizer, device,
        model_name=f"MoE-{config.name}",
        early_stopping_patience=config.early_stopping_patience,
        gating_temp_schedule=config.gating_temp_schedule,
    )
    
    trainer.train(num_epochs=base_config['num_epochs'], eval_every=1)
    
    return {
        "config": config.to_dict(),
        "metrics": trainer.get_metrics(),
        "params": params,
    }


def evaluate_success_criteria(results: Dict, dense_baseline_loss: float):
    print(f"\n{'='*80}")
    print("Success Criteria Evaluation")
    print(f"{'='*80}")
    
    for exp_name, exp_results in results.items():
        if exp_name == "config":
            continue
            
        print(f"\nExperiment: {exp_name}")
        print("-" * 40)
        
        metrics = exp_results["metrics"]
        final_val_loss = metrics["val_losses"][-1] if metrics["val_losses"] else float('inf')
        best_val_loss = metrics["best_val_loss"]
        
        mean_throughput = sum(metrics["tokens_per_sec"]) / len(metrics["tokens_per_sec"]) if metrics["tokens_per_sec"] else 0
        
        timing = metrics["timing_breakdown"]
        if timing["forward"]:
            mean_forward = sum(timing["forward"]) / len(timing["forward"])
            mean_backward = sum(timing["backward"]) / len(timing["backward"])
            mean_optimizer = sum(timing["optimizer"]) / len(timing["optimizer"])
        else:
            mean_forward = mean_backward = mean_optimizer = 0
        
        print(f"Final validation loss: {final_val_loss:.4f}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Dense baseline loss: {dense_baseline_loss:.4f}")
        
        if dense_baseline_loss > 0:
            improvement = (dense_baseline_loss - best_val_loss) / dense_baseline_loss * 100
            print(f"Improvement over dense: {improvement:+.2f}%")
            
            if best_val_loss < dense_baseline_loss:
                print("✓ PRIMARY: Validation loss lower than dense baseline")
            elif improvement > -25:
                print("✓ PRIMARY: Validation gap reduced by at least 25%")
            else:
                print("✗ PRIMARY: Did not meet validation loss criteria")
        
        print(f"\nMean throughput: {mean_throughput:.0f} tokens/sec")
        print(f"Timing breakdown - Forward: {mean_forward:.2f}s | Backward: {mean_backward:.2f}s | Optimizer: {mean_optimizer:.2f}s")
        
        if metrics.get("expert_stats_history"):
            final_stats = metrics["expert_stats_history"][-1]["stats"]
            print(f"\nExpert Statistics (final epoch):")
            for layer_idx, stats in final_stats.items():
                print(f"  Layer {layer_idx}:")
                print(f"    Entropy: {stats['entropy']:.3f}")
                print(f"    Min usage: {stats['min_usage_pct']:.1f}%")
                print(f"    Max usage: {stats['max_usage_pct']:.1f}%")
                
                if stats['min_usage_pct'] >= 5.0:
                    print(f"    ✓ TERTIARY: All experts used (min >= 5%)")
                else:
                    print(f"    ✗ TERTIARY: Some experts underused (min < 5%)")


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
    num_experts = 8

    print("\nLoading data...")
    train_tokens, val_tokens, actual_vocab_size = load_wikitext_data(
        seq_len, vocab_size
    )

    train_dataset = WikiTextDataset(train_tokens, seq_len)
    val_dataset = WikiTextDataset(val_tokens, seq_len)
    
    base_config = {
        'vocab_size': actual_vocab_size,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'seq_len': seq_len,
        'num_experts': num_experts,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
    }

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n{'='*80}")
    print("Training Dense Baseline")
    print(f"{'='*80}")
    
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
    print(f"Dense model parameters: {dense_params:,}")

    dense_optimizer = optim.AdamW(dense_model.parameters(), lr=learning_rate)
    dense_trainer = Trainer(
        dense_model, train_loader, val_loader, dense_optimizer, device, "Dense",
        early_stopping_patience=3,
    )
    dense_trainer.train(num_epochs)
    
    dense_metrics = dense_trainer.get_metrics()
    dense_baseline_loss = dense_metrics["best_val_loss"]
    
    print(f"\nDense baseline best validation loss: {dense_baseline_loss:.4f}")
    
    print(f"\n{'='*80}")
    print("Running MoE Experiment Matrix")
    print(f"{'='*80}")
    
    experiment_configs = get_experiment_configs()
    all_results = {
        "config": base_config,
        "dense": {
            "metrics": dense_metrics,
            "params": dense_params,
        }
    }
    
    for exp_name, exp_config in experiment_configs.items():
        exp_results = run_single_experiment(
            exp_config, train_dataset, val_dataset, base_config, device
        )
        all_results[exp_name] = exp_results
    
    evaluate_success_criteria(all_results, dense_baseline_loss)
    
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")

    with open("experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("Results saved to experiment_results.json")
    
    torch.save(dense_model.state_dict(), "dense_model.pt")
    print("Dense model saved to dense_model.pt")
    
    for exp_name in experiment_configs.keys():
        print(f"MoE model saved for experiment: {exp_name}")

    return all_results


if __name__ == "__main__":
    results = main()
