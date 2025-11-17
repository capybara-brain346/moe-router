# Trainer Module Documentation

## Overview

The `trainer.py` module provides training infrastructure for transformer models with Mixture-of-Experts (MoE) support. It implements training loops, evaluation, experiment configuration management, and comprehensive metric tracking.

## Key Components

### Trainer Class

The main training orchestrator that handles model training, validation, and metric collection.

#### Initialization Parameters

- `model` (nn.Module): The neural network model to train
- `train_loader` (DataLoader): DataLoader for training data
- `val_loader` (DataLoader): DataLoader for validation data
- `optimizer` (optim.Optimizer): PyTorch optimizer instance
- `device` (torch.device): Device for computation (CPU/GPU)
- `model_name` (str): Identifier for the model (default: "model")
- `early_stopping_patience` (int): Number of epochs without improvement before stopping (default: 0, disabled)
- `gating_temp_schedule` (Optional[Dict]): Temperature annealing schedule for MoE gating

#### Attributes

**Loss Tracking:**

- `train_losses`: Total training losses per epoch
- `val_losses`: Total validation losses per epoch
- `train_ce_losses`: Cross-entropy training losses per epoch
- `val_ce_losses`: Cross-entropy validation losses per epoch
- `train_aux_losses`: Auxiliary (load balancing) losses per epoch
- `val_aux_losses`: Auxiliary validation losses per epoch

**Performance Metrics:**

- `tokens_per_sec`: Training throughput per epoch
- `steps`: Total training steps per epoch
- `current_step`: Current global training step

**Timing Breakdown:**

- `timing_breakdown`: Dict with 'forward', 'backward', 'optimizer' timing lists

**Early Stopping:**

- `best_val_loss`: Best validation loss observed
- `patience_counter`: Epochs without improvement
- `stopped_early`: Whether early stopping was triggered

**Expert Statistics:**

- `expert_stats_history`: Historical expert usage statistics for MoE layers

#### Methods

##### `train_epoch() -> Dict[str, float]`

Executes one training epoch over the entire training dataset.

**Process:**

1. Iterates through training batches
2. Performs forward pass with loss computation
3. Executes backward pass with gradient clipping (max norm: 1.0)
4. Updates model parameters via optimizer
5. Tracks timing for each phase (forward, backward, optimizer)
6. Computes throughput metrics

**Returns:**
Dictionary containing:

- `loss`: Average total loss
- `ce_loss`: Average cross-entropy loss
- `aux_loss`: Average auxiliary loss
- `tokens_per_sec`: Training throughput
- `forward_time`: Total forward pass time
- `backward_time`: Total backward pass time
- `optimizer_time`: Total optimizer step time

##### `evaluate() -> Dict[str, float]`

Evaluates the model on validation data without gradient computation.

**Returns:**
Dictionary containing:

- `loss`: Average total validation loss
- `ce_loss`: Average cross-entropy validation loss
- `aux_loss`: Average auxiliary validation loss

##### `train(num_epochs: int, eval_every: int = 1)`

Main training loop that orchestrates the entire training process.

**Features:**

1. **Temperature Annealing**: Linearly anneals gating temperature if schedule provided
2. **Periodic Evaluation**: Validates model every `eval_every` epochs
3. **Early Stopping**: Monitors validation loss and stops if no improvement
4. **Expert Statistics**: Tracks and logs expert usage patterns for MoE models
5. **Comprehensive Logging**: Prints detailed metrics including loss breakdown, throughput, and timing

**Temperature Schedule:**
If `gating_temp_schedule` is provided:

```python
{
    'start': 1.0,    # Initial temperature
    'end': 0.5,      # Final temperature
    'epochs': 5      # Annealing period
}
```

Temperature is linearly annealed from `start` to `end` over `epochs`, then held constant.

**Expert Statistics Logging:**
For MoE models, logs per-layer:

- Entropy: Distribution uniformity (higher = more balanced)
- Min usage: Percentage of tokens routed to least-used expert
- Max usage: Percentage of tokens routed to most-used expert

##### `get_metrics() -> Dict`

Returns comprehensive training metrics for analysis and visualization.

**Returns:**
Dictionary containing all tracked metrics:

- Loss histories (train/val, CE/aux)
- Throughput measurements
- Timing breakdowns
- Expert statistics history
- Early stopping information

### ExperimentConfig Class

Configuration dataclass for experiment parameters.

#### Parameters

**Model Architecture:**

- `expert_dim` (int): Hidden dimension of expert FFN layers

**MoE Configuration:**

- `top_k` (int): Number of experts to route each token to (default: 1)
- `aux_weight` (float): Weight for load balancing loss (default: 0.01)
- `capacity_factor` (float): Expert capacity multiplier (default: 1.25)

**Regularization:**

- `dropout` (float): Dropout probability (default: 0.1)
- `weight_decay` (float): L2 regularization strength (default: 0.0)
- `label_smoothing` (float): Label smoothing factor (default: 0.0)

**Training:**

- `batch_size` (int): Training batch size (default: 16)
- `early_stopping_patience` (int): Early stopping patience (default: 0)
- `gating_temp_schedule` (Optional[Dict]): Temperature annealing configuration

**Identification:**

- `name` (str): Experiment identifier

#### Methods

##### `to_dict()`

Serializes configuration to dictionary for logging and saving.

### Utility Functions

#### `count_parameters(model: nn.Module) -> int`

Counts trainable parameters in a model.

**Returns:** Total number of trainable parameters

#### `get_experiment_configs() -> Dict[str, ExperimentConfig]`

Returns predefined experiment configurations for comparative studies.

**Configurations:**

1. **baseline**: Standard MoE configuration

   - top_k=1, standard regularization
   - Purpose: Primary MoE baseline

2. **top2**: Uses top-2 expert routing

   - top_k=2, double expert capacity
   - Purpose: Test multi-expert routing

3. **strong_reg**: Heavy regularization

   - Higher dropout (0.2), weight decay (1e-3), label smoothing (0.1)
   - Purpose: Test overfitting prevention

4. **throughput_opt**: Optimized for speed
   - Larger batch size (32)
   - Purpose: Maximize training throughput

#### `run_single_experiment(config, train_dataset, val_dataset, base_config, device) -> Dict`

Executes a complete experiment with given configuration.

**Process:**

1. Creates data loaders with specified batch size
2. Initializes GPT model with MoE layers at positions [2, 4]
3. Sets up AdamW optimizer with configuration-specific parameters
4. Creates Trainer instance with early stopping and temperature scheduling
5. Trains for specified number of epochs
6. Returns results including metrics and parameter count

**Returns:**
Dictionary containing:

- `config`: Experiment configuration
- `metrics`: Training and validation metrics
- `params`: Model parameter count

#### `evaluate_success_criteria(results: Dict, dense_baseline_loss: float)`

Evaluates experiment results against success criteria.

**Criteria Evaluated:**

1. **PRIMARY - Model Quality:**

   - Validation loss lower than dense baseline, OR
   - Validation gap reduced by at least 25%

2. **SECONDARY - Performance:**

   - Throughput metrics
   - Timing breakdown analysis

3. **TERTIARY - Expert Utilization:**
   - All experts used (min usage >= 5%)
   - Distribution entropy

**Output:**
Prints detailed evaluation for each experiment with pass/fail indicators.

#### `main() -> Dict`

Main entry point that orchestrates the complete experimental pipeline.

**Pipeline:**

1. **Setup:**

   - Device selection (CUDA if available)
   - Hyperparameter definition
   - Data loading

2. **Dense Baseline:**

   - Trains standard transformer without MoE
   - Establishes performance baseline
   - Early stopping with patience=3

3. **MoE Experiments:**

   - Runs all configured experiments
   - Compares against dense baseline

4. **Evaluation:**

   - Assesses success criteria for all experiments

5. **Serialization:**
   - Saves results to `experiment_results.json`
   - Saves dense model weights to `dense_model.pt`

**Configuration:**

```python
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
```

**Returns:** Complete results dictionary with all experiments

## Design Patterns

### Separation of Concerns

The module separates:

- Training logic (Trainer)
- Experiment configuration (ExperimentConfig)
- Orchestration (main, run_single_experiment)
- Evaluation (evaluate_success_criteria)

### Metric Collection

Comprehensive metric tracking enables:

- Performance debugging via timing breakdowns
- Model analysis via expert statistics
- Result reproducibility via complete configuration logging

### Early Stopping

Prevents overfitting and reduces training time:

- Monitors validation loss
- Configurable patience
- Tracks best model state

### Temperature Annealing

Improves MoE training stability:

- High initial temperature: exploration
- Low final temperature: exploitation
- Linear annealing schedule

## Usage Example

```python
from trainer import main, get_experiment_configs, run_single_experiment

results = main()

configs = get_experiment_configs()
for name, config in configs.items():
    print(f"{name}: {config.to_dict()}")

device = torch.device("cuda")
result = run_single_experiment(
    configs["baseline"],
    train_dataset,
    val_dataset,
    base_config,
    device
)
```

## Dependencies

- `torch`: Core PyTorch functionality
- `transformer`: GPTModel implementation
- `dataset`: WikiTextDataset and data loading

## Output Files

- `experiment_results.json`: Complete experimental results and configurations
- `dense_model.pt`: Dense baseline model weights
