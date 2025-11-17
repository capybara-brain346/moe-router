# MoE (Mixture-of-Experts) Module Documentation

## Overview

The `moe.py` module implements a Mixture-of-Experts layer with token-level routing, load balancing, and usage tracking. It enables sparse conditional computation where each token is processed by a subset of expert networks.

## Core Concepts

### Mixture-of-Experts

MoE is a sparse neural network architecture where:
- Multiple expert networks exist in parallel
- A gating network routes each input to a subset of experts
- Only activated experts process their assigned inputs
- Enables model capacity scaling without proportional computation increase

### Token-Level Routing

Each token independently chooses which experts to use based on:
- Learned routing logits from gating network
- Top-k selection mechanism
- Temperature-controlled softmax for exploration/exploitation

### Load Balancing

Auxiliary loss encourages balanced expert utilization to:
- Prevent expert collapse (all tokens to one expert)
- Ensure all experts learn meaningful specializations
- Improve model efficiency and quality

## Key Components

### Expert Class

Individual feed-forward expert network.

#### Parameters

- `hidden_dim` (int): Input/output dimension
- `ffn_dim` (int): Intermediate hidden dimension
- `dropout` (float): Dropout probability (default: 0.1)

#### Architecture

```
x → Linear(hidden_dim → ffn_dim) → GELU → Dropout → Linear(ffn_dim → hidden_dim)
```

**Design**: Identical to standard transformer FFN, allowing experts to specialize during training.

#### forward(x) -> Tensor

**Input:**
- `x`: Token representations [num_tokens, hidden_dim]

**Output:**
- Transformed representations [num_tokens, hidden_dim]

**Note:** Each expert is a simple two-layer MLP with GELU activation.

### MoELayer Class

Complete Mixture-of-Experts layer with routing and load balancing.

#### Parameters

**Architecture:**
- `hidden_dim` (int): Input/output dimension
- `num_experts` (int): Number of parallel experts
- `ffn_dim` (int): Expert intermediate dimension
- `dropout` (float): Dropout probability (default: 0.1)

**Routing:**
- `top_k` (int): Number of experts per token (default: 1)
- `capacity_factor` (float): Expert capacity multiplier (default: 1.25, currently unused in implementation)
- `gating_temperature` (float): Softmax temperature for routing (default: 1.0)

**Training:**
- `load_balance_weight` (float): Weight for auxiliary load balancing loss (default: 0.01)

#### Architecture Components

1. **Router**: Linear layer projecting tokens to expert logits
2. **Experts**: ModuleList of expert networks
3. **Statistics**: Tracking counters for expert usage analysis

#### Attributes

**Routing:**
- `router`: Linear(hidden_dim → num_experts), no bias
- `experts`: ModuleList of Expert instances

**Statistics Tracking:**
- `expert_counts`: Tensor tracking token counts per expert
- `expert_router_probs`: Accumulated router probabilities per expert
- `num_tokens_processed`: Total tokens processed (for normalization)

**Note:** Statistics only accumulate during evaluation mode (not training).

#### forward(x) -> Tuple[Tensor, Tensor]

Main forward pass implementing expert routing and execution.

**Input:**
- `x`: Token representations [batch_size, seq_len, hidden_dim]

**Output:**
- `output`: Expert-processed representations [batch_size, seq_len, hidden_dim]
- `aux_loss`: Load balancing auxiliary loss (scalar)

**Algorithm:**

```python
# 1. Flatten input
x_flat = x.view(-1, hidden_dim)  # [batch*seq_len, hidden_dim]

# 2. Compute routing
router_logits = router(x_flat) / temperature
router_probs = softmax(router_logits)  # [num_tokens, num_experts]

# 3. Select top-k experts
expert_weights, expert_indices = topk(router_probs, k)
expert_weights = normalize(expert_weights)  # Normalize to sum to 1

# 4. Route and process tokens
output = zeros_like(x_flat)
for expert_idx in range(num_experts):
    # Find tokens assigned to this expert
    expert_mask = (expert_indices == expert_idx).any(dim=-1)
    
    if expert_mask.any():
        # Process tokens with expert
        expert_tokens = x_flat[expert_mask]
        expert_output = experts[expert_idx](expert_tokens)
        
        # Weight outputs by routing scores
        token_weights = sum of expert_weights for this expert
        output[expert_mask] += expert_output * token_weights
        
        # Track usage (eval mode only)
        if not training:
            expert_counts[expert_idx] += num_tokens

# 5. Reshape output
output = output.view(batch_size, seq_len, hidden_dim)

# 6. Compute load balancing loss
aux_loss = load_balance_loss(router_probs, expert_indices)

return output, aux_loss
```

**Key Details:**

1. **Temperature Scaling**: `logits / temperature` controls routing sharpness
   - High temp (>1): Softer routing, more exploration
   - Low temp (<1): Sharper routing, more exploitation

2. **Top-k Selection**:
   - If k=1: Use argmax for efficiency (deterministic routing)
   - If k>1: Use topk and normalize weights

3. **Weight Accumulation**: Tokens may receive contributions from multiple experts (when top_k > 1)

4. **Statistics**: Only tracked in eval mode to avoid training overhead

#### _compute_load_balance_loss(router_probs, expert_indices) -> Tensor

Computes auxiliary loss to encourage balanced expert usage.

**Input:**
- `router_probs`: Router softmax probabilities [num_tokens, num_experts]
- `expert_indices`: Selected expert indices [num_tokens, top_k]

**Output:**
- Load balancing loss (scalar)

**Algorithm:**

```python
# 1. Compute fraction of probability mass to each expert
mean_router_probs = router_probs.mean(dim=0)  # [num_experts]

# 2. Compute fraction of tokens assigned to each expert
expert_counts = bincount(expert_indices.flatten())
expert_frequencies = expert_counts / (num_tokens * top_k)  # [num_experts]

# 3. Compute auxiliary loss
load_balance_loss = num_experts * sum(mean_router_probs * expert_frequencies)

return load_balance_weight * load_balance_loss
```

**Mathematical Formulation:**

```
L_aux = α * E * ∑_i f_i * P_i

where:
  α = load_balance_weight
  E = num_experts
  f_i = fraction of tokens assigned to expert i
  P_i = mean router probability for expert i
```

**Intuition:**
- Minimizing `f_i * P_i` encourages experts with high probability to receive fewer tokens
- Factor of `num_experts` normalizes for different expert counts
- Penalizes correlation between routing probability and actual assignments

**Target:** Ideally, both `f_i` and `P_i` should be `1/num_experts` for all i.

#### get_expert_usage() -> Dict[int, int]

Returns raw token counts per expert.

**Returns:**
```python
{
    0: 1500,
    1: 1450,
    2: 1600,
    ...
}
```

**Use Case:** Basic usage statistics for debugging.

#### get_expert_statistics() -> Dict

Returns comprehensive expert usage statistics.

**Returns:**
```python
{
    'usage': {expert_idx: count},
    'percentages': {expert_idx: percentage},
    'entropy': float,
    'min_usage_pct': float,
    'max_usage_pct': float
}
```

**Metrics:**

1. **Usage**: Raw token counts per expert
2. **Percentages**: Usage as percentage of total tokens
3. **Entropy**: Shannon entropy of usage distribution
   ```
   H = -∑_i p_i * log(p_i)
   ```
   - Maximum: log(num_experts) (uniform distribution)
   - Minimum: 0 (all tokens to one expert)
   
4. **Min/Max Usage**: Extremes of usage distribution

**Entropy Interpretation:**
- High entropy (close to log(num_experts)): Well-balanced experts
- Low entropy: Unbalanced, some experts dominate

**Use Case:** Evaluating load balancing effectiveness and expert specialization.

#### reset_expert_counts()

Resets all usage tracking counters to zero.

**Use Case:** Clear statistics before evaluation to get clean per-epoch measurements.

#### set_gating_temperature(temperature: float)

Updates the routing temperature.

**Parameters:**
- `temperature` (float): New temperature value

**Effect:**
- Modifies routing distribution sharpness
- Typically annealed during training: high→low

**Use Case:** Temperature annealing for better convergence.

## Design Patterns

### Statistics Tracking

Statistics are only accumulated during evaluation (not training):
```python
if not self.training:
    self.expert_counts[expert_idx] += count
```

**Rationale:**
- Training: Focus on performance, avoid overhead
- Evaluation: Analyze expert behavior without gradient computation

### Routing Strategies

#### Top-1 Routing

```python
expert_weights, expert_indices = torch.max(router_probs, dim=-1)
```

**Characteristics:**
- Each token uses exactly one expert
- Most efficient (minimum computation)
- Experts develop strong specializations

#### Top-k Routing (k > 1)

```python
expert_weights, expert_indices = torch.topk(router_probs, k, dim=-1)
expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
```

**Characteristics:**
- Each token uses k experts (weighted combination)
- More computation but potentially better quality
- Softer expert boundaries
- Weight normalization ensures proper scaling

### Load Balancing

The auxiliary loss balances two objectives:
1. **Primary Task**: Language modeling (cross-entropy)
2. **Auxiliary Task**: Expert load balancing

**Trade-off:**
- High `load_balance_weight`: Strong balancing, may hurt quality
- Low `load_balance_weight`: Weak balancing, risk of expert collapse

**Typical Values:** 0.001 - 0.1

## Advanced Concepts

### Capacity Factor

Parameter `capacity_factor` is defined but not actively used in current implementation.

**Intended Purpose:**
- Limit tokens per expert: `capacity = (num_tokens / num_experts) * capacity_factor`
- Drop tokens exceeding capacity
- Prevents expert overload in distributed settings

**Current Implementation:** No capacity enforcement (all assigned tokens processed)

### Temperature Annealing

**Schedule:**
```python
# Linear annealing
temp = start_temp - (start_temp - end_temp) * (epoch / anneal_epochs)

# Example: 1.0 → 0.5 over 5 epochs
Epoch 0: temp = 1.0
Epoch 1: temp = 0.9
Epoch 2: temp = 0.8
...
Epoch 5+: temp = 0.5
```

**Benefits:**
- Early training: High temp enables exploration of expert roles
- Late training: Low temp sharpens routing for specialized experts
- Improves convergence and final performance

### Expert Specialization

Experts can specialize based on:
- Token content (semantic clustering)
- Position in sequence (positional patterns)
- Syntactic roles (verbs, nouns, etc.)
- Frequency (common vs. rare tokens)

**Analysis:** Use `get_expert_statistics()` and examine routing patterns to understand specialization.

## Performance Considerations

### Memory

**Parameters:**
```
Router: hidden_dim * num_experts
Experts: num_experts * (2 * hidden_dim * ffn_dim)
Total: O(num_experts * hidden_dim * ffn_dim)
```

**Scaling:** Linear in number of experts, so can dramatically increase capacity.

### Computation

**Per Forward Pass:**
```
Router: O(num_tokens * hidden_dim * num_experts)
Expert Processing: O(num_tokens * hidden_dim * ffn_dim * top_k)
Total: Dominated by expert processing
```

**Efficiency:**
- Dense FFN: All tokens use same network
- MoE (top_k=1): Each token uses 1/num_experts of capacity
- Computation per token: ~1/num_experts of full MoE activation

**Example:**
- 8 experts, top_k=1: Each token uses ~12.5% of total expert capacity
- Similar computation to dense, but 8x parameter count

### Optimization Opportunities

**Current Implementation:** Sequential expert processing
```python
for expert_idx in range(num_experts):
    # Process tokens for this expert
```

**Potential Improvements:**
1. **Batched Expert Processing**: Process all experts in parallel (requires padding)
2. **Distributed Experts**: Place experts on different devices/machines
3. **Capacity Enforcement**: Drop tokens to prevent stragglers
4. **Expert Pruning**: Remove unused experts during inference

## Usage Examples

### Basic MoE Layer

```python
from moe import MoELayer

moe = MoELayer(
    hidden_dim=256,
    num_experts=8,
    ffn_dim=1024,
    dropout=0.1,
    load_balance_weight=0.01,
    top_k=1
)

x = torch.randn(4, 128, 256)  # [batch, seq_len, hidden]
output, aux_loss = moe(x)
```

### Expert Usage Analysis

```python
model.eval()
model.reset_expert_counts()

with torch.no_grad():
    for batch in val_loader:
        model(batch)

stats = model.get_expert_statistics()
for layer_idx, layer_stats in stats.items():
    print(f"Layer {layer_idx}:")
    print(f"  Entropy: {layer_stats['entropy']:.3f}")
    print(f"  Min usage: {layer_stats['min_usage_pct']:.1f}%")
    print(f"  Max usage: {layer_stats['max_usage_pct']:.1f}%")
    
    for expert_idx, pct in layer_stats['percentages'].items():
        print(f"  Expert {expert_idx}: {pct:.1f}%")
```

### Temperature Annealing

```python
moe = MoELayer(hidden_dim=256, num_experts=8, ffn_dim=1024)

for epoch in range(10):
    temp = 1.0 - 0.5 * min(1.0, epoch / 5)
    moe.set_gating_temperature(temp)
    
    train_epoch()
```

### Top-k Routing

```python
moe_top1 = MoELayer(hidden_dim=256, num_experts=8, ffn_dim=1024, top_k=1)

moe_top2 = MoELayer(hidden_dim=256, num_experts=8, ffn_dim=1024, top_k=2)
```

**Trade-off:**
- top_k=1: Faster, stronger specialization
- top_k=2: Higher quality, more computation (2x per token)

## Debugging Tips

### Expert Collapse

**Symptom:** One expert handles most tokens

**Diagnosis:**
```python
stats = moe.get_expert_statistics()
if stats['max_usage_pct'] > 80:
    print("Expert collapse detected!")
```

**Solutions:**
- Increase `load_balance_weight`
- Use higher initial temperature
- Check router initialization

### Unused Experts

**Symptom:** Some experts receive very few tokens

**Diagnosis:**
```python
if stats['min_usage_pct'] < 1:
    print(f"Expert(s) underused: {stats['min_usage_pct']:.2f}%")
```

**Solutions:**
- Increase temperature
- Reduce `load_balance_weight` (may seem counterintuitive)
- Ensure sufficient training data diversity

### High Auxiliary Loss

**Symptom:** Large aux_loss dominating total loss

**Diagnosis:**
```python
print(f"CE Loss: {ce_loss.item():.4f}")
print(f"Aux Loss: {aux_loss.item():.4f}")
```

**Solutions:**
- Reduce `load_balance_weight`
- Check that experts are learning (not collapsed)

## Dependencies

- `torch`: Core PyTorch functionality
- `torch.nn.functional`: Activation functions and operations

