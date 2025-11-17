# Transformer Module Documentation

## Overview

The `transformer.py` module implements a GPT-style decoder-only transformer architecture with optional Mixture-of-Experts (MoE) layers. It provides flexible building blocks for language modeling with sparse expert routing capabilities.

## Architecture

```
Input Tokens
    ↓
Token Embedding + Positional Embedding
    ↓
[TransformerBlock] × N layers
    ├─ Multi-Head Self-Attention (causal)
    └─ Feed-Forward Network (Dense or MoE)
    ↓
Layer Normalization
    ↓
Output Projection (vocab_size)
    ↓
Logits / Loss
```

## Key Components

### MultiHeadAttention Class

Implements scaled dot-product multi-head attention with causal masking.

#### Parameters

- `hidden_dim` (int): Model hidden dimension (must be divisible by num_heads)
- `num_heads` (int): Number of attention heads
- `dropout` (float): Dropout probability (default: 0.1)

#### Architecture

1. **Projection**: Single linear layer projects input to Q, K, V simultaneously
2. **Reshaping**: Splits into multiple heads with dimension `head_dim = hidden_dim // num_heads`
3. **Attention**: Computes scaled dot-product attention per head
4. **Masking**: Applies causal mask to prevent attending to future positions
5. **Combination**: Concatenates heads and projects back to hidden_dim

#### Mathematical Formulation

```
QKV = Linear(x)  # [batch, seq_len, 3*hidden_dim]
Q, K, V = split(QKV)  # Each [batch, num_heads, seq_len, head_dim]

scores = (Q @ K^T) / sqrt(head_dim)
scores = masked_fill(scores, causal_mask, -inf)
attention = softmax(scores) @ V
output = Linear(concat(attention))
```

#### forward(x, mask) -> Tensor

**Input:**
- `x`: Shape [batch_size, seq_len, hidden_dim]
- `mask`: Optional attention mask [1, 1, seq_len, seq_len] or None

**Output:**
- Tensor of shape [batch_size, seq_len, hidden_dim]

**Process:**
1. Project to Q, K, V
2. Reshape to [batch, num_heads, seq_len, head_dim]
3. Compute attention scores with scaling
4. Apply causal mask
5. Apply softmax and dropout
6. Multiply by values
7. Reshape and project output

### DenseFFN Class

Standard feed-forward network with GELU activation.

#### Parameters

- `hidden_dim` (int): Model hidden dimension
- `ffn_dim` (int): Feed-forward intermediate dimension
- `dropout` (float): Dropout probability (default: 0.1)

#### Architecture

```
x → Linear(hidden_dim → ffn_dim) → GELU → Dropout → Linear(ffn_dim → hidden_dim)
```

#### forward(x) -> Tuple[Tensor, Tensor]

**Input:**
- `x`: Shape [batch_size, seq_len, hidden_dim]

**Output:**
- `output`: Transformed tensor [batch_size, seq_len, hidden_dim]
- `aux_loss`: Zero tensor (for API compatibility with MoE)

**Note:** Returns zero auxiliary loss to maintain consistent interface with MoELayer.

### TransformerBlock Class

Single transformer layer with attention and feed-forward sublayers.

#### Parameters

**Architecture:**
- `hidden_dim` (int): Model hidden dimension
- `num_heads` (int): Number of attention heads
- `ffn_dim` (int): Feed-forward intermediate dimension
- `dropout` (float): Dropout probability (default: 0.1)

**MoE Configuration:**
- `use_moe` (bool): Whether to use MoE instead of dense FFN (default: False)
- `num_experts` (int): Number of experts in MoE layer (default: 8)
- `load_balance_weight` (float): Load balancing loss weight (default: 0.01)
- `top_k` (int): Number of experts per token (default: 1)
- `capacity_factor` (float): Expert capacity multiplier (default: 1.25)
- `gating_temperature` (float): Temperature for router softmax (default: 1.0)

#### Architecture Components

1. **Layer Normalization 1**: Pre-norm before attention
2. **Multi-Head Attention**: Causal self-attention
3. **Residual Connection 1**: Add attention output to input
4. **Layer Normalization 2**: Pre-norm before FFN
5. **Feed-Forward Network**: Dense or MoE
6. **Residual Connection 2**: Add FFN output to previous output

#### forward(x, mask) -> Tuple[Tensor, Tensor]

**Input:**
- `x`: Shape [batch_size, seq_len, hidden_dim]
- `mask`: Optional attention mask

**Output:**
- `output`: Transformed tensor [batch_size, seq_len, hidden_dim]
- `aux_loss`: Auxiliary loss (zero for dense, load balancing for MoE)

**Process:**
```python
x = x + dropout(attention(layernorm(x), mask))
ffn_out, aux = ffn(layernorm(x))
x = x + dropout(ffn_out)
return x, aux
```

### GPTModel Class

Complete GPT-style language model with optional MoE layers.

#### Parameters

**Model Architecture:**
- `vocab_size` (int): Vocabulary size
- `hidden_dim` (int): Model hidden dimension
- `num_layers` (int): Number of transformer layers
- `num_heads` (int): Number of attention heads per layer
- `ffn_dim` (int): Feed-forward intermediate dimension
- `max_seq_len` (int): Maximum sequence length (default: 512)
- `dropout` (float): Dropout probability (default: 0.1)

**MoE Configuration:**
- `moe_layers` (Optional[List[int]]): Indices of layers to use MoE (None = all dense)
- `num_experts` (int): Number of experts per MoE layer (default: 8)
- `load_balance_weight` (float): Load balancing loss weight (default: 0.01)
- `top_k` (int): Number of experts per token (default: 1)
- `capacity_factor` (float): Expert capacity multiplier (default: 1.25)
- `gating_temperature` (float): Temperature for router softmax (default: 1.0)

**Training:**
- `label_smoothing` (float): Label smoothing factor for cross-entropy (default: 0.0)

#### Architecture

```python
# Embedding
x = token_embedding(input_ids) + position_embedding(positions)
x = dropout(x)

# Transformer blocks
for block in blocks:
    x, aux_loss = block(x, causal_mask)
    total_aux_loss += aux_loss

# Output
x = layer_norm(x)
logits = linear(x)  # Project to vocabulary
```

#### forward(input_ids, targets) -> Tuple[Tensor, Optional[Tensor], Tensor]

**Input:**
- `input_ids`: Token indices [batch_size, seq_len]
- `targets`: Target token indices [batch_size, seq_len] or None

**Output:**
- `logits`: Predicted logits [batch_size, seq_len, vocab_size]
- `loss`: Cross-entropy + auxiliary loss (if targets provided)
- `aux_loss`: Total auxiliary loss from all MoE layers

**Process:**
1. Embed tokens and add positional encodings
2. Generate causal attention mask (lower triangular)
3. Pass through all transformer blocks, accumulating auxiliary losses
4. Apply final layer normalization
5. Project to vocabulary size
6. Compute loss if targets provided (CE + auxiliary)

#### get_expert_usage() -> Dict[int, Dict[int, int]]

Returns raw expert usage counts for each MoE layer.

**Returns:**
```python
{
    layer_idx: {
        expert_idx: token_count
    }
}
```

**Use Case:** Analyzing which experts are being utilized during inference.

#### get_expert_statistics() -> Dict[int, Dict]

Returns detailed expert statistics for each MoE layer.

**Returns:**
```python
{
    layer_idx: {
        'usage': {expert_idx: count},
        'percentages': {expert_idx: percentage},
        'entropy': float,
        'min_usage_pct': float,
        'max_usage_pct': float
    }
}
```

**Metrics:**
- `entropy`: Shannon entropy of expert distribution (higher = more balanced)
- `min_usage_pct`: Minimum expert usage percentage
- `max_usage_pct`: Maximum expert usage percentage

**Use Case:** Evaluating load balancing and expert utilization.

#### reset_expert_counts()

Resets all expert usage counters to zero across all MoE layers.

**Use Case:** Clear statistics before evaluation phase.

#### set_gating_temperature(temperature: float)

Updates the gating temperature for all MoE layers.

**Parameters:**
- `temperature` (float): New temperature value

**Effect:**
- Higher temperature (>1.0): Softer routing, more exploration
- Lower temperature (<1.0): Sharper routing, more exploitation

**Use Case:** Temperature annealing during training for better convergence.

## Design Decisions

### Pre-Layer Normalization

Uses pre-norm architecture (normalize before sublayer) instead of post-norm:
- **Advantage**: More stable training, especially for deep models
- **Pattern**: `x = x + sublayer(norm(x))`

### Causal Masking

Implements autoregressive generation constraint:
- Tokens can only attend to previous positions
- Mask shape: [1, 1, seq_len, seq_len] broadcasts across batch and heads
- Lower triangular matrix with 1s

### Residual Connections

All sublayers use residual connections:
- Enables gradient flow in deep networks
- Allows learning identity mappings

### Unified FFN Interface

Both DenseFFN and MoELayer return `(output, aux_loss)`:
- **Benefit**: Seamless switching between dense and sparse layers
- **Pattern**: Duck typing for drop-in replacement

### Flexible MoE Placement

`moe_layers` parameter allows arbitrary MoE layer placement:
```python
moe_layers = [2, 4]       # Use MoE at layers 2 and 4
moe_layers = None         # All dense
moe_layers = range(6)     # All MoE
```

**Common Pattern**: Use MoE in middle layers where representations are most abstract.

## Loss Computation

### Cross-Entropy Loss

```python
ce_loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    label_smoothing=label_smoothing
)
```

### Total Loss

```python
total_loss = ce_loss + sum(aux_losses_from_moe_layers)
```

**Components:**
1. **CE Loss**: Language modeling objective
2. **Auxiliary Loss**: Load balancing penalty from MoE layers

## Usage Examples

### Dense Transformer

```python
model = GPTModel(
    vocab_size=10000,
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    ffn_dim=1024,
    max_seq_len=512,
    dropout=0.1,
    moe_layers=None
)

logits, loss, aux_loss = model(input_ids, targets)
```

### Sparse MoE Transformer

```python
model = GPTModel(
    vocab_size=10000,
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    ffn_dim=1024,
    max_seq_len=512,
    dropout=0.1,
    moe_layers=[2, 4],
    num_experts=8,
    top_k=2,
    load_balance_weight=0.01
)

logits, loss, aux_loss = model(input_ids, targets)

expert_stats = model.get_expert_statistics()
for layer_idx, stats in expert_stats.items():
    print(f"Layer {layer_idx} entropy: {stats['entropy']:.3f}")
```

### Temperature Annealing

```python
for epoch in range(num_epochs):
    temp = 1.0 - 0.5 * (epoch / num_epochs)
    model.set_gating_temperature(temp)
    train_epoch()
```

## Performance Considerations

### Memory

- Dense FFN: O(hidden_dim * ffn_dim) parameters per layer
- MoE Layer: O(num_experts * hidden_dim * ffn_dim) parameters per layer
- Trade-off: More parameters but conditional computation

### Computation

- Dense FFN: All tokens use same network
- MoE: Each token uses top_k experts
- Effective computation: ~(top_k / num_experts) of full MoE

### Scaling

For large models:
- Use MoE in subset of layers (e.g., alternating or middle layers)
- Increase num_experts while keeping top_k small
- Balance: more capacity without proportional computation increase

## Dependencies

- `torch`: Core PyTorch functionality
- `moe`: MoELayer implementation

