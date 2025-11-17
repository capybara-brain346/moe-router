# Dataset Module Documentation

## Overview

The `dataset.py` module provides data loading and tokenization utilities for language modeling tasks. It implements a PyTorch Dataset wrapper for sequential text data and a robust data loading pipeline for WikiText-2 with automatic fallback to synthetic data.

## Key Components

### WikiTextDataset Class

PyTorch Dataset for sequential language modeling data.

#### Purpose

Provides sliding window access to tokenized text for autoregressive language modeling:
- Input: Token sequence of length `seq_len`
- Target: Same sequence shifted by one position (next-token prediction)

#### Parameters

- `tokens` (torch.Tensor): Pre-tokenized text as 1D tensor of token IDs
- `seq_len` (int): Sequence length for each training example

#### Attributes

- `tokens`: Stored token tensor
- `seq_len`: Fixed sequence length

#### __len__() -> int

Returns number of complete sequences in the dataset.

**Formula:**
```python
num_sequences = len(tokens) // seq_len
```

**Note:** Partial sequences at the end are truncated.

**Example:**
```python
tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8]
seq_len = 4
length = 9 // 4 = 2  # Two complete sequences
```

#### __getitem__(idx) -> Tuple[Tensor, Tensor]

Returns a single training example (input-target pair).

**Parameters:**
- `idx` (int): Index of the sequence to retrieve

**Returns:**
- `input_ids`: Token sequence [seq_len]
- `targets`: Target sequence [seq_len] (input shifted by 1)

**Algorithm:**

```python
# 1. Calculate slice boundaries
start_idx = idx * seq_len
end_idx = start_idx + seq_len + 1  # +1 for target

# 2. Extract chunk
chunk = tokens[start_idx:end_idx]

# 3. Handle short sequences (padding)
if len(chunk) < seq_len + 1:
    padding = zeros(seq_len + 1 - len(chunk))
    chunk = cat([chunk, padding])

# 4. Split into input and target
input_ids = chunk[:-1]  # First seq_len tokens
targets = chunk[1:]     # Last seq_len tokens (shifted)

return input_ids, targets
```

**Example:**
```python
tokens = [10, 20, 30, 40, 50]
seq_len = 3
idx = 0

chunk = [10, 20, 30, 40]
input_ids = [10, 20, 30]
targets = [20, 30, 40]
```

**Padding Behavior:**
- If sequence at end of dataset is incomplete, pads with zeros
- Ensures consistent tensor shapes for batching
- Padding token ID: 0

### load_wikitext_data Function

Loads and tokenizes WikiText-2 dataset with intelligent caching and error handling.

#### Parameters

- `seq_len` (int): Sequence length for dataset creation (default: 256)
- `vocab_size` (int): Vocabulary size for synthetic data fallback (default: 10000)
- `data_dir` (str): Directory for caching downloaded data (default: "./wikitext")

#### Returns

Tuple of:
- `train_tokens` (torch.Tensor): Training tokens [num_train_tokens]
- `val_tokens` (torch.Tensor): Validation tokens [num_val_tokens]
- `actual_vocab_size` (int): Actual vocabulary size used

#### Data Loading Pipeline

**Phase 1: Download/Cache Management**

```python
# 1. Create data directory
data_path = Path(data_dir)
data_path.mkdir(exist_ok=True)

# 2. Check for cached files
if local_train_file.exists() and size > 1000:
    use cached file
else:
    download from HuggingFace Hub

# 3. Download from HuggingFace
repo_id = "Salesforce/wikitext"
repo_type = "dataset"
files = [
    "wikitext-2-v1/train-00000-of-00001.parquet",
    "wikitext-2-v1/validation-00000-of-00001.parquet"
]
```

**Phase 2: Loading Parquet Files**

```python
train_df = pl.read_parquet(train_file)
val_df = pl.read_parquet(val_file)
```

Uses Polars for efficient parquet reading.

**Phase 3: Tokenization**

```python
# Use tiktoken with GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
actual_vocab_size = enc.n_vocab  # 50257

def tokenize_dataframe(df):
    tokens = []
    for text in df["text"].to_list():
        if text and text.strip():  # Skip empty lines
            tokens.extend(enc.encode(text))
    return torch.tensor(tokens, dtype=torch.long)

train_tokens = tokenize_dataframe(train_df)
val_tokens = tokenize_dataframe(val_df)
```

**Tokenizer Details:**
- **Engine**: tiktoken (Rust-based, very fast)
- **Vocabulary**: GPT-2 BPE (50,257 tokens)
- **Encoding**: Byte-Pair Encoding (BPE)

**Phase 4: Error Handling and Fallback**

```python
try:
    # Attempt WikiText loading
except Exception as e:
    print(f"Failed to load WikiText data ({e})")
    print("Creating synthetic data...")
    
    # Fallback to synthetic data
    train_tokens = torch.randint(0, vocab_size, (100000,))
    val_tokens = torch.randint(0, vocab_size, (10000,))
    return train_tokens, val_tokens, vocab_size
```

**Fallback Scenarios:**
- Network unavailable (no HuggingFace access)
- Missing dependencies (tiktoken, polars, huggingface_hub)
- Corrupted cache files
- Any other loading errors

#### Output Statistics

Function prints useful information:
```
Loading WikiText-2 dataset...
Using cached training file: ./wikitext/train-00000-of-00001.parquet
Using cached validation file: ./wikitext/validation-00000-of-00001.parquet
Loading training and validation data...
Tokenizing datasets...
Train tokens: 2,088,628
Val tokens: 217,646
Vocabulary size: 50,257
```

#### Caching Strategy

**Benefits:**
1. **Speed**: Avoid repeated downloads
2. **Reliability**: Work offline after first download
3. **Efficiency**: HuggingFace cache + local cache

**Cache Locations:**
- Primary: `data_dir/*.parquet` (e.g., `./wikitext/*.parquet`)
- Secondary: `data_dir/.cache/` (HuggingFace Hub cache)

**Cache Validation:**
```python
if local_file.exists() and local_file.stat().st_size > 1000:
    use_cached = True
```

Checks:
- File exists
- File size > 1000 bytes (ensures not corrupted/empty)

## Dataset Characteristics

### WikiText-2

**Source:** Wikipedia articles

**Statistics:**
- Training tokens: ~2.1M
- Validation tokens: ~217K
- Vocabulary: 50,257 (GPT-2 BPE)

**Characteristics:**
- High-quality English text
- Well-formed sentences and paragraphs
- Diverse topics (encyclopedia coverage)
- Standard benchmark for language modeling

**Citation:**
```
@misc{merity2016pointer,
    title={Pointer Sentinel Mixture Models},
    author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
    year={2016},
    eprint={1609.07843},
    archivePrefix={arXiv}
}
```

### Synthetic Data (Fallback)

**Properties:**
- Uniformly random token IDs
- No semantic structure
- Training: 100K tokens
- Validation: 10K tokens
- Vocabulary: User-specified (default 10,000)

**Use Cases:**
- Testing/debugging when network unavailable
- Quick iteration without data download
- Unit testing

## Design Patterns

### Robust Error Handling

**Philosophy:** Never fail due to data loading issues

**Strategy:**
```python
try:
    # Attempt real data loading
    return real_data
except Exception as e:
    # Log error
    print(f"Failed: {e}")
    # Fallback to synthetic
    return synthetic_data
```

**Benefits:**
- Development continues even without network
- Tests can run in CI/CD without external dependencies
- Graceful degradation

### Efficient Tokenization

**Implementation:**
```python
tokens = []
for text in df["text"].to_list():
    if text and text.strip():
        tokens.extend(enc.encode(text))
return torch.tensor(tokens, dtype=torch.long)
```

**Optimizations:**
1. Skip empty lines (avoid tokenizing whitespace)
2. Batch collection before tensor conversion
3. Use tiktoken (Rust-based, much faster than Python tokenizers)

### Memory-Efficient Dataset

**Pattern:** Store raw tokens, compute examples on-the-fly

```python
class WikiTextDataset:
    def __init__(self, tokens, seq_len):
        self.tokens = tokens  # Store once
        self.seq_len = seq_len
    
    def __getitem__(self, idx):
        # Slice on demand, no pre-computed storage
        return self.tokens[start:end]
```

**Benefits:**
- Memory: O(num_tokens) instead of O(num_sequences * seq_len)
- Flexibility: Can change seq_len without reprocessing
- Speed: No preprocessing overhead

## Usage Examples

### Basic Usage

```python
from dataset import load_wikitext_data, WikiTextDataset
from torch.utils.data import DataLoader

train_tokens, val_tokens, vocab_size = load_wikitext_data(
    seq_len=256,
    vocab_size=10000,
    data_dir="./data"
)

train_dataset = WikiTextDataset(train_tokens, seq_len=256)
val_dataset = WikiTextDataset(val_tokens, seq_len=256)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for input_ids, targets in train_loader:
    # input_ids: [batch_size, seq_len]
    # targets: [batch_size, seq_len]
    outputs = model(input_ids, targets)
```

### Custom Data Directory

```python
train_tokens, val_tokens, vocab_size = load_wikitext_data(
    data_dir="/mnt/data/wikitext"
)
```

### Inspecting Data

```python
train_tokens, val_tokens, vocab_size = load_wikitext_data()

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")
print(f"Vocabulary size: {vocab_size:,}")

dataset = WikiTextDataset(train_tokens, seq_len=128)
print(f"Number of sequences: {len(dataset)}")

input_ids, targets = dataset[0]
print(f"Input shape: {input_ids.shape}")
print(f"Target shape: {targets.shape}")
```

### Decoding Tokens

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
train_tokens, _, _ = load_wikitext_data()

sample = train_tokens[:100]
text = enc.decode(sample.tolist())
print(text)
```

## Performance Considerations

### Download Size

**WikiText-2 Parquet Files:**
- Training: ~2-3 MB compressed
- Validation: ~200-300 KB compressed

**Download Time:** 1-5 seconds on typical connections

### Tokenization Speed

**tiktoken Performance:**
- ~1-10 MB/s text processing
- WikiText-2: Tokenizes in <1 second

**Comparison:**
- tiktoken (Rust): 10-100x faster than pure Python tokenizers
- HuggingFace tokenizers: Similar speed (also Rust-based)

### Memory Usage

**Raw Tokens:**
- Train: ~2M tokens × 8 bytes (long) = ~16 MB
- Val: ~200K tokens × 8 bytes = ~1.6 MB
- Total: <20 MB

**Dataset Overhead:** Negligible (only stores reference to tokens)

### DataLoader Performance

**Bottleneck:** Usually not data loading (tokens are in memory)

**Optimization:**
```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

## Dependencies

### Required

- `torch`: Tensor operations and Dataset interface
- `polars`: Fast parquet file reading

### Optional (for WikiText)

- `tiktoken`: Fast BPE tokenization (GPT-2 vocabulary)
- `huggingface_hub`: Download datasets from HuggingFace

### Fallback (if missing)

- Uses synthetic random data
- No external dependencies needed

## Troubleshooting

### "Failed to load WikiText data"

**Causes:**
1. Missing dependencies (tiktoken, polars, huggingface_hub)
2. Network issues
3. HuggingFace Hub unavailable

**Solution:**
```bash
pip install tiktoken polars huggingface_hub
```

Or: Accept synthetic data fallback for testing.

### Cached File Corrupted

**Symptom:** Error reading parquet file

**Solution:**
```bash
rm -rf ./wikitext/*.parquet
rm -rf ./wikitext/.cache
```

Re-run to download fresh files.

### Out of Memory

**Symptom:** OOM when creating dataset

**Solution:**
- Reduce vocabulary size for synthetic data
- Use memory mapping for very large datasets (requires modification)

### Slow Training

**Check:**
1. DataLoader num_workers (increase for better CPU utilization)
2. Batch size (increase if GPU underutilized)
3. pin_memory=True (faster CPU→GPU transfer)

## Future Enhancements

### Potential Improvements

1. **Memory Mapping:**
   ```python
   tokens = np.memmap('tokens.dat', dtype=np.int64, mode='r')
   ```
   For datasets too large for RAM.

2. **Custom Tokenizers:**
   Support other vocabularies (SentencePiece, custom BPE).

3. **Multiple Datasets:**
   Combine WikiText with other sources (BookCorpus, C4).

4. **Data Augmentation:**
   Add noise, masking, or other augmentation techniques.

5. **Streaming:**
   Process data on-the-fly without full download.

6. **Preprocessing Cache:**
   Cache tokenized output for faster subsequent loads.

## Integration

### With Trainer

```python
from dataset import load_wikitext_data, WikiTextDataset
from trainer import Trainer

train_tokens, val_tokens, vocab_size = load_wikitext_data(seq_len=256)
train_dataset = WikiTextDataset(train_tokens, seq_len=256)
val_dataset = WikiTextDataset(val_tokens, seq_len=256)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

trainer = Trainer(model, train_loader, val_loader, optimizer, device)
trainer.train(num_epochs=10)
```

### With Custom Data

```python
custom_tokens = torch.tensor([...])  # Your tokenized data
dataset = WikiTextDataset(custom_tokens, seq_len=512)
```

The WikiTextDataset class is agnostic to token source—works with any 1D token tensor.

