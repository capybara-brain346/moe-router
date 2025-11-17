import torch
from torch.utils.data import Dataset
import polars as pl
from pathlib import Path
import tiktoken


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


def load_wikitext_data(
    seq_len: int = 256,
    vocab_size: int = 10000,
    data_dir: str = "./wikitext",
):
    try:
        from huggingface_hub import hf_hub_download

        print("Loading WikiText-2 dataset...")

        data_path = Path(data_dir)
        data_path.mkdir(exist_ok=True)

        repo_id = "Salesforce/wikitext"
        repo_type = "dataset"

        train_filename = "wikitext-2-v1/train-00000-of-00001.parquet"
        local_train = data_path / "train-00000-of-00001.parquet"

        if local_train.exists() and local_train.stat().st_size > 1000:
            print(f"Using cached training file: {local_train}")
            train_file = str(local_train)
        else:
            print(f"Downloading {train_filename}...")
            train_file = hf_hub_download(
                repo_id=repo_id,
                filename=train_filename,
                repo_type=repo_type,
                cache_dir=str(data_path / ".cache"),
            )

        val_filename = "wikitext-2-v1/validation-00000-of-00001.parquet"
        local_val = data_path / "validation-00000-of-00001.parquet"

        if local_val.exists() and local_val.stat().st_size > 1000:
            print(f"Using cached validation file: {local_val}")
            val_file = str(local_val)
        else:
            print(f"Downloading {val_filename}...")
            val_file = hf_hub_download(
                repo_id=repo_id,
                filename=val_filename,
                repo_type=repo_type,
                cache_dir=str(data_path / ".cache"),
            )

        print("Loading training and validation data...")
        train_df = pl.read_parquet(train_file)
        val_df = pl.read_parquet(val_file)

        try:
            enc = tiktoken.get_encoding("gpt2")
            actual_vocab_size = enc.n_vocab

            def tokenize_dataframe(df):
                tokens = []
                for text in df["text"].to_list():
                    if text and text.strip():
                        tokens.extend(enc.encode(text))
                return torch.tensor(tokens, dtype=torch.long)

        except Exception as e:
            raise ValueError(f"Error occured while tokenizing: {e}")

        print("Tokenizing datasets...")
        train_tokens = tokenize_dataframe(train_df)
        val_tokens = tokenize_dataframe(val_df)

        print(f"Train tokens: {len(train_tokens):,}")
        print(f"Val tokens: {len(val_tokens):,}")
        print(f"Vocabulary size: {actual_vocab_size:,}")

        return train_tokens, val_tokens, actual_vocab_size

    except Exception as e:
        print(f"Failed to load WikiText data ({e.__class__.__name__}: {e})")
        print("Creating synthetic data...")
        train_tokens = torch.randint(0, vocab_size, (100000,))
        val_tokens = torch.randint(0, vocab_size, (10000,))
        return train_tokens, val_tokens, vocab_size
