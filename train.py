import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from moe_model import MoERouter, TransformerWithMoE
import time


class SyntheticTextDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_seq = self.data[idx, :-1]
        target_seq = self.data[idx, 1:]
        return input_seq, target_seq


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        
        logits, aux_loss = model(input_ids)
        
        ce_loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        loss = ce_loss + aux_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_aux_loss += aux_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_aux_loss = total_aux_loss / num_batches
    
    return avg_loss, avg_ce_loss, avg_aux_loss


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_aux_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits, aux_loss = model(input_ids)
            
            ce_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1)
            )
            
            loss = ce_loss + aux_loss
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_aux_loss += aux_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_aux_loss = total_aux_loss / num_batches
    
    return avg_loss, avg_ce_loss, avg_aux_loss


def test_moe_router():
    print("=" * 60)
    print("Testing MoE Router Component")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 8
    hidden_dim = 64
    num_experts = 4
    ffn_dim = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    moe_router = MoERouter(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        ffn_dim=ffn_dim,
        load_balance_weight=0.01
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, hidden_dim).to(device)
    
    print(f"Input shape: {x.shape}")
    
    output, aux_loss = moe_router(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.6f}")
    
    assert output.shape == x.shape, "Output shape mismatch!"
    assert aux_loss.item() >= 0, "Aux loss should be non-negative!"
    
    print("\n✓ MoE Router test passed!\n")


def train_model():
    print("=" * 60)
    print("Training Transformer with MoE")
    print("=" * 60)
    
    vocab_size = 1000
    hidden_dim = 128
    num_experts = 8
    ffn_dim = 256
    num_layers = 2
    seq_len = 32
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    print("Model Configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of experts: {num_experts}")
    print(f"  FFN dimension: {ffn_dim}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of epochs: {num_epochs}\n")
    
    train_dataset = SyntheticTextDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
    val_dataset = SyntheticTextDataset(num_samples=200, seq_len=seq_len, vocab_size=vocab_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TransformerWithMoE(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        max_seq_len=seq_len,
        load_balance_weight=0.01
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}\n")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("=" * 60)
    print("Training Progress")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_ce_loss, train_aux_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        
        val_loss, val_ce_loss, val_aux_loss = evaluate(
            model, val_loader, device
        )
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
        print(f"  Train - Loss: {train_loss:.4f} | CE: {train_ce_loss:.4f} | Aux: {train_aux_loss:.6f}")
        print(f"  Val   - Loss: {val_loss:.4f} | CE: {val_ce_loss:.4f} | Aux: {val_aux_loss:.6f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    torch.save(model.state_dict(), "moe_model.pt")
    print("\nModel saved to moe_model.pt")


if __name__ == "__main__":
    test_moe_router()
    train_model()

