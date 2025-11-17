import torch
import tiktoken
from pathlib import Path
import argparse

from transformer import GPTModel


def simple_generate(
    checkpoint_path: str,
    prompt: str,
    max_length: int = 50,
    temperature: float = 0.8,
    is_moe: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    vocab_size = 10000
    hidden_dim = 256
    num_layers = 6
    num_heads = 8
    ffn_dim = 1024
    max_seq_len = 256

    moe_layers = [2, 4] if is_moe else None

    model = GPTModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        max_seq_len=max_seq_len,
        dropout=0.0,
        moe_layers=moe_layers,
        num_experts=8,
        load_balance_weight=0.01,
        top_k=1,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    input_ids = tokenizer.encode(prompt)
    input_ids = [min(token_id, vocab_size - 1) for token_id in input_ids]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    generated = input_ids.copy()

    print(f"\nPrompt: {prompt}")
    print("Generating...\n")

    with torch.no_grad():
        for _ in range(max_length):
            if input_tensor.size(1) >= max_seq_len:
                input_tensor = input_tensor[:, -max_seq_len:]

            logits, _, _ = model(input_tensor)

            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            next_token_id = next_token.item()
            next_token_id = min(next_token_id, vocab_size - 1)

            generated.append(next_token_id)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)

            if next_token_id == tokenizer.eot_token or next_token_id >= vocab_size:
                break

    output = tokenizer.decode(generated)
    return output


def main():
    parser = argparse.ArgumentParser(description="Simple text generation script")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Max tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature (default: 0.8)"
    )
    parser.add_argument(
        "--moe", action="store_true", help="Use MoE model configuration"
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return

    output = simple_generate(
        args.checkpoint,
        args.prompt,
        args.max_length,
        args.temperature,
        args.moe,
    )

    print("=" * 80)
    print(output)
    print("=" * 80)


if __name__ == "__main__":
    main()
