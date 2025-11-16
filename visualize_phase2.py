import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(filename='phase2_results.json'):
    with open(filename, 'r') as f:
        return json.load(f)


def plot_throughput(results, save_path='plots/throughput.png'):
    Path('plots').mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dense_throughput = results['dense']['tokens_per_sec']
    moe_throughput = results['moe']['tokens_per_sec']
    epochs = list(range(1, len(dense_throughput) + 1))
    
    ax.plot(epochs, dense_throughput, 'o-', label='Dense Model', linewidth=2, markersize=6)
    ax.plot(epochs, moe_throughput, 's-', label='MoE Model', linewidth=2, markersize=6)
    
    dense_avg = np.mean(dense_throughput)
    moe_avg = np.mean(moe_throughput)
    
    ax.axhline(dense_avg, color='blue', linestyle='--', alpha=0.5, 
               label=f'Dense Avg: {dense_avg:.0f} tok/s')
    ax.axhline(moe_avg, color='orange', linestyle='--', alpha=0.5,
               label=f'MoE Avg: {moe_avg:.0f} tok/s')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Tokens/Second', fontsize=12)
    ax.set_title('Training Throughput: Dense vs MoE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved throughput plot to {save_path}")
    plt.close()


def plot_validation_loss(results, save_path='plots/validation_loss.png'):
    Path('plots').mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dense_val_loss = results['dense']['val_ce_losses']
    moe_val_loss = results['moe']['val_ce_losses']
    
    eval_steps = list(range(1, len(dense_val_loss) + 1))
    
    ax.plot(eval_steps, dense_val_loss, 'o-', label='Dense Model', linewidth=2, markersize=6)
    ax.plot(eval_steps, moe_val_loss, 's-', label='MoE Model', linewidth=2, markersize=6)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss (Cross-Entropy)', fontsize=12)
    ax.set_title('Validation Loss vs Training Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    final_dense = dense_val_loss[-1] if dense_val_loss else 0
    final_moe = moe_val_loss[-1] if moe_val_loss else 0
    improvement = ((final_dense - final_moe) / final_dense * 100) if final_dense > 0 else 0
    
    ax.text(0.02, 0.98, 
            f'Final Dense: {final_dense:.4f}\nFinal MoE: {final_moe:.4f}\nImprovement: {improvement:+.2f}%',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved validation loss plot to {save_path}")
    plt.close()


def plot_expert_usage(results, save_path='plots/expert_usage.png'):
    Path('plots').mkdir(exist_ok=True)
    
    expert_usage = results['expert_usage']
    
    if not expert_usage:
        print("No expert usage data found")
        return
    
    num_layers = len(expert_usage)
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 5))
    
    if num_layers == 1:
        axes = [axes]
    
    for idx, (layer_idx, usage) in enumerate(sorted(expert_usage.items(), key=lambda x: int(x[0]))):
        ax = axes[idx]
        
        expert_ids = sorted([int(k) for k in usage.keys()])
        counts = [usage[str(k)] for k in expert_ids]
        total = sum(counts)
        percentages = [100 * c / total for c in counts]
        
        bars = ax.bar(expert_ids, percentages, color='steelblue', edgecolor='black', linewidth=1.2)
        
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=9)
        
        expected_percentage = 100 / len(expert_ids)
        ax.axhline(expected_percentage, color='red', linestyle='--', linewidth=2,
                  label=f'Uniform ({expected_percentage:.1f}%)')
        
        ax.set_xlabel('Expert ID', fontsize=11)
        ax.set_ylabel('Usage (%)', fontsize=11)
        ax.set_title(f'Layer {layer_idx} Expert Usage', fontsize=12, fontweight='bold')
        ax.set_xticks(expert_ids)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(percentages) * 1.2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved expert usage plot to {save_path}")
    plt.close()


def plot_training_loss(results, save_path='plots/training_loss.png'):
    Path('plots').mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dense_train_loss = results['dense']['train_ce_losses']
    moe_train_loss = results['moe']['train_ce_losses']
    epochs = list(range(1, len(dense_train_loss) + 1))
    
    ax.plot(epochs, dense_train_loss, 'o-', label='Dense Model', linewidth=2, markersize=6, alpha=0.7)
    ax.plot(epochs, moe_train_loss, 's-', label='MoE Model', linewidth=2, markersize=6, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss (Cross-Entropy)', fontsize=12)
    ax.set_title('Training Loss: Dense vs MoE', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training loss plot to {save_path}")
    plt.close()


def plot_combined_summary(results, save_path='plots/summary.png'):
    Path('plots').mkdir(exist_ok=True)
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    dense_val_loss = results['dense']['val_ce_losses']
    moe_val_loss = results['moe']['val_ce_losses']
    eval_steps = list(range(1, len(dense_val_loss) + 1))
    ax1.plot(eval_steps, dense_val_loss, 'o-', label='Dense Model', linewidth=2.5, markersize=7)
    ax1.plot(eval_steps, moe_val_loss, 's-', label='MoE Model', linewidth=2.5, markersize=7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    dense_throughput = results['dense']['tokens_per_sec']
    moe_throughput = results['moe']['tokens_per_sec']
    dense_avg = np.mean(dense_throughput)
    moe_avg = np.mean(moe_throughput)
    
    bars = ax2.bar(['Dense', 'MoE'], [dense_avg, moe_avg], 
                   color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Tokens/Second', fontsize=12, fontweight='bold')
    ax2.set_title('Avg Throughput', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    expert_usage = results['expert_usage']
    if expert_usage:
        for idx, (layer_idx, usage) in enumerate(sorted(expert_usage.items(), key=lambda x: int(x[0]))):
            if idx >= 2:
                break
                
            ax = fig.add_subplot(gs[1, idx])
            
            expert_ids = sorted([int(k) for k in usage.keys()])
            counts = [usage[str(k)] for k in expert_ids]
            total = sum(counts)
            percentages = [100 * c / total for c in counts]
            
            bars = ax.bar(expert_ids, percentages, color='mediumseagreen', 
                         edgecolor='black', linewidth=1.2)
            
            expected_percentage = 100 / len(expert_ids)
            ax.axhline(expected_percentage, color='red', linestyle='--', linewidth=2,
                      label='Uniform', alpha=0.7)
            
            ax.set_xlabel('Expert ID', fontsize=11, fontweight='bold')
            ax.set_ylabel('Usage (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'Layer {layer_idx} Expert Usage', fontsize=12, fontweight='bold')
            ax.set_xticks(expert_ids)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
    
    config = results['config']
    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis('off')
    
    info_text = f"""
Model Configuration

Dense Parameters: {config['dense_params']:,}
MoE Parameters: {config['moe_params']:,}
Param Ratio: {config['moe_params']/config['dense_params']:.2f}x

Hidden Dim: {config['hidden_dim']}
Num Layers: {config['num_layers']}
Num Heads: {config['num_heads']}
FFN Dim (Dense): {config['ffn_dim']}
FFN Dim (MoE): {config['moe_ffn_dim']}
Num Experts: {config['num_experts']}

Final Val Loss:
  Dense: {dense_val_loss[-1]:.4f}
  MoE: {moe_val_loss[-1]:.4f}
    """
    
    ax_text.text(0.1, 0.9, info_text, transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    fig.suptitle('Phase 2: Dense vs MoE Transformer Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot to {save_path}")
    plt.close()


def generate_report(results):
    print("\n" + "=" * 80)
    print("PHASE 2 RESULTS SUMMARY")
    print("=" * 80)
    
    config = results['config']
    
    print("\nModel Configuration:")
    print(f"  Dense Model Parameters: {config['dense_params']:,}")
    print(f"  MoE Model Parameters: {config['moe_params']:,}")
    print(f"  Parameter Ratio (MoE/Dense): {config['moe_params']/config['dense_params']:.2f}x")
    print(f"  Number of Experts: {config['num_experts']}")
    print(f"  Hidden Dimension: {config['hidden_dim']}")
    print(f"  Number of Layers: {config['num_layers']}")
    
    print("\nPerformance Metrics:")
    
    dense_throughput = results['dense']['tokens_per_sec']
    moe_throughput = results['moe']['tokens_per_sec']
    dense_avg = np.mean(dense_throughput)
    moe_avg = np.mean(moe_throughput)
    
    print(f"  Throughput:")
    print(f"    Dense: {dense_avg:.0f} tokens/sec")
    print(f"    MoE: {moe_avg:.0f} tokens/sec")
    print(f"    Ratio: {moe_avg/dense_avg:.2f}x")
    
    dense_val = results['dense']['val_ce_losses'][-1]
    moe_val = results['moe']['val_ce_losses'][-1]
    improvement = (dense_val - moe_val) / dense_val * 100
    
    print(f"\n  Final Validation Loss:")
    print(f"    Dense: {dense_val:.4f}")
    print(f"    MoE: {moe_val:.4f}")
    print(f"    Improvement: {improvement:+.2f}%")
    
    print("\n  Expert Load Balance:")
    expert_usage = results['expert_usage']
    for layer_idx, usage in sorted(expert_usage.items(), key=lambda x: int(x[0])):
        counts = list(usage.values())
        total = sum(counts)
        percentages = [100 * c / total for c in counts]
        std_dev = np.std(percentages)
        expected = 100 / len(counts)
        balance_score = 100 * (1 - std_dev / expected)
        
        print(f"    Layer {layer_idx}: {balance_score:.1f}% balanced (std={std_dev:.2f}%)")
    
    print("\n" + "=" * 80)


def main():
    print("Loading Phase 2 results...")
    results = load_results('phase2_results.json')
    
    print("\nGenerating visualizations...")
    
    plot_validation_loss(results)
    plot_throughput(results)
    plot_expert_usage(results)
    plot_training_loss(results)
    plot_combined_summary(results)
    
    generate_report(results)
    
    print("\n" + "=" * 80)
    print("All plots generated successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - plots/validation_loss.png")
    print("  - plots/throughput.png")
    print("  - plots/expert_usage.png")
    print("  - plots/training_loss.png")
    print("  - plots/summary.png (Portfolio Graph)")


if __name__ == "__main__":
    main()

