import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(filename="experiment_results.json"):
    with open(filename, "r") as f:
        return json.load(f)


def plot_throughput(results, save_path="plots/throughput.png"):
    Path("plots").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    dense_throughput = results["dense"]["metrics"]["tokens_per_sec"]
    dense_epochs = list(range(1, len(dense_throughput) + 1))

    colors = ["steelblue", "coral", "mediumseagreen", "orchid", "gold"]
    markers = ["o", "s", "^", "D", "v"]

    ax.plot(
        dense_epochs,
        dense_throughput,
        marker="o",
        linestyle="-",
        label="Dense",
        linewidth=2,
        markersize=6,
        color="black",
    )

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]
    for idx, exp_name in enumerate(exp_names):
        moe_throughput = results[exp_name]["metrics"]["tokens_per_sec"]
        moe_epochs = list(range(1, len(moe_throughput) + 1))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            moe_epochs,
            moe_throughput,
            marker=marker,
            linestyle="-",
            label=f"MoE-{exp_name}",
            linewidth=2,
            markersize=6,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Tokens/Second", fontsize=12)
    ax.set_title("Training Throughput: All Experiments", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved throughput plot to {save_path}")
    plt.close()


def plot_validation_loss(results, save_path="plots/validation_loss.png"):
    Path("plots").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    dense_val_loss = results["dense"]["metrics"]["val_ce_losses"]
    dense_eval_steps = list(range(1, len(dense_val_loss) + 1))

    colors = ["steelblue", "coral", "mediumseagreen", "orchid", "gold"]
    markers = ["o", "s", "^", "D", "v"]

    ax.plot(
        dense_eval_steps,
        dense_val_loss,
        marker="o",
        linestyle="-",
        label="Dense",
        linewidth=2.5,
        markersize=7,
        color="black",
    )

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]
    for idx, exp_name in enumerate(exp_names):
        moe_val_loss = results[exp_name]["metrics"]["val_ce_losses"]
        moe_eval_steps = list(range(1, len(moe_val_loss) + 1))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            moe_eval_steps,
            moe_val_loss,
            marker=marker,
            linestyle="-",
            label=f"MoE-{exp_name}",
            linewidth=2.5,
            markersize=7,
            color=color,
            alpha=0.8,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss (Cross-Entropy)", fontsize=12)
    ax.set_title("Validation Loss: All Experiments", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved validation loss plot to {save_path}")
    plt.close()


def plot_expert_usage(results, save_path="plots/expert_usage.png"):
    Path("plots").mkdir(exist_ok=True)

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]
    if not exp_names:
        print("No MoE experiments found")
        return

    num_experiments = len(exp_names)
    fig, axes = plt.subplots(num_experiments, 2, figsize=(12, 5 * num_experiments))

    if num_experiments == 1:
        axes = axes.reshape(1, -1)

    for exp_idx, exp_name in enumerate(exp_names):
        expert_stats_history = results[exp_name]["metrics"].get(
            "expert_stats_history", []
        )

        if not expert_stats_history:
            continue

        final_stats = expert_stats_history[-1]["stats"]

        for layer_idx_pos, (layer_idx, stats) in enumerate(sorted(final_stats.items())):
            if layer_idx_pos >= 2:
                break

            ax = axes[exp_idx, layer_idx_pos]

            usage = stats["usage"]
            percentages_dict = stats["percentages"]

            expert_ids = sorted([int(k) for k in usage.keys()])
            percentages = [percentages_dict[str(k)] for k in expert_ids]

            bars = ax.bar(
                expert_ids,
                percentages,
                color="steelblue",
                edgecolor="black",
                linewidth=1.2,
            )

            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            expected_percentage = 100 / len(expert_ids)
            ax.axhline(
                expected_percentage,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Uniform ({expected_percentage:.1f}%)",
            )

            ax.set_xlabel("Expert ID", fontsize=10)
            ax.set_ylabel("Usage (%)", fontsize=10)
            ax.set_title(
                f"{exp_name} - Layer {layer_idx} (Entropy: {stats['entropy']:.2f})",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xticks(expert_ids)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, max(percentages) * 1.2 if percentages else 20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved expert usage plot to {save_path}")
    plt.close()


def plot_training_loss(results, save_path="plots/training_loss.png"):
    Path("plots").mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    dense_train_loss = results["dense"]["metrics"]["train_ce_losses"]
    dense_epochs = list(range(1, len(dense_train_loss) + 1))

    colors = ["steelblue", "coral", "mediumseagreen", "orchid", "gold"]
    markers = ["o", "s", "^", "D", "v"]

    ax.plot(
        dense_epochs,
        dense_train_loss,
        marker="o",
        linestyle="-",
        label="Dense",
        linewidth=2,
        markersize=6,
        color="black",
        alpha=0.7,
    )

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]
    for idx, exp_name in enumerate(exp_names):
        moe_train_loss = results[exp_name]["metrics"]["train_ce_losses"]
        moe_epochs = list(range(1, len(moe_train_loss) + 1))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax.plot(
            moe_epochs,
            moe_train_loss,
            marker=marker,
            linestyle="-",
            label=f"MoE-{exp_name}",
            linewidth=2,
            markersize=6,
            color=color,
            alpha=0.7,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss (Cross-Entropy)", fontsize=12)
    ax.set_title("Training Loss: All Experiments", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved training loss plot to {save_path}")
    plt.close()


def plot_timing_breakdown(results, save_path="plots/timing_breakdown.png"):
    Path("plots").mkdir(exist_ok=True)

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]
    if not exp_names:
        print("No MoE experiments found")
        return

    fig, axes = plt.subplots(
        1, len(exp_names) + 1, figsize=(5 * (len(exp_names) + 1), 6)
    )

    if len(exp_names) == 0:
        axes = [axes]

    all_experiments = ["dense"] + exp_names

    for idx, exp_name in enumerate(all_experiments):
        ax = axes[idx]

        timing = results[exp_name]["metrics"]["timing_breakdown"]

        if timing["forward"]:
            forward_time = np.mean(timing["forward"])
            backward_time = np.mean(timing["backward"])
            optimizer_time = np.mean(timing["optimizer"])

            components = ["Forward", "Backward", "Optimizer"]
            times = [forward_time, backward_time, optimizer_time]
            colors = ["#3498db", "#e74c3c", "#2ecc71"]

            bars = ax.bar(
                components, times, color=colors, edgecolor="black", linewidth=1.5
            )

            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{time:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            ax.set_ylabel("Time (seconds)", fontsize=11, fontweight="bold")
            ax.set_title(
                f"{exp_name.capitalize()} Model", fontsize=12, fontweight="bold"
            )
            ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Timing Breakdown per Epoch", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved timing breakdown plot to {save_path}")
    plt.close()


def plot_expert_entropy_evolution(results, save_path="plots/expert_entropy.png"):
    Path("plots").mkdir(exist_ok=True)

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]
    if not exp_names:
        print("No MoE experiments found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["steelblue", "coral", "mediumseagreen", "orchid", "gold"]

    for layer_pos in range(2):
        ax = axes[layer_pos]

        for idx, exp_name in enumerate(exp_names):
            expert_stats_history = results[exp_name]["metrics"].get(
                "expert_stats_history", []
            )

            if not expert_stats_history:
                continue

            epochs = [entry["epoch"] for entry in expert_stats_history]

            layer_keys = list(expert_stats_history[0]["stats"].keys())
            if layer_pos >= len(layer_keys):
                continue

            layer_idx = sorted(layer_keys)[layer_pos]

            entropies = [
                entry["stats"][layer_idx]["entropy"] for entry in expert_stats_history
            ]

            color = colors[idx % len(colors)]
            ax.plot(
                epochs,
                entropies,
                marker="o",
                linestyle="-",
                label=exp_name,
                linewidth=2,
                markersize=6,
                color=color,
            )

        max_entropy = np.log(8)
        ax.axhline(
            max_entropy,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Max Entropy ({max_entropy:.2f})",
            alpha=0.7,
        )

        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Routing Entropy", fontsize=11, fontweight="bold")
        ax.set_title(
            f"Layer {layer_idx} Expert Routing Entropy", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved expert entropy evolution plot to {save_path}")
    plt.close()


def plot_combined_summary(results, save_path="plots/summary.png"):
    Path("plots").mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :2])
    dense_val_loss = results["dense"]["metrics"]["val_ce_losses"]
    dense_eval_steps = list(range(1, len(dense_val_loss) + 1))

    colors = ["steelblue", "coral", "mediumseagreen", "orchid"]
    markers = ["o", "s", "^", "D"]

    ax1.plot(
        dense_eval_steps,
        dense_val_loss,
        marker="o",
        linestyle="-",
        label="Dense",
        linewidth=2.5,
        markersize=7,
        color="black",
    )

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]
    for idx, exp_name in enumerate(exp_names):
        moe_val_loss = results[exp_name]["metrics"]["val_ce_losses"]
        moe_eval_steps = list(range(1, len(moe_val_loss) + 1))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        ax1.plot(
            moe_eval_steps,
            moe_val_loss,
            marker=marker,
            linestyle="-",
            label=f"MoE-{exp_name}",
            linewidth=2.5,
            markersize=7,
            color=color,
            alpha=0.8,
        )

    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Validation Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Validation Loss Comparison", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    throughputs = []
    labels = []

    dense_throughput = results["dense"]["metrics"]["tokens_per_sec"]
    throughputs.append(np.mean(dense_throughput))
    labels.append("Dense")

    for exp_name in exp_names:
        moe_throughput = results[exp_name]["metrics"]["tokens_per_sec"]
        throughputs.append(np.mean(moe_throughput))
        labels.append(f"MoE-\n{exp_name}")

    bars = ax2.bar(
        range(len(labels)),
        throughputs,
        color=["black"] + [colors[i % len(colors)] for i in range(len(exp_names))],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
    )

    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Tokens/Second", fontsize=11, fontweight="bold")
    ax2.set_title("Avg Throughput", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, throughput in zip(bars, throughputs):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{throughput:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax3 = fig.add_subplot(gs[1, :])

    best_losses = []
    exp_labels = []

    best_losses.append(results["dense"]["metrics"]["best_val_loss"])
    exp_labels.append("Dense")

    for exp_name in exp_names:
        best_losses.append(results[exp_name]["metrics"]["best_val_loss"])
        exp_labels.append(exp_name)

    bars = ax3.barh(
        range(len(exp_labels)),
        best_losses,
        color=["black"] + [colors[i % len(colors)] for i in range(len(exp_names))],
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
    )

    ax3.set_yticks(range(len(exp_labels)))
    ax3.set_yticklabels(exp_labels, fontsize=10)
    ax3.set_xlabel("Best Validation Loss", fontsize=11, fontweight="bold")
    ax3.set_title("Best Validation Loss Comparison", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="x")
    ax3.invert_yaxis()

    for bar, loss in zip(bars, best_losses):
        width = bar.get_width()
        ax3.text(
            width,
            bar.get_y() + bar.get_height() / 2.0,
            f" {loss:.4f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    if exp_names and results[exp_names[0]]["metrics"].get("expert_stats_history"):
        first_exp = exp_names[0]
        final_stats = results[first_exp]["metrics"]["expert_stats_history"][-1]["stats"]

        for layer_pos, (layer_idx, stats) in enumerate(sorted(final_stats.items())):
            if layer_pos >= 2:
                break

            ax = fig.add_subplot(gs[2, layer_pos])

            usage = stats["usage"]
            percentages_dict = stats["percentages"]

            expert_ids = sorted([int(k) for k in usage.keys()])
            percentages = [percentages_dict[str(k)] for k in expert_ids]

            bars = ax.bar(
                expert_ids,
                percentages,
                color="steelblue",
                edgecolor="black",
                linewidth=1.2,
            )

            expected_percentage = 100 / len(expert_ids)
            ax.axhline(
                expected_percentage,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Uniform",
                alpha=0.7,
            )

            ax.set_xlabel("Expert ID", fontsize=10, fontweight="bold")
            ax.set_ylabel("Usage (%)", fontsize=10, fontweight="bold")
            ax.set_title(
                f"Baseline - Layer {layer_idx}\nEntropy: {stats['entropy']:.2f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.set_xticks(expert_ids)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

    config = results["config"]
    ax_text = fig.add_subplot(gs[2, 2])
    ax_text.axis("off")

    info_lines = ["Experiment Configuration\n"]
    info_lines.append(f"Dense Params: {results['dense']['params']:,}")

    for exp_name in exp_names:
        params = results[exp_name]["params"]
        info_lines.append(f"{exp_name} Params: {params:,}")

    info_lines.extend(
        [
            f"\nHidden Dim: {config['hidden_dim']}",
            f"Num Layers: {config['num_layers']}",
            f"Num Heads: {config['num_heads']}",
            f"Num Experts: {config['num_experts']}",
        ]
    )

    info_text = "\n".join(info_lines)

    ax_text.text(
        0.1,
        0.9,
        info_text,
        transform=ax_text.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    fig.suptitle(
        "MoE Router Experiment Summary", fontsize=16, fontweight="bold", y=0.98
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved summary plot to {save_path}")
    plt.close()


def generate_report(results):
    print("\n" + "=" * 80)
    print("MoE ROUTER EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    config = results["config"]

    print("\nModel Configuration:")
    print(f"  Dense Model Parameters: {results['dense']['params']:,}")
    print(f"  Number of Experts: {config['num_experts']}")
    print(f"  Hidden Dimension: {config['hidden_dim']}")
    print(f"  Number of Layers: {config['num_layers']}")

    exp_names = [k for k in results.keys() if k not in ["config", "dense"]]

    print("\n  MoE Experiment Configurations:")
    for exp_name in exp_names:
        params = results[exp_name]["params"]
        print(f"    {exp_name}: {params:,} parameters")

    print("\nPerformance Metrics:")

    dense_throughput = results["dense"]["metrics"]["tokens_per_sec"]
    dense_avg = np.mean(dense_throughput)
    dense_val = results["dense"]["metrics"]["best_val_loss"]

    print("\n  Dense Baseline:")
    print(f"    Throughput: {dense_avg:.0f} tokens/sec")
    print(f"    Best Val Loss: {dense_val:.4f}")

    for exp_name in exp_names:
        moe_metrics = results[exp_name]["metrics"]
        moe_throughput = moe_metrics["tokens_per_sec"]
        moe_avg = np.mean(moe_throughput)
        moe_val = moe_metrics["best_val_loss"]
        improvement = (dense_val - moe_val) / dense_val * 100
        throughput_ratio = moe_avg / dense_avg

        print(f"\n  MoE-{exp_name}:")
        print(
            f"    Throughput: {moe_avg:.0f} tokens/sec ({throughput_ratio:.2f}x dense)"
        )
        print(f"    Best Val Loss: {moe_val:.4f}")
        print(f"    Improvement: {improvement:+.2f}%")

        if moe_metrics.get("stopped_early"):
            print("    Early stopping: Yes")

        expert_stats_history = moe_metrics.get("expert_stats_history", [])
        if expert_stats_history:
            final_stats = expert_stats_history[-1]["stats"]
            print("    Expert Load Balance:")
            for layer_idx, stats in sorted(final_stats.items()):
                entropy = stats["entropy"]
                min_usage = stats["min_usage_pct"]
                max_usage = stats["max_usage_pct"]
                print(
                    f"      Layer {layer_idx}: Entropy={entropy:.2f}, Min={min_usage:.1f}%, Max={max_usage:.1f}%"
                )

    print("\n" + "=" * 80)


def main():
    print("Loading experiment results...")
    results = load_results("experiment_results.json")

    print("\nGenerating visualizations...")

    plot_validation_loss(results)
    plot_throughput(results)
    plot_expert_usage(results)
    plot_training_loss(results)
    plot_timing_breakdown(results)
    plot_expert_entropy_evolution(results)
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
    print("  - plots/timing_breakdown.png")
    print("  - plots/expert_entropy.png")
    print("  - plots/summary.png (Comprehensive Summary)")


if __name__ == "__main__":
    main()
