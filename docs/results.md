# MoE vs Dense: Experiment Results and Recommendations

## Executive summary

- The dense model achieved the best validation loss and the highest training throughput. It remains the strongest performer for this setup when evaluating generalization versus raw speed.
- Mixture-of-experts (MoE) variants increased model capacity (roughly 17.3M parameters vs 9.9M for the dense model) but introduced substantial runtime overhead, primarily in the backward pass. That overhead reduced tokens/sec compared with the dense baseline.
- Among MoE variants, `throughput_opt` substantially reduced backward cost compared with the other MoEs and delivered the best throughput among MoEs, but it did not match the dense model in tokens/sec or in best validation loss.
- `top2` reached the lowest training loss but the worst validation loss, suggesting overfitting or instability in gating/generalization.
- Routing entropy and per-expert usage indicate reasonably balanced expert assignment across experiments, but layer- and experiment-level differences remain and may explain some of the generalization differences.

Below is a concise quantitative comparison followed by interpretation and recommended next steps.

---

## Key numbers (single-epoch timing, parameters, best validation loss, avg throughput)

| Model              |     Params | Best validation loss | Avg throughput (tokens/sec) | Forward (s) | Backward (s) | Optimizer (s) | Total per-epoch (s) |
| ------------------ | ---------: | -------------------: | --------------------------: | ----------: | -----------: | ------------: | ------------------: |
| Dense              |  9,924,608 |           **9.3698** |                  **68,825** |        0.40 |         0.20 |          0.03 |                0.63 |
| MoE-baseline       | 17,286,656 |               9.3809 |                      44,976 |        0.75 |         1.17 |          0.05 |                1.97 |
| MoE-top2           | 17,286,656 |               9.3921 |                      37,387 |        0.90 |         1.47 |          0.05 |                2.42 |
| MoE-strong_reg     | 17,286,656 |               9.3798 |                      44,243 |        0.76 |         1.19 |          0.05 |                2.00 |
| MoE-throughput_opt | 17,286,656 |               9.3870 |                      54,784 |        0.56 |         0.96 |          0.03 |                1.55 |

Notes:

- Lower validation loss is better. Dense wins here.
- Throughput is tokens/sec, higher is better. Dense also wins here.
- MoE models have larger parameter count because experts increase model size.

---

## Training and validation dynamics

- **Training loss**: All variants show steadily decreasing training loss. `MoE-top2` reaches the lowest training loss across epochs, indicating higher capacity or faster training fit.
- **Validation loss**: Dense achieves the lowest validation loss (9.3698). MoE variants either match or exceed (worse) the dense validation loss:

  - `MoE-baseline` and `MoE-strong_reg` are close to each other and only marginally worse than dense.
  - `MoE-top2` exhibits the largest gap and a clear upward trend in validation loss after epoch 4, indicating overfitting or gating instability.
  - `MoE-throughput_opt` shows stable validation behavior but does not surpass dense.

- Interpretation: The MoE configurations give higher representational capacity but require careful gating/regularization and optimization to realize generalization benefits. Without such tuning, larger capacity can overfit or destabilize validation performance.

---

## Throughput and timing breakdown

- The dominant cost for MoE models is the backward pass. Backward times:

  - `MoE-top2`: 1.47 s
  - `MoE-baseline`: 1.17 s
  - `MoE-strong_reg`: 1.19 s
  - `MoE-throughput_opt`: 0.96 s
  - Dense: 0.20 s

- `throughput_opt` reduced the backward cost significantly compared with other MoEs, producing the best MoE throughput (54,784 tokens/sec), but still below dense.
- Total per-epoch wall-clock is smallest for dense (0.63s) and largest for `top2` (2.42s).
- Interpretation: MoE overheads are chiefly in expert-specific gradient/communication during backward. Optimizations that reduce communication or reduce expert work in backward propagate directly to throughput gains (as `throughput_opt` demonstrates).

---

## Expert routing behavior: usage and entropy

- **Entropy**: Routing entropy per layer is close to the theoretical maximum (max ≈ 2.08 for 8 experts). That indicates routing is using many experts rather than collapsing to a single expert. Specific observations:

  - `top2` layer 2 entropy reaches very close to max early, which matches its aggressive expert usage behavior.
  - Layer 4 entropies are slightly lower and more variable across experiments.

- **Per-expert usage**: Usage bars across models and layers show modest deviations from perfect uniformity (uniform = 12.5% for 8 experts). Most experiments show usage within roughly 10.6% to 14.7% per expert. A few spots show underused experts (for example `throughput_opt` layer 4 had some experts near 9.8–9.9%).
- Interpretation: Routing is generally balanced, but small non-uniformities exist and could cause local specialization or slight load imbalance. Very skewed usage might lead to undertrained experts and affect generalization.

---

## Trade-offs and interpretation

1. **Dense vs MoE capacity**: MoE increases parameter count substantially but this did not translate into better validation loss in these runs. Denser capacity alone is not sufficient to improve generalization.
2. **Throughput trade-offs**: Dense model is faster and achieves better validation performance. For these experiments, MoE costs (especially backward step) made them slower despite higher capacity. `throughput_opt` shows that MoE overhead can be reduced but not fully eliminated.
3. **MoE-top2 anomaly**: The `top2` variant fits training data best but generalizes worst. This suggests gating choices (top-2 routing) can increase overfitting risk or cause training instability unless balanced with stronger regularization or gating temperature tuning.
4. **Balancing and entropy**: Entropy values close to the max indicate gating is not collapsing, which is good for utilization. However, small routing imbalances can still impact performance. Regularization techniques to encourage balanced loads may help.

---

## Recommendations and prioritized next experiments

Short term (fast experiments)

1. **Tune regularization for `top2`**: Increase load-balancing loss, add expert dropout, or apply stronger L2/weight decay to address overfitting observed in `top2`. Run a few epochs to validate effect on validation loss.
2. **Gating temperature schedule**: Anneal gating temperature or apply a small smoothing to logits. This can stabilize expert assignment and reduce noisy specialization.
3. **Repeat `throughput_opt` with more epochs and learning-rate sweep**: `throughput_opt` reduces overhead and merits further tuning to close the generalization gap with dense.
4. **Measure gradient communication/serialization cost**: Instrument the backward pass to see whether communication or per-expert gradient accumulation dominates. This identifies whether improvements should prioritize compute or communication.

Medium term (architectural/engineering) 5. **Increase k in top-k routing or hybrid gating**: Try k > 2 or soft-gating variants to see whether distributing load to more experts reduces overfitting and improves validation. 6. **Per-expert learning rate scaling**: Try different learning rates or adaptive optimizers applied per expert to avoid under- or overtraining some experts. 7. **Experiment with capacity factor and expert size**: Adjust expert internal dimensionality to change parameter distribution between shared layers and experts. 8. **Profiling at batch-level**: Collect per-batch expert selection distributions, per-expert gradient norms, and memory footprints to identify hotspots for optimization.

Long term 9. **Distributed/expert sharding strategies**: If scaling to many experts, consider fusing or sharding experts to reduce cross-device communication in the backward pass. 10. **Hybrid sparse-dense training schedule**: Start training with denser routing early then introduce sparsity as training stabilizes.

---

## Visualization and metrics to add going forward

If you add more experiments, capture:

- Per-expert gradient norms and update counts.
- Per-batch expert assignment histograms (to detect bursty assignments).
- Wall-clock epoch time vs tokens processed for long runs, to compute effective throughput under steady state.
- Validation-perplexity or task-specific metrics (if available) to supplement cross-entropy.
- Memory usage per device and network bandwidth between devices during backward to better target engineering optimizations.

---

## Bottom line

- For the current setup and hyperparameters, the dense model is the best choice: better validation loss and substantially higher throughput.
- MoE variants add capacity but need careful gating, regularization, and engineering optimization to turn that capacity into consistent generalization gains without incurring prohibitive runtime costs.
- If the objective is to realize MoE capacity advantages (for example to scale parameters much further), prioritize gating regularization and backward-pass communication optimizations. If the objective is immediate best generalization and throughput, use the dense model.

If you want, I can:

- produce a compact slide or one-page PDF summarizing these findings, or
- generate specific hyperparameter sweep suggestions (parameter ranges for load-balancing weight, gating temperature, top-k) and a minimal experiment schedule to test them. Which would be more useful next?
