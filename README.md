# Shadow-Geometric

Reproducibility repo for the ICML 2026 Mechanistic Interpretability Workshop submission:

> **Shadow: In-Situ Geometric Characterization of LLM Reasoning Manifolds via Bayesian Networks and Forman–Ricci Curvature, with Directional Ricci-Tensor Steering**

## Findings

**(F1) Repulsion-based steering outperforms CAA on FFN-output activations.** On Mistral-7B-Instruct-v0.2 at $n{=}600$, CAA's calibrated target-domain probability $P_{\text{tgt}}$ moves in the *wrong* direction with $\alpha$ (`0.074 → 0.065 → 0.068` over $\alpha \in \{1,3,5\}$). All three repulsion arms move $P_{\text{tgt}}$ in the intended direction.

**(F2) Directional Ricci-tensor steering is Pareto-dominant over flat repulsion** on the (steering effect, on-distribution preservation) frontier. At $\alpha{=}5$:

| arm | ΔP_src | ΔPPL |
|---|---|---|
| repel_curv (scalar)        | −0.040 | −0.07 |
| **repel_curv_directional** | **−0.058** | **−0.04** |
| repel_flat                 | −0.091 | **+0.43** (15% PPL increase) |

Tensor produces 64% of flat's source-probability reduction at zero on-distribution cost. Per unit |ΔP_src|, flat costs 4.7 PPL units; tensor costs ≈0.

## Pipeline

Run order (Brev H100 recommended; total ~30 min on H100):

```
collect_ffn_600.py            ← FFN forward hooks → (600, 4096) per layer
        │
        ▼
build_state_ffn.py            ← Sparse PCA + adaptive-k diff K-means
                                  + per-domain FFN-output centroids
        │
        ▼
precompute_steering_tensors_ffn.py   ← bipartite Forman-Ricci κ
                                       + top-D=20 eigendecomposition
        │
        ├─→ run_pipeline_ffn.py            ← BN MI / router / Procrustes / calibration
        │
        └─→ brev_live_steering.py --phase C --inject-target mlp \
                                  --state-file shadow_live_state_ffn.pt
                    │
                    ▼
            score_generations_calibrated.py   ← TF-IDF + LR + temperature scaling
```

`compute_ffn_steering_vectors.py` is included as a standalone helper to
recompute steering vectors against an existing state file (e.g., after
changing the prompt corpus); `build_state_ffn.py` now does this inline,
so it's optional.

## Quick start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full pipeline (Brev H100 recommended)
python pipeline/collect_ffn_600.py                                       # ~5 min
python pipeline/build_state_ffn.py                                       # ~12 min (Sparse PCA dominates)
python pipeline/precompute_steering_tensors_ffn.py                       # ~10 sec
python pipeline/run_pipeline_ffn.py                                      # ~30 sec
python pipeline/brev_live_steering.py --phase C \
    --state-file shadow_live_state_ffn.pt --inject-target mlp           # ~18 min
python pipeline/score_generations_calibrated.py                          # ~10 sec

# Or run everything at once:
bash run_pipeline.sh        # Linux / Brev
.\run_pipeline.ps1          # Windows
```

Outputs land in:
- `activations_ffn_600.pt`, `ffn_600_info.json`
- `shadow_live_state_ffn.pt`
- `steering_tensors.pt`
- `shadow_v3_output/n600_ffn/pipeline_ffn_stats.json`
- `brev_live_results/phaseC_ablation.json`
- `brev_live_results/phaseC_calibrated_scores.json`
- `brev_live_results/phaseC_calibrated_summary.txt`

## Configuration knobs

| Script | Knob | Default | Effect |
|---|---|---|---|
| `build_state_ffn.py` | `K_MIN, K_MAX` | `[4, 14]` | Silhouette sweep range. Below `k_16=10` empirically, scalar curvature-weighted steering collapses to flat (see paper §6.4). |
| `build_state_ffn.py` | `SPCA_ALPHA` | `0.1` | L1 sparsity penalty. Higher → sparser components. Falls back to standard PCA below the nonzero floor. |
| `precompute_steering_tensors_ffn.py` | `D` | `20` | Tensor rank for top-D eigendecomposition. |
| `brev_live_steering.py` | `--inject-target` | `layer` | Set to `mlp` for FFN-consistent injection (matches the geometry). |
| `brev_live_steering.py` | `PHASE_C_REGIME` env | `small` | `small`: single-layer last-token, α∈{1,3,5}. `large`: multi-layer all-position, α∈{5,10,20}. |

## Methodological note: cluster granularity

The bipartite Forman–Ricci formula's per-pair $\rho_{\text{scalar}}(s,t) = |\kappa_{\text{eff}}(s,t)| / |\kappa_{\text{ref}}|$ requires sufficient cluster granularity to differentiate domain pairs. Empirically:

- At `k_16=5` (200 queries / 5 clusters → every domain occupies every cluster substantially): $\rho_{\text{scalar}} \in [0.998, 1.003]$. Constant within 0.5%. Scalar curvature-weighted steering reduces to flat repulsion.
- At `k_16=10` (silhouette-optimal on FFN-output): $\rho_{\text{scalar}} \in [0.66, 1.46]$. Differentiates by ~1.5× across pairs.

Adaptive-$k$ via silhouette sweep is therefore mandatory; fixed small-$k$ settings will produce a degenerate ablation.

## Hardware

- Mistral-7B-Instruct-v0.2 (32 layers, hidden dim 4096, fp16) — requires NVIDIA H100 (80 GB) or equivalent for activation extraction and live steering.
- All other stages (Sparse PCA, K-means, BN, Forman–Ricci, calibrated scoring) run on CPU.

## Limitations

See `docs/paper_workshop.tex` §7 for the full list. Highlights:
- Single Brev H100 run, single seed (42).
- $\arg\max$ flipping not achieved at $\alpha \in \{1,3,5\}$; F2 is on continuous calibrated probabilities.
- Single architecture (Mistral-7B); cross-architecture replication on Llama / Qwen is left for follow-up.
- Per-arm-pair statistical power is borderline at n=54 generations per (arm, α) cell (~1.4–1.7σ on the differences).

## Citation

Anonymous for double-blind review. Citation will be added after camera-ready.

## License

MIT (code) / CC-BY (paper draft).
