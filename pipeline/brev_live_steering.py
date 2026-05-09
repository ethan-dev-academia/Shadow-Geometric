"""
Shadow: Live-Inference Steering Experiment (Brev / GPU)

Runs on a single GPU (H100 recommended, A100 fine). Tests whether Shadow's
contrastive steering vectors actually shift Mistral-7B's generated text when
injected into the residual stream during model.generate().

Phases:
  A. Sanity check #2: do steered activations exhibit the repulsion pattern
     (cross-domain cosine DECREASES) in-generation the same way they do on
     saved activations?

  B. Exp 1a: does generated text itself shift toward the target domain?
     - Alpha sweep: {0.5, 1.0, 2.0, 3.0, 5.0, 8.0}
     - Coherence check via perplexity of generated text
     - Domain classifier on generated text (separate simple classifier)

  C. Exp 1b: attraction-only vs repulsion-only vs contrastive ablation

  D. Exp 1c: BN-gated vs always-steer (uses shadow_live_state.pt)

Usage:
    python brev_live_steering.py --phase A   # just sanity
    python brev_live_steering.py --phase all # full experiment
"""
import argparse, json, pickle, time, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).parent.parent
STATE_PATH = BASE / "shadow_live_state.pt"
OUT_DIR = BASE / "brev_live_results"
OUT_DIR.mkdir(exist_ok=True)

LAYER_IDXS = [8, 16, 24]
LAYER_KEYS = ["layer_8", "layer_16", "layer_24"]
STEER_LAYER_IDX = 16        # legacy; layer where curvature κ is computed
STEER_LAYER_KEY = "layer_16"
STEER_LAYER_IDXS = [8, 16, 24]   # multi-layer steering (workshop default)
STEER_LAYER_KEYS = ["layer_8", "layer_16", "layer_24"]
DOMAINS = ["Medical", "Financial", "Code"]

# Held-out eval prompts — same form as paper but fresh queries
EVAL_PROMPTS = {
    "Medical": [
        "What are the symptoms of asthma?",
        "How does chemotherapy work?",
        "Explain the role of insulin in the body.",
        "What causes high blood pressure?",
        "Describe the immune response to a viral infection.",
    ],
    "Financial": [
        "How do mutual funds work?",
        "What is the difference between a Roth IRA and a traditional IRA?",
        "Explain market capitalization.",
        "How does bond yield change with interest rates?",
        "What is dollar-cost averaging?",
    ],
    "Code": [
        "Write a Python function to check if a string is a palindrome.",
        "Explain the difference between a stack and a queue.",
        "How do you implement merge sort?",
        "What is a hash table and how does it work?",
        "Write a SQL query to find duplicate rows in a table.",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# Activation hook manager
# ═══════════════════════════════════════════════════════════════════
class HookManager:
    """Manages forward hooks on Mistral layers for extraction + steering."""

    def __init__(self, model, inject_target="layer"):
        self.model = model
        self.captured = {k: None for k in LAYER_KEYS}
        self.handles = []
        self.steering_active = False
        self.steer_vecs = {}        # {layer_idx: (4096,) tensor}
        self.alpha = 0.0
        self.sign = 1.0             # +1 = additive (CAA), -1 = repulsion
        self.scale = 1.0            # curvature-derived scaling for repulsion
        self.mode = "caa"           # caa | repel_flat | repel_curv | repel_curv_directional
        self.all_positions = True   # True = steer every token (RepE-style); False = last-token only
        self.inject_target = inject_target  # "layer" (residual stream) or "mlp" (FFN-consistent)
        # Directional Ricci-tensor steering state.
        # dir_packs[layer_idx] = {
        #   "U":         (4096, D)        leading-magnitude eigendirections of T_{l→l'}
        #   "lam_norm":  (D,)             sign-preserving normalized eigenvalues
        #   "v_contrast":(N_dom, N_dom, 4096)  cross-domain contrast vectors μ_tgt − μ_src
        # }
        self.directional_mode = False
        self.dir_packs = {}
        self.dir_src_idx = 0
        self.dir_tgt_idx = 1

    def _capture_hook(self, layer_key):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self.captured[layer_key] = h[:, -1, :].detach().float().cpu().numpy()
            return out
        return hook

    def _steer_hook(self, layer_idx):
        """Per-layer steering hook. Pulls steer_vecs[layer_idx] and applies it."""
        def hook(module, inp, out):
            if not self.steering_active:
                return out
            h = out[0] if isinstance(out, tuple) else out

            # Directional Ricci-tensor branch: GLOBAL operator
            #   delta = sign · alpha · U · diag(lam_norm) · Uᵀ · v_contrast(s,t)
            # where (U, lam_norm) come from the bipartite cross-layer tensor
            # T_{l→l'} (paper Section 4.8) and v_contrast is the precomputed
            # cross-domain contrast (μ_tgt − μ_src) at this layer.
            if self.directional_mode and layer_idx in self.dir_packs:
                pack = self.dir_packs[layer_idx]
                U = pack["U"]                        # (4096, D)
                lam = pack["lam_norm"]               # (D,)
                v_contrast = pack["v_contrast"][self.dir_src_idx,
                                                self.dir_tgt_idx]  # (4096,)
                h = h.clone()
                # Tensor application is input-independent here (operator acts
                # on a fixed contrast vector); compute once per call.
                Tv = U @ (lam * (U.T @ v_contrast.float()))         # (4096,)
                delta = (self.sign * self.alpha * Tv).to(h.dtype).to(h.device)
                if self.all_positions:
                    h = h + delta
                else:
                    h[:, -1, :] = h[:, -1, :] + delta
                if isinstance(out, tuple):
                    return (h,) + out[1:]
                return h

            # Standard (scalar) branch
            if layer_idx not in self.steer_vecs:
                return out
            vec = self.steer_vecs[layer_idx]
            h = h.clone()
            delta = self.sign * self.alpha * self.scale * vec.to(h.dtype).to(h.device)
            if self.all_positions:
                h = h + delta              # broadcasts over seq_len: RepE-style
            else:
                h[:, -1, :] = h[:, -1, :] + delta
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h
        return hook

    def install(self):
        mlayers = self.model.model.layers
        # Steer hook target: layer-output (residual stream) by default; FFN
        # submodule output if self.inject_target == "mlp". The FFN-consistent
        # path injects in the same vector space the BN/clusters/curvature
        # tensor were built from (build_state_ffn.py + precompute_steering_
        # tensors_ffn.py), eliminating the FFN-geometry-vs-residual-injection
        # mismatch.
        inject_target = getattr(self, "inject_target", "layer")
        for idx in STEER_LAYER_IDXS:
            tgt_module = (mlayers[idx].mlp if inject_target == "mlp"
                          else mlayers[idx])
            self.handles.append(tgt_module.register_forward_hook(
                self._steer_hook(idx)))
        # Capture hooks remain on the layer output for diagnostic continuity
        # (Phase A / B logging).
        for idx, key in zip(LAYER_IDXS, LAYER_KEYS):
            self.handles.append(mlayers[idx].register_forward_hook(
                self._capture_hook(key)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def set_steering(self, vecs, alpha, mode="caa", scale=1.0, all_positions=True):
        """Configure the steering hook.

        vecs : either a single ndarray (applied at STEER_LAYER_IDX only) or
               a dict {layer_idx: ndarray} for multi-layer steering.

        mode = "caa"        : additive toward target (sign=+1)
        mode = "repel_flat" : flat repulsion from source (sign=-1, scale=1.0)
        mode = "repel_curv" : curvature-weighted repulsion (sign=-1, scale=|κ|/|κ_ref|)
        """
        if isinstance(vecs, dict):
            self.steer_vecs = {idx: torch.tensor(v, dtype=torch.float32)
                               for idx, v in vecs.items()}
        else:
            self.steer_vecs = {STEER_LAYER_IDX: torch.tensor(vecs, dtype=torch.float32)}
        self.alpha = alpha
        self.scale = scale
        self.mode = mode
        self.sign = 1.0 if mode == "caa" else -1.0
        self.all_positions = all_positions
        self.steering_active = True

    def clear_steering(self):
        self.steering_active = False
        self.steer_vecs = {}
        self.alpha = 0.0
        self.scale = 1.0
        self.sign = 1.0
        self.all_positions = True
        self.directional_mode = False

    def set_directional(self, dir_packs, alpha, src_idx, tgt_idx,
                        sign=+1.0, all_positions=False):
        """Activate global Ricci-tensor steering at the layers in dir_packs.

        dir_packs : dict mapping layer_idx → pack dict (U, lam_norm, v_contrast).
                    Tensors should already be on the same device/dtype as the
                    model's residual stream (or float32 — cast happens at use).
        sign      : +1 pushes activation toward the tensor-shaped target, −1
                    repels from it. Repulsion is the workshop default; +1 is
                    available for completeness.
        """
        self.dir_packs = dir_packs
        self.dir_src_idx = int(src_idx)
        self.dir_tgt_idx = int(tgt_idx)
        self.alpha = float(alpha)
        self.sign = float(sign)
        self.all_positions = bool(all_positions)
        self.mode = "repel_curv_directional"
        self.steer_vecs = {}
        self.directional_mode = True
        self.steering_active = True


# ═══════════════════════════════════════════════════════════════════
# Trajectory scoring (reuse shadow_live_state.pt)
# ═══════════════════════════════════════════════════════════════════
def load_state(path=None):
    p = path if path is not None else STATE_PATH
    return torch.load(p, map_location="cpu", weights_only=False)


def score_trajectory(acts_dict, state):
    """Return (cluster_ids, trajectory_score) for a single query."""
    cluster_ids = {}
    for lk in LAYER_KEYS:
        pca = pickle.loads(state["pca_models"][lk])
        centers = state["kmeans_centers"][lk]
        a = acts_dict[lk].reshape(1, -1)
        reduced = pca.transform(a)
        dists = np.linalg.norm(reduced[:, None, :] - centers[None, :, :], axis=2)
        cluster_ids[lk] = int(dists.argmin(axis=1)[0])

    c8, c16, c24 = cluster_ids["layer_8"], cluster_ids["layer_16"], cluster_ids["layer_24"]
    score = float(np.log(state["T_8_16"][c8, c16]) + np.log(state["T_16_24"][c16, c24]))
    return cluster_ids, score


def cos(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ═══════════════════════════════════════════════════════════════════
# Directional Ricci-tensor pack loader
# ═══════════════════════════════════════════════════════════════════
def load_dir_packs(state, layer_idx_to_pair, device, dtype=torch.float32):
    """Build dir_packs[layer_idx] from saved global tensors.

    Expects state["dir_tensor"][pair] = {"U", "lam_norm", "v_contrast"}.
    `pair` is e.g. "layer_16__layer_24". `v_contrast[s, t]` is μ_tgt(L) −
    μ_src(L); both centroids are computed from the *steered* layer's training
    activations at index s, t.

    state["dir_tensor"] is produced offline (Stage 8 of run_brev_n600.py or
    the standalone precompute_steering_tensors.py companion script).
    """
    out = {}
    if "dir_tensor" not in state:
        return out
    dirs = state["dir_tensor"]
    for layer_idx, pair in layer_idx_to_pair.items():
        if pair not in dirs: continue
        pack_np = dirs[pair]
        out[layer_idx] = {
            "U":          torch.tensor(pack_np["U"],
                                       dtype=dtype, device=device),
            "lam_norm":   torch.tensor(pack_np["lam_norm"],
                                       dtype=dtype, device=device),
            "v_contrast": torch.tensor(pack_np["v_contrast"],
                                       dtype=dtype, device=device),
        }
    return out


# ═══════════════════════════════════════════════════════════════════
# BN-derived Forman-Ricci curvature (workshop novelty)
# ═══════════════════════════════════════════════════════════════════
def build_bn_layer16_graph(state):
    """Build a symmetric weighted adjacency over layer-16 clusters from BN
    transition matrices.

    Edge weight = outgoing-routing similarity (via T_16_24) + incoming-source
    similarity (via T_8_16). Two layer-16 clusters are 'close' if they route
    to similar layer-24 clusters AND receive from similar layer-8 clusters.
    Diagonal is zeroed (self-loops not needed for Forman-Ricci).
    """
    T_8_16 = np.asarray(state["T_8_16"], dtype=np.float64)   # (k8, k16)
    T_16_24 = np.asarray(state["T_16_24"], dtype=np.float64) # (k16, k24)

    # Outgoing similarity: clusters routing to similar layer-24 targets
    # T_16_24 rows are already proper distributions over layer-24 clusters
    A_out = T_16_24 @ T_16_24.T                              # (k16, k16)

    # Incoming similarity: clusters receiving from similar layer-8 sources
    A_in = T_8_16.T @ T_8_16                                 # (k16, k16)

    A = A_out + A_in
    np.fill_diagonal(A, 0.0)
    return A


def forman_ricci_curvature(adj_matrix):
    """Augmented Forman-Ricci curvature on weighted graph (Sreejith et al. 2016).

    For edge (i,j) with weight w_ij:
      F(i,j) = w_ij * (2 - sum_{k!=j} w_ik / max(w_ij, w_ik)
                         - sum_{k!=i} w_jk / max(w_ij, w_jk))

    Returns kappa as a (n, n) symmetric matrix (NaN for non-edges).
    Mirrors experiment_lyapunov_formanricci.py:247.
    """
    n = adj_matrix.shape[0]
    kappa = np.full((n, n), np.nan, dtype=np.float64)
    eps = 1e-12
    for i in range(n):
        for j in range(i + 1, n):
            w_ij = adj_matrix[i, j]
            if w_ij < eps:
                continue
            sum_i = 0.0
            for k in range(n):
                if k != j and adj_matrix[i, k] > eps:
                    sum_i += adj_matrix[i, k] / max(w_ij, adj_matrix[i, k])
            sum_j = 0.0
            for k in range(n):
                if k != i and adj_matrix[j, k] > eps:
                    sum_j += adj_matrix[j, k] / max(w_ij, adj_matrix[j, k])
            F_ij = w_ij * (2.0 - sum_i - sum_j)
            kappa[i, j] = F_ij
            kappa[j, i] = F_ij
    return kappa


def compute_layer16_curvature(state, verbose=True):
    """Compute Forman-Ricci kappa matrix between all layer-16 cluster pairs,
    derived from the BN transition matrices in shadow_live_state.pt.

    Returns:
        kappa: (k16, k16) symmetric, NaN on non-edges
        adj:   (k16, k16) the BN-derived adjacency used to compute kappa
    """
    adj = build_bn_layer16_graph(state)
    kappa = forman_ricci_curvature(adj)
    if verbose:
        valid = kappa[~np.isnan(kappa)]
        print(f"[bn-curv] layer-16 cluster graph: {adj.shape[0]} nodes, "
              f"{(~np.isnan(kappa)).sum() // 2} edges")
        print(f"[bn-curv] kappa: mean={valid.mean():+.4f} std={valid.std():.4f} "
              f"min={valid.min():+.4f} max={valid.max():+.4f}")
        print(f"[bn-curv] negative (bottleneck/hyperbolic): "
              f"{(valid < 0).sum()}/{len(valid)}  "
              f"positive (redundant/spherical): {(valid > 0).sum()}/{len(valid)}")
    return kappa, adj


def compute_domain_cluster_dists(state):
    """Empirical cluster distribution at layer 16 per domain, derived from
    the training data stored in shadow_live_state.pt.

    Returns dict: {domain_name: ndarray of shape (k16,) summing to 1}.
    """
    ch16 = np.asarray(state["train_cluster_hard"]["layer_16"])
    di = np.asarray(state["train_domain_int"])
    domains = state["domains"]
    k16 = state["cluster_k"]["layer_16"]
    dists = {}
    for d_idx, d_name in enumerate(domains):
        mask = di == d_idx
        if mask.sum() == 0:
            dists[d_name] = np.full(k16, 1.0 / k16)
            continue
        counts = np.bincount(ch16[mask], minlength=k16).astype(np.float64)
        dists[d_name] = counts / counts.sum()
    return dists


def domain_pair_curvature(src_domain, tgt_domain, kappa, dists):
    """Joint-occupancy-weighted Forman-Ricci between two domains at layer 16.

    kappa_eff = sum_{c,c'} P(c|src) * P(c'|tgt) * kappa[c,c']
                divided by joint mass on edges (excludes self-pairs which are NaN).
    Returns scalar (typically negative for our BN graphs).
    """
    P_s = dists[src_domain]
    P_t = dists[tgt_domain]
    joint = P_s[:, None] * P_t[None, :]                        # (k16, k16)
    valid = ~np.isnan(kappa)
    K = np.where(valid, kappa, 0.0)
    edge_mass = joint[valid].sum()
    if edge_mass < 1e-12:
        return 0.0
    return float((joint * K).sum() / edge_mass)


def build_curvature_pair_table(state, kappa, dists, verbose=True):
    """Compute kappa_eff for every (src, tgt) domain pair at layer 16.

    Returns:
        table: dict {(src, tgt): kappa_eff}  — for src != tgt
        scale: float — mean |kappa_eff| across pairs (used to normalize alpha)
    """
    domains = state["domains"]
    table = {}
    for s in domains:
        for t in domains:
            if s == t:
                continue
            table[(s, t)] = domain_pair_curvature(s, t, kappa, dists)
    scale = float(np.mean([abs(v) for v in table.values()])) or 1.0
    if verbose:
        print(f"[bn-curv] domain-pair kappa_eff (joint-weighted, layer 16):")
        for (s, t), v in table.items():
            print(f"           {s:10s} -> {t:10s}  kappa_eff = {v:+.4f}  "
                  f"(scale factor = {abs(v)/scale:.3f})")
        print(f"[bn-curv] curvature scale (mean |kappa_eff|) = {scale:.4f}")
    return table, scale


# ═══════════════════════════════════════════════════════════════════
# Phase A: Activation-space repulsion under generation
# ═══════════════════════════════════════════════════════════════════
def phase_A_sanity(model, tokenizer, hooks, state, device):
    """Do steered activations show cross-domain cosine DROP during generation?"""
    print("\n" + "=" * 60)
    print("PHASE A: Activation-space repulsion under generation")
    print("=" * 60)

    steering = state["steering_vectors"]
    # Compute domain centroids from n=600 saved activations
    # (re-derived here from the steering vectors themselves — cheap proxy)

    results = []
    for domain in DOMAINS:
        v = steering[domain][STEER_LAYER_KEY]  # (4096,)
        # Unit-normalize (Panickssery-style)
        v_unit = v / (np.linalg.norm(v) + 1e-10)

        for prompt in EVAL_PROMPTS[domain]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Baseline: no steering
            hooks.clear_steering()
            with torch.no_grad():
                model(**inputs)
            base_act = hooks.captured[STEER_LAYER_KEY][0].copy()

            # Steered toward own domain
            for alpha in [1.0, 3.0, 5.0]:
                hooks.set_steering(v_unit, alpha)
                with torch.no_grad():
                    model(**inputs)
                steered_act = hooks.captured[STEER_LAYER_KEY][0].copy()

                # Cosine to own steering direction & other domains
                delta = steered_act - base_act
                delta_mag = np.linalg.norm(delta)
                cos_to_own_v = cos(delta, v_unit)

                other_doms = [d for d in DOMAINS if d != domain]
                cos_to_other = np.mean([
                    cos(steered_act, steering[od][STEER_LAYER_KEY])
                    for od in other_doms
                ])
                cos_to_other_base = np.mean([
                    cos(base_act, steering[od][STEER_LAYER_KEY])
                    for od in other_doms
                ])

                results.append({
                    "prompt": prompt, "domain": domain, "alpha": alpha,
                    "delta_mag": float(delta_mag),
                    "cos_steering_to_own_v": float(cos_to_own_v),
                    "cos_to_other_base": float(cos_to_other_base),
                    "cos_to_other_steered": float(cos_to_other),
                    "cos_to_other_delta": float(cos_to_other - cos_to_other_base),
                })

    # Aggregate
    print(f"\n  Collected {len(results)} (prompt × alpha) observations")
    by_alpha = {}
    for r in results:
        by_alpha.setdefault(r["alpha"], []).append(r)
    for a, rs in sorted(by_alpha.items()):
        dcos = np.mean([r["cos_to_other_delta"] for r in rs])
        dcos_own = np.mean([r["cos_steering_to_own_v"] for r in rs])
        dmag = np.mean([r["delta_mag"] for r in rs])
        print(f"  α={a}: Δcos(→other) = {dcos:+.4f}  "
              f"cos(Δ, v) = {dcos_own:+.4f}  |Δ| = {dmag:.2f}")

    with open(OUT_DIR / "phaseA_repulsion.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → saved to {OUT_DIR / 'phaseA_repulsion.json'}")

    # Verdict
    best_alpha_dcos = min(by_alpha.keys(), key=lambda a: np.mean([r["cos_to_other_delta"] for r in by_alpha[a]]))
    dcos_best = np.mean([r["cos_to_other_delta"] for r in by_alpha[best_alpha_dcos]])
    if dcos_best < -0.02:
        print(f"\n  ✓ PASS: repulsion effect present (Δcos = {dcos_best:+.4f} at α={best_alpha_dcos})")
        return True
    else:
        print(f"\n  ⚠ WEAK: no clear repulsion (best Δcos = {dcos_best:+.4f})")
        return False


# ═══════════════════════════════════════════════════════════════════
# Phase B: Exp 1a — live generation with steering
# ═══════════════════════════════════════════════════════════════════
def phase_B_generation(model, tokenizer, hooks, state, device):
    print("\n" + "=" * 60)
    print("PHASE B (Exp 1a): Live generation under steering")
    print("=" * 60)

    steering = state["steering_vectors"]
    alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    gen_kwargs = dict(max_new_tokens=60, do_sample=False, use_cache=True,
                      pad_token_id=tokenizer.eos_token_id)

    results = []
    # For each (source_prompt, target_domain, alpha) combination
    for src_domain in DOMAINS:
        for prompt in EVAL_PROMPTS[src_domain][:3]:  # 3 per domain → 9 base prompts
            for tgt_domain in DOMAINS:
                if tgt_domain == src_domain:
                    # Only collect baseline (α=0) for own domain
                    tgt_alphas = [0.0]
                else:
                    tgt_alphas = alphas

                v = steering[tgt_domain][STEER_LAYER_KEY]
                v_unit = v / (np.linalg.norm(v) + 1e-10)

                for alpha in tgt_alphas:
                    if alpha == 0.0:
                        hooks.clear_steering()
                    else:
                        hooks.set_steering(v_unit, alpha)

                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs, **gen_kwargs)
                    gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                                skip_special_tokens=True)
                    results.append({
                        "prompt": prompt, "src_domain": src_domain,
                        "tgt_domain": tgt_domain, "alpha": alpha,
                        "generated": gen_text,
                    })
                    tag = "BL" if alpha == 0 else f"α={alpha}"
                    print(f"  [{src_domain[:3]}→{tgt_domain[:3]} {tag:>6}] {gen_text[:80]}...")

    hooks.clear_steering()
    with open(OUT_DIR / "phaseB_generation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  → saved to {OUT_DIR / 'phaseB_generation.json'}")
    return results


# ═══════════════════════════════════════════════════════════════════
# Metrics: lexical domain classifier + perplexity (coherence)
# ═══════════════════════════════════════════════════════════════════
import re

DOMAIN_KEYWORDS = {
    "Medical": {
        "symptom", "symptoms", "disease", "patient", "patients", "treatment",
        "diagnosis", "clinical", "drug", "drugs", "medication", "therapy",
        "blood", "cell", "cells", "protein", "infection", "virus", "bacteria",
        "immune", "syndrome", "diabetes", "asthma", "cancer", "chemotherapy",
        "insulin", "hormone", "tissue", "organ", "doctor", "hospital",
    },
    "Financial": {
        "investment", "investments", "stock", "stocks", "bond", "bonds",
        "market", "portfolio", "fund", "funds", "tax", "taxes", "interest",
        "yield", "yields", "risk", "asset", "assets", "loan", "loans",
        "rate", "rates", "ira", "roth", "dividend", "capital", "equity",
        "mutual", "broker", "trading", "shares", "credit", "debt",
    },
    "Code": {
        "function", "functions", "algorithm", "algorithms", "array", "arrays",
        "string", "strings", "variable", "loop", "class", "method", "return",
        "sort", "hash", "tree", "list", "def", "python", "sql", "query",
        "stack", "queue", "merge", "binary", "iterate", "recursive",
        "boolean", "integer", "object", "table", "row", "column",
    },
}


def domain_score(text, domain):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in DOMAIN_KEYWORDS[domain]) / len(words)


def domain_classify(text):
    scores = {d: domain_score(text, d) for d in DOMAIN_KEYWORDS}
    if max(scores.values()) == 0:
        return None, scores
    return max(scores, key=scores.get), scores


def text_perplexity(model, tokenizer, text, device):
    """Perplexity of generated text under the (unsteered) model — coherence proxy."""
    if not text.strip():
        return float("nan")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    if enc["input_ids"].shape[1] < 2:
        return float("nan")
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())


def score_generations(records, model, tokenizer, hooks, device):
    """Add perplexity + domain classification to each record in-place."""
    hooks.clear_steering()  # ensure unsteered for PPL compute
    for r in records:
        text = r.get("generated", "")
        ppl = text_perplexity(model, tokenizer, text, device)
        pred, scores = domain_classify(text)
        r["perplexity"] = ppl
        r["pred_domain"] = pred
        r["domain_scores"] = scores
        # If the run had a target domain, record whether classification matches
        if "tgt_domain" in r:
            r["tgt_match"] = (pred == r["tgt_domain"])
        if "src_domain" in r:
            r["src_match"] = (pred == r["src_domain"])
    return records


# ═══════════════════════════════════════════════════════════════════
# Phase C: three-arm ablation (CAA / flat-repulsion / curvature-weighted)
# ═══════════════════════════════════════════════════════════════════
def phase_C_ablation(model, tokenizer, hooks, state, device):
    """Three-arm ablation: this is the workshop's core experiment.

    Arm 1 (caa)        : h += α · v_target           — Panickssery-style additive
    Arm 2 (repel_flat) : h -= α · v_source           — RepE-style flat repulsion
    Arm 3 (repel_curv) : h -= α · |κ_st|/|κ_ref| · v_source — BN-curvature-weighted repulsion

    Same prompts × same alphas across arms, so the only difference is the steering rule.
    """
    print("\n" + "=" * 60)
    print("PHASE C (Exp 1b): Three-arm ablation — CAA / flat-repel / curv-repel")
    print("=" * 60)

    steering = state["steering_vectors"]
    curv_table = state["curv_pair_table"]
    curv_scale = state["curv_scale"]

    # Two regimes selectable via PHASE_C_REGIME env var:
    #   "small"  -> single-layer (16), last-token, alpha in {1,3,5}
    #   "large"  -> multi-layer (8,16,24), all-position, alpha in {5,10,20}
    # Default = small (this is the regime that produced the original 11pp gap).
    import os
    regime = os.environ.get("PHASE_C_REGIME", "small")
    if regime == "small":
        alphas = [1.0, 3.0, 5.0]
        steer_layers_used = [STEER_LAYER_IDX]
        steer_layer_keys_used = [STEER_LAYER_KEY]
        all_positions_flag = False
        print(f"[phaseC] regime=small (single-layer last-token, alpha={alphas})")
    else:
        alphas = [5.0, 10.0, 20.0]
        steer_layers_used = STEER_LAYER_IDXS
        steer_layer_keys_used = STEER_LAYER_KEYS
        all_positions_flag = True
        print(f"[phaseC] regime=large (multi-layer all-position, alpha={alphas})")
    arms = ["caa", "repel_flat", "repel_curv"]
    if state.get("dir_packs_available", False):
        arms.append("repel_curv_directional")
    gen_kwargs = dict(max_new_tokens=60, do_sample=False, use_cache=True,
                      pad_token_id=tokenizer.eos_token_id)

    def unit(v):
        return v / (np.linalg.norm(v) + 1e-10)

    def per_layer_unit_vecs(domain):
        """Per-layer unit-normalized steering vectors for the active steer layers."""
        return {idx: unit(steering[domain][key])
                for idx, key in zip(steer_layers_used, steer_layer_keys_used)}

    results = []
    for src_domain in DOMAINS:
        for prompt in EVAL_PROMPTS[src_domain][:3]:
            for tgt_domain in DOMAINS:
                if tgt_domain == src_domain:
                    continue

                v_src_layers = per_layer_unit_vecs(src_domain)
                v_tgt_layers = per_layer_unit_vecs(tgt_domain)

                kappa_eff = curv_table[(src_domain, tgt_domain)]
                curv_factor = abs(kappa_eff) / (curv_scale + 1e-12)

                for arm in arms:
                    is_directional = arm == "repel_curv_directional"
                    if arm == "caa":
                        steer_vecs, mode_scale = v_tgt_layers, 1.0
                    elif arm == "repel_flat":
                        steer_vecs, mode_scale = v_src_layers, 1.0
                    elif arm == "repel_curv":
                        steer_vecs, mode_scale = v_src_layers, curv_factor
                    else:  # repel_curv_directional
                        steer_vecs, mode_scale = None, 1.0

                    src_idx = DOMAINS.index(src_domain)
                    tgt_idx = DOMAINS.index(tgt_domain)

                    for alpha in alphas:
                        if is_directional:
                            hooks.set_directional(state["dir_packs"], alpha,
                                                  src_idx=src_idx,
                                                  tgt_idx=tgt_idx,
                                                  sign=-1.0,
                                                  all_positions=all_positions_flag)
                        else:
                            hooks.set_steering(steer_vecs, alpha, mode=arm,
                                               scale=mode_scale,
                                               all_positions=all_positions_flag)
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = model.generate(**inputs, **gen_kwargs)
                        gen_text = tokenizer.decode(
                            out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True,
                        )
                        results.append({
                            "prompt": prompt,
                            "src_domain": src_domain,
                            "tgt_domain": tgt_domain,
                            "arm": arm,
                            "alpha": alpha,
                            "kappa_eff": kappa_eff,
                            "curv_factor": curv_factor,
                            "effective_alpha": alpha * mode_scale,
                            "generated": gen_text,
                        })
                        print(f"  [{src_domain[:3]}->{tgt_domain[:3]} {arm:10s} α={alpha} "
                              f"eff={alpha*mode_scale:.2f}] {gen_text[:70]}...")

    hooks.clear_steering()
    print(f"\n  Collected {len(results)} generations. Scoring...")
    score_generations(results, model, tokenizer, hooks, device)

    # Aggregate
    print("\n  === Per-arm summary (averaged over all src×tgt×α) ===")
    by_arm = {}
    for r in results:
        by_arm.setdefault(r["arm"], []).append(r)
    for arm, rs in by_arm.items():
        ppls = [r["perplexity"] for r in rs if not np.isnan(r["perplexity"])]
        tgt_acc = np.mean([r.get("tgt_match", False) for r in rs])
        src_acc = np.mean([r.get("src_match", False) for r in rs])
        print(f"    {arm:10s}: tgt_match={tgt_acc:.2%}  src_match={src_acc:.2%}  "
              f"PPL={np.mean(ppls):.1f}±{np.std(ppls):.1f}")

    print("\n  === Repulsion arms head-to-head (src_match — lower is better) ===")
    for arm in ["repel_flat", "repel_curv"]:
        if arm not in by_arm:
            continue
        for alpha in alphas:
            rs = [r for r in by_arm[arm] if r["alpha"] == alpha]
            if not rs: continue
            src_acc = np.mean([r.get("src_match", False) for r in rs])
            ppls = [r["perplexity"] for r in rs if not np.isnan(r["perplexity"])]
            print(f"    {arm:10s} α={alpha}: src_match={src_acc:.2%}  PPL={np.mean(ppls):.1f}")

    with open(OUT_DIR / "phaseC_ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"  → saved to {OUT_DIR / 'phaseC_ablation.json'}")
    return results


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="A", choices=["A", "B", "C", "all"])
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--state-file", default=None,
                    help="Override state file path (default: shadow_live_state.pt)")
    ap.add_argument("--inject-target", default="layer", choices=["layer", "mlp"],
                    help="Steering hook target: 'layer' (residual-stream output, "
                         "default) or 'mlp' (FFN submodule output, FFN-consistent "
                         "with build_state_ffn.py geometry).")
    args = ap.parse_args()

    state_path = Path(args.state_file) if args.state_file else STATE_PATH
    if not state_path.exists():
        print(f"ERROR: {state_path} missing. Transfer it from local first.")
        sys.exit(1)
    print(f"[state] using state file: {state_path}")

    print(f"[load] model = {args.model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[load] loaded in {time.time()-t0:.1f}s on {device}")
    print(f"[load] model.model.layers = {len(model.model.layers)} layers")

    state = load_state(state_path)
    print(f"[state] pipeline tag = {state.get('pipeline', 'unset (legacy state)')}")
    print(f"[state] n_queries = {state.get('n_queries', 'unknown')}")
    print(f"[state] cluster k = {state['cluster_k']}")
    print(f"[state] T_8_16 shape = {state['T_8_16'].shape}, T_16_24 = {state['T_16_24'].shape}")

    kappa_layer16, bn_adj_layer16 = compute_layer16_curvature(state)
    state["kappa_layer16"] = kappa_layer16
    state["bn_adj_layer16"] = bn_adj_layer16

    domain_dists = compute_domain_cluster_dists(state)
    state["domain_dists_layer16"] = domain_dists
    curv_table, curv_scale = build_curvature_pair_table(state, kappa_layer16, domain_dists)
    state["curv_pair_table"] = curv_table
    state["curv_scale"] = curv_scale

    # Optional: load global directional Ricci-tensor packs from
    # precompute_steering_tensors.py output, if available.
    dir_pt_path = BASE / "steering_tensors.pt"
    if dir_pt_path.exists():
        try:
            tensor_blob = torch.load(dir_pt_path, map_location="cpu",
                                     weights_only=False)
            state["dir_tensor"] = tensor_blob
            packs = load_dir_packs(state,
                                   layer_idx_to_pair={STEER_LAYER_IDX:
                                                      "layer_16__layer_24"},
                                   device=device)
            if packs:
                state["dir_packs"] = packs
                state["dir_packs_available"] = True
                print(f"[dir-tensor] loaded global Ricci-tensor packs from "
                      f"{dir_pt_path}; arms include repel_curv_directional")
        except Exception as e:
            print(f"[dir-tensor] failed to load {dir_pt_path}: {e}")

    print(f"[hook] inject_target = {args.inject_target}  "
          f"({'FFN submodule output' if args.inject_target == 'mlp' else 'layer output (residual stream)'})")
    hooks = HookManager(model, inject_target=args.inject_target)
    hooks.install()

    try:
        if args.phase in ("A", "all"):
            phase_A_sanity(model, tokenizer, hooks, state, device)
        if args.phase in ("B", "all"):
            results_B = phase_B_generation(model, tokenizer, hooks, state, device)
            score_generations(results_B, model, tokenizer, hooks, device)
            with open(OUT_DIR / "phaseB_generation_scored.json", "w") as f:
                json.dump(results_B, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, np.floating) else x)
            print(f"  → scored Phase B saved to {OUT_DIR / 'phaseB_generation_scored.json'}")
        if args.phase in ("C", "all"):
            phase_C_ablation(model, tokenizer, hooks, state, device)
    finally:
        hooks.remove()

    print("\n[done]")


if __name__ == "__main__":
    main()
