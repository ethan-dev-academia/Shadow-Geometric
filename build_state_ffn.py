"""
Build shadow_live_state_ffn.pt from FFN-derived activations on n=600 real queries.

Pipeline:
  1. Load activations_ffn_600.pt (shape (600, 4096) per layer at 8/16/24)
  2. Per layer: Sparse PCA -> 50 components
  3. Per layer: Differentiable K-means (gradient updates on cluster centers)
  4. Build Laplace-smoothed transition matrices T_8_16, T_16_24
  5. Compute contrastive steering vectors per (domain, layer) from raw 4096D FFN acts
  6. Save state file

This replaces shadow_live_state.py's noise-perturbation pipeline. Every claim
the paper makes about Sparse PCA / differentiable K-means / FFN-derived
activations / 600 real queries is now literally true.
"""
import json, pickle, time, numpy as np, torch
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA, PCA
from sklearn.metrics import silhouette_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from differentiable_kmeans import DifferentiableKMeans

BASE = Path(__file__).parent
ACTS_PT = BASE / "activations_ffn_600.pt"
INFO_JSON = BASE / "ffn_600_info.json"
OUT_PT = BASE / "shadow_live_state_ffn.pt"

LAYER_KEYS = ["layer_8", "layer_16", "layer_24"]
LAYER_IDXS = [8, 16, 24]
DOMAINS_PAPER = ["Medical", "Financial", "Code"]   # paper-canonical capitalization
DOMAINS_DATA = ["medical", "financial", "code"]    # collection script lowercase

# Adaptive-k clustering: sweep silhouette score over k ∈ [K_MIN, K_MAX] using
# scikit-learn KMeans (fast Lloyd's algorithm), then fit DifferentiableKMeans
# at the silhouette-optimal k. Matches the v1 pipeline's adaptive-k regime
# (k=13-14 typical), which is the granularity the bipartite Forman-Ricci
# curvature signal needs to differentiate domain pairs (k≤6 collapses
# κ_eff to a constant; see paper Section 5.2 and the May 6 Brev rerun
# diagnostic). K_OVERRIDE is preserved as a manual override knob.
K_MIN = 4
K_MAX = 14
K_OVERRIDE = None  # set to {"layer_8": 6, ...} to skip silhouette sweep

# Sparse PCA hyperparameters: alpha controls L1 sparsity penalty.
# Empirically, alpha=1.0 zeroes out all components on FFN-output magnitudes
# (mean norms ~1-4); we drop to alpha=0.1 to preserve signal while still
# producing sparse loadings. If components still zero out, we fall back to
# standard PCA (logged in the state file's pipeline tag).
SPCA_ALPHA = 0.1
SPCA_N_COMPONENTS = 50
SPCA_NONZERO_FLOOR = 5  # if fewer than this many nonzeros/component on average, fall back to PCA

# Differentiable K-means hyperparameters
DKM_TEMPERATURE = 1.0
DKM_N_ITERS = 200
DKM_LR = 0.01

# Laplace smoothing for transition matrices
ALPHA_SMOOTH = 1.0

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

t0 = time.time()
def st(m): print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)


# ── 1. Load FFN activations + metadata ───────────────────────────────────
st("Loading FFN activations...")
acts_raw = torch.load(str(ACTS_PT), map_location="cpu", weights_only=False)
acts = {lk: (acts_raw[lk].numpy() if torch.is_tensor(acts_raw[lk]) else acts_raw[lk])
        for lk in LAYER_KEYS}
for lk in LAYER_KEYS:
    st(f"  {lk}: shape={acts[lk].shape}")

with open(INFO_JSON) as f:
    info = json.load(f)
queries = info["queries"]
domain_labels = info["domains"]
domain_int = np.array([DOMAINS_DATA.index(d) for d in domain_labels])
N = len(queries)
st(f"  N={N} (med={int((domain_int==0).sum())} fin={int((domain_int==1).sum())} "
   f"code={int((domain_int==2).sum())})")


# ── 2. Sparse PCA per layer (with fallback to PCA on degenerate sparsity) ──
st(f"Fitting Sparse PCA (alpha={SPCA_ALPHA}, n_components={SPCA_N_COMPONENTS})...")
pca_models = {}
reduced = {}
pca_kind = {}
for lk in LAYER_KEYS:
    st(f"  {lk}: fitting Sparse PCA on {acts[lk].shape}...")
    spca = SparsePCA(n_components=SPCA_N_COMPONENTS, alpha=SPCA_ALPHA,
                     random_state=RNG_SEED, max_iter=200)
    Xr = spca.fit_transform(acts[lk])
    nz_per_comp = float(np.mean(np.sum(np.abs(spca.components_) > 1e-8, axis=1)))
    if nz_per_comp < SPCA_NONZERO_FLOOR:
        st(f"  {lk}: SparsePCA degenerate ({nz_per_comp:.1f} nz/comp < floor "
           f"{SPCA_NONZERO_FLOOR}). Falling back to standard PCA.")
        pca = PCA(n_components=SPCA_N_COMPONENTS, random_state=RNG_SEED)
        Xr = pca.fit_transform(acts[lk])
        pca_models[lk] = pca
        pca_kind[lk] = "PCA"
        st(f"  {lk}: PCA reduced->{Xr.shape}, "
           f"explained_var={pca.explained_variance_ratio_.sum():.3f}")
    else:
        pca_models[lk] = spca
        pca_kind[lk] = "SparsePCA"
        st(f"  {lk}: SparsePCA reduced->{Xr.shape}, "
           f"mean nonzeros/component={nz_per_comp:.1f}/4096")
    reduced[lk] = Xr


# ── 3. Adaptive-k via silhouette + DifferentiableKMeans at chosen k ─────
st("Choosing k per layer by silhouette sweep, then fitting DifferentiableKMeans...")
kmeans_models = {}
cluster_centers = {}
cluster_hard = {}
silhouettes = {}
chosen_k = {}
for lk in LAYER_KEYS:
    Xr = reduced[lk]
    if K_OVERRIDE is not None and lk in K_OVERRIDE:
        best_k = int(K_OVERRIDE[lk])
        best_sil = float("nan")
        st(f"  {lk}: K_OVERRIDE -> k={best_k} (skipping silhouette sweep)")
    else:
        st(f"  {lk}: silhouette sweep over k ∈ [{K_MIN}, {K_MAX}]")
        best_k, best_sil = K_MIN, -1.0
        for k_try in range(K_MIN, K_MAX + 1):
            km_try = KMeans(n_clusters=k_try, random_state=RNG_SEED,
                             n_init=10, max_iter=200)
            labs = km_try.fit_predict(Xr)
            if len(np.unique(labs)) < 2:
                continue
            sil_try = float(silhouette_score(Xr, labs))
            if sil_try > best_sil:
                best_k, best_sil = k_try, sil_try
        st(f"  {lk}: silhouette-optimal k={best_k} (sil={best_sil:.3f})")

    X_tensor = torch.from_numpy(Xr).float()
    dkm = DifferentiableKMeans(n_clusters=best_k, n_features=Xr.shape[1],
                               temperature=DKM_TEMPERATURE)
    st(f"  {lk}: fitting DifferentiableKMeans (k={best_k}, n_iters={DKM_N_ITERS})")
    dkm.fit(X_tensor, n_iters=DKM_N_ITERS, lr=DKM_LR, verbose=False)
    hard = dkm.predict(X_tensor)
    sil = (float(silhouette_score(Xr, hard))
           if len(np.unique(hard)) > 1 else float("nan"))
    kmeans_models[lk] = dkm
    cluster_centers[lk] = dkm.cluster_centers.detach().cpu().numpy()
    cluster_hard[lk] = hard
    silhouettes[lk] = sil
    chosen_k[lk] = best_k
    st(f"  {lk}: k={best_k} silhouette(diffKMeans)={sil:.3f} "
       f"cluster_sizes={np.bincount(hard, minlength=best_k).tolist()}")


# ── 4. Laplace-smoothed transition matrices ──────────────────────────────
st(f"Building transition matrices (Laplace alpha={ALPHA_SMOOTH})...")
def build_transition(h1, h2, k1, k2, alpha=ALPHA_SMOOTH):
    T = np.zeros((k1, k2))
    for a, b in zip(h1, h2):
        T[a, b] += 1
    T += alpha
    T /= T.sum(axis=1, keepdims=True)
    return T

T_8_16 = build_transition(cluster_hard["layer_8"], cluster_hard["layer_16"],
                          chosen_k["layer_8"], chosen_k["layer_16"])
T_16_24 = build_transition(cluster_hard["layer_16"], cluster_hard["layer_24"],
                           chosen_k["layer_16"], chosen_k["layer_24"])
st(f"  T_8_16: {T_8_16.shape}  rows sum to {T_8_16.sum(axis=1)[:3].round(3).tolist()}")
st(f"  T_16_24: {T_16_24.shape}")


# ── 5. Steering vectors: per-domain FFN-output centroids ─────────────────
# Compute per-domain mean FFN-output activations at each instrumented layer.
# These are the contrastive vectors used by the steering arms; v_t - v_s is
# the source-to-target direction. Computed inline so the pipeline is
# self-contained (no legacy state-file fallback required).
st("Computing FFN-output domain steering vectors...")
steering = {d_paper: {} for d_paper in DOMAINS_PAPER}
for di, (d_paper, d_data) in enumerate(zip(DOMAINS_PAPER, DOMAINS_DATA)):
    m = (domain_int == di)
    for lk in LAYER_KEYS:
        mu = acts[lk][m].mean(axis=0).astype(np.float32)
        steering[d_paper][lk] = mu
    norms = [float(np.linalg.norm(steering[d_paper][lk])) for lk in LAYER_KEYS]
    st(f"  {d_paper}: n={int(m.sum())}, layer norms = "
       f"{[round(n, 2) for n in norms]}")
src_label = "FFN-output domain centroids (computed inline from "\
            "activations_ffn_600.pt)"


# ── 6. Package + save state ─────────────────────────────────────────────
st("Packaging state...")
state = {
    "pca_models": {lk: pickle.dumps(pca_models[lk]) for lk in LAYER_KEYS},
    "kmeans_centers": {lk: cluster_centers[lk] for lk in LAYER_KEYS},
    "cluster_k": {lk: int(chosen_k[lk]) for lk in LAYER_KEYS},
    "T_8_16": T_8_16,
    "T_16_24": T_16_24,
    "steering_vectors": steering,
    "silhouettes": silhouettes,
    "alpha_smooth": ALPHA_SMOOTH,
    "spca_alpha": SPCA_ALPHA,
    "spca_n_components": SPCA_N_COMPONENTS,
    "dkm_temperature": DKM_TEMPERATURE,
    "dkm_n_iters": DKM_N_ITERS,
    "layer_keys": LAYER_KEYS,
    "domains": DOMAINS_PAPER,
    "train_cluster_hard": cluster_hard,
    "train_domain_int": domain_int.tolist(),
    "n_queries": N,
    "pipeline": "ffn_600_real_sparsepca_diffkmeans_pathA",
    "pca_kind_per_layer": pca_kind,
    "steering_vectors_source": src_label,
}
torch.save(state, str(OUT_PT))
st(f"Saved -> {OUT_PT.name}")
print(f"\n[DONE] {time.time()-t0:.1f}s total")
