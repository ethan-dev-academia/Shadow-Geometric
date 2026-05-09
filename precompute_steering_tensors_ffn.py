"""
Pre-compute the global directional Ricci-tensor (U, Λ) for the FFN pipeline.

Adapts precompute_steering_tensors.py to consume the FFN state file directly
instead of the v1 residual-stream artifacts. Computes bipartite augmented
Forman-Ricci κ on the cross-layer cluster-transition graph from scratch
(no external kdict required).

Math (paper Section 4.8):

    M(c, c') = κ(c, c') · [(μ_c' − μ_c)(μ_c' − μ_c)ᵀ + ½(Σ_c + Σ_c')]
    T_{ℓ→ℓ'} = Σ_e (N_e / Σ N) · M_e

Top-D=20 eigendecomposition by magnitude with sign-preserving normalization
σ(λ) = sign(λ)·|λ|/max_j|λ_j|.

Inputs:
    activations_ffn_600.pt    raw 4096-D FFN-output activations per layer
    shadow_live_state_ffn.pt  cluster_hard per layer + cluster_k
    ffn_600_info.json         per-query domain labels

Output:
    steering_tensors.pt — torch dict
        {
          "layer_16__layer_24": {
              "U":          float32 (4096, D)
              "lam":        float32 (D,)
              "lam_norm":   float32 (D,)
              "v_contrast": float32 (3, 3, 4096)
              "mu_dom":     float32 (3, 4096)
              "domains":    list[str]
              "D":          int
              "src_layer":  str
              "tgt_layer":  str
          }
        }

brev_live_steering.py auto-picks up steering_tensors.pt if present and
enables the directional (tensor) steering arm in Phase C.

Run on H100 or CPU (top-20 eigh via Lanczos is ~30-60s either way):

    python precompute_steering_tensors_ffn.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import eigsh, LinearOperator

BASE = Path(__file__).parent
ACTS_PT = BASE / "activations_ffn_600.pt"
STATE_PT = BASE / "shadow_live_state_ffn.pt"
INFO_JSON = BASE / "ffn_600_info.json"
OUT_PT = BASE / "steering_tensors.pt"

DOMAINS_PAPER = ["Medical", "Financial", "Code"]
DOMAINS_DATA = ["medical", "financial", "code"]
PAIRS = [("layer_16", "layer_24")]   # steering injects at layer 16
D = 20

t0 = time.time()
def st(m): print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)


def bipartite_forman_ricci(counts: np.ndarray) -> dict:
    """Augmented Forman-Ricci on a bipartite cross-layer cluster-transition
    graph.

    counts : (K_src, K_tgt) integer joint counts N(c, c').
    Returns kdict mapping "src_c__tgt_c" -> kappa(c, c').

    Formula (Sreejith 2016, augmented Forman):
        κ(i,j) = w_ij · (2 - Σ_{k≠j: w_ik>0} w_ik/max(w_ij, w_ik)
                          - Σ_{k≠i: w_jk>0} w_jk/max(w_ij, w_jk))
    where on bipartite graph, neighbors of i ∈ V_src lie in V_tgt and vice
    versa.
    """
    K_src, K_tgt = counts.shape
    kdict = {}
    for u in range(K_src):
        for v in range(K_tgt):
            w_uv = float(counts[u, v])
            if w_uv <= 0: continue
            # neighbors of u (∈ V_src) are in V_tgt: clusters w in V_tgt with counts[u, w] > 0
            sum_u = 0.0
            for w in range(K_tgt):
                if w == v: continue
                w_uw = float(counts[u, w])
                if w_uw <= 0: continue
                sum_u += w_uw / max(w_uv, w_uw)
            # neighbors of v (∈ V_tgt) are in V_src: clusters x in V_src with counts[x, v] > 0
            sum_v = 0.0
            for x in range(K_src):
                if x == u: continue
                w_xv = float(counts[x, v])
                if w_xv <= 0: continue
                sum_v += w_xv / max(w_uv, w_xv)
            kappa = w_uv * (2.0 - sum_u - sum_v)
            kdict[f"layer_src_{u}__layer_tgt_{v}"] = kappa
    return kdict


def build_tensor(H_src, H_tgt, hard_src, hard_tgt, K_src, K_tgt, kdict):
    """Construct global T_{ℓ→ℓ'} as low-rank-plus-diagonal LinearOperator
    and return its top-D eigenpairs.
    """
    n, d = H_src.shape

    mu_s = np.zeros((K_src, d)); var_s = np.zeros((K_src, d))
    for c in range(K_src):
        m = hard_src == c
        if m.sum() < 2: continue
        mu_s[c] = H_src[m].mean(axis=0)
        var_s[c] = H_src[m].var(axis=0)

    mu_t = np.zeros((K_tgt, d)); var_t = np.zeros((K_tgt, d))
    for c in range(K_tgt):
        m = hard_tgt == c
        if m.sum() < 2: continue
        mu_t[c] = H_tgt[m].mean(axis=0)
        var_t[c] = H_tgt[m].var(axis=0)

    counts = np.zeros((K_src, K_tgt))
    for a, b in zip(hard_src, hard_tgt):
        counts[a, b] += 1
    total = counts.sum()
    if total < 1:
        raise RuntimeError("empty bipartite transition graph")

    rank1_dirs = []
    rank1_coefs = []
    diag = np.zeros(d)
    n_edges_used = 0
    for k_str, kappa in kdict.items():
        # parse "layer_src_U__layer_tgt_V"
        parts = k_str.split("__")
        u = int(parts[0].split("_")[-1])
        v = int(parts[1].split("_")[-1])
        if u >= K_src or v >= K_tgt: continue
        n_e = counts[u, v]
        if n_e < 1: continue
        w_e = float(n_e / total)
        d_vec = mu_t[v] - mu_s[u]
        if np.dot(d_vec, d_vec) < 1e-9: continue
        rank1_dirs.append(d_vec)
        rank1_coefs.append(w_e * float(kappa))
        diag += w_e * float(kappa) * 0.5 * (var_s[u] + var_t[v])
        n_edges_used += 1
    if not rank1_dirs:
        raise RuntimeError("no usable edges in kdict")

    Drank = np.stack(rank1_dirs, axis=1).astype(np.float64)   # (d, E)
    coefs = np.array(rank1_coefs, dtype=np.float64)            # (E,)

    def mv(x):
        return Drank @ (coefs * (Drank.T @ x)) + diag * x

    op = LinearOperator((d, d), matvec=mv, dtype=np.float64)
    st(f"  eigsh on {d}-D LinearOperator (E={n_edges_used} edges)...")
    lam, U = eigsh(op, k=D, which="LM")
    order = np.argsort(-np.abs(lam))
    lam = lam[order]; U = U[:, order]
    max_abs = float(np.abs(lam).max()) + 1e-12
    lam_norm = np.sign(lam) * (np.abs(lam) / max_abs)
    return (U.astype(np.float32),
            lam.astype(np.float32),
            lam_norm.astype(np.float32),
            mu_s)


def per_domain_centroids(H, domain_int, n_dom):
    n, d = H.shape
    out = np.zeros((n_dom, d), dtype=np.float64)
    for di in range(n_dom):
        m = domain_int == di
        if m.sum() < 1: continue
        out[di] = H[m].mean(axis=0)
    return out


def main():
    st(f"Loading FFN activations + state...")
    acts_raw = torch.load(str(ACTS_PT), map_location="cpu", weights_only=False)
    state = torch.load(str(STATE_PT), map_location="cpu", weights_only=False)
    with open(INFO_JSON) as f:
        info = json.load(f)

    LAYER_KEYS = state["layer_keys"]
    cluster_k = state["cluster_k"]
    cluster_hard = {lk: np.asarray(state["train_cluster_hard"][lk])
                    for lk in LAYER_KEYS}
    acts = {lk: (acts_raw[lk].numpy() if torch.is_tensor(acts_raw[lk])
                 else acts_raw[lk])
            for lk in LAYER_KEYS}

    domain_labels = info["domains"]
    domain_int = np.array([DOMAINS_DATA.index(d) for d in domain_labels])
    st(f"  layers: {LAYER_KEYS}")
    st(f"  cluster_k: {cluster_k}")
    st(f"  N={len(domain_int)} (med={int((domain_int==0).sum())} "
       f"fin={int((domain_int==1).sum())} code={int((domain_int==2).sum())})")

    blob = {}
    for src_lk, tgt_lk in PAIRS:
        K_src, K_tgt = cluster_k[src_lk], cluster_k[tgt_lk]
        H_src = acts[src_lk].astype(np.float64)
        H_tgt = acts[tgt_lk].astype(np.float64)
        hard_src, hard_tgt = cluster_hard[src_lk], cluster_hard[tgt_lk]

        # bipartite joint counts -> bipartite Forman-Ricci κ
        counts = np.zeros((K_src, K_tgt))
        for a, b in zip(hard_src, hard_tgt):
            counts[a, b] += 1
        st(f"[{src_lk}→{tgt_lk}] joint-count matrix shape={counts.shape}, "
           f"nonzero edges={int((counts > 0).sum())}/{K_src*K_tgt}")
        kdict = bipartite_forman_ricci(counts)
        kappa_vals = np.array(list(kdict.values()))
        st(f"  bipartite κ stats: n_edges={len(kdict)}  "
           f"mean|κ|={np.abs(kappa_vals).mean():.3f}  "
           f"%neg={(kappa_vals < 0).mean()*100:.1f}%  "
           f"range=[{kappa_vals.min():.2f}, {kappa_vals.max():.2f}]")

        # build tensor + top-D eigendecomposition
        st(f"  building T_{{{src_lk}→{tgt_lk}}} and top-{D} eigendecomposition...")
        U, lam, lam_norm, _ = build_tensor(
            H_src, H_tgt, hard_src, hard_tgt, K_src, K_tgt, kdict)

        mu_dom = per_domain_centroids(H_src, domain_int, len(DOMAINS_PAPER))
        v_contrast = mu_dom[None, :, :] - mu_dom[:, None, :]

        pair_key = f"{src_lk}__{tgt_lk}"
        blob[pair_key] = {
            "U":          U,
            "lam":        lam,
            "lam_norm":   lam_norm,
            "v_contrast": v_contrast.astype(np.float32),
            "mu_dom":     mu_dom.astype(np.float32),
            "domains":    DOMAINS_PAPER,
            "D":          D,
            "src_layer":  src_lk,
            "tgt_layer":  tgt_lk,
            "kappa_stats": {
                "mean_abs": float(np.abs(kappa_vals).mean()),
                "pct_neg": float((kappa_vals < 0).mean() * 100),
                "min": float(kappa_vals.min()),
                "max": float(kappa_vals.max()),
                "n_edges": int(len(kdict)),
            },
        }
        st(f"  |λ| range [{abs(lam).min():.3g}, {abs(lam).max():.3g}]   "
           f"sign +{(lam > 0).sum()} / -{(lam < 0).sum()}")

    torch.save(blob, str(OUT_PT))
    sz = OUT_PT.stat().st_size / (1024 * 1024)
    st(f"Saved {len(blob)} pair tensor(s) -> {OUT_PT.name}  ({sz:.1f} MB)")
    print(f"\n[DONE] {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
