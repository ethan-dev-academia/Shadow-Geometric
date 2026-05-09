"""
Recompute FFN-output domain steering vectors and overwrite the stale
legacy residual-stream vectors currently sitting in shadow_live_state_ffn.pt.

Why: build_state_ffn.py loads steering_vectors from the legacy
shadow_live_state.pt because the original Path A architecture mixed FFN
geometry with residual-stream injection. With brev_live_steering.py now
hooking mlayers[idx].mlp (Path B, FFN-consistent), we need FFN-derived
contrastive vectors.

Math: for each (domain d, layer ℓ), the steering vector is the per-domain
mean of the FFN-output activations at the last token:

    v_d,ℓ = mean_{q ∈ D_d} h_ℓ^FFN(q)

Then the contrastive direction from source s to target t is v_t,ℓ - v_s,ℓ
(this is what brev_live_steering.py constructs internally).

In:  activations_ffn_600.pt, ffn_600_info.json, shadow_live_state_ffn.pt
Out: shadow_live_state_ffn.pt with state["steering_vectors"] replaced;
     also writes steering_vectors_ffn.pt as a separate artifact.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).parent.parent
ACTS_PT = BASE / "activations_ffn_600.pt"
INFO_JSON = BASE / "ffn_600_info.json"
STATE_PT = BASE / "shadow_live_state_ffn.pt"
SIDECAR_PT = BASE / "steering_vectors_ffn.pt"

DOMAINS_PAPER = ["Medical", "Financial", "Code"]
DOMAINS_DATA = ["medical", "financial", "code"]

t0 = time.time()
def st(m): print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)


def main():
    st("Loading FFN activations + info + state...")
    acts_raw = torch.load(str(ACTS_PT), map_location="cpu", weights_only=False)
    state = torch.load(str(STATE_PT), map_location="cpu", weights_only=False)
    with open(INFO_JSON) as f:
        info = json.load(f)
    LAYER_KEYS = state["layer_keys"]
    domain_labels = info["domains"]
    domain_int = np.array([DOMAINS_DATA.index(d) for d in domain_labels])

    acts = {lk: (acts_raw[lk].numpy() if torch.is_tensor(acts_raw[lk])
                 else acts_raw[lk])
            for lk in LAYER_KEYS}

    # Per-domain centroid at each layer (FFN-output, 4096-D, last token).
    new_steering = {d_paper: {} for d_paper in DOMAINS_PAPER}
    for di, (d_paper, d_data) in enumerate(zip(DOMAINS_PAPER, DOMAINS_DATA)):
        m = (domain_int == di)
        st(f"  domain={d_paper}: n={int(m.sum())}")
        for lk in LAYER_KEYS:
            mu = acts[lk][m].mean(axis=0).astype(np.float32)  # (4096,)
            new_steering[d_paper][lk] = mu
            st(f"    {lk}: ‖μ‖={np.linalg.norm(mu):.3f}")

    # Show the FFN-output discriminability we'll be working with.
    st("FFN-output domain centroid pairwise cosines per layer:")
    for lk in LAYER_KEYS:
        mu_M = new_steering["Medical"][lk]
        mu_F = new_steering["Financial"][lk]
        mu_C = new_steering["Code"][lk]
        def cos(a, b):
            return float(np.dot(a, b) /
                         (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        st(f"  {lk}: Med·Fin={cos(mu_M, mu_F):+.3f}  "
           f"Med·Cod={cos(mu_M, mu_C):+.3f}  "
           f"Fin·Cod={cos(mu_F, mu_C):+.3f}")

    # Sidecar artifact (independent of state).
    torch.save(new_steering, str(SIDECAR_PT))
    st(f"Wrote sidecar -> {SIDECAR_PT.name}")

    # In-place state update.
    state["steering_vectors"] = new_steering
    state["steering_vectors_source"] = "FFN-output domain centroids (recomputed)"
    torch.save(state, str(STATE_PT))
    st(f"Updated state -> {STATE_PT.name}")
    print(f"\n[DONE] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
