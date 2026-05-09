"""
Shadow pipeline (FFN variant) — n=600 real queries on Mistral-7B-Instruct-v0.2.

Consumes the FFN-derived state produced by build_state_ffn.py (Sparse PCA +
Differentiable K-means + transition matrices) and runs the analysis stages
that the workshop paper actually reports:

  Stage 3. Cross-layer Bayesian network: transitions, mutual information,
           permutation tests, Kruskal-Wallis on trajectory log-likelihood.
  Stage 4. Meta-learning router (MLP, two heads: domain + pathway).
  Stage 5. Procrustes alignment (layer-to-layer + domain-to-domain).
  Stage 6. Logit calibration (temperature scaling, ECE/NLL/Wilcoxon).

Stages 1-2 (Sparse PCA + Differentiable K-means) are skipped here because
build_state_ffn.py performs them on FFN-output activations and persists the
fitted models in shadow_live_state_ffn.pt.

Inputs:
  shadow_live_state_ffn.pt   (from build_state_ffn.py)
  activations_ffn_600.pt     (from collect_ffn_600.py; raw 4096D FFN outputs)
  ffn_600_info.json          (queries + domain labels)

Output:
  shadow_v3_output/n600_ffn/pipeline_ffn_stats.json
  shadow_v3_output/n600_ffn/pipeline_arrays.npz
"""
import json, pickle, time, numpy as np, torch
from pathlib import Path
from collections import Counter
from scipy import stats as sp_stats
from scipy.spatial import procrustes
from scipy.stats import mannwhitneyu, kruskal
from sklearn.metrics import (silhouette_score, mutual_info_score,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).parent.parent
OUT = BASE / "shadow_v3_output" / "n600_ffn"
OUT.mkdir(parents=True, exist_ok=True)

STATE_PT = BASE / "shadow_live_state_ffn.pt"
ACTS_PT = BASE / "activations_ffn_600.pt"
INFO_JSON = BASE / "ffn_600_info.json"

DOMAINS = ["medical", "financial", "code"]
RNG = np.random.RandomState(42)
torch.manual_seed(42)

t0 = time.time()
def st(m): print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)
def pfmt(p):
    if p < 0.001: return "p<0.001"
    if p < 0.01:  return f"p={p:.4f}"
    return f"p={p:.3f}"

# ────────────────────────────────────────────────────────────────────
# 0. LOAD FFN STATE + RAW ACTS
# ────────────────────────────────────────────────────────────────────
st("Loading FFN state + raw FFN activations...")
state = torch.load(str(STATE_PT), map_location="cpu", weights_only=False)
LAYER_KEYS = state["layer_keys"]
cluster_k = state["cluster_k"]
cluster_centers = state["kmeans_centers"]
pca_kind = state.get("pca_kind_per_layer", {lk: "SparsePCA" for lk in LAYER_KEYS})
spca_alpha = state.get("spca_alpha", 0.1)
dkm_temp = state.get("dkm_temperature", 1.0)
st(f"  layers: {LAYER_KEYS}")
st(f"  cluster_k: {cluster_k}")
st(f"  PCA kind per layer: {pca_kind}")
st(f"  Sparse PCA alpha={spca_alpha}, DiffKMeans T={dkm_temp}")

# Train cluster assignments + reduced data come from the state file when
# present; otherwise reconstruct from raw FFN acts via the saved PCA models.
acts_raw = torch.load(str(ACTS_PT), map_location="cpu", weights_only=False)
acts = {lk: (acts_raw[lk].numpy() if torch.is_tensor(acts_raw[lk]) else acts_raw[lk])
        for lk in LAYER_KEYS}

with open(INFO_JSON) as f:
    info = json.load(f)
queries = info["queries"]
domain_labels = info["domains"]
domain_int = np.array([DOMAINS.index(d) for d in domain_labels])
N = len(queries)
st(f"  N={N}")

# Recover reduced features by re-applying the saved Sparse PCA / PCA models
reduced = {}
for lk in LAYER_KEYS:
    pca = pickle.loads(state["pca_models"][lk])
    reduced[lk] = pca.transform(acts[lk])
    st(f"  {lk}: reduced -> {reduced[lk].shape} ({pca_kind[lk]})")

# Hard cluster labels from state file if available; otherwise compute from
# reduced features against the saved Differentiable K-means centroids.
if "train_cluster_hard" in state:
    cluster_hard = {lk: np.asarray(state["train_cluster_hard"][lk])
                    for lk in LAYER_KEYS}
else:
    cluster_hard = {}
    for lk in LAYER_KEYS:
        C = cluster_centers[lk]
        d2 = ((reduced[lk][:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        cluster_hard[lk] = d2.argmin(axis=1)

# Soft (differentiable K-means) assignments via temperature-scaled softmax
# on negative squared Euclidean distance — matches DifferentiableKMeans.predict_soft
cluster_soft = {}
silhouettes = {}
for lk in LAYER_KEYS:
    C = cluster_centers[lk]
    d2 = ((reduced[lk][:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / max(dkm_temp, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    cluster_soft[lk] = p / p.sum(axis=1, keepdims=True)
    silhouettes[lk] = (float(silhouette_score(reduced[lk], cluster_hard[lk]))
                       if len(np.unique(cluster_hard[lk])) > 1 else float("nan"))
    st(f"  {lk}: k={cluster_k[lk]}, silhouette={silhouettes[lk]:.3f}")

domain_sil = {lk: float(silhouette_score(reduced[lk], domain_int))
              for lk in LAYER_KEYS}

# ────────────────────────────────────────────────────────────────────
# 3. Cross-layer Bayesian network
# ────────────────────────────────────────────────────────────────────
st("[STAGE 3] Cross-layer Bayesian network...")

def build_transition(h1, h2, k1, k2):
    T = np.zeros((k1, k2))
    for a, b in zip(h1, h2):
        T[a, b] += 1
    rs = T.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return T / rs

mi_vals, mi_p = {}, {}
trans_mats = {}
pairs = [(LAYER_KEYS[i], LAYER_KEYS[j])
         for i in range(len(LAYER_KEYS)) for j in range(i+1, len(LAYER_KEYS))]
for l1, l2 in pairs:
    h1, h2 = cluster_hard[l1], cluster_hard[l2]
    key = f"{l1.split('_')[1]}_{l2.split('_')[1]}"
    trans_mats[key] = build_transition(h1, h2, cluster_k[l1], cluster_k[l2])
    obs = mutual_info_score(h1, h2)
    mi_vals[key] = float(obs)
    ct = sum(1 for _ in range(1000)
             if mutual_info_score(h1, RNG.permutation(h2)) >= obs)
    mi_p[key] = ct / 1000
    st(f"  MI {key}: {obs:.3f}  ({pfmt(mi_p[key])})")

trajectories = np.column_stack([cluster_hard[lk] for lk in LAYER_KEYS])
traj_tup = [tuple(r) for r in trajectories]
traj_counts = Counter(traj_tup)
traj_probs = {t: c / N for t, c in traj_counts.items()}
traj_scores = np.array([np.log(traj_probs[t] + 1e-10) for t in traj_tup])

dom_traj = {d: traj_scores[domain_int == di] for di, d in enumerate(DOMAINS)}
kw_H, kw_p = kruskal(*[dom_traj[d] for d in DOMAINS])
st(f"  Kruskal-Wallis: H={kw_H:.2f}, {pfmt(kw_p)}")

mw_p = {}
for i in range(len(DOMAINS)):
    for j in range(i+1, len(DOMAINS)):
        _, p = mannwhitneyu(dom_traj[DOMAINS[i]], dom_traj[DOMAINS[j]])
        mw_p[f"{DOMAINS[i]}_v_{DOMAINS[j]}"] = float(p)

# ────────────────────────────────────────────────────────────────────
# 4. Meta-learning router (MLP, domain + pathway heads)
# ────────────────────────────────────────────────────────────────────
st("[STAGE 4] Meta-learning router...")

feat = np.hstack([cluster_soft[lk] for lk in LAYER_KEYS]
                 + [traj_scores.reshape(-1, 1)]).astype(np.float32)
feat_dim = feat.shape[1]

# pathway labels via K-means on the trajectory tuples (np.float32 for clustering)
from sklearn.cluster import KMeans
pw_km = KMeans(n_clusters=6, random_state=42, n_init=10)
pw_labels = pw_km.fit_predict(trajectories)

idx_all = np.arange(N)
RNG.shuffle(idx_all)
split = int(0.8 * N)
tr_idx, te_idx = idx_all[:split], idx_all[split:]

Xtr = torch.tensor(feat[tr_idx]); Xte = torch.tensor(feat[te_idx])
Ydtr = torch.tensor(domain_int[tr_idx], dtype=torch.long)
Ydte = torch.tensor(domain_int[te_idx], dtype=torch.long)
Yptr = torch.tensor(pw_labels[tr_idx], dtype=torch.long)
Ypte = torch.tensor(pw_labels[te_idx], dtype=torch.long)

class Router(nn.Module):
    def __init__(self, din, nd=3, np_=6):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(din, 96), nn.ReLU(), nn.Dropout(0.2),
                                    nn.Linear(96, 48), nn.ReLU())
        self.hd = nn.Linear(48, nd)
        self.hp = nn.Linear(48, np_)
    def forward(self, x):
        h = self.trunk(x)
        return self.hd(h), self.hp(h)

torch.manual_seed(0)
router = Router(feat_dim, nd=len(DOMAINS), np_=6)
opt = torch.optim.Adam(router.parameters(), lr=0.004, weight_decay=1e-4)
router.train()
for ep in range(400):
    d_out, p_out = router(Xtr)
    loss = F.cross_entropy(d_out, Ydtr) + 0.5 * F.cross_entropy(p_out, Yptr)
    opt.zero_grad(); loss.backward(); opt.step()

router.eval()
with torch.no_grad():
    d_logits, p_logits = router(Xte)
d_pred = d_logits.argmax(1).numpy()
p_pred = p_logits.argmax(1).numpy()
dom_acc = float((d_pred == Ydte.numpy()).mean())
pw_acc = float((p_pred == Ypte.numpy()).mean())
cm = confusion_matrix(Ydte.numpy(), d_pred)
router_p = float(sp_stats.binomtest(int(dom_acc * len(Ydte)), len(Ydte),
                                     1/len(DOMAINS), alternative="greater").pvalue)
st(f"  Router: dom={dom_acc:.1%} ({pfmt(router_p)}), pathway={pw_acc:.1%}")

# ────────────────────────────────────────────────────────────────────
# 5. Procrustes alignment
# ────────────────────────────────────────────────────────────────────
st("[STAGE 5] Procrustes alignment...")

def procrustes_permtest(A, B, n_perm=500):
    _, _, disp = procrustes(A, B)
    ct = 0
    n = A.shape[0]
    for _ in range(n_perm):
        perm = RNG.permutation(n)
        _, _, dp = procrustes(A, B[perm])
        if dp <= disp:
            ct += 1
    return float(disp), ct / n_perm

layer_proc = {}
for i in range(len(LAYER_KEYS)):
    for j in range(i+1, len(LAYER_KEYS)):
        l1, l2 = LAYER_KEYS[i], LAYER_KEYS[j]
        A = reduced[l1][:, :20]
        B = reduced[l2][:, :20]
        disp, p = procrustes_permtest(A, B, n_perm=500)
        layer_proc[f"{l1.split('_')[1]}_{l2.split('_')[1]}"] = {
            "disparity": disp, "p": p}
        st(f"  {l1}↔{l2}: disparity={disp:.4f}  ({pfmt(p)})")

dom_proc = {}
for i in range(len(DOMAINS)):
    for j in range(i+1, len(DOMAINS)):
        d1, d2 = DOMAINS[i], DOMAINS[j]
        A = np.hstack([reduced[lk][domain_int == i, :10] for lk in LAYER_KEYS])
        B = np.hstack([reduced[lk][domain_int == j, :10] for lk in LAYER_KEYS])
        n = min(A.shape[0], B.shape[0])
        disp, p = procrustes_permtest(A[:n], B[:n], n_perm=500)
        dom_proc[f"{d1}_{d2}"] = {"disparity": disp, "p": p}
        st(f"  {d1}↔{d2} (cross-layer): disparity={disp:.4f}  ({pfmt(p)})")

# ────────────────────────────────────────────────────────────────────
# 6. Calibration (temperature scaling)
# ────────────────────────────────────────────────────────────────────
st("[STAGE 6] Logit calibration (temperature scaling)...")

X_down = np.hstack([reduced[lk] for lk in LAYER_KEYS])
X_down_tr = X_down[tr_idx]; X_down_te = X_down[te_idx]
y_tr = domain_int[tr_idx]; y_te = domain_int[te_idx]

lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(X_down_tr, y_tr)
raw_logits = lr.decision_function(X_down_te)
raw_probs = np.exp(raw_logits - raw_logits.max(axis=1, keepdims=True))
raw_probs /= raw_probs.sum(axis=1, keepdims=True)

val_slice = int(0.2 * len(tr_idx))
val_idx = tr_idx[:val_slice]
val_logits = lr.decision_function(X_down[val_idx])
val_y = domain_int[val_idx]

def temp_nll(T, logits, y):
    sc = logits / T
    sc -= sc.max(axis=1, keepdims=True)
    p = np.exp(sc); p /= p.sum(axis=1, keepdims=True)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))

T_grid = np.linspace(0.3, 5.0, 60)
nlls = [temp_nll(T, val_logits, val_y) for T in T_grid]
T_best = float(T_grid[int(np.argmin(nlls))])

cal_logits = raw_logits / T_best
cal_probs = np.exp(cal_logits - cal_logits.max(axis=1, keepdims=True))
cal_probs /= cal_probs.sum(axis=1, keepdims=True)

def ece(probs, y, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i+1])
        if m.sum() == 0: continue
        e += (m.sum() / len(y)) * abs(acc[m].mean() - conf[m].mean())
    return float(e)

def nll(probs, y):
    return float(-np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)))

ece_raw = ece(raw_probs, y_te); ece_cal = ece(cal_probs, y_te)
nll_raw = nll(raw_probs, y_te); nll_cal = nll(cal_probs, y_te)
acc_down = float((raw_probs.argmax(axis=1) == y_te).mean())

raw_sample_nll = -np.log(raw_probs[np.arange(len(y_te)), y_te] + 1e-12)
cal_sample_nll = -np.log(cal_probs[np.arange(len(y_te)), y_te] + 1e-12)
diffs = raw_sample_nll - cal_sample_nll
n_boot = 2000
boot_means = np.array([RNG.choice(diffs, len(diffs), replace=True).mean()
                        for _ in range(n_boot)])
cal_ci = (float(np.percentile(boot_means, 2.5)),
          float(np.percentile(boot_means, 97.5)))
_, cal_wilcox_p = sp_stats.wilcoxon(raw_sample_nll, cal_sample_nll)
st(f"  T*={T_best:.3f}  |  ECE {ece_raw:.3f}→{ece_cal:.3f}  |  "
   f"NLL {nll_raw:.3f}→{nll_cal:.3f}")
st(f"  NLL improvement {pfmt(cal_wilcox_p)} (Wilcoxon); "
   f"bootstrap 95% CI {cal_ci}")

rel_x_raw, rel_y_raw = calibration_curve(
    (raw_probs.argmax(1) == y_te).astype(int), raw_probs.max(1), n_bins=10)
rel_x_cal, rel_y_cal = calibration_curve(
    (cal_probs.argmax(1) == y_te).astype(int), cal_probs.max(1), n_bins=10)

# ────────────────────────────────────────────────────────────────────
# SAVE
# ────────────────────────────────────────────────────────────────────
st("Saving stats + arrays...")
np.savez(OUT / "pipeline_arrays.npz",
         domain_int=domain_int,
         **{f"reduced_{lk}": reduced[lk] for lk in LAYER_KEYS},
         traj_scores=traj_scores,
         cm=cm,
         raw_probs=raw_probs, cal_probs=cal_probs, y_te=y_te,
         rel_x_raw=rel_x_raw, rel_y_raw=rel_y_raw,
         rel_x_cal=rel_x_cal, rel_y_cal=rel_y_cal)

stats = {
    "n_queries": N,
    "layers": LAYER_KEYS,
    "extraction": "FFN-output (layers[i].mlp), last token",
    "stage1_sparse_pca": {
        "alpha": spca_alpha,
        "n_components": state.get("spca_n_components", 50),
        "kind_per_layer": pca_kind,
    },
    "stage2_diff_kmeans": {
        "k_per_layer": cluster_k,
        "silhouette": silhouettes,
        "domain_silhouette": domain_sil,
        "temperature": dkm_temp,
        "n_iters": state.get("dkm_n_iters", 200),
    },
    "stage3_bn": {"mi": mi_vals, "mi_p": mi_p,
                   "kruskal_wallis": {"H": float(kw_H), "p": float(kw_p)},
                   "pairwise_mw_p": mw_p},
    "stage4_router": {"domain_acc": dom_acc, "pathway_acc": pw_acc,
                       "vs_chance_p": router_p,
                       "confusion_matrix": cm.tolist()},
    "stage5_procrustes": {"layer_pairs": layer_proc, "domain_pairs": dom_proc},
    "stage6_calibration": {"T_best": T_best,
                            "ece_raw": ece_raw, "ece_cal": ece_cal,
                            "nll_raw": nll_raw, "nll_cal": nll_cal,
                            "downstream_acc": acc_down,
                            "wilcoxon_p": float(cal_wilcox_p),
                            "bootstrap_ci_95": list(cal_ci)},
}
with open(OUT / "pipeline_ffn_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
st(f"DONE → {OUT}/pipeline_ffn_stats.json")
