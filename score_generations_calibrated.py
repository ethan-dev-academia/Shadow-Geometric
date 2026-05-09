"""
Score Phase C generations with a calibrated text-domain classifier.

Trains TF-IDF + LogisticRegression on the original 600 queries (the same
prompt distribution used to extract activations for the geometry pipeline),
then applies temperature scaling on a held-out slice. Each generation in
brev_live_results/phaseC_ablation.json is scored to produce calibrated
domain probabilities.

Per-arm task accuracy is reported under two definitions:

    src_preserved_acc :  P(pred = src_domain | generation)  averaged per arm.
                         The repulsion-arms claim is "tensor preserves
                         on-source generation behavior at less cost than
                         scalar"; this is the metric to compare.
                         Higher = less drift from source.

    tgt_match_acc     :  P(pred = tgt_domain | generation)  averaged per arm.
                         The CAA arm's claim is "attraction toward target";
                         this is the metric for that arm.

Output:
    brev_live_results/phaseC_calibrated_scores.json
    brev_live_results/phaseC_calibrated_summary.txt   (human-readable table)

CPU-only. ~10s on a laptop.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
QUERIES_JSON = BASE / "queries_600.json"
PHASE_C_JSON = BASE / "brev_live_results" / "phaseC_ablation.json"
OUT_JSON = BASE / "brev_live_results" / "phaseC_calibrated_scores.json"
OUT_TXT = BASE / "brev_live_results" / "phaseC_calibrated_summary.txt"

DOMAINS_PAPER = ["Medical", "Financial", "Code"]
DOMAINS_DATA = ["medical", "financial", "code"]
RNG_SEED = 42

t0 = time.time()
def st(m): print(f"  [{time.time()-t0:6.1f}s] {m}", flush=True)


def fit_calibrated_classifier(queries, labels, seed=RNG_SEED):
    """TF-IDF + LogReg, temperature-scaled on a held-out 20% slice.

    Returns (vectorizer, clf, T_best). Apply at inference time as:
        X = vectorizer.transform([text])
        logits = clf.decision_function(X)
        probs = softmax(logits / T_best)
    """
    y = np.array([DOMAINS_DATA.index(d) for d in labels])
    X_train, X_val, y_train, y_val = train_test_split(
        queries, y, test_size=0.2, random_state=seed, stratify=y)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95)
    Xt_train = vec.fit_transform(X_train)
    Xt_val = vec.transform(X_val)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xt_train, y_train)

    raw_logits_val = clf.decision_function(Xt_val)
    if raw_logits_val.ndim == 1:
        # Binary case shouldn't happen with 3 domains; guard anyway.
        raw_logits_val = np.column_stack([-raw_logits_val, raw_logits_val])

    def temp_nll(T):
        sc = raw_logits_val / T
        sc -= sc.max(axis=1, keepdims=True)
        p = np.exp(sc); p /= p.sum(axis=1, keepdims=True)
        return -np.mean(np.log(p[np.arange(len(y_val)), y_val] + 1e-12))

    T_grid = np.linspace(0.3, 5.0, 60)
    nlls = [temp_nll(T) for T in T_grid]
    T_best = float(T_grid[int(np.argmin(nlls))])

    # Held-out validation accuracy + NLL pre/post calibration
    raw_probs = np.exp(raw_logits_val - raw_logits_val.max(axis=1, keepdims=True))
    raw_probs /= raw_probs.sum(axis=1, keepdims=True)
    cal_logits = raw_logits_val / T_best
    cal_probs = np.exp(cal_logits - cal_logits.max(axis=1, keepdims=True))
    cal_probs /= cal_probs.sum(axis=1, keepdims=True)

    val_acc = float((raw_probs.argmax(axis=1) == y_val).mean())
    nll_raw = float(log_loss(y_val, raw_probs, labels=[0, 1, 2]))
    nll_cal = float(log_loss(y_val, cal_probs, labels=[0, 1, 2]))
    st(f"  TF-IDF+LR val_acc={val_acc:.3f}  NLL raw={nll_raw:.3f} -> "
       f"cal={nll_cal:.3f}  T*={T_best:.2f}")

    return vec, clf, T_best, {"val_acc": val_acc,
                              "nll_raw": nll_raw,
                              "nll_cal": nll_cal,
                              "T_best": T_best}


def score(vec, clf, T_best, texts):
    Xt = vec.transform(texts)
    logits = clf.decision_function(Xt)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    logits = logits / T_best
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def main():
    st(f"Loading queries from {QUERIES_JSON.name}")
    with open(QUERIES_JSON) as f:
        qd = json.load(f)
    queries, labels = qd["queries"], qd["domains"]
    st(f"  {len(queries)} queries  ("
       f"{sum(d=='medical' for d in labels)} med / "
       f"{sum(d=='financial' for d in labels)} fin / "
       f"{sum(d=='code' for d in labels)} code)")

    st("Fitting TF-IDF + LogReg + temperature scaling...")
    vec, clf, T_best, cal_meta = fit_calibrated_classifier(queries, labels)

    st(f"Loading Phase C generations from {PHASE_C_JSON}")
    with open(PHASE_C_JSON) as f:
        records = json.load(f)
    st(f"  {len(records)} generations")

    texts = [r["generated"] for r in records]
    probs = score(vec, clf, T_best, texts)
    pred_idx = probs.argmax(axis=1)

    # Augment each record with calibrated scores
    arm_groups = defaultdict(list)
    for r, p, pi in zip(records, probs, pred_idx):
        src_idx = DOMAINS_DATA.index(r["src_domain"].lower())
        tgt_idx = DOMAINS_DATA.index(r["tgt_domain"].lower())
        r["calibrated_probs"] = {DOMAINS_PAPER[i]: float(p[i]) for i in range(3)}
        r["calibrated_pred"] = DOMAINS_PAPER[int(pi)]
        r["P_src"] = float(p[src_idx])
        r["P_tgt"] = float(p[tgt_idx])
        r["pred_eq_src"] = bool(pi == src_idx)
        r["pred_eq_tgt"] = bool(pi == tgt_idx)
        arm_groups[r["arm"]].append(r)

    # Per-arm summary
    summary = {}
    for arm, rs in arm_groups.items():
        P_src = np.array([r["P_src"] for r in rs])
        P_tgt = np.array([r["P_tgt"] for r in rs])
        ppl = np.array([r["perplexity"] for r in rs])
        src_pres = float(np.mean([r["pred_eq_src"] for r in rs]))
        tgt_match = float(np.mean([r["pred_eq_tgt"] for r in rs]))
        summary[arm] = {
            "n": len(rs),
            "src_preserved_acc": src_pres,
            "tgt_match_acc": tgt_match,
            "P_src_mean": float(P_src.mean()),
            "P_src_std": float(P_src.std()),
            "P_tgt_mean": float(P_tgt.mean()),
            "P_tgt_std": float(P_tgt.std()),
            "ppl_mean": float(ppl.mean()),
            "ppl_std": float(ppl.std()),
        }

    # Per-arm × per-α breakdown for the repulsion arms (workshop-paper table)
    per_alpha = defaultdict(dict)
    for arm in arm_groups:
        for alpha in sorted({r["alpha"] for r in arm_groups[arm]}):
            rs = [r for r in arm_groups[arm] if r["alpha"] == alpha]
            P_src = np.array([r["P_src"] for r in rs])
            P_tgt = np.array([r["P_tgt"] for r in rs])
            ppl = np.array([r["perplexity"] for r in rs])
            per_alpha[arm][float(alpha)] = {
                "n": len(rs),
                "src_preserved_acc": float(np.mean([r["pred_eq_src"] for r in rs])),
                "tgt_match_acc": float(np.mean([r["pred_eq_tgt"] for r in rs])),
                "P_src_mean": float(P_src.mean()),
                "P_src_std": float(P_src.std()),
                "P_tgt_mean": float(P_tgt.mean()),
                "P_tgt_std": float(P_tgt.std()),
                "ppl_mean": float(ppl.mean()),
                "ppl_std": float(ppl.std()),
            }

    # Save augmented records + summary
    out = {
        "calibration": cal_meta,
        "n_generations": len(records),
        "per_arm_summary": summary,
        "per_arm_per_alpha": {k: dict(v) for k, v in per_alpha.items()},
        "records": records,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    st(f"Saved scored records -> {OUT_JSON}")

    # Human-readable table
    lines = []
    lines.append("=" * 78)
    lines.append("Calibrated Phase C scoring (TF-IDF + LogReg + temp scaling)")
    lines.append(f"Calibration: T*={cal_meta['T_best']:.2f}  "
                 f"val_acc={cal_meta['val_acc']:.3f}  "
                 f"NLL {cal_meta['nll_raw']:.3f} -> {cal_meta['nll_cal']:.3f}")
    lines.append("=" * 78)
    lines.append(f"{'arm':28s} {'n':>4s} "
                 f"{'src_pres':>10s} {'tgt_match':>10s} "
                 f"{'P_src':>14s} {'P_tgt':>14s} {'PPL':>10s}")
    for arm in sorted(summary):
        s = summary[arm]
        lines.append(
            f"{arm:28s} {s['n']:>4d} "
            f"{s['src_preserved_acc']:>10.3f} {s['tgt_match_acc']:>10.3f} "
            f"{s['P_src_mean']:>6.3f}±{s['P_src_std']:.3f} "
            f"{s['P_tgt_mean']:>6.3f}±{s['P_tgt_std']:.3f} "
            f"{s['ppl_mean']:>5.2f}±{s['ppl_std']:.1f}")
    lines.append("")
    lines.append("Per-arm × per-α (src_preserved_acc — lower = more steering effect):")
    for arm in sorted(per_alpha):
        for alpha in sorted(per_alpha[arm]):
            s = per_alpha[arm][alpha]
            lines.append(f"  {arm:28s} α={alpha:.1f}  "
                         f"src_pres={s['src_preserved_acc']:.3f}  "
                         f"tgt_match={s['tgt_match_acc']:.3f}  "
                         f"P_src={s['P_src_mean']:.3f}  "
                         f"P_tgt={s['P_tgt_mean']:.3f}  "
                         f"PPL={s['ppl_mean']:.2f}")
    body = "\n".join(lines)
    print("\n" + body + "\n")
    OUT_TXT.write_text(body + "\n")
    st(f"Saved summary table -> {OUT_TXT}")


if __name__ == "__main__":
    main()
