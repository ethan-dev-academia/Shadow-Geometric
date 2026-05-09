"""
Collect FFN-output activations from Mistral-7B-Instruct-v0.2 across 600 real queries.

Key difference from collect_mistral_600real.py: hooks layers[i].mlp specifically,
not layers[i]. This captures the FFN submodule's direct output (the contribution
that gets added to the residual stream), not the post-block residual stream itself.

Output: activations_ffn_600.pt with shape (n_queries, hidden_dim) per layer.
"""
import json, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).parent.parent
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
TARGET_LAYERS = [8, 16, 24]
OUT_PT = BASE / "activations_ffn_600.pt"
OUT_INFO = BASE / "ffn_600_info.json"

device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
print(f"[INFO] device={device}")

with open(BASE / "data" / "queries_600.json") as f:
    data = json.load(f)
queries = data["queries"]
domains = data["domains"]
print(f"[INFO] Loaded {len(queries)} queries "
      f"({sum(d=='medical' for d in domains)} med / "
      f"{sum(d=='financial' for d in domains)} fin / "
      f"{sum(d=='code' for d in domains)} code)")

print(f"[INFO] Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, low_cpu_mem_usage=True)
model = model.to(device)
model.eval()

layers = model.model.layers
print(f"[INFO] Model has {len(layers)} transformer blocks; "
      f"hooking .mlp on layers {TARGET_LAYERS}")

buf = {f"layer_{i}": [] for i in TARGET_LAYERS}
handles = []


def make_ffn_hook(layer_idx):
    """Hook on layers[i].mlp captures the FFN's direct output before residual add."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        buf[f"layer_{layer_idx}"].append(
            h[:, -1, :].detach().to(torch.float32).cpu())
    return hook


for li in TARGET_LAYERS:
    if li < len(layers):
        # CRITICAL: hook .mlp specifically, not the whole layer
        handles.append(layers[li].mlp.register_forward_hook(make_ffn_hook(li)))
    else:
        raise RuntimeError(f"Layer {li} out of range ({len(layers)} layers)")

t0 = time.time()
for i, q in enumerate(queries):
    inputs = tok(q, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        model(**inputs)
    if (i + 1) % 50 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(queries) - i - 1) / rate
        print(f"  [{elapsed:6.1f}s] {i+1:3d}/{len(queries)}  "
              f"rate={rate:.2f} q/s  eta={eta/60:.1f} min")

for h in handles:
    h.remove()

stacked = {k: torch.cat(v, dim=0) for k, v in buf.items() if v}
print(f"\n[INFO] FFN activation tensors:")
for k, t in stacked.items():
    print(f"  {k}: shape={tuple(t.shape)}  mean_norm={t.norm(dim=1).mean():.2f}")

torch.save(stacked, str(OUT_PT))
with open(OUT_INFO, "w") as f:
    json.dump({
        "model": MODEL,
        "n_queries": len(queries),
        "target_layers": TARGET_LAYERS,
        "hook_target": "layers[i].mlp (FFN submodule output)",
        "queries": queries,
        "domains": domains,
        "hidden_size": int(model.config.hidden_size),
    }, f, indent=2)

print(f"\n[DONE] {time.time()-t0:.1f}s total -> {OUT_PT.name}")
