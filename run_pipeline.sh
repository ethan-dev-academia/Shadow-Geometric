#!/usr/bin/env bash
# Full Shadow-Geometric pipeline runner (Brev H100 recommended; ~35 min total)
# Usage: bash run_pipeline.sh
set -euo pipefail

echo "==> [1/6] Collecting FFN activations (~5 min)..."
python pipeline/collect_ffn_600.py

echo "==> [2/6] Building state file (~12 min)..."
python pipeline/build_state_ffn.py

echo "==> [3/6] Precomputing steering tensors (~10 sec)..."
python pipeline/precompute_steering_tensors_ffn.py

echo "==> [4/6] Running BN pipeline (~30 sec)..."
python pipeline/run_pipeline_ffn.py

echo "==> [5/6] Live steering experiment (~18 min)..."
python pipeline/brev_live_steering.py --phase C --state-file shadow_live_state_ffn.pt --inject-target mlp

echo "==> [6/6] Calibrated scoring (~10 sec)..."
python pipeline/score_generations_calibrated.py

echo ""
echo "Done. Outputs:"
echo "  activations_ffn_600.pt, ffn_600_info.json"
echo "  shadow_live_state_ffn.pt, steering_tensors.pt"
echo "  shadow_v3_output/n600_ffn/"
echo "  brev_live_results/"
