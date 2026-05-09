# Full Shadow-Geometric pipeline runner (Brev H100 recommended; ~35 min total)
# Usage: .\run_pipeline.ps1  (from repo root with venv activated)

$ErrorActionPreference = "Stop"

Write-Host "==> [1/6] Collecting FFN activations (~5 min)..."
python pipeline/collect_ffn_600.py

Write-Host "==> [2/6] Building state file (~12 min)..."
python pipeline/build_state_ffn.py

Write-Host "==> [3/6] Precomputing steering tensors (~10 sec)..."
python pipeline/precompute_steering_tensors_ffn.py

Write-Host "==> [4/6] Running BN pipeline (~30 sec)..."
python pipeline/run_pipeline_ffn.py

Write-Host "==> [5/6] Live steering experiment (~18 min)..."
python pipeline/brev_live_steering.py --phase C --state-file shadow_live_state_ffn.pt --inject-target mlp

Write-Host "==> [6/6] Calibrated scoring (~10 sec)..."
python pipeline/score_generations_calibrated.py

Write-Host ""
Write-Host "Done. Outputs:"
Write-Host "  activations_ffn_600.pt, ffn_600_info.json"
Write-Host "  shadow_live_state_ffn.pt, steering_tensors.pt"
Write-Host "  shadow_v3_output/n600_ffn/"
Write-Host "  brev_live_results/"
