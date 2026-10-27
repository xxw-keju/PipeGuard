#!/usr/bin/env bash
# Unified Suite 2 pipeline: full recapture -> structural-diversity audit ->
# CUDA grounding gate -> NVIDIA sanitizer execution evidence.
# Run on the GPU box from the checker repo root. Outputs are versioned by the
# checker git SHA so ledgers are traceable to a freeze point.
#
#   PIPEGUARD_TL_PYTHON=... PIPEGUARD_TRITON_PYTHON=... \
#     tools/run_suite2_pipeline.sh /path/to/output-root [checker-python]
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
OUT_ROOT=${1:?usage: run_suite2_pipeline.sh <output-root> [checker-python]}
PY=${2:-$(command -v python3)}
SHA=$(git -C "$ROOT" rev-parse --short HEAD)
DIRTY=$(git -C "$ROOT" status --porcelain | wc -l)
RUN="$OUT_ROOT/suite2-pipeline-${SHA}$( [ "$DIRTY" -gt 0 ] && echo -dirty )"
mkdir -p "$RUN"
echo "pipeline run dir: $RUN (checker @ $SHA, dirty files: $DIRTY)"

echo "[1/4] full recapture (210 cases)"
"$PY" "$ROOT/tools/run_suite2_selected210.py" \
  --output "$RUN/recapture" --jobs 8 --timeout 180 \
  2>&1 | tee "$RUN/recapture.log" | tail -3

echo "[2/4] structural-diversity audit (strict: unique == n per cell)"
"$PY" "$ROOT/tools/suite2_structural_audit.py" \
  --run-dir "$RUN/recapture" --out "$RUN/structural-audit.json" \
  | tee "$RUN/structural-audit.txt" || echo "AUDIT FAILED (see report)"

echo "[3/4] CUDA grounding gate"
"$PY" "$ROOT/tools/suite2_cuda_grounding.py" \
  --run-dir "$RUN/recapture" --out "$RUN/grounding.json" \
  | tee "$RUN/grounding.txt" || echo "GROUNDING FAILED (see report)"

echo "[4/4] NVIDIA sanitizer execution evidence (racecheck+synccheck, 4 GPUs x3)"
"$PY" "$ROOT/tools/run_suite2_nvidia_all.py" \
  --output "$RUN/nvidia" --timeout 60 --gpus 0,0,0,1,1,1,2,2,2,3,3,3 \
  2>&1 | tee "$RUN/nvidia.log" | tail -5

echo
echo "pipeline complete: $RUN"
echo "  recapture summary : $RUN/recapture/summary.json"
echo "  structural audit  : $RUN/structural-audit.json"
echo "  grounding         : $RUN/grounding.json"
echo "  sanitizer matrix  : $RUN/nvidia/summary.json"
