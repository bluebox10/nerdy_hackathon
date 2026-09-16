#!/usr/bin/env bash
# Wait for the held-out evals, then rebuild every shipped artifact.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[all] waiting for evals…"
while [ ! -f results/.done_evals ]; do sleep 30; done
echo "[all] evals done, finalising $(date -Is)"
bash scripts/run_finalise.sh
touch results/.done_final
echo "[all] FINALISED $(date -Is)"
