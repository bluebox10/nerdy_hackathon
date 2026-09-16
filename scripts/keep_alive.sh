#!/usr/bin/env bash
# Supervises the API server and the Cloudflare quick tunnel.
# A quick-tunnel hostname changes on restart, so we only ever restart a tunnel that has
# actually died, and we record the live hostname where the submission filler can read it.
cd "$(dirname "$0")/.."
export HF_HOME=/home/nvidia/23BRS1361_lpdev/.hf_cache CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PORT=${PORT:-8077}

start_api() {
  ( cd serve && nohup ../.venv/bin/python -m uvicorn app:app --host 127.0.0.1 \
      --port "$PORT" --log-level warning >> ../logs/server.log 2>&1 & )
  echo "[keepalive] api restarted"
}
start_tunnel() {
  nohup ./bin/cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate \
    >> logs/tunnel.log 2>&1 &
  echo "[keepalive] tunnel restarted"
}

while true; do
  curl -sf -m 5 "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 || start_api
  pgrep -f "cloudflared tunnel --url" >/dev/null || start_tunnel
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/tunnel.log | tail -1)
  if [ -n "$URL" ]; then
    printf '%s' "$URL" > serve/artifacts/PUBLIC_URL
  fi
  sleep 30
done
