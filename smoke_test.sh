#!/usr/bin/env bash
# smoke_test.sh
#
# End-to-end pipeline validation on a small, real slice of LoCoMo — not the
# full 5,882-record / 20-60 minute embed run. Fetches 1 real conversation
# (~500-600 dialogue turns), embeds it, and runs both the batch and streamed
# legs against a real Qdrant container, all in a couple of minutes. Exercises
# every script (fetch_dataset.py, embed_corpus.py, run_benchmark.py) and every
# code path (batch upsert, streamed upsert, checkpoints, recall/latency,
# footprint, container restart) without touching data/, embeddings/, or
# results.json — so it's safe to run against a clone that already has the
# full published results checked out, and safe for CI.
#
# Usage:
#   ./smoke_test.sh                 # uses its own qdrant-agent-memory-bench-smoke container
#   ./smoke_test.sh <container-name>  # use a different container name
#
# Requires Docker and the packages in requirements.txt (pip install -r
# requirements.txt) to already be available on PATH.

set -euo pipefail

CONTAINER="${1:-qdrant-agent-memory-bench-smoke}"
PORT="${SMOKE_PORT:-6344}"
WORKDIR="$(mktemp -d)"
trap 'echo "--- cleaning up ---"; docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; rm -rf "$WORKDIR"' EXIT

echo "=== smoke test: workdir=$WORKDIR container=$CONTAINER port=$PORT ==="

echo "--- starting a throwaway Qdrant container ---"
docker run -d --name "$CONTAINER" \
  -p "${PORT}:6333" \
  -e QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=128 \
  qdrant/qdrant:v1.19.0 >/dev/null

for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT}/readyz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "--- fetching 1 real LoCoMo conversation ---"
python3 fetch_dataset.py --out-dir "$WORKDIR/data" --max-conversations 1

echo "--- embedding it (small enough to finish in seconds, not 20-60 min) ---"
python3 embed_corpus.py --data-dir "$WORKDIR/data" --embed-dir "$WORKDIR/embeddings" --query-sample-size 20

echo "--- running both legs against the throwaway container ---"
python3 run_benchmark.py \
  --qdrant-url "http://localhost:${PORT}" \
  --container "$CONTAINER" \
  --data-dir "$WORKDIR/data" \
  --embed-dir "$WORKDIR/embeddings" \
  --out "$WORKDIR/results.json" \
  --checkpoint-every 100 \
  --settle-seconds 2

echo "=== smoke test passed: pipeline runs end-to-end, see $WORKDIR/results.json ==="
