#!/usr/bin/env bash
# Load-test the containerized app and apply the performance gate.
#
#   backend/loadtest/run.sh              # build + start the stack, test, gate, tear down
#   KEEP_STACK=1 backend/loadtest/run.sh # leave the stack running afterwards
#
# Locust runs in its own container on the compose network (service "loadtest",
# profile "loadtest") and hits http://api:8000 directly. Exit status is the
# gate's: 0 pass, 1 threshold breached, 2 results unusable.
#
# Tunables (env): TARGET_RPS, WARMUP_SECONDS, TEST_SECONDS, PERF_MAX_P99_MS,
# PERF_MAX_ERROR_RATE, PERF_MIN_RPS_RATIO.
set -euo pipefail

cd "$(dirname "$0")/../.."
docker compose up -d --build --wait postgres redis mlflow trainer api
if [[ "${KEEP_STACK:-0}" != "1" ]]; then
  trap 'docker compose down' EXIT
fi

# The API reports healthy once it has loaded the Production model.
docker compose exec -T api python -c "
import json, sys, time, urllib.request
for _ in range(60):
    body = json.load(urllib.request.urlopen('http://localhost:8000/health'))
    if body['model_loaded']:
        sys.exit(0)
    time.sleep(2)
sys.exit('API never loaded a model')
"
# Give /recommendation some history to work with.
docker compose exec -T api python -m app.seed

docker compose --profile loadtest run --rm --build loadtest
