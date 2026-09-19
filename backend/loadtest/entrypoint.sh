#!/bin/sh
# Run the load profile against the API, then gate on the results.
set -eu
RESULTS="${RESULTS_DIR:-/results}"
mkdir -p "$RESULTS"
rm -f "$RESULTS"/locust_*.csv
locust -f /app/loadtest/locustfile.py --headless \
  --host "${API_URL:-http://api:8000}" \
  --csv "$RESULTS/locust" --only-summary --exit-code-on-error 0
exec python -m loadtest.check_results "$RESULTS/locust_stats.csv"
