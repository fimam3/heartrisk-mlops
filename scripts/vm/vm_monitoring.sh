#!/usr/bin/env bash
# Step 5 - start the API with Prometheus and Grafana, and generate traffic for the dashboard.
#
# Usage:  bash scripts/vm/vm_monitoring.sh up        # start API (port 8000), Prometheus (9090), Grafana (3000)
#         bash scripts/vm/vm_monitoring.sh traffic   # send ~60 valid and 8 invalid requests (repeat before screenshots)
#         bash scripts/vm/vm_monitoring.sh logs      # last request log lines from the API
#         bash scripts/vm/vm_monitoring.sh down      # stop everything (only after you have your screenshots)
#
# Grafana login: admin / admin  (dashboard: "HeartRisk API"). On a headless VM use SSH tunnels:
#   ssh -L 3000:localhost:3000 -L 9090:localhost:9090 <user>@<vm>
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
use_host_docker
docker compose version >/dev/null 2>&1 || die "The 'docker compose' plugin is missing. Try: sudo apt-get install -y docker-compose-plugin"

COMPOSE=(docker compose -f docker-compose.monitoring.yml)

case "${1:-up}" in
  up)
    # The plain Docker test container uses the same port 8000.
    docker rm -f hr >/dev/null 2>&1 || true
    say "Start the monitoring stack"
    "${COMPOSE[@]}" up --build -d
    wait_for_health || { "${COMPOSE[@]}" logs api --tail 30; die "The API did not become healthy in time."; }
    "${COMPOSE[@]}" ps
    echo
    echo "API:         http://localhost:8000/docs"
    echo "Prometheus:  http://localhost:9090/targets"
    echo "Grafana:     http://localhost:3000   (admin / admin) -> dashboard 'HeartRisk API'"
    echo
    echo "Next:  bash scripts/vm/vm_monitoring.sh traffic"
    ;;

  traffic)
    say "Sending 60 valid requests"
    for _ in $(seq 1 60); do
      curl -s -X POST "$API_URL/predict" -H "Content-Type: application/json" -d @"$SAMPLE" >/dev/null
    done
    say "Sending 8 invalid requests (these show up as errors)"
    for _ in $(seq 1 8); do
      echo '{"age":300}' | curl -s -X POST "$API_URL/predict" -H "Content-Type: application/json" -d @- >/dev/null
    done
    ok "Done. Wait about 10 seconds, then refresh Grafana (time range: last 15 minutes)."
    ;;

  logs)
    "${COMPOSE[@]}" logs api --tail 10
    ;;

  down)
    "${COMPOSE[@]}" down
    ok "Monitoring stack stopped"
    ;;

  *)
    die "Usage: bash scripts/vm/vm_monitoring.sh [up|traffic|logs|down]"
    ;;
esac
