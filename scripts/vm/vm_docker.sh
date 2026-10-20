#!/usr/bin/env bash
# Step 3 - build the Docker image, run the container and test the /predict endpoint.
#
# Usage:  bash scripts/vm/vm_docker.sh         # build, run, test (container keeps running)
#         bash scripts/vm/vm_docker.sh stop    # remove the container (do this before the monitoring step)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
use_host_docker

case "${1:-up}" in
  up)
    say "Build the image"
    docker build -t heartrisk-api:1.0 .

    say "Run the container on port 8000"
    docker rm -f hr >/dev/null 2>&1 || true
    docker run -d --name hr -p 8000:8000 heartrisk-api:1.0
    if ! wait_for_health; then
      docker logs hr || true
      die "The API did not become healthy in time. See the container log above."
    fi

    demo_requests "$API_URL"

    say "Container status and request logs"
    docker ps --filter name=hr
    docker logs hr --tail 5
    echo
    echo "Container 'hr' is still running. Stop it with:  bash scripts/vm/vm_docker.sh stop"
    ;;
  stop)
    docker rm -f hr >/dev/null 2>&1 && ok "Container hr removed" || warn "Container hr was not running"
    ;;
  *)
    die "Usage: bash scripts/vm/vm_docker.sh [up|stop]"
    ;;
esac
