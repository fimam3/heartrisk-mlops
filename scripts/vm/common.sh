#!/usr/bin/env bash
# Shared helpers for the scripts in scripts/vm/. This file is sourced, not run directly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs

SAMPLE="$ROOT/sample.json"
API_URL="http://localhost:8000"

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# Run a command as root (directly, or through sudo). Returns 1 if neither is possible.
as_root() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif have sudo; then
    sudo "$@"
  else
    return 1
  fi
}

# Activate the project's virtual environment and set PYTHONPATH.
activate_venv() {
  [ -f "$ROOT/.venv/bin/activate" ] || die "No .venv found. Run first: bash scripts/vm/vm_setup.sh"
  set +u
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  set -u
  export PYTHONPATH="$ROOT/src"
}

# Make plain `docker` talk to this machine's Docker (not Minikube's), and check it works.
use_host_docker() {
  have docker || die "Docker is not installed. Run: bash scripts/vm/vm_setup.sh"
  unset DOCKER_HOST DOCKER_TLS_VERIFY DOCKER_CERT_PATH MINIKUBE_ACTIVE_DOCKERD 2>/dev/null || true
  docker info >/dev/null 2>&1 || die "This user cannot use Docker yet. Try: newgrp docker   (or log out and back in), then run the script again."
}

# Wait until an HTTP endpoint answers. Usage: wait_for_health [url] [tries]
wait_for_health() {
  local url="${1:-$API_URL/health}" tries="${2:-40}" i
  for i in $(seq 1 "$tries"); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

# Send one valid and one invalid prediction request. Usage: demo_requests [base_url]
demo_requests() {
  local base="${1:-$API_URL}"
  say "GET $base/health"
  curl -s "$base/health"; echo
  say "POST $base/predict  (valid patient from sample.json)"
  curl -s -X POST "$base/predict" -H "Content-Type: application/json" -d @"$SAMPLE"; echo
  say "POST $base/predict  (invalid input -> expect HTTP 422)"
  echo '{"age":300}' | curl -s -w '\nHTTP status: %{http_code}\n' -X POST "$base/predict" \
    -H "Content-Type: application/json" -d @-
}
