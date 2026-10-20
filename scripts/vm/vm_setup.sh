#!/usr/bin/env bash
# Step 1 - install everything the assignment needs on an Ubuntu/Debian VM, then build the Python environment.
# Safe to run more than once: anything already installed is skipped.
#
# Usage:  bash scripts/vm/vm_setup.sh            # git, Docker, kubectl, Minikube, Python 3.13 + packages
#         bash scripts/vm/vm_setup.sh --no-k8s   # skip kubectl and Minikube
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WITH_K8S=1
[ "${1:-}" = "--no-k8s" ] && WITH_K8S=0
DOCKER_NOTE=""

case "$(uname -m)" in
  x86_64)  ARCH=amd64 ;;
  aarch64) ARCH=arm64 ;;
  *) die "Unsupported CPU architecture: $(uname -m)" ;;
esac

# Install a downloaded binary into /usr/local/bin, or ~/.local/bin when there is no root access.
install_bin() {
  local file="$1" name="$2"
  if as_root install -m 0755 "$file" "/usr/local/bin/$name" 2>/dev/null; then
    return 0
  fi
  mkdir -p "$HOME/.local/bin"
  install -m 0755 "$file" "$HOME/.local/bin/$name"
  export PATH="$HOME/.local/bin:$PATH"
  warn "$name was installed into ~/.local/bin. Add it to your PATH:  export PATH=\$HOME/.local/bin:\$PATH"
}

say "System packages (git, curl, unzip)"
if have apt-get; then
  as_root apt-get update -y >/dev/null 2>&1 || true
  as_root apt-get install -y git curl unzip ca-certificates >/dev/null 2>&1 \
    && ok "git, curl and unzip are installed" \
    || warn "Could not install system packages (no sudo?). Continuing with what is already installed."
fi
have git  || die "git is missing and could not be installed."
have curl || die "curl is missing and could not be installed."

say "Docker"
if have docker; then
  ok "Docker is already installed: $(docker --version)"
else
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  as_root sh /tmp/get-docker.sh || die "Docker could not be installed. Ask the VM administrator to install Docker and the compose plugin."
  as_root usermod -aG docker "$USER" || true
  DOCKER_NOTE="Docker was just installed. Run 'newgrp docker' (or log out and back in) before using Docker."
fi
if have docker && ! docker info >/dev/null 2>&1; then
  DOCKER_NOTE="${DOCKER_NOTE:-Your user cannot use Docker yet. Run 'newgrp docker' (or log out and back in).}"
fi
if have docker && docker compose version >/dev/null 2>&1; then
  ok "Docker Compose: $(docker compose version --short 2>/dev/null || echo available)"
else
  warn "The 'docker compose' plugin is missing. Try: sudo apt-get install -y docker-compose-plugin"
fi

if [ "$WITH_K8S" -eq 1 ]; then
  say "kubectl"
  if have kubectl; then
    ok "kubectl is already installed"
  else
    ver="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
    curl -fsSL -o /tmp/kubectl "https://dl.k8s.io/release/${ver}/bin/linux/${ARCH}/kubectl"
    install_bin /tmp/kubectl kubectl
    ok "kubectl ${ver} installed"
  fi

  say "Minikube"
  if have minikube; then
    ok "Minikube is already installed"
  else
    curl -fsSL -o /tmp/minikube "https://storage.googleapis.com/minikube/releases/latest/minikube-linux-${ARCH}"
    install_bin /tmp/minikube minikube
    ok "Minikube installed"
  fi
fi

say "Python 3.13 environment"
export PATH="$HOME/.local/bin:$PATH"
if ! have uv; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
have uv || die "uv could not be installed."
uv python install 3.13
if [ ! -f "$ROOT/.venv/bin/activate" ]; then
  uv venv --python 3.13 --seed "$ROOT/.venv"
fi
activate_venv
python -m pip install --quiet --upgrade pip
python -m pip install -r requirements.txt
python - <<'PY'
import sys, sklearn, pandas, numpy, mlflow, fastapi
print("Python", sys.version.split()[0], "| scikit-learn", sklearn.__version__, "| pandas", pandas.__version__,
      "| numpy", numpy.__version__, "| mlflow", mlflow.__version__, "| fastapi", fastapi.__version__)
PY

say "Project files check"
for f in data/raw/processed.cleveland.data models/heart_pipeline.joblib models/model_card.json sample.json \
         .github/workflows/ci.yml Dockerfile k8s/deployment.yaml docker-compose.monitoring.yml; do
  if [ -f "$f" ]; then ok "$f"; else warn "MISSING: $f  (did you clone the full repository?)"; fi
done

say "Setup finished"
[ -n "$DOCKER_NOTE" ] && warn "$DOCKER_NOTE"
echo "Next:  bash scripts/vm/vm_pipeline.sh"
