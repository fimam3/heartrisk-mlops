#!/usr/bin/env bash
# Step 4 - deploy the API to Kubernetes (Minikube) and test it.
#
# Usage:  bash scripts/vm/vm_k8s.sh up       # start Minikube, build the image inside it, apply k8s/*.yaml
#         bash scripts/vm/vm_k8s.sh test     # call /health and /predict through the Kubernetes Service
#         bash scripts/vm/vm_k8s.sh status   # pods, service, autoscaler
#         bash scripts/vm/vm_k8s.sh tunnel   # give the LoadBalancer an EXTERNAL-IP (keep this terminal open)
#         bash scripts/vm/vm_k8s.sh down     # delete the deployment
#
# Optional environment variables: MK_CPUS (default 2), MK_MEMORY in MB (default 3072)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NS=heartrisk
have kubectl  || die "kubectl is not installed. Run: bash scripts/vm/vm_setup.sh"
have minikube || die "Minikube is not installed. Run: bash scripts/vm/vm_setup.sh"

case "${1:-up}" in
  up)
    use_host_docker
    if minikube status >/dev/null 2>&1; then
      ok "Minikube is already running"
    else
      say "Start Minikube"
      FORCE=""
      [ "$(id -u)" -eq 0 ] && FORCE="--force"   # the docker driver refuses to run as root otherwise
      minikube start --driver=docker --cpus="${MK_CPUS:-2}" --memory="${MK_MEMORY:-3072}" $FORCE
    fi
    minikube addons enable metrics-server >/dev/null 2>&1 || warn "Could not enable metrics-server (the autoscaler will show <unknown>)"

    say "Build the image inside Minikube"
    eval "$(minikube -p minikube docker-env)"
    docker build -t heartrisk-api:1.0 .

    say "Apply the Kubernetes manifests"
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/
    kubectl -n "$NS" rollout status deploy/heartrisk-api --timeout=240s

    say "Deployment status"
    kubectl -n "$NS" get pods,svc,hpa
    echo
    echo "Next:  bash scripts/vm/vm_k8s.sh test"
    echo "For a LoadBalancer EXTERNAL-IP, run in a second terminal:  bash scripts/vm/vm_k8s.sh tunnel"
    ;;

  test)
    say "Forward a local port to the Kubernetes Service"
    kubectl -n "$NS" port-forward svc/heartrisk-api 8080:80 >logs/port-forward.log 2>&1 &
    PF=$!
    trap 'kill $PF 2>/dev/null || true' EXIT
    wait_for_health http://localhost:8080/health 20 || { cat logs/port-forward.log; die "Could not reach the Service. Is it deployed? Run: bash scripts/vm/vm_k8s.sh up"; }
    demo_requests http://localhost:8080

    say "Pod logs (JSON request lines)"
    kubectl -n "$NS" logs deploy/heartrisk-api --tail 5
    say "Cluster state"
    kubectl -n "$NS" get pods,svc,hpa
    ;;

  status)
    kubectl -n "$NS" get pods,svc,hpa
    ;;

  tunnel)
    say "minikube tunnel (keep this terminal open; it may ask for your sudo password)"
    echo "In another terminal run:  kubectl -n $NS get svc   then   curl http://<EXTERNAL-IP>/health"
    exec minikube tunnel
    ;;

  down)
    kubectl delete -f k8s/ --ignore-not-found
    kubectl delete namespace "$NS" --ignore-not-found
    ok "Deployment removed (Minikube is still running; stop it with: minikube stop)"
    ;;

  *)
    die "Usage: bash scripts/vm/vm_k8s.sh [up|test|status|tunnel|down]"
    ;;
esac
