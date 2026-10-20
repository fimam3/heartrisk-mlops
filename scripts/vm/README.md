# VM run scripts

Helper scripts that reproduce the whole assignment on a Linux VM (Ubuntu/Debian). Run them from the repository root,
in this order, with `bash` (no need to make them executable):

| Order | Command | What it does |
|---|---|---|
| 1 | `bash scripts/vm/vm_setup.sh` | Installs git, Docker, kubectl, Minikube and Python 3.13, creates `.venv`, installs `requirements.txt` |
| 2 | `bash scripts/vm/vm_pipeline.sh` | Cleans data, EDA, trains and tracks with MLflow, quality gate, pytest, flake8 |
| 2b | `bash scripts/vm/vm_mlflow.sh` | Opens the MLflow UI on port 5000 |
| 3 | `bash scripts/vm/vm_docker.sh` | Builds the image, runs it, tests `/health` and `/predict`. `stop` removes the container. |
| 4 | `bash scripts/vm/vm_k8s.sh up` then `test` | Starts Minikube, deploys `k8s/`, tests through the Service. `tunnel` exposes the LoadBalancer. `down` removes it. |
| 5 | `bash scripts/vm/vm_monitoring.sh up` then `traffic` | Starts API + Prometheus + Grafana and generates traffic. `down` stops it. |

Notes:
- After `vm_setup.sh` installs Docker for the first time, run `newgrp docker` (or log out and in) once.
- Run `bash scripts/vm/vm_docker.sh stop` before step 5, because both use port 8000.
- On a VM without a desktop, forward ports with `ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 5000:localhost:5000 <user>@<vm>`.
