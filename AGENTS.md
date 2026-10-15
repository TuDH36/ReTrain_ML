# Repository Guidelines

## Scope and Structure

This repository contains the standalone AI Retrain service. `retrain_app/` contains
the FastAPI endpoints, subprocess supervisor, SQLite job ledger, callback delivery,
ES/MinIO clients, and training pipeline. `tests/` contains synthetic fixtures and
isolated HTTP integration tests. Deployment uses `Dockerfile`, `compose.yaml`, and
the settings documented in `.env.example` and `README.md`.

## Development and Verification

Use Python 3.11 and the pinned ML dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
docker compose up -d --build
```

Copy `.env.example` to `.env` and configure the environment before starting Docker.
The tests must not write to real ES or MinIO. Tests requiring sibling Predict source
or existing VND artifacts may skip in a standalone checkout.

## Coding and Business Rules

Use four-space indentation and snake_case. Preserve the feature ordering, cleaning
rules, categorical encoding offset, and compatibility with the Predict service.
The confirmed retraining policy fits all retained records, removes every exact
feature conflict group, and evaluates on the training set without early stopping.
Mark these metrics explicitly as training metrics.

BE dispatches jobs through HTTP; AI does not consume Redis. Keep job acceptance
durable and idempotent. Allow one active job per tenant. Preserve ordered callback
retries, tenant isolation, and cancellation during publication. Emit CANCELLED only
after the process exits and its uploaded objects have been cleaned up.

## Git and Delivery

The repository root is this directory; origin is
`https://github.com/TuDH36/ReTrain_ML.git`, branch `main`.
The user requests that subsequent completed changes to this service be committed
and pushed here. Use specific imperative commit subjects, review staged files,
and run checks appropriate to the change before pushing. Never force-push or
change the parent repository's remote.

Keep `.env`, credentials, real alert records, model binaries, and local runtime
state out of commits. Commit only placeholder configuration in `.env.example`.
