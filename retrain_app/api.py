import logging
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings
from .cancellation import Cancellations
from .contracts import CancelRequest, CancelResponse, JobError, StartRequest
from .errors import training_error
from .store import Store, TERMINAL
from .supervisor import Supervisor


def create_app(settings=None, *, run_supervisor=True):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        settings.validate()
        app.state.store = Store(settings.db_path)
        app.state.cancellations = Cancellations(settings, app.state.store)
        supervisor = Supervisor(settings, app.state.store)
        if run_supervisor:
            supervisor.start()
        try:
            yield
        finally:
            await app.state.cancellations.close()
            if run_supervisor:
                await supervisor.close()

    app = FastAPI(title="AI Retrain", version="1.0.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def auth(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        if not credentials or not secrets.compare_digest(credentials.credentials, settings.api_token):
            raise HTTPException(401, "UNAUTHORIZED", headers={"WWW-Authenticate": "Bearer"})

    def reply(row):
        body = {key: row[key] for key in ("job_id", "tenant_id", "target_version", "status")}
        if row["status"] == "FAILED":
            body.update(training_error(row["error_code"]))
        elif row["error_code"]:
            body["error_code"] = row["error_code"]
        body["manifest_key"] = f"tenants/{row['tenant_id']}/versions/{row['target_version']}/manifest.json"
        return body

    @app.exception_handler(JobError)
    async def job_error(request, exc):
        code = 404 if exc.code == "JOB_NOT_FOUND" else 409
        if exc.code == "WORKER_QUEUE_FULL":
            code = 503
        return JSONResponse({"detail": exc.code}, status_code=code)

    @app.get("/health")
    def health():
        return {"status": "ok", "callback_enabled": settings.callback_enabled,
                "pending_callbacks": app.state.store.pending_callback_count()}

    @app.post("/api/v1/ai/retrain/start", dependencies=[Depends(auth)])
    def start(body: StartRequest):
        if not settings.callback_enabled:
            raise HTTPException(503, "CALLBACK_NOT_ENABLED")
        if not settings.es_mapping_confirmed:
            raise HTTPException(503, "Confirm ES mapping in .env: ES_MAPPING_CONFIRMED=true")
        if len(body.added_alert_ids) + len(body.removed_alert_ids) > settings.max_alerts:
            raise HTTPException(422, "TOO_MANY_ALERT_IDS")
        row, created = app.state.store.accept(body.model_dump(), settings.max_pending_jobs)
        return JSONResponse(reply(row), status_code=202 if created or row["status"] not in TERMINAL else 200)

    @app.post("/api/v1/ai/retrain/cancel", dependencies=[Depends(auth)],
              response_model=CancelResponse, response_model_exclude_none=True,
              responses={404: {"model": CancelResponse, "description": "CANCEL_FAILED: job not found for this tenant."},
                         409: {"model": CancelResponse, "description": "CANCEL_FAILED: job can no longer be cancelled."},
                         500: {"model": CancelResponse, "description": "CANCEL_FAILED: internal cancellation error."},
                         503: {"model": CancelResponse, "description": "CANCEL_FAILED: stop/cleanup failed or cancellation timed out."}},
              description="Waits in this HTTP request until cancellation succeeds or fails; never returns 202 or CANCELLING. "
                          "Returns 200 CANCELLED only after the process has exited and cleanup is confirmed. "
                          "CANCEL_FAILED always includes error_code. Calling again after a failed attempt retries cancellation; "
                          "simultaneous calls join the same attempt. Configure the BE/proxy timeout above "
                          "CANCEL_ATTEMPT_TIMEOUT (default 180 seconds). No cancellation callback is sent.")
    async def cancel(body: CancelRequest):
        try:
            row = await app.state.cancellations.cancel(body.job_id, body.tenant_id)
        except JobError as exc:
            result = {"job_id": body.job_id, "tenant_id": body.tenant_id,
                      "status": "CANCEL_FAILED", "error_code": exc.code}
            return JSONResponse(result, status_code=404 if exc.code == "JOB_NOT_FOUND" else 409)
        except Exception as exc:
            logging.error("cancel_request_failed job=%s error_type=%s", body.job_id, type(exc).__name__)
            return JSONResponse({"job_id": body.job_id, "tenant_id": body.tenant_id,
                                 "status": "CANCEL_FAILED", "error_code": "CANCEL_INTERNAL_ERROR"}, status_code=500)
        result = {key: row[key] for key in ("job_id", "tenant_id", "status")}
        if row["status"] == "CANCEL_FAILED":
            result["error_code"] = row["error_code"]
        return JSONResponse(result, status_code=200 if row["status"] == "CANCELLED" else 503)

    return app
