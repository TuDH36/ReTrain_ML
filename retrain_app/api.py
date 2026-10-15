import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings
from .contracts import CancelRequest, CancelResponse, JobError, StartRequest
from .store import Store, TERMINAL
from .supervisor import Supervisor


def create_app(settings=None, *, run_supervisor=True):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        settings.validate()
        app.state.store = Store(settings.db_path)
        supervisor = Supervisor(settings, app.state.store)
        if run_supervisor:
            supervisor.start()
        try:
            yield
        finally:
            if run_supervisor:
                await supervisor.close()

    app = FastAPI(title="AI Retrain", version="1.0.0", lifespan=lifespan)
    bearer = HTTPBearer(auto_error=False)

    def auth(credentials: HTTPAuthorizationCredentials = Depends(bearer)):
        if not credentials or not secrets.compare_digest(credentials.credentials, settings.api_token):
            raise HTTPException(401, "UNAUTHORIZED", headers={"WWW-Authenticate": "Bearer"})

    def reply(row):
        body = {key: row[key] for key in ("job_id", "tenant_id", "target_version", "status")}
        if row["error_code"]:
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
              responses={202: {"model": CancelResponse, "description": "Cancellation in progress; tenant remains locked."},
                         503: {"model": CancelResponse, "description": "Cancellation attempt failed; call again after backoff to retry."}},
              description="Call repeatedly with the same job_id and tenant_id. A failed attempt is reported as "
                          "503 CANCEL_FAILED once; the following call starts a retry and returns 202 CANCELLING. "
                          "Only 200 CANCELLED confirms the process stopped and cleanup finished. "
                          "Cancellation statuses are not sent to the training callback.")
    def cancel(body: CancelRequest):
        row = app.state.store.request_cancel(body.job_id, body.tenant_id)
        result = {key: row[key] for key in ("job_id", "tenant_id", "status")}
        if row["status"] == "CANCEL_FAILED":
            result["error_code"] = row["error_code"]
        return JSONResponse(result, status_code={"CANCELLED": 200, "CANCELLING": 202, "CANCEL_FAILED": 503}[row["status"]])

    return app
