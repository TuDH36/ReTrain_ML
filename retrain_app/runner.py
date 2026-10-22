"""One isolated OS process for one job. Invoked only by Supervisor."""
import ctypes
import json
import logging
import os
import shutil
import signal
import sys
import time


def bind_parent():
    # Set this BEFORE heavy ML imports. A killed API cannot leave a training child alive.
    if sys.platform.startswith("linux"):
        expected = int(os.environ["RETRAIN_PARENT_PID"])
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != expected:
            raise SystemExit(1)


def run(job_id):
    from botocore.exceptions import BotoCoreError, ClientError
    from .config import Settings
    from .contracts import Cancelled, JobError
    from .elasticsearch import Alerts
    from .storage import Models
    from .store import CANCEL_PENDING, Store
    from .training import fit_bundle, selected_ids

    cfg = Settings.from_env()
    store = Store(cfg.db_path)
    job = store.get(job_id)
    request = json.loads(job["request_json"])
    root = cfg.state_dir.resolve() / "work" / job_id
    output = root / "prepared"
    try:
        models = Models(cfg)
        if job["status"] == "PUBLISH":
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        else:
            if job["status"] in CANCEL_PENDING:
                raise Cancelled()
            if models.read_manifest(request["tenant_id"], request["target_version"], optional=True) is not None:
                raise JobError("TARGET_VERSION_ALREADY_EXISTS")
            base = models.read_manifest(request["tenant_id"], request["base_version"])
            models.validate_base(base)
            ids = selected_ids(base["training_alert_ids"], request, cfg.max_alerts)
            models.download_base(request["tenant_id"], request["base_version"], base, root / "base")
            rows = Alerts(cfg).load(request["tenant_id"], ids, base_manifest=base,
                                    added_alert_ids=request["added_alert_ids"])
            manifest = fit_bundle(cfg, request, base, root / "base", output, rows,
                                  lambda: store.transition(job_id, {"TRAIN"}, "EVALUATE"))
            # Cancellation is checked on this transition and throughout publication.
            store.transition(job_id, {"EVALUATE"}, "PUBLISH")
        while True:
            try:
                store.check_cancel(job_id)
                models.publish(manifest, output, lambda: store.check_cancel(job_id))
                break
            except (BotoCoreError, ClientError) as exc:
                # Retry publication, including an ambiguous manifest PUT. Never report FAILED
                # while a committed manifest might exist. Persisted bundle survives restart.
                logging.warning("publish_pending job=%s error_type=%s", job_id, type(exc).__name__)
                store.check_cancel(job_id)
                time.sleep(5)
        store.transition(job_id, {"PUBLISH"}, "DONE")
    except Cancelled:
        pass  # Parent must observe actual exit before committing CANCELLED.
    except Exception as exc:
        code = exc.code if isinstance(exc, JobError) else type(exc).__name__.upper()
        logging.error("job_failed job=%s code=%s", job_id, code)
        try:
            store.transition(job_id, {"TRAIN", "EVALUATE", "PUBLISH"}, "FAILED", code)
        except (Cancelled, JobError):
            pass
    finally:
        # Only our own local workspace. Never remove a remote tenant/version prefix.
        if store.get(job_id)["status"] in {"DONE", "FAILED"}:
            shutil.rmtree(root, ignore_errors=True)


def cleanup(job_id):
    from .config import Settings
    from .storage import Models
    from .store import Store

    cfg = Settings.from_env()
    store = Store(cfg.db_path)
    job = store.get(job_id)
    request = json.loads(job["request_json"])
    # TRAIN/EVALUATE cannot have written any target objects: no network needed to cancel.
    if job["cancel_from"] == "PUBLISH":
        try:
            Models(cfg).rollback(request["tenant_id"], request["target_version"], job_id)
        except Exception as exc:
            logging.warning("cancel_cleanup_failed job=%s error_type=%s", job_id, type(exc).__name__)
            store.cancel_failed(job_id, "MINIO_DELETE_FAILED", job["cancel_attempt_at"])
            return False  # The waiting HTTP request reports this failure to BE.
    root = cfg.state_dir.resolve() / "work" / job_id
    try:
        if root.exists():
            shutil.rmtree(root)
    except OSError as exc:
        logging.warning("cancel_local_cleanup_failed job=%s error_type=%s", job_id, type(exc).__name__)
        store.cancel_failed(job_id, "LOCAL_DELETE_FAILED", job["cancel_attempt_at"])
        return False
    # Parent waits for THIS process to exit before committing CANCELLED (no callback).
    return True


if __name__ == "__main__":
    bind_parent()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--cleanup" in sys.argv[2:]:
        raise SystemExit(0 if cleanup(sys.argv[1]) else 1)
    run(sys.argv[1])
