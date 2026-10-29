import asyncio
import logging
import os
import signal
import subprocess
import sys
import time

import httpx

from .errors import training_error
from .store import CANCEL_PENDING

log = logging.getLogger(__name__)

CALLBACK_STATUSES = {
    "TRAIN": "TRAINING",
    "EVALUATE": "EVALUATING",
    "PUBLISH": "PACKAGING",
    "DONE": "SUCCESS",
    "FAILED": "FAILED",
}


class InstanceLock:
    """Refuse a second API worker sharing the same state directory."""

    def __init__(self, path):
        self.file = open(path, "a+b")
        self.file.seek(0)
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("Run exactly one API process per state volume") from None

    def close(self):
        self.file.close()


class Callbacks:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store

    async def deliver(self, client, event):
        if event["status"] in {"CANCELLED", "CANCEL_FAILED"}:
            # Compatibility with undelivered entries from the older callback contract.
            # BE receives cancellation results in the waiting POST /cancel request.
            self.store.callback_result(event["job_id"], event["seq"], True)
            return
        if not self.settings.callback_enabled:
            return
        body = {
            "job_id": event["job_id"],
            "status": CALLBACK_STATUSES[event["status"]],
            "error_message": None,
        }
        if event["status"] == "FAILED":
            job = self.store.get(event["job_id"])
            error = training_error(job["error_code"])
            # Keep SOAR's three-field callback contract. This field contains one
            # of four public codes; BE maps it to the user-facing explanation.
            body["error_message"] = error["error_code"]
            log.warning("training_failed_callback job=%s error_code=%s reason=%s",
                        event["job_id"], error["error_code"], job["error_code"])
        headers = {"X-AI-Service-Token": self.settings.callback_token}
        http_status = None
        try:
            response = await client.post(self.settings.callback_url, json=body, headers=headers)
            http_status = response.status_code
            ok = http_status == 200
        except httpx.HTTPError:
            ok = False
        delay = min(2 ** min(event["attempts"], 10), self.settings.callback_retry_max_seconds)
        self.store.callback_result(event["job_id"], event["seq"], ok, delay)
        if not ok:
            # Keep this event pending: a later phase must never overtake a rejected one.
            log.warning("callback_pending job=%s seq=%s http_status=%s", event["job_id"], event["seq"], http_status)

    async def run(self):
        async with httpx.AsyncClient(timeout=self.settings.callback_timeout, follow_redirects=False) as client:
            while True:
                try:
                    await asyncio.gather(*(self.deliver(client, e) for e in self.store.due_callbacks()))
                except Exception:
                    log.exception("callback_dispatch_failed")
                await asyncio.sleep(0.5)


class Supervisor:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.children = {}
        self.restart_after = {}
        self.tasks = []

    def start(self):
        self.lock = InstanceLock(self.settings.state_dir / "service.lock")
        self.store.recover()
        self.tasks = [asyncio.create_task(self.run()),
                      asyncio.create_task(Callbacks(self.settings, self.store).run())]

    def spawn(self, job):
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(self.settings.train_threads)
        env["RETRAIN_PARENT_PID"] = str(os.getpid())
        # Child receives the same settings via environment, never via CLI secrets.
        for name in self.settings.__dataclass_fields__:
            val = getattr(self.settings, name)
            env[name.upper()] = str(val).lower() if isinstance(val, bool) else str(val)
        cleanup = job["status"] == "CANCELLING"
        command = [sys.executable, "-m", "retrain_app.runner", job["job_id"]]
        if cleanup:
            command.append("--cleanup")
        proc = subprocess.Popen(command,
                                env=env, start_new_session=(os.name != "nt"))
        self.children[job["job_id"]] = {"process": proc, "started": time.monotonic(), "term_at": None,
                                        "cleanup": cleanup, "cancel_attempt_at": job.get("cancel_attempt_at")}

    @staticmethod
    def signal_child(entry, force=False):
        proc = entry["process"]
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.kill() if force else proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    async def run(self):
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("supervisor_tick_failed")
            await asyncio.sleep(0.25)

    def tick(self):
        now = time.monotonic()
        for job_id, entry in list(self.children.items()):
            job = self.store.get(job_id)
            proc = entry["process"]
            if proc.poll() is not None:
                proc.wait()
                if entry["cleanup"]:
                    if proc.returncode == 0:
                        self.store.cancel_cleaned(job_id)
                    elif job["status"] == "CANCELLING" and job["cancel_attempt_at"] == entry.get("cancel_attempt_at"):
                        self.store.cancel_failed(job_id, "CANCEL_CLEANUP_FAILED", job["cancel_attempt_at"])
                else:
                    self.store.stopped(job_id, "TRAIN_TIMEOUT" if entry.get("timed_out") else "WORKER_PROCESS_EXITED")
                del self.children[job_id]
                self.restart_after[job_id] = now + (0 if job["status"] in CANCEL_PENDING else 10)
                continue
            if job["status"] == "CANCELLING" and time.time() - job["cancel_attempt_at"] >= self.settings.cancel_attempt_timeout:
                self.store.cancel_failed(job_id, "CANCEL_TIMEOUT", job["cancel_attempt_at"])
                # Keep tracking the process. Never unlock or start another publisher
                # while a timed-out process might still be running or writing to S3.
                continue
            if job["status"] == "CANCEL_FAILED":
                continue
            if entry["cleanup"]:
                continue
            cooperative_cancel = job["status"] == "CANCELLING" and job["cancel_from"] == "PUBLISH"
            timed_out = now - entry["started"] > self.settings.max_train_seconds and job["status"] != "PUBLISH" and not cooperative_cancel
            if (job["status"] == "CANCELLING" and not cooperative_cancel) or timed_out:
                entry["timed_out"] = timed_out
                try:
                    if entry["term_at"] is None:
                        self.signal_child(entry)
                        entry["term_at"] = now
                    elif now - entry["term_at"] > self.settings.cancel_grace_seconds:
                        self.signal_child(entry, force=True)
                except OSError:
                    if job["status"] in CANCEL_PENDING:
                        self.store.cancel_failed(job_id, "CANCEL_PROCESS_STOP_FAILED", job["cancel_attempt_at"])
                    else:
                        log.exception("process_stop_failed job=%s", job_id)
        for job in self.store.active():
            if len(self.children) >= self.settings.max_concurrent_jobs:
                break
            job_id = job["job_id"]
            if job_id in self.children or now < self.restart_after.get(job_id, 0):
                continue
            if job["status"] not in {"QUEUED", "PUBLISH", "CANCELLING"}:
                continue
            if job["status"] == "QUEUED":
                self.store.transition(job_id, {"QUEUED"}, "TRAIN")
            try:
                self.spawn(job)
            except OSError:
                if job["status"] == "CANCELLING":
                    self.store.cancel_failed(job_id, "CANCEL_PROCESS_START_FAILED", job["cancel_attempt_at"])
                else:
                    self.store.stopped(job_id, "PROCESS_START_FAILED")
                self.restart_after[job_id] = now + 10

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for entry in self.children.values():
            self.signal_child(entry)
        deadline = time.monotonic() + self.settings.cancel_grace_seconds
        while any(e["process"].poll() is None for e in self.children.values()) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        for job_id, entry in self.children.items():
            self.signal_child(entry, force=True)
            await asyncio.to_thread(entry["process"].wait)
            self.store.stopped(job_id, "SERVICE_STOPPED")
        self.lock.close()
