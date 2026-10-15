"""Durable local job ledger and ordered callback outbox (one service replica)."""
import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager

from .contracts import Cancelled, JobError

TERMINAL = {"DONE", "FAILED", "CANCELLED"}
CANCEL_PENDING = {"CANCELLING", "CANCEL_FAILED"}
ACTIVE_SQL = "'QUEUED','TRAIN','EVALUATE','PUBLISH','CANCELLING','CANCEL_FAILED'"


class Store:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(f"""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
                    target_version TEXT NOT NULL, request_json TEXT NOT NULL,
                    request_hash TEXT NOT NULL, status TEXT NOT NULL,
                    seq INTEGER NOT NULL DEFAULT 0, error_code TEXT, cancel_from TEXT,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    UNIQUE(tenant_id, target_version)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_tenant
                    ON jobs(tenant_id) WHERE status IN ({ACTIVE_SQL});
                CREATE TABLE IF NOT EXISTS callbacks (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id), seq INTEGER NOT NULL,
                    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    next_try REAL NOT NULL DEFAULT 0, delivered INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(job_id, seq)
                );
            """)
        # Upgrade existing volumes without losing jobs or pending callbacks.
        with self.transaction() as db:
            columns = {r["name"] for r in db.execute("PRAGMA table_info(jobs)")}
            if "cancel_failure_reported" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN cancel_failure_reported INTEGER NOT NULL DEFAULT 0")
            if "cancel_attempt_at" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN cancel_attempt_at REAL")
                db.execute("UPDATE jobs SET cancel_attempt_at=updated_at WHERE status='CANCELLING'")
            index = db.execute("SELECT sql FROM sqlite_master WHERE name='one_active_per_tenant'").fetchone()
            if index and "CANCEL_FAILED" not in index[0]:
                db.execute("DROP INDEX one_active_per_tenant")
                db.execute(f"CREATE UNIQUE INDEX one_active_per_tenant ON jobs(tenant_id) WHERE status IN ({ACTIVE_SQL})")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def get(self, job_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def active(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute(
                f"SELECT * FROM jobs WHERE status IN ({ACTIVE_SQL}) ORDER BY created_at"
            )]

    def accept(self, request, max_pending):
        body = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(body.encode()).hexdigest()
        now = time.time()
        with self.transaction() as db:
            old = db.execute("SELECT * FROM jobs WHERE job_id=?", (request["job_id"],)).fetchone()
            if old:
                if old["request_hash"] != digest:
                    raise JobError("JOB_ID_REUSED_WITH_DIFFERENT_REQUEST")
                return dict(old), False
            if db.execute(f"SELECT COUNT(*) FROM jobs WHERE status IN ({ACTIVE_SQL})").fetchone()[0] >= max_pending:
                raise JobError("WORKER_QUEUE_FULL")
            try:
                db.execute("""INSERT INTO jobs
                    (job_id,tenant_id,target_version,request_json,request_hash,status,created_at,updated_at)
                    VALUES (?,?,?,?,?,'QUEUED',?,?)""",
                    (request["job_id"], request["tenant_id"], request["target_version"], body, digest, now, now))
            except sqlite3.IntegrityError:
                raise JobError("TENANT_BUSY_OR_TARGET_VERSION_RESERVED") from None
        return self.get(request["job_id"]), True

    def _set(self, db, job_id, status, error_code=None, emit=True):
        row = db.execute("SELECT seq FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        seq = row[0] + int(emit)
        db.execute("UPDATE jobs SET status=?,seq=?,error_code=?,updated_at=? WHERE job_id=?",
                   (status, seq, error_code, time.time(), job_id))
        if emit:
            db.execute("INSERT INTO callbacks(job_id,seq,status) VALUES (?,?,?)", (job_id, seq, status))

    def transition(self, job_id, expected, status, error_code=None):
        with self.transaction() as db:
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row and row[0] in CANCEL_PENDING:
                raise Cancelled()
            if not row or row[0] not in expected:
                raise JobError("INVALID_JOB_TRANSITION")
            self._set(db, job_id, status, error_code)

    def request_cancel(self, job_id, tenant):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=? AND tenant_id=?", (job_id, tenant)).fetchone()
            if not row:
                raise JobError("JOB_NOT_FOUND")
            status = row["status"]
            if status in {"DONE", "FAILED"}:
                raise JobError("JOB_NOT_CANCELLABLE")
            if status == "QUEUED":
                self._set(db, job_id, "CANCELLED", emit=False)
            elif status == "CANCEL_FAILED":
                if not row["cancel_failure_reported"]:
                    # Report the failed attempt once before the next call retries it.
                    # Polling cannot silently clear a failure before BE sees it.
                    db.execute("UPDATE jobs SET cancel_failure_reported=1 WHERE job_id=?", (job_id,))
                    return dict(row)
                db.execute("UPDATE jobs SET cancel_attempt_at=?,cancel_failure_reported=0 WHERE job_id=?",
                           (time.time(), job_id))
                self._set(db, job_id, "CANCELLING", emit=False)
            elif status not in {"CANCELLING", "CANCELLED"}:
                db.execute("UPDATE jobs SET cancel_from=?,cancel_attempt_at=? WHERE job_id=?",
                           (status, time.time(), job_id))
                self._set(db, job_id, "CANCELLING", emit=False)
            # Return this transaction's result even if the child finishes immediately.
            return dict(db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone())

    def cancel_failed(self, job_id, error_code):
        """A cancellation failure is NOT terminal and never emits a training callback."""
        with self.transaction() as db:
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row and row[0] == "CANCELLING":
                db.execute("UPDATE jobs SET cancel_failure_reported=0 WHERE job_id=?", (job_id,))
                self._set(db, job_id, "CANCEL_FAILED", error_code, emit=False)

    def stopped(self, job_id, error_code="WORKER_PROCESS_EXITED"):
        """Only the supervisor calls this, AFTER reaping the subprocess."""
        with self.transaction() as db:
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row[0] not in TERMINAL and row[0] not in {"PUBLISH", *CANCEL_PENDING}:
                self._set(db, job_id, "FAILED", error_code)

    def cancel_cleaned(self, job_id):
        """Called only after the cleanup child successfully exited."""
        with self.transaction() as db:
            row = db.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row[0] in CANCEL_PENDING:
                self._set(db, job_id, "CANCELLED", emit=False)

    def check_cancel(self, job_id):
        if self.get(job_id)["status"] in CANCEL_PENDING:
            raise Cancelled()

    def recover(self):
        # Linux children are killed on parent death. No auto-retraining after restart.
        for job in self.active():
            if job["status"] in {"TRAIN", "EVALUATE", "CANCELLING"}:
                self.stopped(job["job_id"], "WORKER_RESTARTED")
        # CANCELLING resumes cleanup; CANCEL_FAILED waits for BE to retry /cancel.
        # PUBLISH resumes its durable prepared bundle.

    def due_callbacks(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("""
                SELECT c.* FROM callbacks c WHERE delivered=0 AND next_try<=?
                AND NOT EXISTS (SELECT 1 FROM callbacks p
                    WHERE p.job_id=c.job_id AND p.seq<c.seq AND p.delivered=0)
                ORDER BY next_try LIMIT 20
            """, (time.time(),))]

    def callback_result(self, job_id, seq, ok, delay=0):
        with self.transaction() as db:
            db.execute("""UPDATE callbacks SET delivered=?,attempts=attempts+1,next_try=?
                          WHERE job_id=? AND seq=?""", (int(ok), time.time() + delay, job_id, seq))

    def pending_callback_count(self):
        with self.connection() as db:
            return db.execute("SELECT COUNT(*) FROM callbacks WHERE delivered=0").fetchone()[0]
