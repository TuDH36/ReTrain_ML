"""Wait for a cancellation result without blocking other API requests."""
import asyncio
import time

from .contracts import JobError


class Cancellations:
    def __init__(self, settings, store):
        self.settings, self.store = settings, store
        self.tasks = {}

    async def cancel(self, job_id, tenant_id):
        key = (job_id, tenant_id)
        task = self.tasks.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._wait(job_id, tenant_id))
            self.tasks[key] = task
            task.add_done_callback(lambda done: self._finished(key, done))
        # A disconnected caller must not interrupt cancellation or other callers.
        return await asyncio.shield(task)

    def _finished(self, key, task):
        if self.tasks.get(key) is task:
            del self.tasks[key]
        if not task.cancelled():
            task.exception()  # Retrieve errors even when every caller disconnected.

    async def _wait(self, job_id, tenant_id):
        row = await asyncio.to_thread(self.store.request_cancel, job_id, tenant_id)
        if row["status"] == "CANCELLED":
            return row
        attempt = row["cancel_attempt_at"]
        remaining = max(0, self.settings.cancel_attempt_timeout - (time.time() - attempt))
        deadline = time.monotonic() + remaining
        while row["status"] == "CANCELLING":
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await asyncio.to_thread(self.store.cancel_failed, job_id, "CANCEL_TIMEOUT", attempt)
            else:
                await asyncio.sleep(min(0.2, remaining))
            row = await asyncio.to_thread(self.store.get, job_id)
            if row is None:
                raise JobError("JOB_NOT_FOUND")
        if row["status"] not in {"CANCELLED", "CANCEL_FAILED"}:
            raise RuntimeError("Unexpected cancellation result")
        return row

    async def close(self):
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
