from dataclasses import replace

import pytest

from retrain_app.contracts import JobError
from retrain_app.elasticsearch import Alerts


def alert(identifier="RealID", tenant="VND", label="TRUE_POSITIVE"):
    return {"_id": identifier, "found": True,
            "_source": {"id": identifier, "customerCode": tenant, "rawData": {"ai_training": {"label": label}}}}


def event(identifier, command, ref="RealID"):
    return {"_id": identifier, "_index": "case_events", "_source": {"alertId": ref, "event": {"command": command}}}


def page(hits, total=2):
    return {"hits": {"total": {"value": total, "relation": "eq"}, "hits": hits}, "_scroll_id": "scroll"}


def test_reads_all_pages_and_deterministic_first_meaningful(settings):
    client = Alerts(settings)
    responses = [{"docs": [alert()]}, page([event("event-b", "later")]),
                 page([event("event-a", "first")]), {}]
    calls = []
    def call(method, path, body):
        calls.append((method, path, body))
        return responses.pop(0)
    client.call = call
    rows = client.load("VND", ["RealID"])
    assert rows == [{"alert_id": "RealID", "label": 1, "events": {"command": "first"}}]
    assert any(path == "/_search/scroll" and method == "POST" for method, path, _ in calls)
    assert calls[-1][0] == "DELETE"
    assert calls[0][2]["docs"][0]["_id"] == "RealID"


@pytest.mark.parametrize("document,code", [
    (alert(tenant="BIDV"), "TENANT_MISMATCH"),
    (alert(label="MODEL_PREDICTED_TP"), "TRAINING_LABEL_INVALID"),
    ({"_id": "RealID", "found": False}, "ALERT_NOT_FOUND"),
])
def test_bad_alerts_rejected_before_event_lookup(settings, document, code):
    client = Alerts(settings)
    client.call = lambda *args: {"docs": [document]}
    with pytest.raises(JobError, match=code):
        client.load("VND", ["RealID"])


def test_truncated_event_results_are_not_treated_as_complete(settings):
    client = Alerts(settings)
    responses = [page([event("e1", "a")], total=3), page([], total=3), {}]
    client.call = lambda *args: responses.pop(0)
    with pytest.raises(JobError, match="INCOMPLETE_EVENTS"):
        client.events(["RealID"])
