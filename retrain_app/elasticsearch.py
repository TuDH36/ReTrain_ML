import json
from collections import defaultdict
from urllib.parse import quote

import requests

from .contracts import JobError
from .labels import TrainingLabels
from .preprocessing import is_meaningful


def field(document, path):
    if path in document:
        return document[path]
    node = document
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


class Alerts:
    def __init__(self, settings, session=None):
        self.settings = settings
        self.session = session or requests.Session()
        self.session.verify = settings.es_ca_bundle or settings.es_verify_tls
        if settings.es_api_key:
            self.session.headers["Authorization"] = f"ApiKey {settings.es_api_key}"
        elif settings.es_username:
            self.session.auth = (settings.es_username, settings.es_password)

    def call(self, method, path, body):
        response = self.session.request(method, self.settings.es_url.rstrip("/") + path,
                                        json=body, timeout=self.settings.io_timeout)
        response.raise_for_status()
        result = response.json()
        if result.get("timed_out") or result.get("_shards", {}).get("failed", 0):
            raise JobError("ES_PARTIAL_RESULT")
        return result

    def events(self, refs):
        cfg = self.settings
        query = {"query": {"terms": {cfg.es_event_link_field: refs}},
                 "sort": ["_doc"], "size": cfg.es_batch_size, "track_total_hits": True}
        routing = f"&routing={quote(cfg.es_routing, safe='')}" if cfg.es_routing else ""
        page = self.call("POST", f"/{quote(cfg.es_event_index, safe='')}/_search?scroll=2m{routing}", query)
        scroll_id = page.get("_scroll_id")
        total = page["hits"]["total"]
        if isinstance(total, dict) and total.get("relation") != "eq":
            raise JobError("ES_INEXACT_TOTAL")
        expected = total["value"] if isinstance(total, dict) else total
        seen, result = set(), defaultdict(list)
        source_link = cfg.es_event_link_field.removesuffix(".keyword")
        allowed = set(refs)
        try:
            while True:
                hits = page["hits"]["hits"]
                if not hits:
                    break
                for hit in hits:
                    key = (hit["_index"], hit["_id"])
                    if key in seen:
                        raise JobError("ES_DUPLICATE_EVENT_PAGE")
                    seen.add(key)
                    source = hit["_source"]
                    ref = field(source, source_link)
                    if not isinstance(ref, str) or ref not in allowed:
                        raise JobError("ES_EVENT_LINK_MISMATCH")
                    event = field(source, cfg.es_event_data_field)
                    if isinstance(event, str):
                        try:
                            event = json.loads(event)
                        except ValueError:
                            raise JobError("ES_EVENT_JSON_INVALID") from None
                    if not isinstance(event, dict):
                        raise JobError("ES_EVENT_OBJECT_REQUIRED")
                    result[ref].append((hit["_id"], event))
                if len(seen) == expected:
                    break
                if not scroll_id:
                    raise JobError("ES_SCROLL_ID_MISSING")
                page = self.call("POST", "/_search/scroll", {"scroll": "2m", "scroll_id": scroll_id})
                scroll_id = page.get("_scroll_id", scroll_id)
            if len(seen) != expected:
                raise JobError("ES_INCOMPLETE_EVENTS")
        finally:
            if scroll_id:
                try:
                    self.call("DELETE", "/_search/scroll", {"scroll_id": [scroll_id]})
                except (requests.RequestException, JobError):
                    pass
        return result

    def load(self, tenant, ids, *, base_manifest, added_alert_ids):
        cfg = self.settings
        labels = TrainingLabels(cfg, base_manifest, added_alert_ids)
        rows = []
        all_refs = set()
        for offset in range(0, len(ids), cfg.es_batch_size):
            batch = ids[offset:offset + cfg.es_batch_size]
            docs = [{"_id": i, **({"routing": cfg.es_routing} if cfg.es_routing else {})} for i in batch]
            response = self.call("POST", f"/{quote(cfg.es_alert_index, safe='')}/_mget", {"docs": docs})
            hits = response.get("docs", [])
            if len(hits) != len(batch) or {h.get("_id") for h in hits} != set(batch):
                raise JobError("ES_INCOMPLETE_ALERTS")
            by_ref = {}
            for hit in hits:
                if not hit.get("found") or hit.get("error"):
                    raise JobError("ES_ALERT_NOT_FOUND")
                source = hit["_source"]
                if str(field(source, cfg.es_tenant_field) or "").upper() != tenant:
                    raise JobError("ES_ALERT_TENANT_MISMATCH")
                label = labels.resolve(hit["_id"], field(source, cfg.es_added_label_field),
                                       field(source, cfg.es_label_field))
                ref = hit["_id"] if cfg.es_event_link_source == "_id" else field(source, cfg.es_event_link_source)
                if not isinstance(ref, str) or not ref or ref in all_refs:
                    raise JobError("ES_EVENT_REFERENCE_MISSING_OR_NONUNIQUE")
                all_refs.add(ref)
                by_ref[ref] = (hit["_id"], label)
            events = self.events(list(by_ref))
            for ref, (alert_id, label) in by_ref.items():
                if not events.get(ref):
                    raise JobError("ES_ALERT_EVENTS_MISSING")
                # Same first-meaningful rule as Predict. BE should send this same event order.
                merged = {}
                for _, event in sorted(events[ref], key=lambda item: item[0]):
                    for key, value in event.items():
                        if key not in merged or (not is_meaningful(merged[key]) and is_meaningful(value)):
                            merged[key] = value
                rows.append({"alert_id": alert_id, "label": label, "events": merged})
        return sorted(rows, key=lambda row: row["alert_id"])
