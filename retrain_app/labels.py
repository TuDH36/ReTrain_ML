"""Preserve inherited training labels; correct only newly reported mistakes."""
from .contracts import JobError


LABEL_POLICY = "invert_added_ai_label_preserve_inherited"


def validate_training_labels(manifest):
    labels = manifest.get("training_alert_labels")
    if "training_alert_labels" not in manifest:
        # Older versions read ground truth directly from ES. New versions must
        # always carry the exact labels used, including previously corrected ones.
        if manifest.get("training", {}).get("label_policy") == LABEL_POLICY:
            raise JobError("BASE_TRAINING_LABELS_MISSING_OR_INVALID")
        return None
    if (not isinstance(labels, dict)
            or set(labels) != set(manifest["training_alert_ids"])
            or any(type(value) is not int or value not in (0, 1) for value in labels.values())):
        raise JobError("BASE_TRAINING_LABELS_MISSING_OR_INVALID")
    return labels


class TrainingLabels:
    def __init__(self, settings, base_manifest, added_alert_ids):
        self.settings = settings
        self.base_ids = set(base_manifest["training_alert_ids"])
        self.base_labels = validate_training_labels(base_manifest)
        # An ID already in the base is inherited, even if BE resends it as added.
        self.added_ids = set(added_alert_ids) - self.base_ids

    def resolve(self, alert_id, ai_label, legacy_label):
        if alert_id in self.base_ids:
            if self.base_labels is not None:
                return self.base_labels[alert_id]
            labels = {self.settings.es_positive_label: 1, self.settings.es_negative_label: 0}
            if not isinstance(legacy_label, str) or legacy_label not in labels:
                raise JobError("ES_TRAINING_LABEL_INVALID")
            return labels[legacy_label]
        if alert_id not in self.added_ids:
            raise JobError("TRAINING_ALERT_NOT_SELECTED")
        normalized = " ".join(ai_label.strip().lower().replace("_", " ").split()) if isinstance(ai_label, str) else ""
        # aiLabel is the incorrect prediction reported by SOC, not ground truth.
        corrected = {"true positive": 0, "false positive": 1}
        if normalized not in corrected:
            raise JobError("ES_TRAINING_LABEL_INVALID")
        return corrected[normalized]
