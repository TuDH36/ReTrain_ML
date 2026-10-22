"""Full refit, exact conflict removal, and explicitly in-sample evaluation."""
import json
import os
import re
from datetime import datetime, timezone

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .contracts import JobError
from .labels import LABEL_POLICY
from .field_mapping import FieldMapper, flatten_json
from .preprocessing import _to_str, clean_value, is_meaningful
from .storage import ARTIFACTS, file_info


def selected_ids(base_ids, request, limit):
    base = set(base_ids)
    removed = set(request["removed_alert_ids"])
    if not removed <= base:
        raise JobError("REMOVED_IDS_NOT_IN_BASE")
    result = sorted((base - removed) | set(request["added_alert_ids"]))
    if not result:
        raise JobError("EMPTY_SELECTED_DATASET")
    if len(result) > limit:
        raise JobError("TOO_MANY_TRAINING_ALERTS")
    return result


def prepare(rows, text_cols, cat_cols, aliases):
    columns = list(dict.fromkeys(text_cols + cat_cols))
    if any(not c.startswith("events.") for c in columns):
        raise JobError("FEATURES_MUST_COME_FROM_EVENTS")
    if any(c not in columns for c in aliases.values()):
        raise JobError("INVALID_FEATURE_ALIAS")
    mapper = FieldMapper(columns, aliases)
    values, ids, labels = [], [], []
    for row in rows:
        mapped, _ = mapper.map_payload(flatten_json(row["events"]))
        if not any(is_meaningful(mapped.get(c)) for c in columns):
            continue
        # Notebook clean_data occurs BEFORE exact conflict grouping. IDs are never cleaned.
        values.append({c: re.sub(r"\\+", "/", _to_str(mapped.get(c, "-")).lower()).strip() or "-"
                       for c in columns})
        ids.append(row["alert_id"])
        labels.append(row["label"])
    if not values:
        raise JobError("NO_USABLE_FEATURE_ROWS")
    frame = pd.DataFrame(values, columns=columns)
    label_series = pd.Series(labels, index=frame.index)
    work = frame.assign(__label=label_series)
    conflict = work.groupby(columns, dropna=False, sort=False)["__label"].transform("nunique") > 1
    good = frame.loc[~conflict].copy()
    retained = [v for i, v in enumerate(ids) if not conflict.iloc[i]]
    y = label_series.loc[~conflict].to_numpy(dtype=np.int32)
    if len(set(y)) != 2:
        raise JobError("TRAINING_REQUIRES_BOTH_TP_AND_FP_AFTER_FILTERING")
    for col in columns:
        good[col] = good[col].map(clean_value)
    counts = {"selected_count": len(rows), "empty_feature_count": len(rows) - len(values),
              "conflict_count": int(conflict.sum()), "training_count": len(retained)}
    return good, y, retained, counts


def features(bundle, frame, text_cols, cat_cols, fit=False):
    text = frame[text_cols].agg(" ".join, axis=1)
    char = bundle["tfidf_char.pkl"]
    word = bundle["tfidf_word.pkl"]
    encoder = bundle["cat_encoder.pkl"]
    char_x = char.fit_transform(text) if fit else char.transform(text)
    word_x = word.fit_transform(text) if fit else word.transform(text)
    cat_x = encoder.fit_transform(frame[cat_cols]) if fit else encoder.transform(frame[cat_cols])
    # Use ACTUAL vocabulary dimensions, not the vectorizers' configured max_features.
    offset = char_x.shape[1] + word_x.shape[1]
    matrix = hstack([char_x, word_x, csr_matrix(cat_x + 1)], format="csr")
    return matrix, list(range(offset, offset + len(cat_cols)))


def metrics(y, probabilities, threshold):
    prediction = (probabilities >= threshold).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def write_json_durable(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def fit_bundle(settings, request, base_manifest, base_dir, output_dir, rows, on_evaluate):
    expected_version = base_manifest.get("sklearn_version")
    if expected_version and expected_version != sklearn.__version__:
        raise JobError("SKLEARN_VERSION_MISMATCH")
    base = {name: joblib.load(base_dir / name) for name in ARTIFACTS}
    cat_cols = list(getattr(base["cat_encoder.pkl"], "feature_names_in_", []))
    if not cat_cols or len(set(cat_cols)) != len(cat_cols):
        raise JobError("BASE_CATEGORICAL_COLUMNS_MISSING")
    text_cols = base_manifest["text_cols"]
    frame, y, ids, counts = prepare(rows, text_cols, cat_cols, base_manifest.get("field_aliases", {}))
    bundle = {name: clone(model) for name, model in base.items()}
    classifier = bundle["lgbm_clf.pkl"]
    classifier.set_params(n_jobs=settings.train_threads, verbosity=-1)
    matrix, categorical_indices = features(bundle, frame, text_cols, cat_cols, fit=True)
    # User decision: fit ALL retained records. No split, eval_set, or early stopping.
    classifier.fit(matrix, y, categorical_feature=categorical_indices)
    on_evaluate()
    probabilities = classifier.predict_proba(matrix)[:, 1]
    if not np.isfinite(probabilities).all() or list(classifier.classes_) != [0, 1]:
        raise JobError("INVALID_MODEL_PREDICTIONS")
    threshold = float(base_manifest.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise JobError("INVALID_THRESHOLD")
    scores = {str(t): metrics(y, probabilities, t) for t in sorted({0.5, 0.9, threshold})}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in bundle.items():
        with (output_dir / name).open("wb") as stream:
            joblib.dump(model, stream)
            stream.flush()
            os.fsync(stream.fileno())
    reloaded = {name: joblib.load(output_dir / name) for name in ARTIFACTS}
    reloaded_matrix, _ = features(reloaded, frame, text_cols, cat_cols)
    actual = reloaded["lgbm_clf.pkl"].predict_proba(reloaded_matrix)[:, 1]
    if not np.allclose(probabilities, actual, rtol=1e-12, atol=1e-12):
        raise JobError("ARTIFACT_RELOAD_PREDICTION_MISMATCH")
    inherited = set(base_manifest["training_alert_ids"])
    actual_ids = set(ids)
    manifest = {key: base_manifest[key] for key in
                ("text_cols", "threshold", "min_important_fields", "required_fields", "field_aliases")
                if key in base_manifest}
    manifest.update({
        "schema_version": 1, "tenant": request["tenant_id"], "model_version": request["target_version"],
        "base_version": request["base_version"], "job_id": request["job_id"],
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "created_by": request["created_by"], "training_alert_ids": ids,
        "training_alert_labels": {alert_id: int(label) for alert_id, label in zip(ids, y)},
        "added_count": len(actual_ids - inherited), "removed_count": len(inherited - actual_ids),
        "sklearn_version": sklearn.__version__, "lightgbm_version": lightgbm.__version__,
        "_n_cat_cols": len(cat_cols), "artifacts": [file_info(output_dir / name) for name in sorted(ARTIFACTS)],
        "evaluation": {"dataset": "train", "is_independent_test": False, "sample_count": len(ids),
                       "positive_label": 1, "metrics_by_threshold": scores},
        "training": {"method": "full_refit", "validation": "none", "early_stopping": False,
                     "label_policy": LABEL_POLICY,
                     "n_estimators": classifier.get_params()["n_estimators"],
                     "conflict_policy": "drop_all_exact_groups_before_noise_cleaning", **counts,
                     "requested_added_count": len(request["added_alert_ids"]),
                     "requested_removed_count": len(request["removed_alert_ids"])},
        "data_source": {"alert_index": settings.es_alert_index, "event_index": settings.es_event_index,
                        "tenant_field": settings.es_tenant_field, "label_field": settings.es_label_field,
                        "added_label_field": settings.es_added_label_field,
                        "inherited_label_source": "base_manifest.training_alert_labels_or_legacy_label_field",
                        "event_link_source": settings.es_event_link_source,
                        "event_link_field": settings.es_event_link_field,
                        "event_data_field": settings.es_event_data_field,
                        "event_order": "ES_event_id_ascending_first_meaningful",
                        "snapshot": "ES_values_read_at_job_execution"},
    })
    write_json_durable(output_dir / "manifest.json", manifest)
    return manifest
