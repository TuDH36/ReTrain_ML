import importlib.util
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from retrain_app.contracts import JobError
from retrain_app.storage import ARTIFACTS, file_info
from retrain_app.training import features, fit_bundle, prepare, selected_ids


def synthetic_base(directory):
    directory.mkdir(parents=True)
    cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat.fit(pd.DataFrame({"events.host": ["host-a", "host-b"]}))
    objects = {
        "cat_encoder.pkl": cat,
        "tfidf_char.pkl": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=1000),
        "tfidf_word.pkl": TfidfVectorizer(ngram_range=(1, 2), max_features=1000),
        "lgbm_clf.pkl": LGBMClassifier(n_estimators=20, num_leaves=7, min_child_samples=2, random_state=42, n_jobs=1, verbosity=-1),
    }
    objects["tfidf_char.pkl"].fit(["original baseline command"])
    objects["tfidf_word.pkl"].fit(["original baseline command"])
    for name, obj in objects.items():
        joblib.dump(obj, directory / name)
    return {"tenant": "VND", "model_version": "v1.0", "threshold": 0.5,
            "sklearn_version": sklearn.__version__, "text_cols": ["events.command"],
            "training_alert_ids": [f"Baseline-{i}" for i in range(8)],
            "field_aliases": {}, "min_important_fields": 1, "required_fields": [],
            "artifacts": [file_info(directory / n) for n in sorted(ARTIFACTS)]}


def synthetic_rows():
    return [{"alert_id": f"CaseSensitiveID-{i}", "label": i % 2,
             "events": {"command": "malicious shell attack" if i % 2 else "normal trusted backup",
                        "host": "host-a" if i % 2 else "host-b"}} for i in range(40)]


def test_full_refit_train_metrics_reload_and_actual_categorical_offset(settings, request_body, tmp_path):
    base_dir = tmp_path / "base"
    base = synthetic_base(base_dir)
    rows = synthetic_rows()
    stages = []
    output = tmp_path / "result"
    manifest = fit_bundle(settings, request_body, base, base_dir, output, rows, lambda: stages.append("EVALUATE"))
    assert stages == ["EVALUATE"]
    assert manifest["training_alert_ids"] == [r["alert_id"] for r in rows]
    assert manifest["evaluation"]["dataset"] == "train"
    assert manifest["evaluation"]["sample_count"] == 40
    assert not manifest["training"]["early_stopping"]
    assert manifest["evaluation"]["metrics_by_threshold"]["0.5"]["f1"] > 0.9
    assert {p.name for p in output.iterdir()} == ARTIFACTS | {"manifest.json"}
    bundle = {name: joblib.load(output / name) for name in ARTIFACTS}
    # Vocabulary is re-fit, not reused from the original base vectorizers.
    assert "baseline" not in bundle["tfidf_word.pkl"].vocabulary_
    frame, y, _, _ = prepare(rows, ["events.command"], ["events.host"], {})
    matrix, categorical = features(bundle, frame, ["events.command"], ["events.host"])
    assert categorical == [matrix.shape[1] - 1] and categorical[0] != 2000
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_exact_conflicts_removed_before_noise_and_ids_preserved():
    rows = synthetic_rows()[:4]
    rows += [
        {"alert_id": "ConflictA", "label": 0, "events": {"command": "Same COMMAND", "host": "host-z"}},
        {"alert_id": "ConflictB", "label": 1, "events": {"command": "same command", "host": "host-z"}},
        {"alert_id": "Empty", "label": 1, "events": {"command": "-", "host": "none"}},
    ]
    _, y, ids, counts = prepare(rows, ["events.command"], ["events.host"], {})
    assert counts == {"selected_count": 7, "empty_feature_count": 1, "conflict_count": 2, "training_count": 4}
    assert ids == [r["alert_id"] for r in rows[:4]]
    assert len(y) == 4


def test_dataset_delta_not_new_only_and_single_class_rejected(request_body):
    request = dict(request_body, removed_alert_ids=["Old_A"])
    assert selected_ids(["Old_A", "Old_B"], request, 100) == ["NewID_A", "NewID_B", "Old_B"]
    with pytest.raises(JobError, match="REMOVED_IDS_NOT_IN_BASE"):
        selected_ids(["Other"], request, 100)
    with pytest.raises(JobError, match="BOTH_TP_AND_FP"):
        prepare([synthetic_rows()[0]], ["events.command"], ["events.host"], {})


def test_preprocessing_matches_prediction_source_if_available():
    # Runtime stays standalone. In this repository, ensure copied cleaning rules did not drift.
    source = Path(__file__).resolve().parents[2] / "Product/api/app/preprocessing.py"
    if not source.exists():
        pytest.skip("Predict source is not part of a standalone deployment")
    spec = importlib.util.spec_from_file_location("prediction_preprocessing", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from retrain_app.preprocessing import clean_value, is_meaningful
    for value in [None, "-", "NaN", r"C:\Windows\System32", "aBcDef01234567890123456789", 1720000000123,
                  ["PowerShell", "-enc"], True, 200.0, "2026-10-15T12:13:14Z"]:
        assert clean_value(value) == module.clean_value(value)
        assert is_meaningful(value) == module.is_meaningful(value)


def test_repository_base_bundle_can_be_refit_with_synthetic_rows(settings, request_body, tmp_path):
    default = Path(__file__).resolve().parents[2] / "model/VND"
    base_dir = Path(os.environ.get("BASE_BUNDLE_DIR", str(default)))
    if not (base_dir / "manifest.json").exists():
        pytest.skip("Existing VND artifacts are optional for standalone deployments")
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    from retrain_app.storage import Models
    Models.validate_base(manifest)
    for entry in manifest["artifacts"]:
        assert file_info(base_dir / entry["file"])["sha256"] == entry["sha256"]
    encoder = joblib.load(base_dir / "cat_encoder.pkl")
    columns = list(dict.fromkeys(manifest["text_cols"] + list(encoder.feature_names_in_)))
    rows = [{"alert_id": f"Synthetic-{i}", "label": i % 2,
             "events": {col: "attack shell" if i % 2 else "trusted backup" for col in columns}}
            for i in range(60)]
    result = fit_bundle(settings, request_body, manifest, base_dir, tmp_path / "actual-base-refit", rows, lambda: None)
    assert result["training"]["training_count"] == 60
    assert result["_n_cat_cols"] == len(encoder.feature_names_in_)
    assert result["training_alert_ids"] == [r["alert_id"] for r in rows]
