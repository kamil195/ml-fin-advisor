# RETRAIN_COMMAND.md

As of the dependency pinning in `requirements.txt` (`scikit-learn>=1.8,<2.0`),
existing joblib artifacts in `models/serving/` may have been produced with a
different scikit-learn minor version. Pickle/joblib files are generally
forward-compatible within a major line, but if serving refuses to load a model
("module not found", "n_estimators" / attribute errors, or a version-mismatch
warning), retrain rather than fighting the loader.

## Command

Run the full pipeline (data generation → features → training → evaluation →
MLflow → serving artifacts):

```bash
# activate the project venv first
python run_pipeline.py --users 50 --months 12
```

That regenerates everything written to `models/serving/`:

- `classifier_lgb.joblib`, `classifier_meta.joblib`
- `scaler.joblib`, `label_encoder.joblib`, `tfidf_vectorizer.joblib`, `svd_reducer.joblib`
- `feature_columns.json`
- `forecast_results.json`, `budget_results.json`

## Verify after retraining

```bash
python -m pytest -q tests/unit
python -c "from src.serving.app import create_app; create_app(); print('serving app OK')"
```

## When NOT to retrain

If the artifacts load cleanly (no import/attribute errors) the existing models
can keep serving. Retrain only when a load failure shows a version mismatch —
this file exists so you never have to edit the loader "just to make it work".