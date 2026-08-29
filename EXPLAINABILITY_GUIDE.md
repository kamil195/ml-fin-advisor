# Explainability Guide — ML Fin-Advisor

How SHAP and Anchor-style explanations actually work in **this** codebase:
where they are computed, what each field means, and where the limits are.

---

## 1. Where explanations come from

### 1.1 SHAP attributions — `POST /v1/classify`

Entry point: `src/serving/routes/classify.py` → `_compute_shap()`.

Flow:

1. The trained LightGBM base classifier (`classifier_lgb.joblib`) produces
   class probabilities.
2. If `shap` is importable, `shap.TreeExplainer(model)` is built on the base
   classifier and `explainer.shap_values(X)` is called for the single-row
   feature vector `X` (`[text_svd_* ..., numerical_*]`, scaled the same way
   as training).
3. **Shape handling (multiclass):**
   - `shap < ~0.45` returns a *list* of `(n_samples, n_features)` arrays,
     one per class → we index `[class_idx]`.
   - `shap >= ~0.45` returns a 3-D `(n_samples, n_features, n_classes)`
     array → we must take `[0, :, class_idx]`. This exact bug caused every
     `/v1/classify` request to 500 with `shap` installed, which is why the
     code and `tests/unit/test_classify_endpoint.py` both cover the 2-D
     *and* 3-D paths.
4. The predicted class's attribution vector is reduced to the `top_k=5`
   features by absolute value and returned as `shap_features[]`
   (`feature`, `value`, `shap_value`).
5. **Fallback:** if `shap` is missing or raises, or the model has no tree
   explainer, we use `model.feature_importances_` and label the response
   `attribution_method = "global_importance"`. This is a global ranking,
   identical for every request — **not** an explanation of this prediction.
   The API surfaces this honestly via `attribution_method`.

### 1.2 Anchor-style rules — `_build_anchor_rule()`

Real `anchor-exp` explanations leverage model-specific local-surrogate
sampling. In this codebase the anchor is a *deterministic, human-readable
IF-THEN rule* built from the transaction's own features:

- `MCC = <code>` (merchant category code)
- amount bucket (`< $20`, `$20–$100`, `$100–$500`, `≥ $500`)
- `channel = <POS|ONLINE|ATM|TRANSFER|RECURRING>`
- time-of-day band (late-night / morning / afternoon / evening)

Returned as `anchor_rule` on `ClassifyResponse`. Because it is constructed
from the same feature values the model consumes, it is always reproducible
and never requires an external sampler at serving time.

### 1.3 What is NOT explained

- The **meta-learner** stacking step (base LGBM → meta model) is included in
  the probability path, but SHAP is computed on the **base** classifier only.
  Attribution therefore covers the primary model, not the stacked head.
- `global_importance` fallback values must never be presented as
  per-transaction explanations in the UI.

---

## 2. GDPR / compliance pitfalls (3)

1. **Fallback attributions silently masquerading as explanations.**
   If `shap` is unavailable, `attribution_method` becomes
   `global_importance`. Failing to surface this distinction — or stripping
   the field in the frontend — misrepresents a global model ranking as a
   per-instance explanation, which conflicts with the "right to
   explanation" expectations under GDPR Art. 22(3) and similar automated
   decision-making regimes. **Mitigation:** always render
   `attribution_method` and treat `global_importance` as "why this model
   in general", not "why THIS transaction".

2. **Velocity fraud flags on financial data.** The fraud analysis returns
   velocity counts and flags derived from transaction frequency. These are
   indirect inferences about a person's income, lifestyle and behaviour —
   personal data under GDPR. If fraud results are stored or used for:
   blocking, review queues, or risk scoring, this can be a purely automated
   decision with legal effect. **Mitigation:** human-in-the-loop review
   before any blocking action, documented retention limits, and a clear
   avenue for the user to contest the outcome (Art. 22(3)).

3. **Stale cache / model-version drift making explanations wrong.**
   Explanations are cached (`CACHE_TTLS["explanations"] = 30d`). After a
   model retrain, users can still be served 30-day-old attributions for the
   *old* model while the classifier is already the *new* one — an
   explainability mismatch that erodes auditability. **Mitigation:** include
   the model version hash in the cache key and invalidate the
   `explanations` namespace on every model rollout.

---

## 3. Keeping it honest

- Never log `shap_values` raw arrays; they can encode transaction-level PII.
- When increasing `top_k`, remember every extra feature shown is a
  disclosure of modeling behaviour — keep the default at 5.
- If a future model ships with a different explainer (e.g. an LLM), the
  `attribution_method` contract must be extended, not bypassed.