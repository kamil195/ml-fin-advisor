"""
Endpoint-level tests for POST /v1/classify and POST /consumer/classify/live.

These exist because a bug in ``_compute_shap`` made /v1/classify return 500 on
every request whenever the ``shap`` package was installed, and the entire test
suite stayed green — nothing exercised the endpoint at all.

The two attribution paths must BOTH be covered:

  * ``shap`` installed   → TreeExplainer, which for multiclass LightGBM returns
    a 3-D (n_samples, n_features, n_classes) array in shap >= ~0.45.
  * ``shap`` absent      → global feature-importance fallback. This is the path
    production takes today, because ``shap`` is not in requirements-serve.txt.
"""

from __future__ import annotations

import builtins

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


TXN = {
    "user_id": "test_user",
    "timestamp": "2026-03-02T14:30:00",
    "amount": -6.75,
    "currency": "USD",
    "merchant_name": "Starbucks",
    "merchant_mcc": 5814,
    "account_type": "CHECKING",
    "channel": "POS",
    "raw_description": "STARBUCKS STORE #1234",
}


@pytest.fixture
def client(auth_env):
    """Authenticated test client.

    The endpoint tests call financial routes that are protected by
    ``require_auth``; the shared ``auth_env`` fixture from conftest.py injects
    the mocked Supabase JWT resolver and sets ``SUPABASE_URL`` so verification
    succeeds. These tests never reach a real Supabase instance.
    """
    # TestClient must be used as a context manager, otherwise lifespan never
    # runs and app.state.classifier is missing entirely.
    with TestClient(create_app()) as c:
        yield c


def _skip_if_no_model(resp):
    if resp.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")


def test_classify_returns_200(client, make_auth_headers):
    resp = client.post(
        "/v1/classify", json={"transaction": TXN}, headers=make_auth_headers()
    )
    _skip_if_no_model(resp)
    assert resp.status_code == 200, resp.text


def test_classify_response_shape(client, make_auth_headers):
    resp = client.post(
        "/v1/classify", json={"transaction": TXN}, headers=make_auth_headers()
    )
    _skip_if_no_model(resp)
    body = resp.json()

    for key in (
        "category_l1",
        "category_l2",
        "confidence",
        "top_3",
        "shap_features",
        "attribution_method",
        "anchor_rule",
    ):
        assert key in body, f"missing {key}"

    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top_3"]) == 3
    assert body["attribution_method"] in {"shap", "global_importance"}


def test_classify_attributions_are_scalars(client, make_auth_headers):
    """
    Regression test for the 3-D SHAP array bug.

    With shap >= ~0.45 installed, ``shap_values`` is (n_samples, n_features,
    n_classes). Indexing it as if it were 2-D left ``sv`` two-dimensional, so
    the top-k loop compared a numpy array against an int and raised
    "The truth value of an array with more than one element is ambiguous",
    which the route converted into a 500.
    """
    resp = client.post(
        "/v1/classify", json={"transaction": TXN}, headers=make_auth_headers()
    )
    _skip_if_no_model(resp)
    assert resp.status_code == 200, resp.text

    for attr in resp.json()["shap_features"]:
        assert isinstance(attr["feature"], str)
        assert isinstance(attr["shap_value"], (int, float))
        assert isinstance(attr["value"], (int, float))


def test_classify_works_without_shap_installed(client, monkeypatch, make_auth_headers):
    """
    The fallback path is what production runs, since shap is not a declared
    serving dependency. It must return 200 and label itself honestly.
    """
    real_import = builtins.__import__

    def _no_shap(name, *args, **kwargs):
        if name == "shap" or name.startswith("shap."):
            raise ImportError("shap is not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_shap)

    resp = client.post(
        "/v1/classify", json={"transaction": TXN}, headers=make_auth_headers()
    )
    _skip_if_no_model(resp)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["attribution_method"] == "global_importance", (
        "fallback attributions must be labelled as global importance, not "
        "presented as per-transaction SHAP values"
    )


def test_classify_live_returns_200_and_labels_attributions(client, make_auth_headers):
    resp = client.post(
        "/consumer/classify/live",
        json={"merchant": "Starbucks", "amount": -6.75, "date": "2026-03-02"},
        headers=make_auth_headers(),
    )
    _skip_if_no_model(resp)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["model_category"]
    assert body["attribution_method"] in {"shap", "global_importance"}


def test_classify_rejects_malformed_body(client, make_auth_headers):
    resp = client.post(
        "/v1/classify",
        json={"transaction": {"amount": -1.0}},
        headers=make_auth_headers(),
    )
    assert resp.status_code == 422
