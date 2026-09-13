"""AUTH STEP 3 — ownership & cross-user isolation for protected financial routes.

The authenticated JWT ``sub`` is the only authoritative identity: a caller
cannot use a body/path ``user_id`` to reach another user's resources. All
ownership checks return 403 and run AFTER authentication (401) but BEFORE any
cache read or financial computation.

Identity conventions used here:

* ``USER_A`` / ``USER_B`` — two distinct authenticated subjects (mocked JWTs).
* ``DEMO_A`` / ``DEMO_B`` — subjects that exist in the local serving
  artifact's per-user budget data (training-time demo users).
"""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.auth import require_auth
from src.serving.routes import live as live_module

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

USER_A = "user-AAA"
USER_B = "user-BBB"
DEMO_A = "u-00000001-0000-0000-0000-000000000001"
DEMO_B = "u-00000002-0000-0000-0000-000000000002"

TXN_A = {
    "user_id": USER_A,
    "timestamp": "2026-03-05T12:00:00",
    "amount": -100.0,
    "currency": "USD",
    "merchant_name": "FreshMart",
    "merchant_mcc": 5411,
    "account_type": "CHECKING",
    "channel": "POS",
}


def _advise_txn(
    user_id, amount, merchant, mcc=5411, channel="POS",
    l1="FOOD & DINING", l2="Groceries",
) -> dict:
    return {
        "user_id": user_id,
        "timestamp": "2026-03-05T12:00:00",
        "amount": amount,
        "merchant_name": merchant,
        "merchant_mcc": mcc,
        "account_type": "CHECKING",
        "channel": channel,
        "category_l1": l1,
        "category_l2": l2,
        "confidence": 0.95,
    }


def _advise_body(user_id=USER_A) -> dict:
    return {
        "transactions": [
            _advise_txn(user_id, 5000.0, "Payroll - Acme Corp", mcc=0,
                        channel="TRANSFER", l1="FINANCIAL", l2="Income"),
            _advise_txn(user_id, -1500.0, "LandlordCo", mcc=6513,
                        channel="TRANSFER", l1="HOUSING", l2="Rent/Mortgage"),
            _advise_txn(user_id, -300.0, "FreshMart"),
            _advise_txn(user_id, -300.0, "Bistro One", mcc=5812, l2="Restaurants"),
        ]
    }


_CSV_HEADER = ("user_id,timestamp,amount,currency,merchant_name,"
               "merchant_mcc,account_type,channel")


def _csv_body(user_id=USER_A) -> str:
    return "\n".join([
        _CSV_HEADER,
        f"{user_id},2026-03-05T12:00:00,-100.0,USD,FreshMart,5411,CHECKING,POS",
        f"{user_id},2026-03-06T12:00:00,-250.0,USD,Bistro One,5812,CHECKING,POS",
    ]) + "\n"


def _csv_headers():
    return {"Content-Type": "text/csv; charset=utf-8"}


@pytest.fixture
def client_a(auth_env, make_auth_headers):
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers(sub=USER_A))
        yield c


@pytest.fixture
def client_b(auth_env, make_auth_headers):
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers(sub=USER_B))
        yield c


@pytest.fixture
def client_demo_a(auth_env, make_auth_headers):
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers(sub=DEMO_A))
        yield c


@pytest.fixture
def client_short_demo(auth_env, make_auth_headers):
    """A subject equal to the 12-char prefix of a real demo budget user."""
    with TestClient(create_app()) as c:
        c.headers.update(make_auth_headers(sub=DEMO_A[:12]))
        yield c


@pytest.fixture
def app_shared(auth_env, make_auth_headers):
    """One app instance (shared cache) on which per-request identities switch."""
    with TestClient(create_app()) as c:
        yield c


# ── A. path ownership ──────────────────────────────────────────────────────────


def test_forecast_own_path_allowed(client_a):
    r = client_a.get(f"/v1/forecast/{USER_A}")
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 200, r.text
    assert r.json()["user_id"] == USER_A


def test_forecast_other_user_path_forbidden(client_a):
    assert client_a.get(f"/v1/forecast/{USER_B}").status_code == 403


def test_budget_own_demo_subject_allowed(client_demo_a):
    """Own subject that exists in the per-user artifact → exact match serves it."""
    r = client_demo_a.get(f"/v1/budget/{DEMO_A}")
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 200, r.text
    assert r.json()["user_id"] == DEMO_A


def test_budget_other_user_path_forbidden(client_a):
    assert client_a.get(f"/v1/budget/{USER_B}").status_code == 403


def test_budget_own_unknown_subject_never_falls_back(client_b):
    """No first-user demo fallback: own subject missing from the artifact → 404,
    never somebody else's budget presented under the caller's id."""
    r = client_b.get(f"/v1/budget/{USER_B}")
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 404, r.text
    assert "No budget data for user" in r.json()["detail"]


def test_budget_no_prefix_matching(client_short_demo):
    """A prefix of a real demo user id is NOT sufficient (exact match only)."""
    r = client_short_demo.get(f"/v1/budget/{DEMO_A[:12]}")
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 404, r.text


# ── B. body ownership ─────────────────────────────────────────────────────────


def test_classify_own_transaction_allowed(client_a):
    r = client_a.post("/v1/classify", json={"transaction": TXN_A})
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 200, r.text


def test_classify_foreign_transaction_forbidden(client_a):
    foreign = dict(TXN_A, user_id=USER_B)
    assert client_a.post(
        "/v1/classify", json={"transaction": foreign}
    ).status_code == 403


def test_advise_own_history_allowed(client_a):
    assert client_a.post("/consumer/advise", json=_advise_body(USER_A)).status_code == 200


def test_advise_foreign_transaction_forbidden(client_a):
    body = _advise_body(USER_A)
    body["transactions"][2]["user_id"] = USER_B
    assert client_a.post("/consumer/advise", json=body).status_code == 403


def test_advise_top_level_foreign_user_id_forbidden(client_a):
    body = _advise_body(USER_A)
    body["user_id"] = USER_B
    assert client_a.post("/consumer/advise", json=body).status_code == 403


def test_advise_top_level_own_user_id_allowed(client_a):
    body = _advise_body(USER_A)
    body["user_id"] = USER_A
    assert client_a.post("/consumer/advise", json=body).status_code == 200


# ── C. CSV ingestion ownership ────────────────────────────────────────────────


def test_csv_own_rows_associated_with_sub(client_a):
    r = client_a.post(
        "/consumer/transactions/ingest-csv",
        content=_csv_body(USER_A).encode(),
        headers=_csv_headers(),
    )
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["valid_rows"] == 2
    # Ownership is always recorded under the authenticated subject.
    assert {item["user_id"] for item in out["classified"]} == {USER_A}


def test_csv_foreign_rows_forbidden(client_a):
    r = client_a.post(
        "/consumer/transactions/ingest-csv",
        content=_csv_body(USER_B).encode(),
        headers=_csv_headers(),
    )
    assert r.status_code == 403, r.text


def test_csv_mixed_rows_foreign_forbidden(client_a):
    """Even a single foreign row rejects the whole upload (fail closed)."""
    body = _csv_body(USER_A)
    body += f"{USER_B},2026-03-07T12:00:00,-300.0,USD,OtherCo,5311,CHECKING,POS\n"
    r = client_a.post(
        "/consumer/transactions/ingest-csv",
        content=body.encode(),
        headers=_csv_headers(),
    )
    assert r.status_code == 403, r.text


# ── D. live endpoints derive identity from principal.sub ─────────────────────


@pytest.fixture
def live_client(auth_env):
    """Bare app mirroring production wiring: live router behind require_auth."""
    app = FastAPI()
    app.include_router(live_module.router, dependencies=[Depends(require_auth)])
    with TestClient(app) as c:
        yield c


def test_classify_live_uses_principal_sub_for_transaction(
    live_client, make_auth_headers, monkeypatch
):
    """The transaction classified by /consumer/classify/live is always minted
    with the authenticated subject, never a caller-supplied identity."""
    from src.serving.routes.classify import ClassifyResponse

    captured = {}

    async def _stub_classify(cr, req, principal):
        captured["user_id"] = cr.transaction.user_id
        return ClassifyResponse(
            category_l1="FOOD & DINING",
            category_l2="Groceries",
            confidence=0.9,
            top_3=[],
        )

    monkeypatch.setattr(
        "src.serving.routes.classify.classify_transaction", _stub_classify
    )

    r = live_client.post(
        "/consumer/classify/live",
        json={"merchant": "M", "amount": -5, "date": "2026-03-02"},
        headers=make_auth_headers(sub=USER_A),
    )
    assert r.status_code == 200, r.text
    assert captured.get("user_id") == USER_A


def test_forecast_live_uses_principal_sub_for_forecast(
    live_client, make_auth_headers, monkeypatch
):
    """The Prophet fit/predict in /consumer/forecast/live is scoped to the
    authenticated subject."""
    pm = pytest.importorskip("src.models.forecaster.prophet_model")
    seen = []

    class _StubProphet:
        def fit(self, df, **kw):
            seen.append(("fit", kw["user_id"]))

        def predict(self, **kw):
            seen.append(("predict", kw["user_id"]))

            class _FC:
                p10 = [10.0, 10.0, 10.0, 10.0]
                p50 = [20.0, 20.0, 20.0, 20.0]
                p90 = [30.0, 30.0, 30.0, 30.0]

            return _FC()

    monkeypatch.setattr(pm, "ProphetModel", _StubProphet)

    body = {
        "transactions": [
            {"date": "2026-03-01", "merchant": "M1", "amount": -100, "category": "Groceries"},
            {"date": "2026-03-08", "merchant": "M2", "amount": -50, "category": "Groceries"},
        ],
        "horizon_days": 30,
    }
    r = live_client.post(
        "/consumer/forecast/live", json=body, headers=make_auth_headers(sub=USER_B)
    )
    assert r.status_code == 200, r.text
    assert seen, "ProphetModel was never reached"
    assert all(sub == USER_B for _, sub in seen)


def test_live_endpoints_require_token(live_client):
    assert live_client.post(
        "/consumer/classify/live",
        json={"merchant": "M", "amount": -5, "date": "2026-03-02"},
    ).status_code == 401
    assert live_client.post(
        "/consumer/forecast/live", json={"transactions": []}
    ).status_code == 401


# ── E. cache isolation & cross-user matrix ────────────────────────────────────


def test_forecast_cache_entries_are_user_scoped(app_shared, make_auth_headers):
    """B cannot read A's cached forecast: the ownership gate (403) runs before
    any cache read, and keys are namespaced by the authenticated subject."""
    r = app_shared.get(f"/v1/forecast/{USER_A}", headers=make_auth_headers(sub=USER_A))
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code == 200

    # Even when A's forecast is cached on this shared app, B gets 403 for A's path.
    assert app_shared.get(
        f"/v1/forecast/{USER_A}", headers=make_auth_headers(sub=USER_B)
    ).status_code == 403

    cache = app_shared.app.state.cache  # in-memory fallback cache
    assert cache.get("forecasts", USER_A, "30", "all") is not None
    assert cache.get("forecasts", USER_B, "30", "all") is None


def test_budget_cache_entries_are_user_scoped(app_shared, make_auth_headers):
    r = app_shared.get(f"/v1/budget/{DEMO_A}", headers=make_auth_headers(sub=DEMO_A))
    if r.status_code == 503:
        pytest.skip("serving artifacts not present in models/serving/")
    assert r.status_code in (200, 404)

    assert app_shared.get(
        f"/v1/budget/{DEMO_A}", headers=make_auth_headers(sub=DEMO_B)
    ).status_code == 403

    cache = app_shared.app.state.cache
    # B's namespace never exists even though B attempted A's path.
    assert cache.get("budgets", DEMO_B) is None


def test_cross_user_matrix_rejects_both_directions(client_a, client_b):
    """User A cannot reach B's resources and vice versa (path + body)."""
    assert client_a.get(f"/v1/forecast/{USER_B}").status_code == 403
    assert client_a.get(f"/v1/budget/{USER_B}").status_code == 403
    assert client_b.get(f"/v1/forecast/{USER_A}").status_code == 403
    assert client_b.get(f"/v1/budget/{USER_A}").status_code == 403

    # Foreign body identity is rejected in both directions, regardless of
    # whether model artifacts are present (403 is pre-computation).
    for cli, mine, theirs in (
        (client_a, USER_A, USER_B),
        (client_b, USER_B, USER_A),
    ):
        foreign = dict(TXN_A, user_id=theirs)
        assert cli.post("/v1/classify", json={"transaction": foreign}).status_code == 403

        r = cli.post(
            "/consumer/transactions/ingest-csv",
            content=_csv_body(theirs).encode(),
            headers=_csv_headers(),
        )
        assert r.status_code == 403

        advice = _advise_body(mine)
        advice["transactions"][0]["user_id"] = theirs
        assert cli.post("/consumer/advise", json=advice).status_code == 403