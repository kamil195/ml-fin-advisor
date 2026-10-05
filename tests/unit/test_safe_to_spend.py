"""STEP 14 — Safe-to-Spend engine and endpoint tests.

Covers the rules the product promise ("know what you can safely spend before
payday") depends on:

* the arithmetic identity
  ``funds − protected_obligations − expected_spending − safety_buffer − scenario``
* owner isolation: a user's amount is built from ONLY their own persisted rows,
  another user's profile/transactions cannot change it, and no request field can
  select whose finances are computed
* explicit statuses instead of fabricated numbers (``missing_profile_data`` /
  ``missing_balance`` / ``missing_payday`` / ``insufficient_history``) each with
  a ``requirements`` block saying exactly what is missing
* payday resolved from the caller's own profile value or measured from their own
  income-deposit cadence — a payday is never assumed
* protected obligations inferred only from the caller's own observed monthly
  cadence, counted once (never double-counted with the forecast), with every
  exclusion visible and ``Uncategorized`` never treated as protected
* ``limited_history`` labelling, a negative amount that is never floored, and
  templated (never LLM) explanations/assumptions
* cache scoping, ``as_of`` freshness, no caching without an amount, and
  invalidation after ingest / profile update / data deletion
* generic 503/500 errors and a strictly read-only endpoint (scenarios persist
  nothing)

No network and no real Postgres: ``FakeStore`` implements the owner-scoped
repository contract and every client uses a guaranteed in-memory cache.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, time as dtime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.models.forecaster import personal_forecast as pf
from src.services import safe_to_spend as sts
from src.serving.app import create_app
from src.serving.cache import CacheClient
from src.serving.persistence import (
    DETAIL_503,
    DETAIL_NOT_CONFIGURED,
    PersistenceUnavailable,
)
from src.serving.routes import ingest as ingest_module
from src.serving.routes import safe_to_spend as sts_route
from src.utils.constants import HARD_PROTECTED_CATEGORIES

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

USER_A = "user-AAA"
USER_B = "user-BBB"
TODAY = datetime.now(timezone.utc).date()

RENT = "Rent/Mortgage"
UTILITIES = "Utilities"
TAXES = "Taxes"
LOAN = "Loan Payments"

CSV_HEADER = (
    "user_id,timestamp,amount,currency,merchant_name,merchant_mcc,"
    "account_type,channel"
)


@pytest.fixture(autouse=True)
def _no_real_db(monkeypatch):
    """Deterministic startup: no DATABASE_URL (the test installs its store) and
    default rate-limit configuration (a fresh limiter is built per app)."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("RATE_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("API_KEYS", raising=False)


# ── deterministic row builders (relative dates, never a fixed calendar day) ───


def _day(offset: int) -> date:
    return TODAY + timedelta(days=offset)


def _at(offset_days: int, hour: int = 12) -> datetime:
    return datetime.combine(_day(offset_days), dtime(hour), tzinfo=timezone.utc)


def _row(
    offset_days: int,
    amount: float,
    category: str | None = "Groceries",
    *,
    currency: str | None = "USD",
    pending: bool = False,
) -> dict:
    return {
        "occurred_at": _at(offset_days),
        "amount": amount,
        "currency": currency,
        "category_l2": category,
        "is_pending": pending,
    }


def _spend_rows(
    days: int, amount: float = -100.0, category: str = "Groceries"
) -> list[dict]:
    """One deterministic spend per day for ``days`` days, ending yesterday."""
    return [_row(-(days - i), amount, category) for i in range(days)]


def _income_rows(*offsets: int, amount: float = 5000.0) -> list[dict]:
    """Classified income deposits on the given day offsets (payday evidence)."""
    return [_row(offset, amount, "Income") for offset in offsets]


def _norm(rows: list[dict]) -> list[sts.HistoryRow]:
    return sts.normalise_history(rows)


def _compute(
    rows: list[dict],
    *,
    funds: float | None = 10_000.0,
    buffer: float | None = None,
    payday: date | None = None,
    scenario: float = 0.0,
    label: str | None = None,
    profile: bool = True,
    today: date = TODAY,
) -> sts.SafeToSpendComputation:
    """Call the engine exactly the way the serving route does."""
    return sts.compute_safe_to_spend(
        rows=rows,
        today=today,
        available_funds=funds,
        safety_buffer=buffer,
        configured_payday=payday,
        scenario_amount=scenario,
        scenario_label=label,
        profile_present=profile,
    )


# ── serving doubles (owner-scoped repository + in-memory cache) ───────────────


class FakeStore:
    """Owner-scoped repository fake (mirrors PostgresStore's public contract)."""

    def __init__(self) -> None:
        self.profiles: dict[str, dict] = {}
        self.history: dict[str, list[dict]] = {}
        self.fetch_calls: list[str] = []
        self.fail_methods: set[str] = set()
        self.writes: list[tuple[str, str]] = []

    def _maybe_fail(self, method: str) -> None:
        if method in self.fail_methods:
            raise PersistenceUnavailable("InjectedFailure")

    # ── read path used by Safe-to-Spend ────────────────────────────────────
    def get_profile(self, owner_sub: str) -> dict | None:
        self._maybe_fail("get_profile")
        return self.profiles.get(owner_sub)

    def fetch_financial_history(self, owner_sub: str, since: datetime) -> list[dict]:
        self.fetch_calls.append(owner_sub)
        self._maybe_fail("fetch_financial_history")
        return [
            row for row in self.history.get(owner_sub, []) if row["occurred_at"] >= since
        ]

    # ── mutation path (used only to prove Safe-to-Spend never writes) ──────
    def upsert_profile(self, owner_sub: str, data: dict) -> dict:
        self._maybe_fail("upsert_profile")
        self.writes.append(("upsert_profile", owner_sub))
        row = dict(data)
        row["updated_at"] = "u"
        self.profiles[owner_sub] = row
        return row

    def create_ingest_batch(self, owner_sub: str, source: str, row_count: int) -> str:
        self.writes.append(("create_ingest_batch", owner_sub))
        return "batch-0001"

    def insert_transactions(
        self, owner_sub: str, rows: list[dict], ingest_batch_id: str
    ) -> tuple[int, int]:
        self.writes.append(("insert_transactions", owner_sub))
        return len(rows), 0

    def delete_user_data(self, owner_sub: str) -> dict:
        self._maybe_fail("delete_user_data")
        self.writes.append(("delete_user_data", owner_sub))
        self.history.pop(owner_sub, None)
        return {
            "profile_deleted": False,
            "transactions_deleted": 0,
            "batches_deleted": 0,
        }


def _mem_cache() -> CacheClient:
    """Guaranteed in-memory cache (no Redis, no network)."""
    c = CacheClient.__new__(CacheClient)
    c.default_ttl = 60
    c._redis = None
    c._local_cache = {}
    return c


def _profile(
    *,
    funds: float | None = 10_000.0,
    payday: date | None = _day(30),
    buffer: float | None = 500.0,
) -> dict:
    """A saved profile row (keys exactly as PostgresStore returns them)."""
    return {
        "income": 5000.0,
        "savings_target": None,
        "liquid_buffer": funds,
        "total_debt": None,
        "monthly_debt_payments": None,
        "next_payday": payday,
        "safety_buffer": buffer,
        "created_at": "c",
        "updated_at": "u",
    }


@contextmanager
def _client(auth_env, make_auth_headers, store, sub, cache=None):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = store
        c.app.state.cache = cache if cache is not None else _mem_cache()
        c.headers.update(make_auth_headers(sub=sub))
        yield c


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def client(auth_env, make_auth_headers, store):
    with _client(auth_env, make_auth_headers, store, USER_A) as c:
        yield c


def _ready_store(store: FakeStore) -> FakeStore:
    """A's own data: 40 days of groceries plus a monthly rent bill."""
    store.profiles[USER_A] = _profile()
    store.history[USER_A] = _spend_rows(40) + [
        _row(-32, -1500.0, RENT),
        _row(-2, -1500.0, RENT),
    ]
    return store


def _sts_keys(cache: CacheClient, sub: str) -> list[str]:
    return [k for k in cache._local_cache if k.startswith(f"safe_to_spend:{sub}:")]


# ── engine: the arithmetic identity and its honesty rules ─────────────────────


def test_formula_identity_and_every_component_is_reported():
    """(1) ``safe_to_spend`` is exactly the documented five-term identity."""
    comp = _compute(
        _spend_rows(40) + [_row(-32, -1500.0, RENT), _row(-2, -1500.0, RENT)],
        funds=10_000.0,
        buffer=500.0,
        payday=_day(30),
    )
    assert comp.status == sts.STATUS_READY
    assert comp.has_amount is True
    assert comp.current_available_funds == 10_000.0
    assert comp.protected_obligations == 1_500.0  # rent, due today+28
    assert comp.expected_spending_before_payday == pytest.approx(3_000.0)  # 30 x 100
    assert comp.safety_buffer == 500.0
    assert comp.scenario_adjustment == 0.0
    assert comp.safe_to_spend == pytest.approx(5_000.0)
    assert comp.safe_to_spend == round(
        comp.current_available_funds
        - comp.protected_obligations
        - comp.expected_spending_before_payday
        - comp.safety_buffer
        - comp.scenario_adjustment,
        2,
    )
    assert comp.next_payday == _day(30)
    assert comp.days_to_payday == 30
    assert comp.currency == "USD"
    assert comp.payday_source == sts.PAYDAY_SOURCE_PROFILE
    assert comp.forecast_status == pf.STATUS_PERSONALIZED
    assert comp.forecast_fallback_status == pf.FALLBACK_NONE
    assert comp.engine_version == sts.SAFE_TO_SPEND_ENGINE_VERSION
    # the interval is measured, not assumed: a perfectly flat series has none
    assert comp.expected_spending_p10 == pytest.approx(3_000.0)
    assert comp.expected_spending_p90 == pytest.approx(3_000.0)
    assert comp.is_negative is False
    # the explanation is templated from the numbers themselves (never an LLM)
    assert "Safe to Spend = USD 5,000.00" in comp.explanation
    assert "Breakdown:" in comp.explanation
    assert comp.requirements is None  # a status with an amount needs no request


def test_protected_categories_are_counted_once_never_twice():
    """(2) bills are excluded from the forecast and subtracted once, in
    protected obligations — they can never be double-counted."""
    rows = _spend_rows(40) + [_row(-30, -1500.0, RENT), _row(-1, -2000.0, RENT)]
    comp = _compute(rows, funds=10_000.0, buffer=0.0, payday=_day(30))
    full = pf.build_daily_series([(r.occurred_at, r.amount) for r in _norm(rows)])
    assert full.total_spend == pytest.approx(7_500.0)  # the bills ARE in history
    assert comp.expected_spending_before_payday == pytest.approx(3_000.0)  # groceries
    assert comp.protected_obligations == 1_750.0  # median(1500, 2000), due today+29
    assert comp.safe_to_spend == pytest.approx(10_000.0 - 1_750.0 - 3_000.0)


def test_uncategorized_spending_stays_in_the_forecast_and_is_never_protected():
    rows = _norm(_spend_rows(40) + [_row(-5, -400.0, "Uncategorized")])
    assert len(sts.non_protected_spend_rows(rows)) == 41
    assert "Uncategorized" not in sts.PROTECTED_CATEGORY_SET
    plan = sts.upcoming_protected_obligations(rows, TODAY, _day(30))
    assert all(item.category != "Uncategorized" for item in plan.items)


def test_pending_transactions_are_excluded_and_counted_in_the_response():
    """(7) a pending row is never evidence, and the count of exclusions is shown."""
    comp = _compute(
        _spend_rows(40)
        + [
            _row(-3, -4000.0, "Electronics", pending=True),
            _row(-4, -900.0, RENT, pending=True),
        ],
        funds=5_000.0,
        payday=_day(10),
    )
    assert comp.excluded_pending_transactions == 2
    assert comp.expected_spending_before_payday == pytest.approx(1_000.0)
    assert comp.protected_obligations == 0.0  # pending rent is not evidence
    assert comp.safe_to_spend == pytest.approx(4_000.0)
    assert "2 excluded" in " ".join(comp.assumptions)


def test_negative_result_is_reported_as_is_never_floored():
    """(6) a shortfall is stated, not hidden by clamping to zero."""
    comp = _compute(_spend_rows(40), funds=1_000.0, payday=_day(30))
    assert comp.status == sts.STATUS_READY
    assert comp.safe_to_spend == pytest.approx(-2_000.0)
    assert comp.is_negative is True
    assert comp.has_amount is True
    assert "This is negative" in comp.explanation
    assert any("never floored" in note for note in comp.assumptions)


def test_status_precedence_and_no_fabricated_amount():
    """(3) every missing input gets its own status and no number at all."""
    rows = _spend_rows(40)

    no_profile = _compute(rows, funds=10_000.0, payday=_day(30), profile=False)
    assert no_profile.status == sts.STATUS_MISSING_PROFILE_DATA
    assert no_profile.safe_to_spend is None and no_profile.has_amount is False
    assert no_profile.is_negative is False
    assert "No amount is reported" in no_profile.explanation
    assert no_profile.requirements["how_to_resolve"] == ["PUT /consumer/profile"]

    no_balance = _compute(rows, funds=None, payday=_day(30))
    assert no_balance.status == sts.STATUS_MISSING_BALANCE
    assert no_balance.safe_to_spend is None
    assert any("liquid_buffer" in item for item in no_balance.requirements["required"])
    assert no_balance.requirements["how_to_resolve"] == [
        "PUT /consumer/profile with liquid_buffer"
    ]

    no_payday = _compute(rows, funds=10_000.0, payday=None)
    assert no_payday.status == sts.STATUS_MISSING_PAYDAY
    assert no_payday.safe_to_spend is None
    detail = no_payday.requirements["payday_detail"]
    assert detail["reason"] == sts.PAYDAY_REASON_NO_INCOME_HISTORY
    assert detail["income_observations"] == 0
    assert detail["last_income_date"] is None

    # precedence (first match wins): profile → balance → payday
    assert (
        _compute(rows, funds=None, payday=None, profile=False).status
        == sts.STATUS_MISSING_PROFILE_DATA
    )
    assert _compute(rows, funds=None, payday=None).status == sts.STATUS_MISSING_BALANCE


def test_insufficient_history_produces_no_expected_spending():
    comp = _compute(_spend_rows(3), funds=10_000.0, payday=_day(10))
    assert comp.status == sts.STATUS_INSUFFICIENT_HISTORY
    assert comp.expected_spending_before_payday is None
    assert comp.safe_to_spend is None
    assert comp.requirements["required"] == sts.MISSING_INPUTS[
        sts.STATUS_INSUFFICIENT_HISTORY
    ]
    assert comp.requirements["detail"]["observed"]["history_days"] == 3
    assert comp.requirements["how_to_resolve"] == [
        "POST /consumer/transactions/ingest-csv"
    ]


def test_limited_history_is_labelled_and_keeps_a_measured_interval():
    comp = _compute(_spend_rows(20), funds=10_000.0, payday=_day(20))
    assert comp.status == sts.STATUS_LIMITED_HISTORY
    assert comp.has_amount is True
    assert comp.expected_spending_before_payday == pytest.approx(2_000.0)
    assert comp.forecast_status == pf.STATUS_LIMITED
    assert comp.requirements is None
    assert (
        comp.expected_spending_p10
        <= comp.expected_spending_before_payday
        <= comp.expected_spending_p90
    )


def test_status_vocabulary_and_protected_taxonomy_are_single_sources_of_truth():
    assert set(sts.STATUSES) == {
        "missing_profile_data",
        "missing_balance",
        "missing_payday",
        "insufficient_history",
        "limited_history",
        "ready",
    }
    assert sts.STATUSES_WITH_AMOUNT == (sts.STATUS_READY, sts.STATUS_LIMITED_HISTORY)
    assert sts.STATUSES[0] == sts.STATUS_MISSING_PROFILE_DATA
    assert set(sts.PROTECTED_CATEGORIES) == set(HARD_PROTECTED_CATEGORIES)
    assert len(sts.PROTECTED_CATEGORIES) == 6
    assert sts.PROTECTED_CATEGORIES == tuple(sorted(sts.PROTECTED_CATEGORIES))
    assert "Uncategorized" not in sts.PROTECTED_CATEGORY_SET


def test_identical_inputs_produce_identical_output():
    rows = _spend_rows(40) + [_row(-32, -1500.0, RENT), _row(-2, -1500.0, RENT)]
    kwargs = dict(
        funds=10_000.0, buffer=500.0, payday=_day(30), scenario=250.0, label="trip"
    )
    first = _compute(rows, **kwargs)
    second = _compute(rows, **kwargs)
    assert first.as_dict() == second.as_dict()
    assert first.explanation == second.explanation
    assert first.assumptions == second.assumptions
    assert first.protected_obligation_items == second.protected_obligation_items


def test_negative_scenario_amount_is_ignored_not_credited():
    """A scenario can only reduce spending; it can never inflate the amount."""
    base = _compute(_spend_rows(40), funds=10_000.0, payday=_day(10))
    clamped = _compute(_spend_rows(40), funds=10_000.0, payday=_day(10), scenario=-500.0)
    assert clamped.scenario_adjustment == 0.0
    assert clamped.safe_to_spend == base.safe_to_spend


def test_scenario_note_states_nothing_was_persisted():
    comp = _compute(
        _spend_rows(40),
        funds=10_000.0,
        payday=_day(10),
        scenario=250.0,
        label="weekend trip",
    )
    assert comp.scenario_adjustment == 250.0
    assert comp.scenario_label == "weekend trip"
    assert any(
        "no persisted data was read-modify-written" in note for note in comp.assumptions
    )


def test_unconfigured_safety_buffer_is_zero_and_disclosed():
    """(5) an unset buffer is 0.00, reported as an upper bound — never invented."""
    comp = _compute(_spend_rows(40), funds=10_000.0, buffer=None, payday=_day(10))
    assert comp.safety_buffer == 0.0
    assert comp.safety_buffer_source == sts.BUFFER_SOURCE_NOT_CONFIGURED
    notes = " ".join(comp.assumptions)
    assert "No safety buffer is configured" in notes
    assert "upper bound" in notes
    assert "the legacy global/static forecast is never used" in notes

    configured = _compute(_spend_rows(40), funds=10_000.0, buffer=750.0, payday=_day(10))
    assert configured.safety_buffer == 750.0
    assert configured.safety_buffer_source == sts.BUFFER_SOURCE_PROFILE
    assert configured.safe_to_spend == pytest.approx(comp.safe_to_spend - 750.0)


def test_currency_is_taken_from_the_users_own_data_never_assumed():
    assert (
        sts.observed_currency(
            _norm([_row(-1, -5.0), _row(-2, -5.0), _row(-3, -5.0, currency="eur")])
        )
        == "USD"
    )
    # a tie is broken alphabetically, deterministically
    assert (
        sts.observed_currency(
            _norm([_row(-1, -5.0, currency="eur"), _row(-2, -5.0, currency="USD")])
        )
        == "EUR"
    )
    assert sts.observed_currency(_norm([_row(-1, -5.0, currency=None)])) is None
    assert sts.observed_currency([]) is None


def test_money_formatting_and_rounding():
    assert sts.money(None) is None
    assert sts.money(1.239) == 1.24
    assert sts.format_money(None, "USD") == "not available"
    assert sts.format_money(1234.5, "USD") == "USD 1,234.50"
    assert sts.format_money(1234.5, None) == "1,234.50 (currency not observed)"


def test_history_normalisation_keeps_evidence_and_skips_unreadable_rows():
    rows = [
        (_at(-1), -20.0, "USD", "Groceries", "POS", False),
        (_at(-2), -30.0, "USD", "Groceries", "POS", True),
        "not-a-row",
        {"occurred_at": None, "amount": -10.0},
        {
            "occurred_at": "2026-01-01T10:00:00",
            "amount": "-45.5",
            "category_l2": "Groceries",
        },
    ]
    out = _norm(rows)
    assert len(out) == 3
    assert out[0].currency == "USD" and out[0].is_pending is False
    assert out[1].is_pending is True
    assert out[2].amount == -45.5
    assert [r.is_pending for r in sts.settled(out)] == [False, False]
    assert sts.normalise_history([]) == []


# ── engine: payday resolution (a payday is never assumed) ─────────────────────


def test_configured_future_payday_is_used_verbatim():
    res = sts.resolve_payday(_day(12), [], TODAY)
    assert res.resolved is True
    assert res.payday == _day(12)
    assert res.source == sts.PAYDAY_SOURCE_PROFILE
    assert res.reason == sts.PAYDAY_REASON_CONFIGURED
    assert res.configured_payday_stale is False


def test_stale_configured_payday_is_ignored_never_reused():
    res = sts.resolve_payday(_day(-1), [], TODAY)
    assert res.payday is None
    assert res.reason == sts.PAYDAY_REASON_STALE_CONFIG
    assert res.configured_payday_stale is True
    assert "no longer in the future" in res.detail


def test_payday_beyond_the_supported_window_is_refused():
    res = sts.resolve_payday(_day(sts.MAX_DAYS_TO_PAYDAY + 1), [], TODAY)
    assert res.payday is None
    assert res.reason == sts.PAYDAY_REASON_OUT_OF_RANGE
    assert res.configured_payday_stale is False


def test_payday_measured_from_the_users_own_income_cadence():
    res = sts.detect_payday_from_income(_norm(_income_rows(-62, -32, -2)), TODAY)
    assert res.resolved is True
    assert res.source == sts.PAYDAY_SOURCE_MEASURED
    assert res.reason == sts.PAYDAY_REASON_MEASURED
    assert res.payday == _day(28)
    assert res.cadence_days == 30
    assert res.income_observations == 3
    assert res.last_income_date == _day(-2)


def test_stale_profile_payday_falls_back_to_the_measured_cadence():
    rows = _norm(_income_rows(-62, -32, -2))
    res = sts.resolve_payday(_day(-3), rows, TODAY)
    assert res.payday == _day(28)
    assert res.source == sts.PAYDAY_SOURCE_MEASURED
    assert res.configured_payday_stale is True


def test_two_income_deposits_are_not_enough_to_claim_a_payday():
    res = sts.detect_payday_from_income(_norm(_income_rows(-32, -2)), TODAY)
    assert res.payday is None
    assert res.reason == sts.PAYDAY_REASON_NO_INCOME_HISTORY
    assert res.income_observations == 2


def test_irregular_income_is_reported_instead_of_inventing_a_payday():
    res = sts.detect_payday_from_income(_norm(_income_rows(-60, -55, -15)), TODAY)
    assert res.payday is None
    assert res.reason == sts.PAYDAY_REASON_IRREGULAR_INCOME


def test_weekly_income_cadence_is_not_treated_as_a_payday():
    res = sts.detect_payday_from_income(_norm(_income_rows(-22, -15, -8, -1)), TODAY)
    assert res.payday is None
    assert res.reason == sts.PAYDAY_REASON_IRREGULAR_INCOME


def test_semi_monthly_income_cadence_is_accepted():
    res = sts.detect_payday_from_income(_norm(_income_rows(-46, -31, -16, -1)), TODAY)
    assert res.resolved is True
    assert res.cadence_days == 15
    assert res.payday == _day(14)


def test_measured_payday_outside_the_supported_window_is_not_used():
    # Every gap is a clean 30 days, but the implied next payday is *today*.
    res = sts.detect_payday_from_income(_norm(_income_rows(-90, -60, -30)), TODAY)
    assert res.payday is None
    assert res.reason == sts.PAYDAY_REASON_OUT_OF_RANGE
    assert res.cadence_days == 30


def test_income_evidence_requires_a_classified_credit():
    rows = _norm(
        [
            _row(-62, 5000.0, "Income"),
            _row(-32, 5000.0, "Income"),
            _row(-2, -5000.0, "Income"),  # a debit, not a deposit
        ]
    )
    res = sts.detect_payday_from_income(rows, TODAY)
    assert res.income_observations == 2
    assert res.payday is None
    assert res.reason == sts.PAYDAY_REASON_NO_INCOME_HISTORY


# ── engine: protected obligations before payday ───────────────────────────────


def _obligation_rows() -> list[dict]:
    return [
        _row(-32, -1500.0, RENT),
        _row(-2, -1500.0, RENT),
        _row(-1, -90.0, UTILITIES),  # a single payment is not a cadence
    ]


def test_monthly_obligation_due_before_payday_is_counted_once():
    plan = sts.upcoming_protected_obligations(_norm(_obligation_rows()), TODAY, _day(30))
    rent = next(item for item in plan.items if item.category == RENT)
    assert rent.included is True
    assert rent.reason == sts.REASON_INCLUDED
    assert rent.amount == 1_500.0
    assert rent.due_date == _day(28)
    assert rent.median_gap_days == 30
    assert rent.observations == 2
    assert rent.last_paid_on == _day(-2)

    utils = next(item for item in plan.items if item.category == UTILITIES)
    assert utils.included is False
    assert utils.reason == sts.REASON_INSUFFICIENT_OBSERVATIONS
    assert utils.observations == 1
    assert plan.total == 1_500.0
    assert [item.category for item in plan.included] == [RENT]


def test_all_six_protected_categories_are_always_reported():
    """Every exclusion is visible, never silent: all six categories always appear."""
    plan = sts.upcoming_protected_obligations(_norm(_spend_rows(40)), TODAY, _day(30))
    assert {item.category for item in plan.items} == set(sts.PROTECTED_CATEGORIES)
    assert all(
        item.reason == sts.REASON_INSUFFICIENT_OBSERVATIONS for item in plan.items
    )
    assert plan.total == 0.0
    assert plan.included == []


def test_obligation_due_after_payday_is_excluded_not_estimated():
    plan = sts.upcoming_protected_obligations(_norm(_obligation_rows()), TODAY, _day(10))
    rent = next(item for item in plan.items if item.category == RENT)
    assert rent.included is False
    assert rent.reason == sts.REASON_DUE_AFTER_PAYDAY
    assert rent.amount is None and rent.due_date == _day(28)
    assert plan.total == 0.0


def test_bill_already_paid_this_cycle_is_never_counted_twice():
    rows = _norm([_row(-70, -1500.0, RENT), _row(-40, -1500.0, RENT)])
    plan = sts.upcoming_protected_obligations(rows, TODAY, _day(30))
    rent = next(item for item in plan.items if item.category == RENT)
    assert rent.included is False
    assert rent.reason == sts.REASON_ALREADY_PAID_THIS_CYCLE
    assert rent.due_date == _day(-10)
    assert plan.total == 0.0


def test_irregular_cadence_is_reported_and_never_guessed():
    rows = _norm([_row(-100, -100.0, TAXES), _row(-10, -100.0, TAXES)])
    plan = sts.upcoming_protected_obligations(rows, TODAY, _day(30))
    taxes = next(item for item in plan.items if item.category == TAXES)
    assert taxes.included is False
    assert taxes.reason == sts.REASON_IRREGULAR_CADENCE
    assert taxes.median_gap_days == 90
    assert taxes.amount is None
    assert plan.total == 0.0


def test_counted_amount_is_the_median_not_a_one_off_spike():
    rows = _norm(
        [
            _row(-62, -9000.0, LOAN),
            _row(-32, -1000.0, LOAN),
            _row(-2, -1000.0, LOAN),
        ]
    )
    plan = sts.upcoming_protected_obligations(rows, TODAY, _day(30))
    loan = next(item for item in plan.items if item.category == LOAN)
    assert loan.included is True
    assert loan.amount == 1_000.0
    assert loan.due_date == _day(28)


def test_obligation_evidence_ignores_credits_and_pending_rows():
    rows = sts.settled(
        _norm(
            [
                _row(-32, -1500.0, RENT),
                _row(-2, -1500.0, RENT),
                _row(-20, 1500.0, RENT),  # a refund is not a payment
                _row(-17, -1500.0, RENT, pending=True),  # would distort the cadence
            ]
        )
    )
    plan = sts.upcoming_protected_obligations(rows, TODAY, _day(30))
    rent = next(item for item in plan.items if item.category == RENT)
    assert rent.observations == 2
    assert rent.median_gap_days == 30
    assert rent.included is True and rent.amount == 1_500.0


# ── engine: expected spending = the user's OWN forecast over the horizon ──────


def _expected(rows: list[dict], first: date, payday: date, *, tier_rows=None):
    """Build the two series the engine builds, then sum the horizon."""
    settled = sts.settled(_norm(rows))
    series = pf.build_daily_series(
        [(r.occurred_at, r.amount) for r in sts.non_protected_spend_rows(settled)]
    )
    tier_series = (
        None
        if tier_rows is None
        else pf.build_daily_series([(r.occurred_at, r.amount) for r in _norm(tier_rows)])
    )
    return sts.expected_spending_for_horizon(
        series, first, payday, tier_series=tier_series
    )


def test_expected_spending_is_anchored_on_tomorrow_and_ends_on_payday():
    settled = sts.settled(_norm(_spend_rows(40)))
    series = pf.build_daily_series([(r.occurred_at, r.amount) for r in settled])
    assert series.last_date == _day(-1)

    exp = _expected(_spend_rows(40), _day(1), _day(30))
    assert exp.status == pf.STATUS_PERSONALIZED
    assert exp.first_day == _day(1)  # tomorrow, explicitly — not the series end
    assert exp.last_day == _day(30)  # payday
    assert exp.days == 30
    assert exp.total == pytest.approx(3_000.0)
    assert exp.p10 == pytest.approx(3_000.0)
    assert exp.p90 == pytest.approx(3_000.0)
    assert exp.method == pf.PRIMARY_METHOD
    assert exp.requirements is None


def test_horizon_length_drives_the_total():
    short = _expected(_spend_rows(40), _day(1), _day(10))
    long = _expected(_spend_rows(40), _day(1), _day(30))
    assert short.days == 10 and short.total == pytest.approx(1_000.0)
    assert long.days == 30 and long.total == pytest.approx(3_000.0)


def test_expected_spending_reflects_only_the_series_it_is_given():
    a = _expected(_spend_rows(40), _day(1), _day(10))
    b = _expected(_spend_rows(40, amount=-5.0), _day(1), _day(10))
    assert a.total == pytest.approx(1_000.0)
    assert b.total == pytest.approx(50.0)


def test_insufficient_history_yields_no_total_only_requirements():
    exp = _expected(_spend_rows(3), _day(1), _day(10))
    assert exp.available is False
    assert exp.total is None
    assert exp.status == pf.STATUS_INSUFFICIENT
    assert exp.requirements["observed"]["history_days"] == 3
    assert exp.requirements["required"]["history_days"] == pf.MIN_DAYS_LIMITED


def test_empty_horizon_can_never_produce_a_total():
    exp = _expected(_spend_rows(40), _day(1), _day(0))
    assert exp.days == 0
    assert exp.total is None
    assert exp.available is False


def test_governing_tier_is_the_worse_of_full_and_non_protected_history():
    """A user whose non-obligation spending alone looks thin must never be
    labelled more confidently than their own data supports."""
    rows = _spend_rows(12) + [_row(-32, -1500.0, RENT), _row(-2, -1500.0, RENT)]
    settled = sts.settled(_norm(rows))
    series = pf.build_daily_series(
        [(r.occurred_at, r.amount) for r in sts.non_protected_spend_rows(settled)]
    )
    full = pf.build_daily_series([(r.occurred_at, r.amount) for r in settled])
    assert series.history_days == 12
    assert pf.quality_tier(series) == pf.STATUS_INSUFFICIENT
    assert pf.quality_tier(full) != pf.STATUS_INSUFFICIENT

    exp = _expected(rows, _day(1), _day(14), tier_rows=rows)
    assert exp.status == pf.STATUS_INSUFFICIENT
    assert exp.total is None
    assert exp.requirements["observed"]["history_days"] == 12


def test_limited_tier_still_produces_a_labelled_number():
    exp = _expected(_spend_rows(20), _day(1), _day(20))
    assert exp.status == pf.STATUS_LIMITED
    assert exp.total == pytest.approx(2_000.0)
    assert exp.requirements["note"].startswith("Personalized but based on limited")


# ── endpoint: identity, isolation and explicit statuses ───────────────────────


def test_missing_jwt_is_401(auth_env, store):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = _ready_store(store)
        c.app.state.cache = _mem_cache()
        assert c.get("/consumer/safe-to-spend").status_code == 401


def test_invalid_jwt_is_401(auth_env, store):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = _ready_store(store)
        c.app.state.cache = _mem_cache()
        r = c.get(
            "/consumer/safe-to-spend", headers={"Authorization": "Bearer not.a.jwt"}
        )
    assert r.status_code == 401


def test_uses_only_the_callers_own_data(client, store):
    """(1) the amount is built from the caller's own rows and profile."""
    _ready_store(store)
    store.history[USER_B] = _spend_rows(30, amount=-999.0)
    store.profiles[USER_B] = _profile(funds=1.0)

    r = client.get("/consumer/safe-to-spend")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["status"] == sts.STATUS_READY
    assert body["safe_to_spend"] == pytest.approx(5_000.0)
    assert body["current_available_funds"] == 10_000.0
    assert body["protected_obligations"] == 1_500.0
    assert body["expected_spending_before_payday"] == pytest.approx(3_000.0)
    assert body["safety_buffer"] == 500.0
    assert body["currency"] == "USD"
    assert body["next_payday"] == _day(30).isoformat()
    assert body["days_to_payday"] == 30
    assert body["payday_source"] == sts.PAYDAY_SOURCE_PROFILE
    assert body["forecast_status"] == pf.STATUS_PERSONALIZED
    assert body["history_days_used"] == 40
    assert body["transaction_count_used"] == 40  # A's rows, not B's 30
    assert body["excluded_pending_transactions"] == 0
    assert body["engine_version"] == sts.SAFE_TO_SPEND_ENGINE_VERSION
    assert body["is_negative"] is False
    assert body["message"] is None
    assert body["requirements"] is None
    assert body["as_of"] == TODAY.isoformat()
    # only the caller's history was ever queried — never a global or B query
    assert set(store.fetch_calls) == {USER_A}
    assert store.writes == []  # strictly read-only

    breakdown = body["breakdown"]
    assert [c["label"] for c in breakdown][:4] == [
        "Current available funds",
        "Protected obligations before payday",
        "Expected spending before payday",
        "Safety buffer",
    ]
    assert breakdown[0]["sign"] == "+"
    assert all(component["sign"] != "+" for component in breakdown[1:])
    assert all(component["source"] for component in breakdown)

    detail = {item["category"]: item for item in body["protected_obligations_detail"]}
    assert set(detail) == set(sts.PROTECTED_CATEGORIES)
    assert detail[RENT]["included"] is True
    assert detail[RENT]["amount"] == 1_500.0
    assert detail[RENT]["due_date"] == _day(28).isoformat()


def test_another_users_data_cannot_change_the_amount(auth_env, make_auth_headers, store):
    """(2) B's extreme data never influences A's Safe-to-Spend."""
    _ready_store(store)
    cache = _mem_cache()
    with _client(auth_env, make_auth_headers, store, USER_A, cache) as c:
        first = c.get("/consumer/safe-to-spend").json()
        cache.purge_user(USER_A)
        store.history[USER_B] = _spend_rows(60, amount=-99999.0)
        store.profiles[USER_B] = _profile(funds=1.0, buffer=9_999.0, payday=_day(2))
        second = c.get("/consumer/safe-to-spend").json()
    assert first["safe_to_spend"] == second["safe_to_spend"]
    assert first["breakdown"] == second["breakdown"]
    assert set(store.fetch_calls) == {USER_A}


def test_no_request_field_can_select_whose_finances_are_computed(client, store):
    """(3) identity comes from the verified JWT only — never a request field."""
    _ready_store(store)
    r = client.get(f"/consumer/safe-to-spend?user_id={USER_B}&sub={USER_B}")
    assert r.status_code == 200, r.text
    assert set(store.fetch_calls) == {USER_A}
    assert r.json()["current_available_funds"] == 10_000.0
    assert USER_B not in r.text


def test_scenario_body_cannot_select_another_users_data(client, store):
    _ready_store(store)
    r = client.post(
        "/consumer/safe-to-spend/scenario",
        json={"scenario_amount": 100.0, "user_id": USER_B, "owner_sub": USER_B},
    )
    assert r.status_code == 200, r.text
    assert set(store.fetch_calls) == {USER_A}
    assert r.json()["current_available_funds"] == 10_000.0


# ── endpoint: missing inputs are reported, never guessed ─────────────────────


def test_missing_profile_is_reported_not_guessed(client, store):
    r = client.get("/consumer/safe-to-spend")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == sts.STATUS_MISSING_PROFILE_DATA
    assert body["safe_to_spend"] is None
    assert body["requirements"]["how_to_resolve"] == ["PUT /consumer/profile"]
    assert body["message"] == sts_route._MSG_NO_AMOUNT


def test_missing_balance_is_reported_not_guessed(client, store):
    store.profiles[USER_A] = _profile(funds=None)
    store.history[USER_A] = _spend_rows(40)
    body = client.get("/consumer/safe-to-spend").json()
    assert body["status"] == sts.STATUS_MISSING_BALANCE
    assert body["current_available_funds"] is None
    assert body["safe_to_spend"] is None
    assert body["requirements"]["how_to_resolve"] == [
        "PUT /consumer/profile with liquid_buffer"
    ]


def test_missing_payday_is_reported_with_its_reason(client, store):
    store.profiles[USER_A] = _profile(payday=None)
    store.history[USER_A] = _spend_rows(40)
    body = client.get("/consumer/safe-to-spend").json()
    assert body["status"] == sts.STATUS_MISSING_PAYDAY
    assert body["next_payday"] is None
    assert body["days_to_payday"] is None
    assert body["safe_to_spend"] is None
    assert (
        body["requirements"]["payday_detail"]["reason"]
        == sts.PAYDAY_REASON_NO_INCOME_HISTORY
    )


def test_missing_payday_falls_back_to_the_users_own_income_cadence(client, store):
    store.profiles[USER_A] = _profile(payday=None)
    store.history[USER_A] = _spend_rows(40) + _income_rows(-62, -32, -2)
    body = client.get("/consumer/safe-to-spend").json()
    assert body["status"] == sts.STATUS_READY
    assert body["payday_source"] == sts.PAYDAY_SOURCE_MEASURED
    assert body["next_payday"] == _day(28).isoformat()
    assert body["days_to_payday"] == 28
    assert body["expected_spending_before_payday"] == pytest.approx(2_800.0)
    assert body["safe_to_spend"] == pytest.approx(6_700.0)


def test_insufficient_history_is_reported_not_guessed(client, store):
    store.profiles[USER_A] = _profile(payday=_day(10))
    store.history[USER_A] = _spend_rows(3)
    body = client.get("/consumer/safe-to-spend").json()
    assert body["status"] == sts.STATUS_INSUFFICIENT_HISTORY
    assert body["expected_spending_before_payday"] is None
    assert body["safe_to_spend"] is None
    assert body["requirements"]["how_to_resolve"] == [
        "POST /consumer/transactions/ingest-csv"
    ]
    assert body["requirements"]["detail"]["observed"]["history_days"] == 3


def test_limited_history_is_labelled_low_confidence(client, store):
    store.profiles[USER_A] = _profile(payday=_day(20))
    store.history[USER_A] = _spend_rows(20)
    body = client.get("/consumer/safe-to-spend").json()
    assert body["status"] == sts.STATUS_LIMITED_HISTORY
    assert body["safe_to_spend"] is not None
    assert body["forecast_status"] == pf.STATUS_LIMITED
    assert body["message"] == sts_route._MSG_LIMITED


def test_negative_amount_is_returned_with_a_shortfall_flag(client, store):
    store.profiles[USER_A] = _profile(funds=1_000.0, buffer=None, payday=_day(30))
    store.history[USER_A] = _spend_rows(40)
    body = client.get("/consumer/safe-to-spend").json()
    assert body["status"] == sts.STATUS_READY
    assert body["safe_to_spend"] == pytest.approx(-2_000.0)
    assert body["is_negative"] is True
    assert body["safety_buffer_source"] == sts.BUFFER_SOURCE_NOT_CONFIGURED
    assert "negative" in body["explanation"]


def test_currency_is_not_assumed_when_the_caller_has_none(client, store):
    store.profiles[USER_A] = _profile(payday=_day(10))
    store.history[USER_A] = _spend_rows(40)
    for row in store.history[USER_A]:
        row["currency"] = None
    body = client.get("/consumer/safe-to-spend").json()
    assert body["currency"] is None
    assert any(
        "no currency has been observed yet" in note for note in body["assumptions"]
    )


# ── endpoint: scenarios apply to one response and persist nothing ────────────


def test_scenario_is_applied_without_persisting_anything(client, store):
    _ready_store(store)
    base = client.get("/consumer/safe-to-spend").json()

    r = client.post(
        "/consumer/safe-to-spend/scenario",
        json={"scenario_amount": 500.0, "label": "weekend trip"},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["scenario_adjustment"] == 500.0
    assert body["safe_to_spend"] == pytest.approx(base["safe_to_spend"] - 500.0)
    scenario = body["scenario"]
    assert scenario["label"] == "weekend trip"
    assert scenario["amount"] == 500.0
    assert scenario["safe_to_spend_before"] == pytest.approx(base["safe_to_spend"])
    assert scenario["safe_to_spend_after"] == pytest.approx(
        base["safe_to_spend"] - 500.0
    )
    assert scenario["delta"] == pytest.approx(500.0)
    assert scenario["persisted"] is False
    assert store.writes == []  # the scenario persisted nothing at all

    after = client.get("/consumer/safe-to-spend").json()
    assert after["safe_to_spend"] == pytest.approx(base["safe_to_spend"])


def test_zero_scenario_produces_no_scenario_block(client, store):
    _ready_store(store)
    base = client.get("/consumer/safe-to-spend").json()
    body = client.post(
        "/consumer/safe-to-spend/scenario", json={"scenario_amount": 0.0}
    ).json()
    assert body["scenario"] is None
    assert body["scenario_adjustment"] == 0.0
    assert body["safe_to_spend"] == pytest.approx(base["safe_to_spend"])


def test_scenario_can_make_the_amount_negative(client, store):
    _ready_store(store)
    body = client.post(
        "/consumer/safe-to-spend/scenario", json={"scenario_amount": 9_000.0}
    ).json()
    assert body["safe_to_spend"] == pytest.approx(-4_000.0)
    assert body["is_negative"] is True


def test_scenario_validation_rejects_bad_input(client, store):
    _ready_store(store)
    negative = client.post(
        "/consumer/safe-to-spend/scenario", json={"scenario_amount": -1.0}
    )
    assert negative.status_code == 422
    assert client.post("/consumer/safe-to-spend/scenario").status_code == 422
    too_long = client.post(
        "/consumer/safe-to-spend/scenario",
        json={"scenario_amount": 1.0, "label": "x" * 200},
    )
    assert too_long.status_code == 422


# ── endpoint: cache scoping, freshness and invalidation ─────────────────────


def test_cache_key_is_user_scoped_and_engine_aware(client, store):
    _ready_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/safe-to-spend")

    keys = _sts_keys(cache, USER_A)
    assert len(keys) == 1
    key = keys[0]
    assert key.startswith(f"safe_to_spend:{USER_A}:{_day(30).isoformat()}:none:")
    assert key.endswith(sts.SAFE_TO_SPEND_ENGINE_VERSION)
    assert USER_B not in key
    assert "eyJ" not in key and "Bearer" not in key

    # the horizon is part of the key: a new payday cannot serve the old amount
    store.profiles[USER_A] = _profile(payday=_day(15))
    body = client.get("/consumer/safe-to-spend").json()
    assert body["days_to_payday"] == 15
    assert len(_sts_keys(cache, USER_A)) == 2


def test_cached_response_is_reused_only_while_as_of_is_today(client, store):
    _ready_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    first = client.get("/consumer/safe-to-spend").json()

    key = _sts_keys(cache, USER_A)[0]
    expiry, _value = cache._local_cache[key]

    # a today-dated entry is served from cache without recomputing
    marker = dict(first)
    marker["safe_to_spend"] = 111.0
    cache._local_cache[key] = (expiry, marker)
    assert client.get("/consumer/safe-to-spend").json()["safe_to_spend"] == 111.0

    # ...but an entry carrying a *stale* as_of must never be reused
    stale = dict(first)
    stale["safe_to_spend"] = 222.0
    stale["as_of"] = _day(-1).isoformat()
    cache._local_cache[key] = (expiry, stale)
    fresh = client.get("/consumer/safe-to-spend").json()
    assert fresh["as_of"] == TODAY.isoformat()
    assert fresh["safe_to_spend"] == first["safe_to_spend"]


def test_statuses_without_an_amount_are_never_cached(client, store):
    store.profiles[USER_A] = _profile(payday=_day(10))
    store.history[USER_A] = _spend_rows(3)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/safe-to-spend")
    assert _sts_keys(cache, USER_A) == []

    store.profiles.pop(USER_A)
    store.history.pop(USER_A)
    client.get("/consumer/safe-to-spend")  # missing profile/payday path
    assert _sts_keys(cache, USER_A) == []


def test_cache_invalidates_after_profile_update_and_keeps_other_users(client, store):
    """(13) a mutation purges the caller's entries only — never another user's."""
    _ready_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/safe-to-spend")
    cache.set(
        "safe_to_spend",
        USER_B,
        _day(30).isoformat(),
        "none",
        sts.SAFE_TO_SPEND_ENGINE_VERSION,
        value={"as_of": TODAY.isoformat()},
    )
    assert _sts_keys(cache, USER_A) and _sts_keys(cache, USER_B)

    r = client.put(
        "/consumer/profile",
        json={
            "income": 5000.0,
            "liquid_buffer": 12_000.0,
            "next_payday": _day(30).isoformat(),
            "safety_buffer": 0.0,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["cache_purged"] is True
    assert _sts_keys(cache, USER_A) == []
    assert _sts_keys(cache, USER_B), "another user's cache must never be purged"

    refreshed = client.get("/consumer/safe-to-spend").json()
    assert refreshed["current_available_funds"] == 12_000.0
    assert refreshed["safety_buffer"] == 0.0
    assert refreshed["safe_to_spend"] == pytest.approx(7_500.0)  # 12000-1500-3000


def test_cache_invalidates_after_delete(client, store):
    _ready_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/safe-to-spend")
    assert _sts_keys(cache, USER_A)

    r = client.delete("/consumer/data")
    assert r.status_code == 200, r.text
    assert _sts_keys(cache, USER_A) == []


def test_cache_invalidates_after_ingest(client, store, monkeypatch):
    """(12) new transactions change the caller's inputs, so their entry is dropped."""
    _ready_store(store)
    cache = _mem_cache()
    client.app.state.cache = cache
    client.get("/consumer/safe-to-spend")
    assert _sts_keys(cache, USER_A)

    async def _stub_classify(_request, _req, history=None):
        return SimpleNamespace(
            category_l1="Essentials", category_l2="Groceries", confidence=0.9
        )

    monkeypatch.setattr(
        ingest_module, "classify_transaction_with_history", _stub_classify
    )
    body = (
        f"{CSV_HEADER}\n"
        f"{USER_A},2026-04-05T12:00:00,-100.0,USD,FreshMart,5411,CHECKING,POS\n"
    )
    r = client.post(
        "/consumer/transactions/ingest-csv",
        content=body.encode(),
        headers={"Content-Type": "text/csv"},
    )
    assert r.status_code == 200, r.text
    assert _sts_keys(cache, USER_A) == []


# ── endpoint: read-only guarantees and generic failures ─────────────────────


def test_endpoints_never_write_to_the_store(client, store):
    _ready_store(store)
    client.get("/consumer/safe-to-spend")
    client.post("/consumer/safe-to-spend/scenario", json={"scenario_amount": 10.0})
    client.post("/consumer/safe-to-spend/scenario", json={"scenario_amount": 25.0})
    assert store.writes == []


def test_store_failure_returns_a_generic_503(client, store):
    _ready_store(store)
    store.fail_methods.add("fetch_financial_history")
    r = client.get("/consumer/safe-to-spend")
    assert r.status_code == 503
    assert r.json()["detail"] == DETAIL_503
    assert "PersistenceUnavailable" not in r.text


def test_profile_read_failure_returns_a_generic_503(client, store):
    _ready_store(store)
    store.fail_methods.add("get_profile")
    r = client.get("/consumer/safe-to-spend")
    assert r.status_code == 503
    assert r.json()["detail"] == DETAIL_503


def test_unexpected_engine_failure_returns_a_generic_500(client, store, monkeypatch):
    _ready_store(store)

    def _boom(**_kwargs):
        raise RuntimeError("boom-with-internal-detail")

    monkeypatch.setattr(sts, "compute_safe_to_spend", _boom)
    r = client.get("/consumer/safe-to-spend")
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error."
    assert "boom-with-internal-detail" not in r.text


def test_unconfigured_store_returns_503(auth_env, make_auth_headers):
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        c.app.state.store = None
        c.app.state.cache = _mem_cache()
        c.headers.update(make_auth_headers(sub=USER_A))
        r = c.get("/consumer/safe-to-spend")
    assert r.status_code == 503
    assert r.json()["detail"] == DETAIL_NOT_CONFIGURED


def test_result_is_deterministic_across_clients(auth_env, make_auth_headers, store):
    _ready_store(store)
    with _client(auth_env, make_auth_headers, store, USER_A) as c:
        first = c.get("/consumer/safe-to-spend").json()
    with _client(auth_env, make_auth_headers, store, USER_A) as c:
        second = c.get("/consumer/safe-to-spend").json()
    assert first == second