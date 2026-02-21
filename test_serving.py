"""
End-to-end test script for the ML Fin-Advisor serving layer.

Usage:
    python test_serving.py [--base-url http://localhost:8000]

Tests all API endpoints and verifies:
  1. Health / readiness probes
  2. Transaction classification with SHAP + anchor rules
  3. Expense forecasting with p10/p50/p90
  4. Budget recommendations with SHAP, anchor rules, counterfactuals
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


BASE_URL = "http://localhost:8000"
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def _post(path: str, body: dict) -> tuple[int, dict]:
    req = Request(
        f"{BASE_URL}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


def _get(path: str) -> tuple[int, dict]:
    req = Request(f"{BASE_URL}{path}", method="GET")
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))


# ── 1. Health & Readiness ──────────────────────────────────────────────────────


def test_health():
    print("\n── Health & Readiness ──")
    code, data = _get("/health")
    check("GET /health → 200", code == 200)
    check("status = healthy", data.get("status") == "healthy")

    code, data = _get("/ready")
    check("GET /ready → 200", code == 200)
    check("classifier loaded", data.get("components", {}).get("classifier", False))
    check("forecast_data loaded", data.get("components", {}).get("forecast_data", False))
    check("budget_data loaded", data.get("components", {}).get("budget_data", False))


# ── 2. Transaction Classification ─────────────────────────────────────────────


SAMPLE_TRANSACTIONS = [
    {
        "name": "Whole Foods (Groceries)",
        "body": {
            "transaction": {
                "user_id": "u-test-0001",
                "timestamp": "2025-11-15T14:32:00Z",
                "amount": -87.50,
                "currency": "USD",
                "merchant_name": "Whole Foods Market",
                "merchant_mcc": 5411,
                "account_type": "CHECKING",
                "channel": "POS",
                "location_country": "US",
                "raw_description": "WHOLE FOODS MKT #10234 SAN FRANCISCO CA",
            }
        },
        "expected_l2": "Groceries",
    },
    {
        "name": "Netflix (Subscription)",
        "body": {
            "transaction": {
                "user_id": "u-test-0001",
                "timestamp": "2025-11-01T00:00:00Z",
                "amount": -15.99,
                "currency": "USD",
                "merchant_name": "Netflix",
                "merchant_mcc": 4899,
                "account_type": "CREDIT",
                "channel": "RECURRING",
                "location_country": "US",
                "raw_description": "NETFLIX.COM RECURRING MONTHLY",
            }
        },
        "expected_l2": "Subscriptions & Streaming",
    },
    {
        "name": "Shell Gas Station (Fuel)",
        "body": {
            "transaction": {
                "user_id": "u-test-0001",
                "timestamp": "2025-11-10T08:15:00Z",
                "amount": -45.00,
                "currency": "USD",
                "merchant_name": "Shell Gas Station",
                "merchant_mcc": 5541,
                "account_type": "CHECKING",
                "channel": "POS",
                "location_country": "US",
                "raw_description": "SHELL OIL 57442 AUSTIN TX",
            }
        },
        "expected_l2": "Fuel",
    },
]


def test_classify():
    print("\n── Transaction Classification ──")

    for txn in SAMPLE_TRANSACTIONS:
        code, data = _post("/v1/classify", txn["body"])
        check(
            f"POST /classify ({txn['name']}) → 200",
            code == 200,
            f"got {code}",
        )
        if code != 200:
            print(f"    Response: {json.dumps(data, indent=2)[:200]}")
            continue

        predicted = data.get("category_l2", "")
        confidence = data.get("confidence", 0)
        check(
            f"  category_l2 = {txn['expected_l2']}",
            predicted == txn["expected_l2"],
            f"got '{predicted}' (conf={confidence:.3f})",
        )
        check(
            f"  confidence ≥ 0.5",
            confidence >= 0.5,
            f"conf={confidence:.4f}",
        )

        # Verify top-3
        top3 = data.get("top_3", [])
        check(f"  top_3 has entries", len(top3) >= 1, f"len={len(top3)}")

        # Verify SHAP features
        shap = data.get("shap_features", [])
        check(
            f"  SHAP features present",
            len(shap) >= 1,
            f"count={len(shap)}",
        )
        if shap:
            print(f"    SHAP top features:")
            for sf in shap[:3]:
                print(f"      {sf['feature']:30s}  value={sf['value']:8.3f}  shap={sf['shap_value']:+.4f}")

        # Verify anchor rule
        anchor = data.get("anchor_rule", "")
        check(
            f"  anchor rule present",
            len(anchor) > 10 and "IF" in anchor and "THEN" in anchor,
            f"rule='{anchor[:60]}…'" if len(anchor) > 60 else f"rule='{anchor}'",
        )

    print()


# ── 3. Expense Forecasting ────────────────────────────────────────────────────


def test_forecast():
    print("\n── Expense Forecasting ──")

    # Default 30-day horizon
    code, data = _get("/v1/forecast/u-test-0001?horizon=30")
    check("GET /forecast → 200", code == 200, f"got {code}")
    if code != 200:
        print(f"    Response: {json.dumps(data, indent=2)[:200]}")
        return

    forecasts = data.get("forecasts", [])
    check("forecasts non-empty", len(forecasts) >= 1, f"count={len(forecasts)}")

    total = data.get("total_spend", {})
    check("total_spend has p10/p50/p90",
          all(k in total for k in ("p10", "p50", "p90")),
          f"keys={list(total.keys())}")

    # Verify p10 ≤ p50 ≤ p90 for each category
    for fc in forecasts[:5]:
        cat = fc.get("category", "?")
        p10, p50, p90 = fc.get("p10", 0), fc.get("p50", 0), fc.get("p90", 0)
        check(
            f"  {cat}: p10 ≤ p50 ≤ p90",
            p10 <= p50 <= p90,
            f"p10={p10:.0f}, p50={p50:.0f}, p90={p90:.0f}",
        )
        trend = fc.get("trend", "")
        regime = fc.get("regime", "")
        print(f"    trend={trend}, regime={regime}")

    # Test category filtering
    code2, data2 = _get("/v1/forecast/u-test-0001?horizon=30&categories=Groceries,Fuel")
    check("filtered forecast → 200", code2 == 200)
    if code2 == 200:
        n_cats = len(data2.get("forecasts", []))
        check("filtered to ≤ 2 categories", n_cats <= 2, f"count={n_cats}")


# ── 4. Budget Recommendations ─────────────────────────────────────────────────


def test_budget():
    print("\n── Budget Recommendations ──")

    code, data = _get("/v1/budget/u-test-0001")
    check("GET /budget → 200", code == 200, f"got {code}")
    if code != 200:
        print(f"    Response: {json.dumps(data, indent=2)[:300]}")
        return

    check("user_id present", data.get("user_id") == "u-test-0001")
    check("income_estimate > 0", data.get("income_estimate", 0) > 0,
          f"income=${data.get('income_estimate', 0):,.0f}")
    check("savings_target > 0", data.get("savings_target", 0) > 0,
          f"target=${data.get('savings_target', 0):,.0f}")

    recs = data.get("recommendations", [])
    check("recommendations non-empty", len(recs) >= 1, f"count={len(recs)}")

    # Verify each recommendation has SHAP + anchor + counterfactual
    shap_count = 0
    anchor_count = 0
    cf_count = 0

    for rec in recs[:5]:
        cat = rec.get("category", "?")
        budget = rec.get("recommended_budget", 0)
        trend = rec.get("current_trend", 0)

        shap_features = rec.get("shap_top_features", [])
        anchor = rec.get("anchor_rule", "")
        counterfactual = rec.get("counterfactual", "")
        explanation = rec.get("explanation", "")

        if shap_features:
            shap_count += 1
        if anchor and "IF" in anchor:
            anchor_count += 1
        if counterfactual and len(counterfactual) > 10:
            cf_count += 1

        print(f"\n  Category: {cat}")
        print(f"    Budget: ${budget:,.2f}  |  Current: ${trend:,.2f}")
        print(f"    Explanation: {explanation[:100]}…" if len(explanation) > 100 else f"    Explanation: {explanation}")

        if shap_features:
            print(f"    SHAP features ({len(shap_features)}):")
            for sf in shap_features[:3]:
                print(f"      {sf['feature']:30s}  impact={sf['impact']:+.2f}")

        if anchor:
            print(f"    Anchor: {anchor[:100]}")

        if counterfactual:
            print(f"    Counterfactual: {counterfactual[:100]}")

    print()
    check("SHAP features in ≥1 recommendation", shap_count >= 1, f"{shap_count}/{len(recs)}")
    check("anchor rules in ≥1 recommendation", anchor_count >= 1, f"{anchor_count}/{len(recs)}")
    check("counterfactuals in ≥1 recommendation", cf_count >= 1, f"{cf_count}/{len(recs)}")


# ── Main ───────────────────────────────────────────────────────────────────────


def wait_for_server(timeout: int = 30):
    """Wait for the server to start."""
    print(f"Waiting for server at {BASE_URL}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            _get("/health")
            print("Server is ready!\n")
            return True
        except (URLError, ConnectionError):
            time.sleep(1)
    print("Server did not start in time.")
    return False


def main():
    global BASE_URL

    parser = argparse.ArgumentParser(description="Test ML Fin-Advisor API")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for server startup")
    args = parser.parse_args()
    BASE_URL = args.base_url

    print("=" * 70)
    print("  ML FIN-ADVISOR — Serving Layer Integration Tests")
    print("=" * 70)

    if not args.no_wait:
        if not wait_for_server():
            sys.exit(1)

    test_health()
    test_classify()
    test_forecast()
    test_budget()

    # Summary
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print("\n" + "=" * 70)
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        print("\n  Failed tests:")
        for name, ok, detail in results:
            if not ok:
                print(f"    ✗ {name}  ({detail})")
    else:
        print(" — ALL PASSED")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
