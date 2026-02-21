"""
Unit tests for data models, mock generator, and ingestion pipeline.
"""

from __future__ import annotations

import csv
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data.models import (
    ClassificationResult,
    CategoryPrediction,
    Transaction,
    TransactionBatch,
    ValidationReport,
)
from src.utils.constants import (
    AccountType,
    CategoryL1,
    CategoryL2,
    Channel,
    CATEGORY_HIERARCHY,
    lookup_category_by_mcc,
)


# ── Transaction Model Tests ───────────────────────────────────────────────────


class TestTransactionModel:
    """Tests for the Pydantic Transaction model (SPEC §5.1)."""

    def _make_txn(self, **overrides) -> Transaction:
        defaults = {
            "user_id": "u-00000001",
            "timestamp": datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
            "amount": -42.50,
            "currency": "USD",
            "merchant_name": "Whole Foods Market",
            "merchant_mcc": 5411,
            "account_type": AccountType.CHECKING,
            "channel": Channel.POS,
            "location_country": "US",
            "raw_description": "WHOLE FOODS MKT #10234",
        }
        defaults.update(overrides)
        return Transaction(**defaults)

    def test_valid_transaction(self):
        txn = self._make_txn()
        assert txn.amount == -42.50
        assert txn.merchant_name == "Whole Foods Market"
        assert txn.account_type == AccountType.CHECKING
        assert txn.transaction_id  # auto-generated UUID

    def test_currency_uppercased(self):
        txn = self._make_txn(currency="usd")
        assert txn.currency == "USD"

    def test_country_uppercased(self):
        txn = self._make_txn(location_country="us")
        assert txn.location_country == "US"

    def test_optional_location_city(self):
        txn = self._make_txn(location_city=None)
        assert txn.location_city is None

        txn2 = self._make_txn(location_city="San Francisco")
        assert txn2.location_city == "San Francisco"

    def test_category_labels_optional(self):
        txn = self._make_txn()
        assert txn.category_l1 is None
        assert txn.category_l2 is None

    def test_valid_category_hierarchy(self):
        txn = self._make_txn(
            category_l1=CategoryL1.FOOD_AND_DINING,
            category_l2=CategoryL2.GROCERIES,
        )
        assert txn.category_l1 == CategoryL1.FOOD_AND_DINING

    def test_invalid_category_hierarchy_raises(self):
        with pytest.raises(ValueError, match="Category mismatch"):
            self._make_txn(
                category_l1=CategoryL1.TRANSPORTATION,
                category_l2=CategoryL2.GROCERIES,  # belongs to FOOD & DINING
            )

    def test_invalid_mcc_range(self):
        with pytest.raises(Exception):
            self._make_txn(merchant_mcc=99999)

    def test_invalid_currency_length(self):
        with pytest.raises(Exception):
            self._make_txn(currency="US")  # too short

    def test_positive_amount_for_income(self):
        txn = self._make_txn(amount=5500.00)
        assert txn.amount > 0

    def test_serialization_roundtrip(self):
        txn = self._make_txn(
            category_l1=CategoryL1.FOOD_AND_DINING,
            category_l2=CategoryL2.GROCERIES,
        )
        data = txn.model_dump()
        txn2 = Transaction(**data)
        assert txn2.amount == txn.amount
        assert txn2.merchant_name == txn.merchant_name


# ── Constants Tests ───────────────────────────────────────────────────────────


class TestConstants:
    def test_category_hierarchy_covers_all_l2(self):
        for cat in CategoryL2:
            assert cat in CATEGORY_HIERARCHY, f"{cat} missing from hierarchy"

    def test_mcc_lookup_groceries(self):
        assert lookup_category_by_mcc(5411) == CategoryL2.GROCERIES
        assert lookup_category_by_mcc(5450) == CategoryL2.GROCERIES

    def test_mcc_lookup_unknown(self):
        assert lookup_category_by_mcc(9999) is None

    def test_all_l1_have_children(self):
        from src.utils.constants import L1_TO_L2
        for l1 in CategoryL1:
            assert l1 in L1_TO_L2, f"{l1} has no L2 children"
            assert len(L1_TO_L2[l1]) >= 1


# ── Transaction Batch Tests ───────────────────────────────────────────────────


class TestTransactionBatch:
    def test_batch_properties(self):
        txns = [
            Transaction(
                user_id=f"u-{i}",
                timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
                amount=-10.0,
                merchant_name="Test",
                merchant_mcc=5411,
                account_type=AccountType.CHECKING,
                channel=Channel.POS,
            )
            for i in range(3)
        ]
        batch = TransactionBatch(transactions=txns, source="test")
        assert batch.size == 3
        assert len(batch.user_ids) == 3


# ── Mock Generator Tests ─────────────────────────────────────────────────────


class TestMockGenerator:
    def test_generate_small_dataset(self):
        from src.data.mock_generator import generate_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.csv"
            result = generate_dataset(
                n_users=2,
                n_months=2,
                output_path=out,
                seed=123,
            )
            assert result.exists()

            # Read and validate row count
            with open(result, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            # 2 users × 2 months × ~80 txns → should be > 100 rows
            assert len(rows) > 100
            # Check columns
            assert "transaction_id" in rows[0]
            assert "user_id" in rows[0]
            assert "amount" in rows[0]
            assert "category_l2" in rows[0]

    def test_income_transactions_present(self):
        from src.data.mock_generator import generate_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.csv"
            generate_dataset(n_users=1, n_months=1, output_path=out, seed=42)

            with open(out, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            income_rows = [r for r in rows if r["category_l2"] == "Income"]
            assert len(income_rows) >= 1, "Expected at least 1 income transaction per month"

    def test_amounts_are_signed(self):
        from src.data.mock_generator import generate_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.csv"
            generate_dataset(n_users=1, n_months=1, output_path=out, seed=42)

            with open(out, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            amounts = [float(r["amount"]) for r in rows]
            assert any(a > 0 for a in amounts), "Expected some credits (income)"
            assert any(a < 0 for a in amounts), "Expected some debits (spending)"


# ── Ingestion Pipeline Tests ──────────────────────────────────────────────────


class TestIngestion:
    def _create_csv(self, rows: list[dict], path: Path) -> None:
        if not rows:
            path.write_text("")
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _valid_row(self, **overrides) -> dict:
        defaults = {
            "transaction_id": "tid-001",
            "user_id": "u-001",
            "timestamp": "2025-06-15T10:00:00+00:00",
            "amount": "-25.00",
            "currency": "USD",
            "merchant_name": "Starbucks",
            "merchant_mcc": "5814",
            "account_type": "CHECKING",
            "channel": "POS",
            "location_city": "Seattle",
            "location_country": "US",
            "raw_description": "STARBUCKS #1234 SEATTLE WA",
            "is_pending": "False",
            "category_l1": "FOOD & DINING",
            "category_l2": "Coffee Shops",
        }
        defaults.update(overrides)
        return defaults

    def test_ingest_valid_csv(self):
        from src.data.ingestion import ingest_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "valid.csv"
            self._create_csv([self._valid_row()], csv_path)

            batch, report = ingest_csv(csv_path)
            assert report.valid_rows == 1
            assert report.invalid_rows == 0
            assert batch.size == 1
            assert batch.transactions[0].merchant_name == "Starbucks"

    def test_ingest_invalid_row(self):
        from src.data.ingestion import ingest_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "invalid.csv"
            bad_row = self._valid_row(merchant_mcc="not_a_number")
            self._create_csv([self._valid_row(), bad_row], csv_path)

            batch, report = ingest_csv(csv_path)
            assert report.valid_rows == 1
            assert report.invalid_rows == 1
            assert len(report.issues) >= 1

    def test_ingest_missing_file(self):
        from src.data.ingestion import ingest_csv

        with pytest.raises(FileNotFoundError):
            ingest_csv("nonexistent.csv")

    def test_validate_batch_quality_clean(self):
        from src.data.ingestion import validate_batch_quality
        import pandas as pd

        df = pd.DataFrame([{
            "transaction_id": "t1",
            "user_id": "u1",
            "timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
            "amount": -10.0,
            "currency": "USD",
            "merchant_name": "Test",
            "merchant_mcc": 5411,
            "account_type": "CHECKING",
            "channel": "POS",
            "location_country": "US",
            "raw_description": "TEST",
            "is_pending": False,
        }])

        issues = validate_batch_quality(df)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_validate_batch_quality_duplicates(self):
        from src.data.ingestion import validate_batch_quality
        import pandas as pd

        df = pd.DataFrame([
            {"transaction_id": "DUPE", "user_id": "u1", "timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
             "amount": -10.0, "currency": "USD", "merchant_name": "A", "merchant_mcc": 5411,
             "account_type": "CHECKING", "channel": "POS", "location_country": "US",
             "raw_description": "A", "is_pending": False},
            {"transaction_id": "DUPE", "user_id": "u1", "timestamp": pd.Timestamp("2025-01-02", tz="UTC"),
             "amount": -20.0, "currency": "USD", "merchant_name": "B", "merchant_mcc": 5411,
             "account_type": "CHECKING", "channel": "POS", "location_country": "US",
             "raw_description": "B", "is_pending": False},
        ])

        issues = validate_batch_quality(df)
        dupe_issues = [i for i in issues if "duplicate" in i.issue.lower()]
        assert len(dupe_issues) == 1

    def test_end_to_end_generate_and_ingest(self):
        """Generate mock data, then ingest it — full round-trip."""
        from src.data.mock_generator import generate_dataset
        from src.data.ingestion import ingest_csv_to_dataframe

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "e2e.csv"
            generate_dataset(n_users=3, n_months=1, output_path=csv_path, seed=99)

            df, report = ingest_csv_to_dataframe(csv_path, validate=True)

            assert report is not None
            assert len(df) > 0
            assert "amount" in df.columns
            assert "merchant_name" in df.columns
            # Most rows should validate cleanly
            assert report.valid_rows / report.total_rows > 0.95
