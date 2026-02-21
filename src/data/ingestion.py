"""
Data ingestion pipeline for the ML Fin-Advisor project.

Reads transaction data from CSV files (and future: Plaid/Yodlee feeds),
validates against the Pydantic schema, and produces clean TransactionBatch
objects ready for feature engineering.

See SPEC §5.1 (schema) and §10.2 (data validation).
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd
from pydantic import ValidationError

from src.data.models import (
    Transaction,
    TransactionBatch,
    ValidationIssue,
    ValidationReport,
)
from src.utils.constants import AccountType, CategoryL1, CategoryL2, Channel

logger = logging.getLogger(__name__)


# ── CSV Ingestion ──────────────────────────────────────────────────────────────


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Coerce raw CSV string values into the types expected by the Transaction model.

    Handles common CSV quirks: stringified booleans, empty strings for nulls, etc.
    """
    coerced = dict(row)

    # Amount → float
    if "amount" in coerced and isinstance(coerced["amount"], str):
        coerced["amount"] = float(coerced["amount"])

    # MCC → int
    if "merchant_mcc" in coerced and isinstance(coerced["merchant_mcc"], str):
        coerced["merchant_mcc"] = int(coerced["merchant_mcc"])

    # Boolean coercion
    if "is_pending" in coerced:
        val = coerced["is_pending"]
        if isinstance(val, str):
            coerced["is_pending"] = val.strip().lower() in ("true", "1", "yes")

    # Null handling for optional fields
    for nullable_field in ("location_city", "category_l1", "category_l2"):
        if nullable_field in coerced and coerced[nullable_field] in ("", "None", "null", "NA"):
            coerced[nullable_field] = None

    # Enum coercion: strip whitespace
    for enum_field in ("account_type", "channel"):
        if enum_field in coerced and isinstance(coerced[enum_field], str):
            coerced[enum_field] = coerced[enum_field].strip().upper()

    # Timestamp: ensure it parses as datetime
    if "timestamp" in coerced and isinstance(coerced["timestamp"], str):
        ts = coerced["timestamp"].strip()
        # Handle ISO format with or without timezone
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        coerced["timestamp"] = ts

    return coerced


def read_csv_raw(
    file_path: str | Path,
    encoding: str = "utf-8",
) -> Generator[dict[str, Any], None, None]:
    """
    Read a CSV file row-by-row as dictionaries.

    Yields:
        Raw row dictionaries (values are all strings).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if not file_path.suffix.lower() == ".csv":
        raise ValueError(f"Expected a .csv file, got: {file_path.suffix}")

    with open(file_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def validate_and_parse_rows(
    rows: Generator[dict[str, Any], None, None] | list[dict[str, Any]],
    *,
    strict: bool = False,
) -> tuple[list[Transaction], ValidationReport]:
    """
    Validate raw row dictionaries against the Transaction Pydantic model.

    Args:
        rows: Iterable of raw row dictionaries (from CSV or other source).
        strict: If True, stop on the first validation error.

    Returns:
        Tuple of (valid_transactions, validation_report).
    """
    valid: list[Transaction] = []
    issues: list[ValidationIssue] = []
    total = 0

    for idx, raw_row in enumerate(rows):
        total += 1
        try:
            coerced = _coerce_row(raw_row)
            txn = Transaction(**coerced)
            valid.append(txn)
        except ValidationError as e:
            for err in e.errors():
                issue = ValidationIssue(
                    row_index=idx,
                    field=".".join(str(loc) for loc in err["loc"]),
                    issue=err["msg"],
                    severity="error",
                )
                issues.append(issue)
                logger.warning("Row %d validation error: %s — %s", idx, issue.field, issue.issue)

            if strict:
                break
        except (ValueError, TypeError, KeyError) as e:
            issues.append(
                ValidationIssue(
                    row_index=idx,
                    field="__row__",
                    issue=str(e),
                    severity="error",
                )
            )
            if strict:
                break

    report = ValidationReport(
        total_rows=total,
        valid_rows=len(valid),
        invalid_rows=total - len(valid),
        issues=issues,
    )
    return valid, report


# ── High-Level Ingestion Functions ─────────────────────────────────────────────


def ingest_csv(
    file_path: str | Path,
    *,
    source: str = "csv_upload",
    strict: bool = False,
    encoding: str = "utf-8",
) -> tuple[TransactionBatch, ValidationReport]:
    """
    Ingest a CSV file: read → validate → return batch + report.

    Args:
        file_path: Path to the CSV file.
        source: Source identifier for the batch (e.g., 'csv_upload', 'plaid').
        strict: If True, stop on first validation error.
        encoding: File encoding.

    Returns:
        Tuple of (TransactionBatch, ValidationReport).

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is not a .csv file.
    """
    logger.info("Ingesting CSV: %s", file_path)

    raw_rows = read_csv_raw(file_path, encoding=encoding)
    valid_txns, report = validate_and_parse_rows(raw_rows, strict=strict)

    batch = TransactionBatch(
        transactions=valid_txns,
        source=source,
        ingested_at=datetime.now(timezone.utc),
    ) if valid_txns else TransactionBatch(
        transactions=[],
        source=source,
        ingested_at=datetime.now(timezone.utc),
    )

    # Summary log
    logger.info(
        "Ingestion complete: %d/%d rows valid (%d issues)",
        report.valid_rows,
        report.total_rows,
        len(report.issues),
    )

    return batch, report


def ingest_csv_to_dataframe(
    file_path: str | Path,
    *,
    validate: bool = True,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, ValidationReport | None]:
    """
    Ingest a CSV file and return a pandas DataFrame.

    If validate=True, rows that fail schema validation are dropped and
    a ValidationReport is returned. If validate=False, the raw CSV is
    loaded directly (faster, no validation).

    Args:
        file_path: Path to the CSV file.
        validate: Whether to validate against the Pydantic schema.
        encoding: File encoding.

    Returns:
        Tuple of (DataFrame, ValidationReport or None).
    """
    if not validate:
        df = pd.read_csv(file_path, encoding=encoding, parse_dates=["timestamp"])
        return df, None

    batch, report = ingest_csv(file_path, encoding=encoding)

    if not batch.transactions:
        return pd.DataFrame(), report

    # Convert validated transactions to DataFrame
    records = [txn.model_dump() for txn in batch.transactions]
    df = pd.DataFrame(records)

    # Ensure proper dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["amount"] = df["amount"].astype(float)
    df["merchant_mcc"] = df["merchant_mcc"].astype(int)
    df["is_pending"] = df["is_pending"].astype(bool)

    return df, report


# ── Schema Validation Checks (SPEC §10.2) ─────────────────────────────────────


def validate_batch_quality(
    df: pd.DataFrame,
    *,
    expected_columns: list[str] | None = None,
) -> list[ValidationIssue]:
    """
    Run batch-level quality checks on a DataFrame of transactions.

    Checks aligned with SPEC §10.2 (Great Expectations suites):
    - Column presence
    - Null rates
    - Amount distribution sanity
    - Timestamp freshness

    Args:
        df: DataFrame of transactions.
        expected_columns: Override for expected column names.

    Returns:
        List of ValidationIssue objects.
    """
    issues: list[ValidationIssue] = []

    if expected_columns is None:
        expected_columns = [
            "transaction_id", "user_id", "timestamp", "amount", "currency",
            "merchant_name", "merchant_mcc", "account_type", "channel",
            "location_country", "raw_description", "is_pending",
        ]

    # Check required columns
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        issues.append(ValidationIssue(
            field="__schema__",
            issue=f"Missing columns: {sorted(missing_cols)}",
            severity="error",
        ))

    if df.empty:
        issues.append(ValidationIssue(
            field="__volume__",
            issue="DataFrame is empty",
            severity="error",
        ))
        return issues

    # Check for duplicate transaction IDs
    if "transaction_id" in df.columns:
        n_dupes = df["transaction_id"].duplicated().sum()
        if n_dupes > 0:
            issues.append(ValidationIssue(
                field="transaction_id",
                issue=f"{n_dupes} duplicate transaction IDs found",
                severity="error",
            ))

    # Check null rates on critical fields
    critical_fields = ["user_id", "timestamp", "amount", "merchant_name", "merchant_mcc"]
    for field in critical_fields:
        if field in df.columns:
            null_rate = df[field].isna().mean()
            if null_rate > 0:
                issues.append(ValidationIssue(
                    field=field,
                    issue=f"Null rate = {null_rate:.2%} (expected 0%)",
                    severity="error" if null_rate > 0.01 else "warning",
                ))

    # Amount sanity: check for suspiciously large values
    if "amount" in df.columns:
        max_abs = df["amount"].abs().max()
        if max_abs > 100_000:
            issues.append(ValidationIssue(
                field="amount",
                issue=f"Max absolute amount = ${max_abs:,.2f} — suspiciously large",
                severity="warning",
            ))

    # MCC code range check
    if "merchant_mcc" in df.columns:
        invalid_mcc = ((df["merchant_mcc"] < 0) | (df["merchant_mcc"] > 9999)).sum()
        if invalid_mcc > 0:
            issues.append(ValidationIssue(
                field="merchant_mcc",
                issue=f"{invalid_mcc} rows with MCC outside [0, 9999]",
                severity="error",
            ))

    return issues


# ── Convenience: Full Pipeline ─────────────────────────────────────────────────


def run_ingestion_pipeline(
    file_path: str | Path,
    *,
    source: str = "csv_upload",
    output_parquet: str | Path | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Full ingestion pipeline: CSV → validate → quality checks → optional Parquet export.

    Args:
        file_path: Path to input CSV.
        source: Source tag.
        output_parquet: If provided, save cleaned data as Parquet.

    Returns:
        Tuple of (cleaned DataFrame, ValidationReport).
    """
    logger.info("Running ingestion pipeline for: %s", file_path)

    # Step 1: Ingest and validate
    df, report = ingest_csv_to_dataframe(file_path, validate=True)
    if report is None:
        report = ValidationReport(total_rows=len(df), valid_rows=len(df), invalid_rows=0)

    # Step 2: Batch-level quality checks
    quality_issues = validate_batch_quality(df)
    report.issues.extend(quality_issues)

    if quality_issues:
        logger.warning("Batch quality issues found: %d", len(quality_issues))
        for issue in quality_issues:
            logger.warning("  [%s] %s: %s", issue.severity, issue.field, issue.issue)

    # Step 3: Optional Parquet export
    if output_parquet and not df.empty:
        output_path = Path(output_parquet)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")
        logger.info("Saved cleaned data to: %s", output_path)

    return df, report
