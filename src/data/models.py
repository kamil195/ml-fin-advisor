"""
Pydantic data models for the ML Fin-Advisor project.

Implements the raw transaction schema from SPEC §5.1 and related models.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.constants import (
    AccountType,
    CategoryL1,
    CategoryL2,
    Channel,
)


# ── Raw Transaction Schema (SPEC §5.1) ────────────────────────────────────────


class Transaction(BaseModel):
    """
    A single financial transaction as ingested from bank feeds, CSV uploads,
    or manual entry.

    Mirrors the schema defined in SPEC §5.1. Amounts are signed:
    negative = debit (money out), positive = credit (money in).
    """

    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique transaction identifier (UUID).",
    )
    user_id: str = Field(
        ...,
        description="Unique user identifier (UUID).",
    )
    timestamp: datetime = Field(
        ...,
        description="Transaction timestamp in UTC.",
    )
    amount: float = Field(
        ...,
        description="Signed amount. Negative = debit, positive = credit.",
    )
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code.",
    )
    merchant_name: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Merchant or payee name.",
    )
    merchant_mcc: int = Field(
        ...,
        ge=0,
        le=9999,
        description="4-digit Merchant Category Code (ISO 18245).",
    )
    account_type: AccountType = Field(
        ...,
        description="Type of account the transaction belongs to.",
    )
    channel: Channel = Field(
        ...,
        description="Transaction channel (POS, ONLINE, ATM, TRANSFER, RECURRING).",
    )
    location_city: str | None = Field(
        default=None,
        description="City where the transaction occurred (nullable).",
    )
    location_country: str = Field(
        default="US",
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code.",
    )
    raw_description: str = Field(
        default="",
        max_length=2000,
        description="Raw transaction description from bank feed.",
    )
    is_pending: bool = Field(
        default=False,
        description="Whether the transaction is still pending settlement.",
    )

    # ── Optional label (populated after classification or user correction) ──

    category_l1: CategoryL1 | None = Field(
        default=None,
        description="Level-1 category label (assigned by classifier or user).",
    )
    category_l2: CategoryL2 | None = Field(
        default=None,
        description="Level-2 category label (assigned by classifier or user).",
    )

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        return v.upper()

    @field_validator("location_country")
    @classmethod
    def country_uppercase(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def validate_category_consistency(self) -> Transaction:
        """If both L1 and L2 are set, ensure L2 belongs to L1."""
        from src.utils.constants import CATEGORY_HIERARCHY

        if self.category_l1 is not None and self.category_l2 is not None:
            expected_l1 = CATEGORY_HIERARCHY.get(self.category_l2)
            if expected_l1 != self.category_l1:
                raise ValueError(
                    f"Category mismatch: L2 '{self.category_l2.value}' belongs to "
                    f"'{expected_l1.value if expected_l1 else 'unknown'}', "
                    f"not '{self.category_l1.value}'."
                )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "user_id": "u-00000001-0000-0000-0000-000000000001",
                    "timestamp": "2025-11-15T14:32:00Z",
                    "amount": -42.50,
                    "currency": "USD",
                    "merchant_name": "Whole Foods Market",
                    "merchant_mcc": 5411,
                    "account_type": "CHECKING",
                    "channel": "POS",
                    "location_city": "San Francisco",
                    "location_country": "US",
                    "raw_description": "WHOLE FOODS MKT #10234 SAN FRANCISCO CA",
                    "is_pending": False,
                    "category_l1": "FOOD & DINING",
                    "category_l2": "Groceries",
                }
            ]
        }
    }


# ── Batch Wrapper ──────────────────────────────────────────────────────────────


class TransactionBatch(BaseModel):
    """A batch of transactions for bulk ingestion."""

    transactions: list[Transaction] = Field(
        ...,
        min_length=1,
        description="List of transactions in the batch.",
    )
    source: str = Field(
        default="unknown",
        description="Source identifier (e.g., 'plaid', 'csv_upload', 'manual').",
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the batch was ingested.",
    )

    @property
    def size(self) -> int:
        return len(self.transactions)

    @property
    def user_ids(self) -> set[str]:
        return {t.user_id for t in self.transactions}


# ── Classification Result ──────────────────────────────────────────────────────


class CategoryPrediction(BaseModel):
    """Single category prediction with confidence."""

    category: CategoryL2
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class ClassificationResult(BaseModel):
    """
    Output of the transaction classifier (SPEC §11.2.1).
    """

    category_l1: CategoryL1
    category_l2: CategoryL2
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    top_3: list[CategoryPrediction] = Field(max_length=3)
    is_impulse: bool = False
    impulse_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


# ── Forecast Result ────────────────────────────────────────────────────────────


class CategoryForecast(BaseModel):
    """Probabilistic forecast for a single category (SPEC §11.2.2)."""

    category: CategoryL2
    p10: float = Field(description="10th percentile forecast")
    p50: float = Field(description="Median (50th percentile) forecast")
    p90: float = Field(description="90th percentile forecast")
    trend: str = Field(description="Trend direction: stable, increasing, decreasing")
    regime: str = Field(description="Current spending regime: normal, elevated, reduced, irregular")


class ForecastResult(BaseModel):
    """Full forecast response for a user."""

    user_id: str
    generated_at: datetime
    horizon_days: int
    forecasts: list[CategoryForecast]
    total_spend: dict[str, float] = Field(
        description="Aggregate spend forecast: p10, p50, p90"
    )


# ── Budget Recommendation ─────────────────────────────────────────────────────


class SHAPFeature(BaseModel):
    """A single SHAP feature-impact pair."""

    feature: str
    impact: float


class BudgetRecommendation(BaseModel):
    """Single category budget recommendation with explanation (SPEC §11.2.3)."""

    category: CategoryL2
    recommended_budget: float
    current_trend: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    explanation: str
    shap_top_features: list[SHAPFeature] = Field(default_factory=list)
    anchor_rule: str = ""
    counterfactual: str = ""


class BudgetResult(BaseModel):
    """Full budget recommendation response for a user."""

    user_id: str
    period: str = Field(description="Budget period, e.g. '2026-03'")
    income_estimate: float
    savings_target: float
    recommendations: list[BudgetRecommendation]


# ── Validation Report ──────────────────────────────────────────────────────────


class ValidationIssue(BaseModel):
    """A single data validation issue."""

    row_index: int | None = None
    field: str
    issue: str
    severity: str = Field(description="'error' or 'warning'")


class ValidationReport(BaseModel):
    """Result of validating a transaction batch."""

    total_rows: int
    valid_rows: int
    invalid_rows: int
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return self.invalid_rows == 0
