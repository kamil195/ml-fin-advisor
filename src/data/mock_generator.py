"""
Mock data generator for development and testing.

Produces realistic synthetic transaction data matching the schema in SPEC §5.1.
Default: 100 users × 24 months × ~80 transactions/month ≈ 192 000 rows.

Usage:
    python -m src.data.mock_generator                   # defaults
    python -m src.data.mock_generator --users 10 --months 6 --output data/small.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.utils.constants import (
    AccountType,
    CategoryL2,
    Channel,
    CATEGORY_HIERARCHY,
)

# ── Reproducibility ────────────────────────────────────────────────────────────

SEED = 42
RNG = random.Random(SEED)

# ── Merchant Catalogue ─────────────────────────────────────────────────────────
# Each entry: (merchant_name, mcc, category_l2, typical_amount_range, channel_weights)

MerchantProfile = tuple[str, int, CategoryL2, tuple[float, float], dict[Channel, float]]

MERCHANT_CATALOGUE: list[MerchantProfile] = [
    # ── HOUSING ────────────────────────────────────────────
    ("Apartment Mgmt Corp", 6513, CategoryL2.RENT_MORTGAGE, (900, 2500), {Channel.TRANSFER: 1.0}),
    ("City Water & Electric", 4900, CategoryL2.UTILITIES, (60, 250), {Channel.RECURRING: 0.8, Channel.ONLINE: 0.2}),
    ("National Home Insurance", 6300, CategoryL2.HOME_INSURANCE, (80, 200), {Channel.RECURRING: 1.0}),
    ("Handy Home Repairs", 1520, CategoryL2.MAINTENANCE_REPAIRS, (50, 500), {Channel.POS: 0.7, Channel.ONLINE: 0.3}),

    # ── FOOD & DINING ─────────────────────────────────────
    ("Whole Foods Market", 5411, CategoryL2.GROCERIES, (25, 180), {Channel.POS: 0.9, Channel.ONLINE: 0.1}),
    ("Trader Joe's", 5411, CategoryL2.GROCERIES, (20, 120), {Channel.POS: 1.0}),
    ("Kroger Supermarket", 5411, CategoryL2.GROCERIES, (15, 200), {Channel.POS: 0.95, Channel.ONLINE: 0.05}),
    ("Costco Wholesale", 5411, CategoryL2.GROCERIES, (80, 350), {Channel.POS: 1.0}),
    ("Olive Garden", 5812, CategoryL2.RESTAURANTS, (18, 85), {Channel.POS: 1.0}),
    ("Chipotle Mexican Grill", 5812, CategoryL2.RESTAURANTS, (10, 25), {Channel.POS: 0.7, Channel.ONLINE: 0.3}),
    ("The Capital Grille", 5812, CategoryL2.RESTAURANTS, (60, 250), {Channel.POS: 1.0}),
    ("Starbucks Coffee", 5814, CategoryL2.COFFEE_SHOPS, (3, 9), {Channel.POS: 0.8, Channel.ONLINE: 0.2}),
    ("Blue Bottle Coffee", 5814, CategoryL2.COFFEE_SHOPS, (4, 12), {Channel.POS: 1.0}),
    ("DoorDash", 5814, CategoryL2.FOOD_DELIVERY, (15, 55), {Channel.ONLINE: 1.0}),
    ("Uber Eats", 5814, CategoryL2.FOOD_DELIVERY, (12, 50), {Channel.ONLINE: 1.0}),
    ("The Pub Downtown", 5813, CategoryL2.ALCOHOL_BARS, (15, 80), {Channel.POS: 1.0}),

    # ── TRANSPORTATION ─────────────────────────────────────
    ("Shell Gas Station", 5541, CategoryL2.FUEL, (25, 75), {Channel.POS: 1.0}),
    ("Chevron", 5541, CategoryL2.FUEL, (30, 80), {Channel.POS: 1.0}),
    ("Metro Transit Authority", 4111, CategoryL2.PUBLIC_TRANSIT, (2.50, 50), {Channel.POS: 0.5, Channel.RECURRING: 0.5}),
    ("Uber Rides", 4121, CategoryL2.RIDE_SHARE, (8, 45), {Channel.ONLINE: 1.0}),
    ("Lyft", 4121, CategoryL2.RIDE_SHARE, (7, 40), {Channel.ONLINE: 1.0}),
    ("City Parking Garage", 7523, CategoryL2.PARKING_TOLLS, (5, 30), {Channel.POS: 1.0}),
    ("Jiffy Lube Auto Service", 7538, CategoryL2.VEHICLE_MAINTENANCE, (30, 400), {Channel.POS: 1.0}),

    # ── SHOPPING & ENTERTAINMENT ───────────────────────────
    ("Nordstrom", 5311, CategoryL2.CLOTHING_ACCESSORIES, (30, 300), {Channel.POS: 0.5, Channel.ONLINE: 0.5}),
    ("H&M Fashion", 5311, CategoryL2.CLOTHING_ACCESSORIES, (15, 120), {Channel.POS: 0.6, Channel.ONLINE: 0.4}),
    ("Best Buy Electronics", 5732, CategoryL2.ELECTRONICS, (20, 800), {Channel.POS: 0.4, Channel.ONLINE: 0.6}),
    ("Apple Store", 5732, CategoryL2.ELECTRONICS, (30, 1500), {Channel.POS: 0.5, Channel.ONLINE: 0.5}),
    ("Netflix", 4899, CategoryL2.SUBSCRIPTIONS_STREAMING, (9.99, 22.99), {Channel.RECURRING: 1.0}),
    ("Spotify", 4899, CategoryL2.SUBSCRIPTIONS_STREAMING, (9.99, 15.99), {Channel.RECURRING: 1.0}),
    ("Disney+", 4899, CategoryL2.SUBSCRIPTIONS_STREAMING, (7.99, 13.99), {Channel.RECURRING: 1.0}),
    ("REI Co-op", 5945, CategoryL2.HOBBIES_SPORTS, (20, 250), {Channel.POS: 0.5, Channel.ONLINE: 0.5}),
    ("Amazon Books", 5942, CategoryL2.BOOKS_MEDIA, (8, 40), {Channel.ONLINE: 1.0}),
    ("Charitable Foundation", 8099, CategoryL2.GIFTS_DONATIONS, (10, 200), {Channel.ONLINE: 0.8, Channel.TRANSFER: 0.2}),

    # ── HEALTH & PERSONAL ──────────────────────────────────
    ("CVS Pharmacy", 5912, CategoryL2.HEALTHCARE_PHARMACY, (5, 120), {Channel.POS: 0.9, Channel.ONLINE: 0.1}),
    ("Dr. Smith Medical", 8011, CategoryL2.HEALTHCARE_PHARMACY, (20, 350), {Channel.POS: 1.0}),
    ("Planet Fitness", 7941, CategoryL2.FITNESS_GYM, (10, 50), {Channel.RECURRING: 1.0}),
    ("Great Clips Salon", 7298, CategoryL2.PERSONAL_CARE, (15, 60), {Channel.POS: 1.0}),
    ("PetSmart", 742, CategoryL2.PET_CARE, (10, 150), {Channel.POS: 0.7, Channel.ONLINE: 0.3}),

    # ── FINANCIAL ──────────────────────────────────────────
    ("Vanguard Transfer", 6010, CategoryL2.SAVINGS_INVESTMENTS, (100, 1000), {Channel.TRANSFER: 1.0}),
    ("Student Loan Corp", 6012, CategoryL2.LOAN_PAYMENTS, (150, 600), {Channel.RECURRING: 1.0}),
    ("AllState Insurance", 6300, CategoryL2.INSURANCE_PREMIUMS, (80, 250), {Channel.RECURRING: 1.0}),
    ("Bank Monthly Fee", 6012, CategoryL2.FEES_CHARGES, (5, 35), {Channel.RECURRING: 1.0}),
    ("IRS Tax Payment", 9311, CategoryL2.TAXES, (200, 3000), {Channel.TRANSFER: 1.0}),
]

# Countries for random assignment
COUNTRIES = ["US", "US", "US", "US", "CA", "GB", "DE", "FR", "AU"]

CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "San Francisco", "Seattle", "Denver", "Austin", "Miami",
    "Boston", "Portland", "Nashville", "San Diego", "Atlanta",
]


# ── User Profile Generation ───────────────────────────────────────────────────


class UserProfile:
    """Encapsulates a synthetic user's spending behavior."""

    def __init__(self, user_id: str, rng: random.Random) -> None:
        self.user_id = user_id
        self.rng = rng

        # Income & pay cycle
        self.monthly_income = rng.gauss(5500, 1500)
        self.monthly_income = max(2000, min(15000, self.monthly_income))
        self.pay_day = rng.choice([1, 15])  # 1st or 15th
        self.pay_frequency_days = rng.choice([14, 15, 30])

        # Account preferences
        self.primary_account = rng.choice(list(AccountType))
        self.country = rng.choice(COUNTRIES)
        self.city = rng.choice(CITIES)

        # Spending personality: selects a subset of merchants the user frequents
        n_merchants = rng.randint(15, len(MERCHANT_CATALOGUE))
        self.preferred_merchants: list[MerchantProfile] = rng.sample(
            MERCHANT_CATALOGUE, k=n_merchants
        )

        # Category spending weights (some users spend more on dining, etc.)
        self.category_affinity: dict[CategoryL2, float] = {}
        for m in self.preferred_merchants:
            cat = m[2]
            self.category_affinity[cat] = self.category_affinity.get(cat, 0) + rng.uniform(0.5, 2.0)

        # Subscriptions: recurring monthly charges (always appear)
        self.subscriptions = [
            m for m in self.preferred_merchants if Channel.RECURRING in m[4]
        ]

    def monthly_txn_count(self) -> int:
        """Target ~80 transactions/month with some variation."""
        return max(40, int(self.rng.gauss(80, 15)))


# ── Transaction Generation ─────────────────────────────────────────────────────


def _pick_channel(channel_weights: dict[Channel, float], rng: random.Random) -> Channel:
    """Weighted random selection of a transaction channel."""
    channels = list(channel_weights.keys())
    weights = list(channel_weights.values())
    return rng.choices(channels, weights=weights, k=1)[0]


def _generate_amount(low: float, high: float, rng: random.Random) -> float:
    """Generate a realistic transaction amount within range."""
    # Use log-normal-ish distribution for more realistic amounts
    mid = (low + high) / 2
    spread = (high - low) / 4
    amount = rng.gauss(mid, spread)
    amount = max(low, min(high, amount))
    return round(amount, 2)


def _random_timestamp(
    year: int, month: int, rng: random.Random
) -> datetime:
    """Generate a random timestamp within a given month."""
    import calendar

    _, days_in_month = calendar.monthrange(year, month)
    day = rng.randint(1, days_in_month)
    hour = rng.randint(6, 23)  # most transactions between 6 AM and 11 PM
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _generate_raw_description(merchant_name: str, city: str, rng: random.Random) -> str:
    """Generate a realistic raw bank description."""
    # Simulate messy bank descriptions
    variations = [
        f"{merchant_name.upper()} {city.upper()[:3]}",
        f"{merchant_name.upper().replace(' ', '')} #{rng.randint(100, 99999)}",
        f"POS {merchant_name.upper()} {city.upper()} {''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}",
        f"{merchant_name.upper()} - {''.join(rng.choices('0123456789', k=4))}",
        f"PURCHASE {merchant_name.upper()} {rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}",
    ]
    return rng.choice(variations)


def generate_user_month(
    user: UserProfile,
    year: int,
    month: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Generate all transactions for one user in one month."""
    transactions: list[dict[str, Any]] = []
    txn_count = user.monthly_txn_count()

    # Always add income deposit(s)
    income_txn = {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user.user_id,
        "timestamp": datetime(year, month, min(user.pay_day, 28), 8, 0, 0, tzinfo=timezone.utc).isoformat(),
        "amount": round(user.monthly_income / (30 / user.pay_frequency_days), 2),
        "currency": "USD",
        "merchant_name": "Employer Direct Deposit",
        "merchant_mcc": 6012,
        "account_type": AccountType.CHECKING.value,
        "channel": Channel.TRANSFER.value,
        "location_city": user.city,
        "location_country": user.country,
        "raw_description": f"DIRECT DEP EMPLOYER PAYROLL {year}{month:02d}",
        "is_pending": False,
        "category_l1": "FINANCIAL",
        "category_l2": CategoryL2.INCOME.value,
    }
    transactions.append(income_txn)

    # If biweekly, add second paycheck
    if user.pay_frequency_days <= 15:
        second_pay_day = min(user.pay_day + user.pay_frequency_days, 28)
        income_txn2 = income_txn.copy()
        income_txn2["transaction_id"] = str(uuid.uuid4())
        income_txn2["timestamp"] = datetime(
            year, month, second_pay_day, 8, 0, 0, tzinfo=timezone.utc
        ).isoformat()
        income_txn2["raw_description"] = f"DIRECT DEP EMPLOYER PAYROLL {year}{month:02d} B"
        transactions.append(income_txn2)

    # Always add subscriptions (recurring charges)
    for merchant in user.subscriptions:
        name, mcc, cat, (low, high), ch_weights = merchant
        sub_txn = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user.user_id,
            "timestamp": datetime(year, month, rng.randint(1, 5), 0, 0, 0, tzinfo=timezone.utc).isoformat(),
            "amount": -round(rng.uniform(low, high), 2),
            "currency": "USD",
            "merchant_name": name,
            "merchant_mcc": mcc,
            "account_type": user.primary_account.value,
            "channel": Channel.RECURRING.value,
            "location_city": None,
            "location_country": user.country,
            "raw_description": f"RECURRING {name.upper()} MONTHLY",
            "is_pending": False,
            "category_l1": CATEGORY_HIERARCHY[cat].value,
            "category_l2": cat.value,
        }
        transactions.append(sub_txn)

    # Fill remaining with random spending from preferred merchants
    remaining = txn_count - len(transactions)
    for _ in range(max(0, remaining)):
        merchant = rng.choice(user.preferred_merchants)
        name, mcc, cat, (low, high), ch_weights = merchant

        # Skip income (handled above) and some subscriptions (already added)
        if cat == CategoryL2.INCOME:
            continue

        ts = _random_timestamp(year, month, rng)
        amount = -_generate_amount(low, high, rng)  # debit = negative
        channel = _pick_channel(ch_weights, rng)

        txn = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user.user_id,
            "timestamp": ts.isoformat(),
            "amount": amount,
            "currency": "USD",
            "merchant_name": name,
            "merchant_mcc": mcc,
            "account_type": user.primary_account.value,
            "channel": channel.value,
            "location_city": user.city if channel == Channel.POS else None,
            "location_country": user.country,
            "raw_description": _generate_raw_description(name, user.city, rng),
            "is_pending": rng.random() < 0.02,  # 2% pending
            "category_l1": CATEGORY_HIERARCHY[cat].value,
            "category_l2": cat.value,
        }
        transactions.append(txn)

    return transactions


# ── Main Generator ─────────────────────────────────────────────────────────────


def generate_dataset(
    n_users: int = 100,
    n_months: int = 24,
    start_year: int = 2024,
    start_month: int = 3,
    output_path: str | Path = "data/mock_transactions.csv",
    seed: int = SEED,
) -> Path:
    """
    Generate the complete mock dataset.

    Args:
        n_users: Number of synthetic users.
        n_months: Number of months of history to generate.
        start_year: Year for the first month of data.
        start_month: Month (1-12) for the first month of data.
        output_path: Path for the output CSV file.
        seed: Random seed for reproducibility.

    Returns:
        Path to the generated CSV file.
    """
    rng = random.Random(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV columns (matches Transaction model fields)
    fieldnames = [
        "transaction_id", "user_id", "timestamp", "amount", "currency",
        "merchant_name", "merchant_mcc", "account_type", "channel",
        "location_city", "location_country", "raw_description", "is_pending",
        "category_l1", "category_l2",
    ]

    # Create user profiles
    users = [
        UserProfile(
            user_id=f"u-{i:08d}-0000-0000-0000-{i:012d}",
            rng=random.Random(rng.randint(0, 2**32)),
        )
        for i in range(1, n_users + 1)
    ]

    total_txns = 0
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for user_idx, user in enumerate(users, 1):
            year, month = start_year, start_month

            for _ in range(n_months):
                month_txns = generate_user_month(user, year, month, rng)

                for txn in month_txns:
                    writer.writerow(txn)
                    total_txns += 1

                # Advance to next month
                month += 1
                if month > 12:
                    month = 1
                    year += 1

            if user_idx % 10 == 0:
                print(f"  Generated data for {user_idx}/{n_users} users...")

    print(f"\nDone! Generated {total_txns:,} transactions for {n_users} users.")
    print(f"Output: {output_path.resolve()}")
    return output_path


# ── CLI Entry Point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate mock financial transaction data for ML Fin-Advisor."
    )
    parser.add_argument("--users", type=int, default=100, help="Number of users (default: 100)")
    parser.add_argument("--months", type=int, default=24, help="Months of history (default: 24)")
    parser.add_argument("--start-year", type=int, default=2024, help="Start year (default: 2024)")
    parser.add_argument("--start-month", type=int, default=3, help="Start month (default: 3)")
    parser.add_argument("--output", type=str, default="data/mock_transactions.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    generate_dataset(
        n_users=args.users,
        n_months=args.months,
        start_year=args.start_year,
        start_month=args.start_month,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
