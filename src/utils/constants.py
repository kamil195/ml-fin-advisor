"""
Constants and enumerations for the ML Fin-Advisor project.

Defines the category taxonomy (SPEC §5.3), enums, and MCC mappings (SPEC Appendix B).
"""

from enum import Enum


# ── Account & Channel Enums ────────────────────────────────────────────────────


class AccountType(str, Enum):
    """Bank account type."""

    CHECKING = "CHECKING"
    SAVINGS = "SAVINGS"
    CREDIT = "CREDIT"
    INVESTMENT = "INVESTMENT"


class Channel(str, Enum):
    """Transaction channel."""

    POS = "POS"
    ONLINE = "ONLINE"
    ATM = "ATM"
    TRANSFER = "TRANSFER"
    RECURRING = "RECURRING"


# ── Category Taxonomy (SPEC §5.3) ─────────────────────────────────────────────


class CategoryL1(str, Enum):
    """Level-1 spending categories."""

    HOUSING = "HOUSING"
    FOOD_AND_DINING = "FOOD & DINING"
    TRANSPORTATION = "TRANSPORTATION"
    SHOPPING_AND_ENTERTAINMENT = "SHOPPING & ENTERTAINMENT"
    HEALTH_AND_PERSONAL = "HEALTH & PERSONAL"
    FINANCIAL = "FINANCIAL"


class CategoryL2(str, Enum):
    """Level-2 spending categories (30 total)."""

    # HOUSING
    RENT_MORTGAGE = "Rent/Mortgage"
    UTILITIES = "Utilities"
    HOME_INSURANCE = "Home Insurance"
    MAINTENANCE_REPAIRS = "Maintenance & Repairs"

    # FOOD & DINING
    GROCERIES = "Groceries"
    RESTAURANTS = "Restaurants"
    COFFEE_SHOPS = "Coffee Shops"
    FOOD_DELIVERY = "Food Delivery"
    ALCOHOL_BARS = "Alcohol & Bars"

    # TRANSPORTATION
    FUEL = "Fuel"
    PUBLIC_TRANSIT = "Public Transit"
    RIDE_SHARE = "Ride-Share"
    PARKING_TOLLS = "Parking & Tolls"
    VEHICLE_MAINTENANCE = "Vehicle Maintenance"

    # SHOPPING & ENTERTAINMENT
    CLOTHING_ACCESSORIES = "Clothing & Accessories"
    ELECTRONICS = "Electronics"
    SUBSCRIPTIONS_STREAMING = "Subscriptions & Streaming"
    HOBBIES_SPORTS = "Hobbies & Sports"
    BOOKS_MEDIA = "Books & Media"
    GIFTS_DONATIONS = "Gifts & Donations"

    # HEALTH & PERSONAL
    HEALTHCARE_PHARMACY = "Healthcare & Pharmacy"
    FITNESS_GYM = "Fitness & Gym"
    PERSONAL_CARE = "Personal Care"
    PET_CARE = "Pet Care"

    # FINANCIAL
    SAVINGS_INVESTMENTS = "Savings & Investments"
    LOAN_PAYMENTS = "Loan Payments"
    INSURANCE_PREMIUMS = "Insurance Premiums"
    FEES_CHARGES = "Fees & Charges"
    TAXES = "Taxes"
    INCOME = "Income"


# Hierarchical mapping: L2 → L1
CATEGORY_HIERARCHY: dict[CategoryL2, CategoryL1] = {
    # HOUSING
    CategoryL2.RENT_MORTGAGE: CategoryL1.HOUSING,
    CategoryL2.UTILITIES: CategoryL1.HOUSING,
    CategoryL2.HOME_INSURANCE: CategoryL1.HOUSING,
    CategoryL2.MAINTENANCE_REPAIRS: CategoryL1.HOUSING,
    # FOOD & DINING
    CategoryL2.GROCERIES: CategoryL1.FOOD_AND_DINING,
    CategoryL2.RESTAURANTS: CategoryL1.FOOD_AND_DINING,
    CategoryL2.COFFEE_SHOPS: CategoryL1.FOOD_AND_DINING,
    CategoryL2.FOOD_DELIVERY: CategoryL1.FOOD_AND_DINING,
    CategoryL2.ALCOHOL_BARS: CategoryL1.FOOD_AND_DINING,
    # TRANSPORTATION
    CategoryL2.FUEL: CategoryL1.TRANSPORTATION,
    CategoryL2.PUBLIC_TRANSIT: CategoryL1.TRANSPORTATION,
    CategoryL2.RIDE_SHARE: CategoryL1.TRANSPORTATION,
    CategoryL2.PARKING_TOLLS: CategoryL1.TRANSPORTATION,
    CategoryL2.VEHICLE_MAINTENANCE: CategoryL1.TRANSPORTATION,
    # SHOPPING & ENTERTAINMENT
    CategoryL2.CLOTHING_ACCESSORIES: CategoryL1.SHOPPING_AND_ENTERTAINMENT,
    CategoryL2.ELECTRONICS: CategoryL1.SHOPPING_AND_ENTERTAINMENT,
    CategoryL2.SUBSCRIPTIONS_STREAMING: CategoryL1.SHOPPING_AND_ENTERTAINMENT,
    CategoryL2.HOBBIES_SPORTS: CategoryL1.SHOPPING_AND_ENTERTAINMENT,
    CategoryL2.BOOKS_MEDIA: CategoryL1.SHOPPING_AND_ENTERTAINMENT,
    CategoryL2.GIFTS_DONATIONS: CategoryL1.SHOPPING_AND_ENTERTAINMENT,
    # HEALTH & PERSONAL
    CategoryL2.HEALTHCARE_PHARMACY: CategoryL1.HEALTH_AND_PERSONAL,
    CategoryL2.FITNESS_GYM: CategoryL1.HEALTH_AND_PERSONAL,
    CategoryL2.PERSONAL_CARE: CategoryL1.HEALTH_AND_PERSONAL,
    CategoryL2.PET_CARE: CategoryL1.HEALTH_AND_PERSONAL,
    # FINANCIAL
    CategoryL2.SAVINGS_INVESTMENTS: CategoryL1.FINANCIAL,
    CategoryL2.LOAN_PAYMENTS: CategoryL1.FINANCIAL,
    CategoryL2.INSURANCE_PREMIUMS: CategoryL1.FINANCIAL,
    CategoryL2.FEES_CHARGES: CategoryL1.FINANCIAL,
    CategoryL2.TAXES: CategoryL1.FINANCIAL,
    CategoryL2.INCOME: CategoryL1.FINANCIAL,
}

# Reverse mapping: L1 → list[L2]
L1_TO_L2: dict[CategoryL1, list[CategoryL2]] = {}
for l2, l1 in CATEGORY_HIERARCHY.items():
    L1_TO_L2.setdefault(l1, []).append(l2)


# ── Discretionary vs. Non-Discretionary ────────────────────────────────────────

DISCRETIONARY_CATEGORIES: set[CategoryL2] = {
    CategoryL2.RESTAURANTS,
    CategoryL2.COFFEE_SHOPS,
    CategoryL2.FOOD_DELIVERY,
    CategoryL2.ALCOHOL_BARS,
    CategoryL2.RIDE_SHARE,
    CategoryL2.CLOTHING_ACCESSORIES,
    CategoryL2.ELECTRONICS,
    CategoryL2.SUBSCRIPTIONS_STREAMING,
    CategoryL2.HOBBIES_SPORTS,
    CategoryL2.BOOKS_MEDIA,
    CategoryL2.GIFTS_DONATIONS,
    CategoryL2.PERSONAL_CARE,
}


# ── MCC Code Mappings (SPEC Appendix B) ───────────────────────────────────────

# Maps MCC code ranges to (L1, L2) category tuples.
# Format: (mcc_low, mcc_high) → CategoryL2
MCC_TO_CATEGORY: dict[tuple[int, int], CategoryL2] = {
    # FOOD & DINING
    (5411, 5499): CategoryL2.GROCERIES,
    (5812, 5812): CategoryL2.RESTAURANTS,
    (5813, 5813): CategoryL2.ALCOHOL_BARS,
    (5814, 5814): CategoryL2.FOOD_DELIVERY,
    # TRANSPORTATION
    (5541, 5542): CategoryL2.FUEL,
    (4111, 4131): CategoryL2.PUBLIC_TRANSIT,
    (4121, 4121): CategoryL2.RIDE_SHARE,  # overlaps transit intentionally
    (7523, 7523): CategoryL2.PARKING_TOLLS,
    (7531, 7531): CategoryL2.PARKING_TOLLS,
    (7538, 7538): CategoryL2.VEHICLE_MAINTENANCE,
    # SHOPPING & ENTERTAINMENT
    (5311, 5399): CategoryL2.CLOTHING_ACCESSORIES,
    (5732, 5733): CategoryL2.ELECTRONICS,
    (5735, 5735): CategoryL2.BOOKS_MEDIA,
    (5942, 5942): CategoryL2.BOOKS_MEDIA,
    (5945, 5945): CategoryL2.HOBBIES_SPORTS,
    (5947, 5947): CategoryL2.GIFTS_DONATIONS,
    # HEALTH & PERSONAL
    (8011, 8099): CategoryL2.HEALTHCARE_PHARMACY,
    (5912, 5912): CategoryL2.HEALTHCARE_PHARMACY,
    (7941, 7941): CategoryL2.FITNESS_GYM,
    (7298, 7298): CategoryL2.PERSONAL_CARE,
    (742, 742): CategoryL2.PET_CARE,
    # HOUSING
    (4900, 4900): CategoryL2.UTILITIES,
    # FINANCIAL
    (6010, 6012): CategoryL2.FEES_CHARGES,
    (6051, 6051): CategoryL2.FEES_CHARGES,
}


def lookup_category_by_mcc(mcc: int) -> CategoryL2 | None:
    """Look up L2 category by MCC code. Returns None if no mapping found."""
    for (low, high), category in MCC_TO_CATEGORY.items():
        if low <= mcc <= high:
            return category
    return None
