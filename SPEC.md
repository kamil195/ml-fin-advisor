# Personal Finance Advisor with Behavior Modeling

## A Supervised Learning Approach to Transaction Classification, Time-Series Expense Forecasting, and Interpretable Budget Recommendations

**Version:** 1.0  
**Date:** February 19, 2026  
**Status:** Draft  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Success Criteria](#3-goals--success-criteria)
4. [System Architecture](#4-system-architecture)
5. [Data Specification](#5-data-specification)
6. [Module 1 — Transaction Classification](#6-module-1--transaction-classification)
7. [Module 2 — Time-Series Expense Forecasting](#7-module-2--time-series-expense-forecasting)
8. [Module 3 — Interpretable Budget Recommendations](#8-module-3--interpretable-budget-recommendations)
9. [Behavior Modeling Layer](#9-behavior-modeling-layer)
10. [Training & Evaluation Pipeline](#10-training--evaluation-pipeline)
11. [Serving & Inference](#11-serving--inference)
12. [Ethical Considerations & Fairness](#12-ethical-considerations--fairness)
13. [Project Structure](#13-project-structure)
14. [Milestones & Timeline](#14-milestones--timeline)
15. [Appendices](#15-appendices)

---

## 1. Executive Summary

This specification defines a **supervised-learning personal finance advisor** that ingests raw bank/card transaction data and delivers three core capabilities:

| Capability | Technique | Output |
|---|---|---|
| **Transaction Classification** | Multi-class supervised learning (gradient-boosted trees + fine-tuned transformer embeddings) | Category label + confidence score per transaction |
| **Expense Forecasting** | Time-series models (Prophet / N-BEATS / Temporal Fusion Transformer) | 30/60/90-day rolling expense forecasts per category |
| **Budget Recommendations** | Constraint-based optimization with interpretable rule extraction (SHAP + anchors) | Per-category budget ceilings with natural-language explanations |

A cross-cutting **Behavior Modeling Layer** captures user spending patterns, detects regime changes (e.g., lifestyle inflation), and feeds behavioral features into all three modules.

---

## 2. Problem Statement

### 2.1 Context

Consumer financial management tools today either (a) classify transactions with rigid rule-based systems that break on merchant-name variations, or (b) forecast spending with naïve averages that ignore seasonality and behavioral shifts. Users receive budgets that feel arbitrary and are therefore ignored.

### 2.2 Gaps Addressed

| Gap | How This System Addresses It |
|---|---|
| Brittle merchant-to-category mappings | Learned embeddings over merchant names, MCC codes, amounts, and temporal context |
| Static forecasts | Multi-horizon probabilistic forecasts that adapt to regime changes |
| Opaque recommendations | Every budget suggestion accompanied by feature-attribution explanations (SHAP values, anchor rules) |
| No behavioral awareness | Explicit behavior model capturing habit formation, impulse-vs-planned spending, and income-cycle alignment |

---

## 3. Goals & Success Criteria

### 3.1 Functional Goals

| ID | Goal | Measurable Target |
|---|---|---|
| G1 | Classify transactions into ≥ 30 spending categories | Macro-F1 ≥ 0.92 on held-out test set |
| G2 | Forecast next-30-day total spend per category | MAPE ≤ 12% across top-10 categories by volume |
| G3 | Produce per-category budget ceilings | ≥ 78% user acceptance rate (A/B test) |
| G4 | Provide human-readable explanations for every recommendation | 100% of recommendations include ≥ 1 interpretable rule |
| G5 | Detect behavioral regime changes within 7 days of onset | Precision ≥ 0.85, Recall ≥ 0.80 |

### 3.2 Non-Functional Goals

| ID | Goal | Target |
|---|---|---|
| NF1 | Classification latency (p99) | ≤ 50 ms per transaction |
| NF2 | Forecast generation latency | ≤ 5 s for full user profile |
| NF3 | Model retraining cadence | Weekly (incremental); monthly (full) |
| NF4 | Data privacy | All PII tokenized; no raw merchant names stored post-feature-extraction |
| NF5 | Horizontal scalability | Handle ≥ 10 M users with linear cost scaling |

---

## 4. System Architecture

### 4.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                        │
│  Bank Feeds (Plaid/Yodlee)  ·  CSV Upload  ·  Manual Entry API    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING PIPELINE                   │
│                                                                     │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Text Features│  │ Numerical Feats  │  │ Temporal Features    │  │
│  │ (merchant    │  │ (amount, balance │  │ (day-of-week, pay-  │  │
│  │  embeddings) │  │  velocity, etc.) │  │  cycle phase, etc.) │  │
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│         └──────────────┬────┴───────────────────────┘              │
│                        ▼                                            │
│              ┌──────────────────┐                                   │
│              │ BEHAVIOR MODELING│                                   │
│              │     LAYER        │                                   │
│              └────────┬─────────┘                                   │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────────┐
   │ MODULE 1    │ │ MODULE 2 │ │ MODULE 3     │
   │ Transaction │ │ Expense  │ │ Budget       │
   │ Classifier  │ │ Forecast │ │ Recommender  │
   └──────┬──────┘ └────┬─────┘ └──────┬───────┘
          │              │              │
          └──────────────┼──────────────┘
                         ▼
              ┌─────────────────────┐
              │   SERVING LAYER     │
              │  REST API / gRPC    │
              │  + Explanation UI   │
              └─────────────────────┘
```

### 4.2 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Framework | PyTorch 2.x, scikit-learn, XGBoost, LightGBM |
| Time-Series | Nixtla `neuralforecast` (N-BEATS, TFT), Prophet |
| NLP Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Interpretability | SHAP, Alibi (Anchors), LIME |
| Orchestration | Apache Airflow / Prefect |
| Feature Store | Feast (offline: Parquet on S3; online: Redis) |
| Model Registry | MLflow |
| Serving | FastAPI + ONNX Runtime / TorchServe |
| Monitoring | Evidently AI (data drift), Prometheus + Grafana |
| Infrastructure | Kubernetes (EKS), Terraform |

---

## 5. Data Specification

### 5.1 Raw Transaction Schema

```
Transaction {
    transaction_id   : str (UUID)
    user_id          : str (UUID)
    timestamp        : datetime (UTC)
    amount           : float (signed; negative = debit)
    currency         : str (ISO 4217)
    merchant_name    : str
    merchant_mcc     : int (4-digit Merchant Category Code)
    account_type     : enum [CHECKING, SAVINGS, CREDIT, INVESTMENT]
    channel          : enum [POS, ONLINE, ATM, TRANSFER, RECURRING]
    location_city    : str | null
    location_country : str (ISO 3166-1 alpha-2)
    raw_description  : str
    is_pending       : bool
}
```

### 5.2 Derived Feature Groups

#### 5.2.1 Text Features

| Feature | Derivation | Dimension |
|---|---|---|
| `merchant_embedding` | Sentence-Transformer encoding of `merchant_name + raw_description` | 384 |
| `merchant_name_tokens` | Character-trigram TF-IDF (top 5 000 features, SVD-reduced) | 64 |
| `mcc_embedding` | Learned embedding via category-label supervision | 16 |

#### 5.2.2 Numerical Features

| Feature | Derivation |
|---|---|
| `log_amount` | `log1p(abs(amount))` |
| `amount_zscore_user` | Z-score of amount within user's historical distribution |
| `amount_pct_of_income` | `abs(amount) / estimated_monthly_income` |
| `balance_after` | Running balance post-transaction (when available) |
| `rolling_spend_7d` | Sum of debits in trailing 7-day window |
| `rolling_spend_30d` | Sum of debits in trailing 30-day window |
| `txn_count_24h` | Number of transactions in preceding 24 hours |

#### 5.2.3 Temporal Features

| Feature | Derivation |
|---|---|
| `hour_of_day` | Cyclical encoding (sin/cos) |
| `day_of_week` | Cyclical encoding (sin/cos) |
| `day_of_month` | Cyclical encoding (sin/cos) |
| `days_since_payday` | Distance to nearest detected income deposit |
| `is_weekend` | Boolean |
| `is_holiday` | Boolean (country-aware via `holidays` library) |
| `month_phase` | Categorical: `early` (1–10), `mid` (11–20), `late` (21–end) |

#### 5.2.4 Behavioral Features (from Behavior Modeling Layer — see §9)

| Feature | Derivation |
|---|---|
| `spending_regime` | Current regime label (e.g., `normal`, `elevated`, `reduced`) |
| `impulse_score` | Probability that this transaction is impulse-driven |
| `habit_strength` | Recurrence strength of this merchant/category (0–1) |
| `income_cycle_phase` | Normalized position within detected pay cycle (0.0–1.0) |
| `lifestyle_drift_30d` | % change in median category spend vs. 90-day baseline |

### 5.3 Label Schema (Transaction Categories)

Hierarchical 2-level taxonomy (6 L1 + 30 L2):

```
HOUSING
  ├── Rent/Mortgage
  ├── Utilities
  ├── Home Insurance
  └── Maintenance & Repairs

FOOD & DINING
  ├── Groceries
  ├── Restaurants
  ├── Coffee Shops
  ├── Food Delivery
  └── Alcohol & Bars

TRANSPORTATION
  ├── Fuel
  ├── Public Transit
  ├── Ride-Share
  ├── Parking & Tolls
  └── Vehicle Maintenance

SHOPPING & ENTERTAINMENT
  ├── Clothing & Accessories
  ├── Electronics
  ├── Subscriptions & Streaming
  ├── Hobbies & Sports
  ├── Books & Media
  └── Gifts & Donations

HEALTH & PERSONAL
  ├── Healthcare & Pharmacy
  ├── Fitness & Gym
  ├── Personal Care
  └── Pet Care

FINANCIAL
  ├── Savings & Investments
  ├── Loan Payments
  ├── Insurance Premiums
  ├── Fees & Charges
  ├── Taxes
  └── Income (credit-side)
```

### 5.4 Data Volume Assumptions

| Entity | Estimated Scale |
|---|---|
| Users | 10 M |
| Transactions / user / month | ~80 |
| Training corpus (historical) | 2 years × 10 M users × 80 txns/mo ≈ 19.2 B rows |
| Feature vector width (dense) | ~500 dimensions |

### 5.5 Data Splits

| Split | Allocation | Strategy |
|---|---|---|
| Train | 70% | Temporal split: all data before T − 90 days |
| Validation | 15% | T − 90 days to T − 30 days |
| Test | 15% | Most recent 30 days |
| Out-of-time hold-out | Separate | Next calendar month (post-deployment baseline) |

> **Note:** Splits are **per-user temporal** to prevent label leakage from future transactions.

---

## 6. Module 1 — Transaction Classification

### 6.1 Objective

Assign each incoming transaction a **category label** (L2) with a calibrated confidence score. Support user corrections as online feedback for model refinement.

### 6.2 Model Architecture

```
                  ┌───────────────────┐
                  │  Raw Transaction  │
                  └────────┬──────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌───────────┐  ┌──────────────┐
   │  Text Tower │  │ Numerical │  │  Temporal +   │
   │ (frozen     │  │  Features │  │  Behavioral   │
   │  SentTrans  │  │   (batch  │  │   Features    │
   │  + learned  │  │    norm)  │  │  (batch norm) │
   │  projection)│  └─────┬─────┘  └──────┬────────┘
   └──────┬──────┘        │               │
          │               │               │
          └───────────────┬───────────────┘
                          ▼
                 ┌─────────────────┐
                 │  Concatenation  │
                 │  + 2-layer MLP  │
                 │  (512 → 256)    │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  Gradient-Boost │
                 │  Meta-Learner   │
                 │  (LightGBM on   │
                 │   MLP logits +  │
                 │   raw features) │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  Softmax Output │
                 │  (30 classes)   │
                 └─────────────────┘
```

**Rationale for stacked architecture:**
- The **text tower** captures semantic similarity among merchant names regardless of spelling variation.
- The **MLP** learns cross-feature interactions between embeddings, amounts, and temporal signals.
- The **LightGBM meta-learner** acts as a calibration/correction layer that captures residual patterns (especially long-tail merchants) and provides well-calibrated probabilities out-of-the-box.

### 6.3 Training Procedure

| Aspect | Detail |
|---|---|
| Loss | Focal loss (γ = 2.0) to handle class imbalance |
| Optimizer (MLP) | AdamW, lr = 3e-4, cosine schedule, 20 epochs |
| Meta-learner | LightGBM with 5-fold CV on MLP outputs (out-of-fold predictions) |
| Class weighting | Inverse-frequency weighting with cap at 10× |
| Data augmentation | Merchant name typo injection (random char swap/drop, 10% of samples) |
| User correction feedback | Treat as high-confidence labels (weight = 3× in loss); retrain weekly |

### 6.4 Evaluation Metrics

| Metric | Target | Rationale |
|---|---|---|
| **Macro-F1** | ≥ 0.92 | Primary metric; ensures balanced performance across all categories |
| Top-3 Accuracy | ≥ 0.98 | Fallback UX: show top-3 suggestions if confidence < threshold |
| ECE (Expected Calibration Error) | ≤ 0.03 | Confidence scores must be reliable for downstream modules |
| Per-class Recall (min) | ≥ 0.80 | No category should be systematically missed |
| Latency (p99) | ≤ 50 ms | Real-time classification on transaction arrival |

### 6.5 Handling Edge Cases

| Scenario | Strategy |
|---|---|
| Unknown / new merchant | Fall back to MCC code + amount heuristics; flag for human review if confidence < 0.5 |
| Multi-category transactions (e.g., Walmart) | Use amount + time-of-day heuristics; if ambiguous, assign most frequent category for that user at that merchant |
| International transactions | Currency-normalized amount features + country-aware MCC mapping |
| Recurring vs. one-time | Channel feature (`RECURRING` flag) + recurrence detection from behavior model |

---

## 7. Module 2 — Time-Series Expense Forecasting

### 7.1 Objective

Produce **probabilistic multi-horizon forecasts** of per-category spending for each user at 30-, 60-, and 90-day horizons.

### 7.2 Forecasting Targets

| Target Series | Granularity | Aggregation |
|---|---|---|
| Total monthly spend | Weekly | Sum of all debits |
| Per-category spend (top-10 L2 categories per user) | Weekly | Sum per category |
| Discretionary vs. non-discretionary | Weekly | Sum per group |

### 7.3 Model Selection

A **model tournament** approach: train multiple models, select per-user based on validation performance.

| Model | Strengths | When Selected |
|---|---|---|
| **Prophet** | Strong seasonality decomposition, robust to missing data | Users with < 6 months history; stable spending patterns |
| **N-BEATS** | High accuracy on univariate series, no feature engineering needed | Users with 6–18 months history; moderate variability |
| **Temporal Fusion Transformer (TFT)** | Handles covariates (income, behavioral features), multi-horizon, built-in attention for interpretability | Users with > 18 months history; complex patterns |

### 7.4 Covariate Inputs (for TFT)

| Covariate | Type | Description |
|---|---|---|
| `income_amount` | Known future | Predicted next income deposit(s) |
| `is_holiday_week` | Known future | Binary flag per week |
| `spending_regime` | Observed past | From behavior model |
| `category_trend_3m` | Observed past | Linear trend coefficient over trailing 3 months |
| `inflation_index` | Known future | CPI-based category-level price index |

### 7.5 Training & Validation

| Aspect | Detail |
|---|---|
| Window | Expanding window: train on all history up to T, validate on T to T+30d |
| Retraining | Weekly incremental update; full retrain monthly |
| Backtest | 6-fold temporal cross-validation (each fold shifts 30 days forward) |
| Probabilistic output | 10th, 50th, 90th percentile forecasts (quantile regression for TFT/N-BEATS; uncertainty intervals for Prophet) |

### 7.6 Evaluation Metrics

| Metric | Target | Scope |
|---|---|---|
| **MAPE** (median prediction) | ≤ 12% | Top-10 categories, 30-day horizon |
| MAPE (60-day) | ≤ 18% | Top-10 categories |
| MAPE (90-day) | ≤ 25% | Top-10 categories |
| **CRPS** (Continuous Ranked Probability Score) | Minimize | Evaluates full predictive distribution quality |
| **Coverage** (90% PI) | 85–95% | Prediction intervals should be neither too wide nor too narrow |
| WAPE (Weighted Absolute Percentage Error) | ≤ 10% | Total spend across all categories |

### 7.7 Regime-Aware Forecasting

The behavior model (§9) feeds a **regime indicator** into the forecaster:

1. When a regime change is detected (e.g., user starts spending significantly more on dining), the model:
   - Increases the learning rate for recent observations (exponential weighting).
   - Widens prediction intervals for 2 forecast cycles to reflect increased uncertainty.
2. Forecasts include a **regime annotation**: `"Forecast adjusted: elevated dining spend detected since Jan 15"`.

---

## 8. Module 3 — Interpretable Budget Recommendations

### 8.1 Objective

Generate **actionable, personalized budget ceilings** per spending category, each accompanied by a natural-language explanation grounded in the user's own data.

### 8.2 Recommendation Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│   STEP 1: Baseline Budget from Forecast                        │
│   budget_baseline[c] = forecast_p50[c] (30-day)                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│   STEP 2: Savings Goal Integration                             │
│   If user has savings target S:                                │
│     gap = current_income - sum(budget_baseline) - S            │
│     If gap < 0: distribute |gap| as cuts across discretionary  │
│     categories, weighted by elasticity scores                  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│   STEP 3: Behavioral Feasibility Check                         │
│   For each category c:                                         │
│     max_reduction[c] = f(habit_strength[c],                    │
│                          historical_variance[c],               │
│                          user_compliance_history[c])            │
│     budget[c] = max(budget_baseline[c] - cut[c],              │
│                     budget_baseline[c] * (1 - max_reduction))  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│   STEP 4: Constraint Optimization                              │
│   Solve:                                                       │
│     minimize  Σ_c w_c · |budget[c] - user_preference[c]|      │
│     subject to:                                                │
│       Σ_c budget[c] ≤ income - savings_target                 │
│       budget[c] ≥ floor[c]  (essential minimums)              │
│       budget[c] ≤ ceiling[c] (behavioral max reduction)       │
│   Solver: scipy.optimize.linprog (or cvxpy for QP variant)    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│   STEP 5: Explanation Generation                               │
│   For each adjusted category:                                  │
│     - SHAP waterfall: which features drove this budget level   │
│     - Anchor rule: "IF dining_frequency > 12/month AND         │
│       avg_meal > $28 THEN reduce dining budget by 15%"         │
│     - Peer comparison: "You spend 23% more on dining than      │
│       users with similar income"                               │
│     - Trend narrative: "Your grocery spending has increased    │
│       8% over the last 3 months"                               │
└──────────────────────────────────────────────────────────────────┘
```

### 8.3 Interpretability Methods

| Method | Purpose | Library |
|---|---|---|
| **SHAP (TreeExplainer)** | Global + local feature importance for the budget optimization model | `shap` |
| **Anchor Rules** | IF-THEN rules that "anchor" a prediction with high precision | `alibi` |
| **Counterfactual Explanations** | "If you reduced coffee-shop visits from 18 to 10/month, you'd save ~$64" | Custom (nearest-neighbor in feature space) |
| **Peer Benchmarking** | Contextual comparison against cohort (income band × region × household size) | Percentile computation on anonymized aggregate data |

### 8.4 Explanation Templates

```python
TEMPLATES = {
    "over_budget_habit": (
        "You've spent ${amount} on {category} this month, which is "
        "${over_amount} over your budget. This appears to be a recurring "
        "pattern — you've exceeded this budget in {n_months} of the last "
        "6 months. Consider {suggestion}."
    ),
    "savings_opportunity": (
        "Reducing {category} by {pct}% (about ${save_amount}/month) "
        "would help you reach your {goal_name} goal {time_saved} sooner. "
        "Your spending on {category} is in the {percentile}th percentile "
        "compared to similar users."
    ),
    "regime_change_alert": (
        "Your {category} spending has {direction} by {change_pct}% since "
        "{change_date}. At this rate, you'll spend an estimated "
        "${projected} this month — ${delta} {over_under} your budget."
    ),
    "positive_reinforcement": (
        "Great job! You've stayed within your {category} budget for "
        "{streak} consecutive months, saving a total of ${total_saved}."
    ),
}
```

### 8.5 User Feedback Loop

| Signal | Usage |
|---|---|
| Budget accepted (no edit) | Positive label: model was well-calibrated |
| Budget manually adjusted | Implicit preference signal → update user preference vector |
| Budget ignored (overspent without acknowledgment) | Indicates budget was unrealistic → increase `floor[c]` for that user/category |
| Explicit "too aggressive" / "too lenient" | Direct preference → adjust elasticity score ± 0.1 |

### 8.6 Evaluation

| Metric | Target | Measurement |
|---|---|---|
| **Acceptance Rate** | ≥ 78% | % of recommended budgets kept without user modification |
| Budget Adherence | ≥ 65% | % of months where user stays within recommended budget |
| Explanation Helpfulness (survey) | ≥ 4.0 / 5.0 | Post-recommendation micro-survey |
| Savings Goal Achievement | ≥ 40% of users with goals hit them within 12 months | Longitudinal tracking |

---

## 9. Behavior Modeling Layer

### 9.1 Purpose

Captures latent user spending behaviors that are not directly observable from individual transactions. Provides behavioral features consumed by all three modules.

### 9.2 Components

#### 9.2.1 Spending Regime Detection

- **Method:** Online Bayesian changepoint detection (BOCPD) on per-category weekly spend.
- **States:** `normal`, `elevated`, `reduced`, `irregular`
- **Output:** Current regime label + posterior probability + estimated change date.
- **Parameters:**
  - Hazard function: constant `λ = 1/60` (expected regime duration ≈ 60 days)
  - Observation model: Gaussian with conjugate Normal-Inverse-Gamma prior

#### 9.2.2 Impulse Score

Estimates the probability that a given transaction is impulsive (unplanned).

| Signal | Weight | Rationale |
|---|---|---|
| Time since last same-merchant visit unusually short | 0.20 | Repeat visits in quick succession suggest impulse |
| Transaction at unusual hour for user | 0.15 | Late-night purchases correlate with impulse spending |
| Amount significantly above user's median for category | 0.20 | Unusual ticket size |
| Occurs within 48h of income deposit | 0.15 | "Payday splurge" effect |
| Category is discretionary | 0.15 | Essentials are rarely impulsive |
| No prior transaction at this merchant | 0.15 | Novel merchant exploration |

- **Model:** Logistic regression on the above features (interpretability is critical here).
- **Training labels:** Derived from user self-reports + heuristic pseudo-labels (validated by labeling team).

#### 9.2.3 Habit Strength Index

Measures how ingrained a spending pattern is.

$$H(c, u) = \alpha \cdot \text{Recurrence}(c,u) + \beta \cdot \text{Consistency}(c,u) + \gamma \cdot \text{Duration}(c,u)$$

Where:
- $\text{Recurrence}$ = frequency relative to expected (e.g., daily coffee = 1.0)
- $\text{Consistency}$ = 1 − coefficient of variation of inter-purchase intervals
- $\text{Duration}$ = months since first observed purchase in category, capped at 12
- $\alpha = 0.4, \beta = 0.35, \gamma = 0.25$ (tunable per cohort)

#### 9.2.4 Income Cycle Alignment

- Detect income deposits via amount clustering + recurrence (semi-supervised: user confirms during onboarding).
- Compute `days_since_payday / pay_period_length` → normalized cycle position $\in [0, 1]$.
- Enables analysis of **pay-cycle spending curves** (front-loaded vs. even vs. end-loaded spenders).

#### 9.2.5 Lifestyle Drift Detector

- Compare rolling 30-day median spend per category against 90-day baseline.
- Alert if drift exceeds ± 2σ of historical variation for ≥ 2 consecutive periods.
- Feeds into forecaster (§7.7) and budget recommender (§8.2 Step 3).

### 9.3 Behavioral Feature Store

All behavioral features are computed asynchronously and materialized in the **feature store** (Feast):
- **Offline store:** Parquet files on S3 (for training).
- **Online store:** Redis (for real-time inference, TTL = 24h, refresh every 6h).

---

## 10. Training & Evaluation Pipeline

### 10.1 Pipeline DAG

```
                    ┌────────────┐
                    │  Raw Data  │
                    │  Ingestion │
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │  Data      │
                    │  Validation│  ← Great Expectations
                    └─────┬──────┘
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
          ┌──────────┐ ┌──────┐ ┌──────────┐
          │ Feature  │ │Label │ │ Behavior │
          │ Engineer │ │ QA   │ │ Model    │
          └────┬─────┘ └──┬───┘ └────┬─────┘
               │          │          │
               └──────────┼──────────┘
                          ▼
               ┌─────────────────────┐
               │   Model Training    │
               │  (Classification,   │
               │   Forecasting,      │
               │   Budget Optimizer) │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │    Evaluation &     │
               │    Comparison       │
               │  (vs. champion)     │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   Registry &        │
               │   Promotion         │  ← MLflow
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   Canary Rollout    │
               │   (5% → 25% → 100%)│
               └─────────────────────┘
```

### 10.2 Data Validation (Great Expectations)

| Suite | Checks |
|---|---|
| Schema | Column presence, types, nullability |
| Volume | Daily transaction count within ± 3σ of 30-day rolling mean |
| Distribution | Amount distribution KL-divergence < 0.1 vs. reference |
| Label | Category distribution chi-squared test p > 0.01 |
| Freshness | Max timestamp within 2h of current time |

### 10.3 Model Comparison Protocol

For each candidate model:
1. Evaluate on **test set** using primary metrics (§6.4, §7.6, §8.6).
2. Compare against current **champion model** using paired bootstrap test (n=10 000, α=0.05).
3. If candidate is statistically significantly better on primary metric **and** not significantly worse on any secondary metric → promote.
4. Run **shadow mode** for 72h in production (log predictions, don't serve) before canary rollout.

### 10.4 Experiment Tracking

All experiments tracked in **MLflow** with:
- Hyperparameters
- Metrics (train, val, test)
- Artifacts (model binary, SHAP summary plots, confusion matrices)
- Data version hash (DVC)
- Git commit SHA

---

## 11. Serving & Inference

### 11.1 Inference Modes

| Mode | Trigger | Latency Target | Model Format |
|---|---|---|---|
| **Real-time classification** | New transaction arrives | ≤ 50 ms (p99) | ONNX (MLP) + LightGBM binary |
| **Batch forecasting** | Nightly job (02:00 UTC) | ≤ 5 s / user | PyTorch (TFT) / Prophet pickle |
| **On-demand budget refresh** | User opens budget tab or monthly trigger | ≤ 2 s | Pre-computed forecast + optimizer |

### 11.2 API Specification

#### 11.2.1 Classify Transaction

```
POST /v1/classify
Request:
{
  "transaction": { <Transaction object per §5.1> }
}

Response:
{
  "category_l1": "FOOD & DINING",
  "category_l2": "Coffee Shops",
  "confidence": 0.94,
  "top_3": [
    {"category": "Coffee Shops", "confidence": 0.94},
    {"category": "Restaurants", "confidence": 0.04},
    {"category": "Groceries", "confidence": 0.01}
  ],
  "is_impulse": false,
  "impulse_score": 0.12
}
```

#### 11.2.2 Get Forecast

```
GET /v1/forecast/{user_id}?horizon=30&categories=all

Response:
{
  "user_id": "...",
  "generated_at": "2026-02-19T14:00:00Z",
  "horizon_days": 30,
  "forecasts": [
    {
      "category": "Groceries",
      "p10": 320.00,
      "p50": 385.00,
      "p90": 460.00,
      "trend": "stable",
      "regime": "normal"
    },
    ...
  ],
  "total_spend": {"p10": 2800, "p50": 3250, "p90": 3720}
}
```

#### 11.2.3 Get Budget Recommendations

```
GET /v1/budget/{user_id}

Response:
{
  "user_id": "...",
  "period": "2026-03",
  "income_estimate": 5500.00,
  "savings_target": 550.00,
  "recommendations": [
    {
      "category": "Restaurants",
      "recommended_budget": 280.00,
      "current_trend": 340.00,
      "confidence": 0.87,
      "explanation": "Reducing dining out by 18% (~$60/month) would help you reach your emergency fund goal 2 months sooner. Your dining spend is in the 72nd percentile for your income bracket.",
      "shap_top_features": [
        {"feature": "dining_frequency", "impact": +45.0},
        {"feature": "avg_meal_cost", "impact": +32.0},
        {"feature": "income_pct_dining", "impact": -12.0}
      ],
      "anchor_rule": "IF dining_visits > 14/month AND avg_ticket > $24 THEN suggest 15% reduction",
      "counterfactual": "If you cooked at home 2 more nights/week, estimated monthly savings: $96"
    },
    ...
  ]
}
```

### 11.3 Caching Strategy

| Data | Cache Layer | TTL | Invalidation |
|---|---|---|---|
| User feature vectors | Redis | 6h | On new transaction |
| Forecasts | Redis | 24h | On nightly batch run |
| Budget recommendations | PostgreSQL + Redis | 7d | On user preference change or forecast refresh |
| SHAP explanations | PostgreSQL | 30d | On model version change |

### 11.4 Scalability

- **Classification service:** Stateless, horizontally scaled behind load balancer. Auto-scale on CPU utilization > 60%.
- **Forecast service:** GPU-backed pods (for TFT inference), scaled on queue depth.
- **Feature computation:** Spark on EMR for batch features; Flink for streaming features (transaction velocity, running balances).

---

## 12. Ethical Considerations & Fairness

### 12.1 Bias Mitigation

| Risk | Mitigation |
|---|---|
| Income-level bias in recommendations (e.g., always suggesting cuts for lower-income users) | Budget recommendations normalize to income %; floors set per category to preserve dignity |
| Gender/demographic bias in peer comparisons | Cohorts defined by income band + region only; no demographic segmentation |
| Merchant-name bias (non-English merchants misclassified) | Multilingual sentence-transformer; explicit evaluation on non-English merchant subsets |
| Model performance disparity across user segments | Fairness audit: stratified evaluation by income quintile, account age, geographic region |

### 12.2 Fairness Metrics

| Metric | Threshold |
|---|---|
| Equal Opportunity difference (classification recall) across income quintiles | ≤ 0.05 |
| Demographic parity of "impulse" labeling across income bands | ≤ 0.08 |
| Recommendation aggressiveness (mean % cut suggested) parity across cohorts | Within ± 10% relative |

### 12.3 Privacy & Security

| Requirement | Implementation |
|---|---|
| PII minimization | Merchant names hashed after embedding extraction; raw descriptions dropped post-feature-extraction |
| Data encryption | AES-256 at rest; TLS 1.3 in transit |
| Access control | Row-level security; ML engineers access only anonymized/aggregated data |
| Right to deletion | Full user data purge pipeline (GDPR/CCPA compliant, ≤ 72h SLA) |
| Model inversion protection | Differential privacy noise (ε = 8) added to published aggregate statistics |
| Explainability audit trail | All explanations logged with model version, feature values, and timestamp |

### 12.4 Responsible AI Guardrails

- **No automated financial decisions:** The system provides *recommendations*, never auto-executes transactions or account changes.
- **Confidence thresholds:** Recommendations with confidence < 0.6 include a disclaimer: *"This suggestion is based on limited data — please review carefully."*
- **Human escalation:** Users can flag any recommendation for expert review. Flagged cases feed into a quality assurance queue.
- **Tone guidelines:** Explanations never use shame-inducing language ("you wasted", "you overspent irresponsibly"). Always neutral or encouraging.

---

## 13. Project Structure

```
ml-fin-advisor/
├── README.md
├── pyproject.toml                   # Project metadata & dependency management
├── Makefile                         # Common dev commands
│
├── configs/
│   ├── model_config.yaml            # Hyperparameters for all modules
│   ├── feature_config.yaml          # Feature definitions & transformations
│   ├── serving_config.yaml          # API & caching configuration
│   └── fairness_config.yaml         # Fairness thresholds & audit settings
│
├── src/
│   ├── data/
│   │   ├── ingestion.py             # Bank feed connectors (Plaid, CSV, API)
│   │   ├── validation.py            # Great Expectations suites
│   │   ├── preprocessing.py         # Cleaning, normalization, deduplication
│   │   └── splits.py                # Temporal train/val/test splitting
│   │
│   ├── features/
│   │   ├── text_features.py         # Merchant embeddings, TF-IDF
│   │   ├── numerical_features.py    # Amount transforms, rolling aggregates
│   │   ├── temporal_features.py     # Cyclical encoding, holiday flags
│   │   ├── behavioral_features.py   # Regime, impulse, habit, cycle features
│   │   └── feature_store.py         # Feast integration (online + offline)
│   │
│   ├── models/
│   │   ├── classifier/
│   │   │   ├── text_tower.py        # Sentence-Transformer + projection
│   │   │   ├── mlp.py               # Multi-modal fusion MLP
│   │   │   ├── meta_learner.py      # LightGBM stacking layer
│   │   │   └── train.py             # Classification training loop
│   │   │
│   │   ├── forecaster/
│   │   │   ├── prophet_model.py     # Prophet wrapper
│   │   │   ├── nbeats_model.py      # N-BEATS wrapper
│   │   │   ├── tft_model.py         # Temporal Fusion Transformer
│   │   │   ├── model_selector.py    # Per-user model tournament
│   │   │   └── train.py             # Forecasting training loop
│   │   │
│   │   ├── recommender/
│   │   │   ├── budget_optimizer.py   # Constraint optimization (scipy/cvxpy)
│   │   │   ├── feasibility.py       # Behavioral feasibility checks
│   │   │   ├── explanations.py      # SHAP, Anchors, counterfactuals
│   │   │   └── templates.py         # Natural language templates
│   │   │
│   │   └── behavior/
│   │       ├── regime_detector.py    # Bayesian changepoint detection
│   │       ├── impulse_scorer.py     # Impulse probability model
│   │       ├── habit_index.py        # Habit strength computation
│   │       └── income_cycle.py       # Pay-cycle detection & alignment
│   │
│   ├── serving/
│   │   ├── app.py                   # FastAPI application
│   │   ├── routes/
│   │   │   ├── classify.py          # POST /v1/classify
│   │   │   ├── forecast.py          # GET /v1/forecast/{user_id}
│   │   │   └── budget.py            # GET /v1/budget/{user_id}
│   │   ├── middleware.py            # Auth, rate limiting, logging
│   │   └── cache.py                 # Redis caching layer
│   │
│   ├── evaluation/
│   │   ├── classification_metrics.py
│   │   ├── forecast_metrics.py
│   │   ├── recommendation_metrics.py
│   │   ├── fairness_audit.py        # Stratified fairness evaluation
│   │   └── model_comparison.py      # Champion/challenger testing
│   │
│   └── utils/
│       ├── logging.py
│       ├── privacy.py               # PII hashing, anonymization
│       └── constants.py             # Category taxonomy, enums
│
├── pipelines/
│   ├── training_pipeline.py         # Airflow/Prefect DAG for training
│   ├── feature_pipeline.py          # Feature computation DAG
│   └── inference_pipeline.py        # Batch inference DAG
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb    # Feature importance & correlation
│   ├── 03_model_experiments.ipynb   # Model prototyping
│   └── 04_fairness_analysis.ipynb   # Fairness audit visualizations
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py
│   │   ├── test_classifier.py
│   │   ├── test_forecaster.py
│   │   ├── test_recommender.py
│   │   └── test_behavior.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_serving.py
│   └── fixtures/
│       └── sample_transactions.json
│
├── infrastructure/
│   ├── terraform/                   # IaC for AWS resources
│   ├── k8s/                         # Kubernetes manifests
│   └── docker/
│       ├── Dockerfile.train         # Training image
│       └── Dockerfile.serve         # Serving image
│
└── docs/
    ├── architecture.md
    ├── data_dictionary.md
    └── runbook.md
```

---

## 14. Milestones & Timeline

| Phase | Duration | Key Deliverables |
|---|---|---|
| **Phase 0: Foundation** | Weeks 1–3 | Data pipeline, schema validation, feature store setup, dev environment |
| **Phase 1: Classification** | Weeks 4–8 | Text tower, MLP, meta-learner; macro-F1 ≥ 0.90 on internal test set |
| **Phase 2: Behavior Model** | Weeks 6–10 | Regime detection, impulse scorer, habit index (parallel with Phase 1) |
| **Phase 3: Forecasting** | Weeks 9–14 | Prophet baseline → N-BEATS → TFT; model tournament; MAPE ≤ 12% |
| **Phase 4: Recommendations** | Weeks 13–17 | Budget optimizer, explanation engine, template system |
| **Phase 5: Integration & Serving** | Weeks 16–19 | API, caching, end-to-end latency targets met |
| **Phase 6: Fairness & Hardening** | Weeks 18–21 | Fairness audit, privacy review, guardrails, load testing |
| **Phase 7: Beta & Iteration** | Weeks 22–26 | Canary rollout (5% → 25%), A/B test budget acceptance rate, iterate |
| **Phase 8: GA** | Week 27 | Full rollout, monitoring dashboards, runbook finalized |

---

## 15. Appendices

### Appendix A: Hyperparameter Defaults

#### A.1 Transaction Classifier (MLP)

| Parameter | Value |
|---|---|
| Hidden layers | [512, 256] |
| Activation | GELU |
| Dropout | 0.3 |
| Batch size | 2048 |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Focal loss γ | 2.0 |
| Label smoothing | 0.05 |

#### A.2 Transaction Classifier (LightGBM Meta-Learner)

| Parameter | Value |
|---|---|
| num_leaves | 127 |
| max_depth | 8 |
| learning_rate | 0.05 |
| n_estimators | 500 |
| min_child_samples | 50 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |

#### A.3 Temporal Fusion Transformer

| Parameter | Value |
|---|---|
| Hidden size | 64 |
| Attention heads | 4 |
| LSTM layers | 2 |
| Dropout | 0.1 |
| Learning rate | 1e-3 |
| Quantiles | [0.1, 0.5, 0.9] |
| Max encoder length | 52 weeks |
| Max prediction length | 13 weeks |

#### A.4 N-BEATS

| Parameter | Value |
|---|---|
| Stack types | [trend, seasonality, generic] |
| Blocks per stack | 3 |
| Hidden size | 256 |
| Theta dims | [4, 8, 4] |
| Lookback multiple | 5× horizon |

### Appendix B: Category Mapping — MCC Code Reference

| MCC Range | Mapped L1 Category |
|---|---|
| 5411–5499 | FOOD & DINING (Groceries) |
| 5812–5814 | FOOD & DINING (Restaurants) |
| 5541–5542 | TRANSPORTATION (Fuel) |
| 4111–4131 | TRANSPORTATION (Public Transit) |
| 5311–5399 | SHOPPING & ENTERTAINMENT |
| 6010–6012 | FINANCIAL (ATM / Cash) |
| 8011–8099 | HEALTH & PERSONAL |
| ... | (Full mapping in `src/utils/constants.py`) |

### Appendix C: Glossary

| Term | Definition |
|---|---|
| **BOCPD** | Bayesian Online Changepoint Detection — a sequential algorithm for detecting abrupt changes in time-series generating processes |
| **CRPS** | Continuous Ranked Probability Score — a proper scoring rule for evaluating probabilistic forecasts |
| **ECE** | Expected Calibration Error — measures how well predicted probabilities match actual frequencies |
| **Focal Loss** | A modified cross-entropy loss that down-weights easy examples, focusing training on hard/rare classes |
| **MAPE** | Mean Absolute Percentage Error — average of absolute percentage errors across predictions |
| **MCC** | Merchant Category Code — a 4-digit ISO 18245 code assigned by card networks to classify merchant type |
| **SHAP** | SHapley Additive exPlanations — a game-theoretic approach to explain individual model predictions |
| **TFT** | Temporal Fusion Transformer — an attention-based architecture for multi-horizon time-series forecasting with interpretable components |
| **WAPE** | Weighted Absolute Percentage Error — sum of absolute errors divided by sum of actuals; robust to near-zero values |

---

*End of Specification*
