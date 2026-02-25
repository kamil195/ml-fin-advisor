"""Debug the forecaster logic."""
import pandas as pd, numpy as np

df = pd.read_csv("data/raw/transactions.csv", parse_dates=["timestamp"])

# Apply same temporal split as pipeline
max_dt = df["timestamp"].max()
test_cut = max_dt - pd.DateOffset(months=2)
val_cut = test_cut - pd.DateOffset(months=2)

train = df[df["timestamp"] < val_cut]
val = df[(df["timestamp"] >= val_cut) & (df["timestamp"] < test_cut)]

# Filter debits
debits = train[train["amount"] < 0].copy()
debits["abs_amount"] = debits["amount"].abs()
debits["yr_month"] = debits["timestamp"].dt.to_period("M")

val_debits = val[val["amount"] < 0].copy()  
val_debits["abs_amount"] = val_debits["amount"].abs()
val_debits["yr_month"] = val_debits["timestamp"].dt.to_period("M")

n_val_months = val_debits["yr_month"].nunique()
print(f"Train months: {debits['yr_month'].nunique()}")
print(f"Val months:   {n_val_months}")
print(f"Train period: {debits['timestamp'].min()} to {debits['timestamp'].max()}")
print(f"Val period:   {val_debits['timestamp'].min()} to {val_debits['timestamp'].max()}")
print()

cats = debits["category_l2"].value_counts().head(5).index
for cat in cats:
    ct = debits[debits["category_l2"] == cat]
    cv = val_debits[val_debits["category_l2"] == cat]
    
    monthly = ct.groupby("yr_month")["abs_amount"].sum()
    mean_f = monthly.mean()
    
    user_monthly = ct.groupby(["user_id", "yr_month"])["abs_amount"].sum().groupby("user_id").mean()
    user_f = user_monthly.sum()
    
    forecast_monthly = 0.50 * mean_f + 0.25 * mean_f + 0.25 * user_f
    forecast_total = forecast_monthly * n_val_months
    actual_total = cv["abs_amount"].sum()
    
    ape = abs(forecast_total - actual_total) / actual_total * 100
    print(f"{cat:30s}")
    print(f"  Monthly totals: {monthly.values}")
    print(f"  Mean monthly: {mean_f:.0f}, User sum: {user_f:.0f}")
    print(f"  Forecast total ({n_val_months} mo): {forecast_total:.0f}")
    print(f"  Actual total: {actual_total:.0f}")
    print(f"  APE: {ape:.1f}%")
    print()
