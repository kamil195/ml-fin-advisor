"""
Consumer-facing Streamlit app  –  "5 free analyses then pay" flow.

Pages:
  1. Sign Up / Sign In  (by e-mail)
  2. Dashboard          (usage meter · past results · upgrade CTA)
  3. Analyse            (classify transaction with paywall)
  4. History            (read-only past analyses)
"""

import streamlit as st
import requests
from datetime import datetime

# ── Page Config ─────────────────────────────────────────────────

st.set_page_config(page_title="Fin Advisor", page_icon="💰", layout="wide")

API_URL = st.secrets.get("API_URL", "https://fin-advisor-sa6h.onrender.com")

# ── Helpers ─────────────────────────────────────────────────────


def _api(method: str, path: str, **kwargs):
    """Call the consumer API and return (ok, data_or_error)."""
    url = f"{API_URL}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=60, **kwargs)
        data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        return resp.status_code, data
    except requests.exceptions.Timeout:
        return 0, "Request timed out — the server may be cold-starting. Wait 30 s and retry."
    except Exception as e:
        return 0, str(e)


def _register(email: str, name: str = ""):
    return _api("post", "/consumer/register", json={"email": email, "display_name": name})


def _status(uid: str):
    return _api("get", f"/consumer/status/{uid}")


def _analyse(uid: str, txn: dict):
    return _api("post", "/consumer/analyse", json={"user_id": uid, "transaction": txn})


def _upgrade(uid: str):
    return _api("post", "/consumer/upgrade", json={"user_id": uid})


def _history(uid: str, limit: int = 20):
    return _api("get", f"/consumer/history/{uid}?limit={limit}")


# ── Session state defaults ──────────────────────────────────────

for key, default in {
    "user": None,          # dict with user_id, email, etc.
    "signed_in": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar: sign-up / sign-in ──────────────────────────────────

with st.sidebar:
    st.header("🔑 Account")

    if st.session_state.signed_in and st.session_state.user:
        u = st.session_state.user
        st.success(f"Signed in as **{u.get('display_name', u['email'])}**")
        st.caption(f"ID: `{u['user_id']}`")

        # Refresh status
        code, data = _status(u["user_id"])
        if code == 200:
            st.session_state.user.update(data)
            u = st.session_state.user

            # Usage meter
            used = u["free_tier_used"]
            limit = u["free_tier_limit"]
            sub = u["subscription_status"]

            if sub == "paid":
                st.info("✅ **Pro plan** — unlimited analyses")
            else:
                pct = min(used / limit, 1.0)
                st.progress(pct, text=f"{used}/{limit} free analyses used")
                if used >= limit:
                    st.warning("Free tier exhausted — upgrade to continue.")

        if st.button("Sign out"):
            st.session_state.user = None
            st.session_state.signed_in = False
            st.rerun()
    else:
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            display_name = st.text_input("Display Name (optional)", placeholder="Alex")
            submitted = st.form_submit_button("Sign Up / Sign In", type="primary")

        if submitted and email:
            with st.spinner("Connecting…"):
                code, data = _register(email, display_name)
            if code == 200:
                st.session_state.user = data
                st.session_state.signed_in = True
                st.success(f"Welcome, **{data.get('display_name', email)}**!")
                st.rerun()
            else:
                st.error(f"Registration failed: {data}")

    st.divider()
    st.caption(f"API: `{API_URL}`")


# ── Guard: must be signed in ────────────────────────────────────

if not st.session_state.signed_in:
    st.title("💰 Personal Finance Advisor")
    st.markdown(
        """
        **Try it free** — get **5 AI-powered transaction analyses** at no cost.

        ### How it works
        1. **Sign up** with your email in the sidebar
        2. **Classify transactions** instantly with ML
        3. **Upgrade** to Pro for unlimited access

        ### What you get
        | Feature | Free | Pro ($49.99/mo) |
        |---|:---:|:---:|
        | Transaction classification | 5 total | **Unlimited** |
        | SHAP explanations | ✅ | ✅ |
        | Anchor rules | ✅ | ✅ |
        | Analysis history | ✅ | ✅ |
        | Priority support | — | ✅ |
        """
    )
    st.stop()

# ── Signed-in content ──────────────────────────────────────────

u = st.session_state.user
st.title(f"💰 Welcome, {u.get('display_name', u['email'])}")

tab_analyse, tab_history, tab_account = st.tabs([
    "🔍 Analyse Transaction",
    "📜 History",
    "⚙️ Account & Upgrade",
])

# ── Tab 1: Analyse ──────────────────────────────────────────────

with tab_analyse:
    st.header("Classify a Transaction")

    # Show remaining
    sub_status = u.get("subscription_status", "free")
    remaining = u.get("remaining_free", 0)
    used = u.get("free_tier_used", 0)
    limit = u.get("free_tier_limit", 5)

    if sub_status == "paid":
        st.info("✅ Pro plan active — unlimited analyses")
    else:
        col_meter, col_msg = st.columns([1, 2])
        with col_meter:
            st.metric("Remaining Free", f"{remaining}/{limit}")
        with col_msg:
            if remaining == 0:
                st.error("🔒 Free tier exhausted. Upgrade to continue analysing.")
            elif remaining <= 2:
                st.warning(f"⚠️ Only {remaining} free analyses left!")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        merchant = st.text_input("Merchant Name", "WAL-MART")
        amount = st.number_input("Amount ($)", value=-45.67, format="%.2f")
        mcc = st.number_input("MCC Code", value=5411, step=1)
        channel = st.selectbox("Channel", ["POS", "ONLINE", "ATM", "RECURRING"])
        account_type = st.selectbox("Account Type", ["CHECKING", "SAVINGS", "CREDIT"])

    can_go = sub_status == "paid" or remaining > 0

    if col1.button("🔍 Classify", type="primary", disabled=not can_go):
        txn = {
            "user_id": u["user_id"],
            "account_type": account_type,
            "merchant_name": merchant,
            "amount": amount,
            "merchant_mcc": mcc,
            "channel": channel,
            "timestamp": datetime.now().isoformat(),
        }
        with st.spinner("Classifying…"):
            code, data = _analyse(u["user_id"], txn)

        if code == 200:
            with col2:
                st.success(f"✅ **{data['category_l2']}**")
                st.metric("Confidence", f"{data['confidence']*100:.2f}%")

                if "shap_features" in data and data["shap_features"]:
                    st.subheader("Top Features")
                    for feat in data["shap_features"][:5]:
                        st.text(f"• {feat['feature']}: {feat['shap_value']:.3f}")

                if "anchor_rule" in data:
                    st.info(f"📌 {data['anchor_rule']}")

                # Update local usage
                consumer_meta = data.get("_consumer", {})
                if consumer_meta:
                    st.session_state.user["free_tier_used"] = consumer_meta["free_tier_used"]
                    st.session_state.user["remaining_free"] = consumer_meta["remaining_free"]
                    st.session_state.user["subscription_status"] = consumer_meta["subscription_status"]

        elif code == 402:
            detail = data if isinstance(data, dict) else {"message": str(data)}
            err = detail.get("detail", detail)
            if isinstance(err, dict):
                st.error(f"🔒 {err.get('message', 'Free tier exhausted')}")
                upgrade_url = err.get("upgrade_url", "")
                if upgrade_url:
                    st.link_button("⬆️ Upgrade to Pro", upgrade_url)
            else:
                st.error(str(err))
        else:
            st.error(f"Error ({code}): {data}")

    if not can_go:
        st.divider()
        st.markdown("### 🔓 Unlock Unlimited Analyses")
        st.markdown("Upgrade to **Pro** for **$49.99/month** and get unlimited access.")
        code2, data2 = _upgrade(u["user_id"])
        if code2 == 200:
            st.link_button("⬆️ Upgrade Now", data2["checkout_url"], type="primary")

# ── Tab 2: History ──────────────────────────────────────────────

with tab_history:
    st.header("📜 Analysis History")

    code, data = _history(u["user_id"])
    if code == 200 and data.get("items"):
        st.caption(f"Showing {len(data['items'])} of {data['total']} analyses")
        for item in data["items"]:
            req = item.get("request_summary", {})
            res = item.get("response_data", {})
            ts = datetime.fromtimestamp(item["created_at"]).strftime("%Y-%m-%d %H:%M")
            with st.expander(
                f"**{res.get('category_l2', 'Unknown')}** — "
                f"{req.get('merchant', '?')} · ${abs(req.get('amount', 0)):.2f} · {ts}"
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Confidence", f"{res.get('confidence', 0)*100:.2f}%")
                    if "anchor_rule" in res:
                        st.info(f"📌 {res['anchor_rule']}")
                with col_b:
                    if "shap_features" in res:
                        for feat in res["shap_features"][:3]:
                            st.text(f"• {feat['feature']}: {feat['shap_value']:.3f}")
    elif code == 200:
        st.info("No analyses yet — classify a transaction to get started!")
    else:
        st.error(f"Could not load history: {data}")

# ── Tab 3: Account & Upgrade ───────────────────────────────────

with tab_account:
    st.header("⚙️ Account")

    col_info, col_plan = st.columns(2)

    with col_info:
        st.subheader("Profile")
        st.text(f"Email:    {u['email']}")
        st.text(f"User ID:  {u['user_id']}")
        st.text(f"Name:     {u.get('display_name', '—')}")

    with col_plan:
        st.subheader("Plan & Usage")
        sub = u.get("subscription_status", "free")
        used = u.get("free_tier_used", 0)
        limit = u.get("free_tier_limit", 5)

        if sub == "paid":
            st.success("✅ **Pro Plan** — unlimited analyses")
            st.balloons()
        else:
            st.metric("Plan", "Free")
            st.progress(min(used / limit, 1.0), text=f"{used}/{limit} free analyses used")

            if used >= limit:
                st.warning("Your free analyses are used up.")

            st.divider()
            st.markdown("### Upgrade to Pro")
            st.markdown(
                """
                - **Unlimited** transaction classifications
                - Full SHAP + Anchor explanations
                - Priority support
                - **$49.99/month** — cancel anytime
                """
            )
            code3, data3 = _upgrade(u["user_id"])
            if code3 == 200:
                st.link_button("⬆️ Upgrade to Pro", data3["checkout_url"], type="primary")
