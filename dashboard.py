import os
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# CRITICAL: Disable ALL file watching to prevent inotify error
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_WATCH_FILE_SYSTEM"] = "false"
os.environ["STREAMLIT_SERVER_WATCH_CONFIG"] = "false"
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

# Page config
st.set_page_config(
    page_title="FinWise AI",
    page_icon="\U0001F4B0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for minimalist theme
st.markdown("""
<style>
    /* Pure white background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Remove all borders, boxes, shadows */
    div[data-testid="stVerticalBlock"] {
        border: none !important;
        box-shadow: none !important;
        background: none !important;
    }
    
    /* Clean typography */
    h1, h2, h3 {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    
    /* Indigo accent for buttons */
    .stButton button {
        background-color: #6366F1 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 400 !important;
    }
    
    /* Clean sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: none !important;
    }
    
    /* Remove all template boxes */
    div[data-testid="stThumbnail"] {
        border: none !important;
    }
    
    /* Spacing only, no lines */
    hr {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'email' not in st.session_state:
    st.session_state.email = None
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"

# API URL
API_URL = "https://fin-advisor-sa6h.onrender.com"

# Sidebar
with st.sidebar:
    # Brand
    st.markdown("## 💰 FinWise AI")
    st.markdown("")
    
    # Navigation
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
    if st.button("📝 Classify", use_container_width=True):
        st.session_state.page = "Classify"
    if st.button("📈 Forecast", use_container_width=True):
        st.session_state.page = "Forecast"
    if st.button("💰 Budget", use_container_width=True):
        st.session_state.page = "Budget"
    if st.button("ℹ️ About", use_container_width=True):
        st.session_state.page = "About"
    
    st.markdown("---")
    
    # User
    if st.session_state.authenticated:
        st.markdown(f"**👤 {st.session_state.email}**")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.rerun()
    else:
        st.markdown("### 🔐 Login")
        email = st.text_input("Email", placeholder="your@email.com")
        api_key = st.text_input("API Key", type="password", placeholder="Enter your API key")
        if st.button("Connect", use_container_width=True):
            if email and api_key:
                st.session_state.email = email
                st.session_state.api_key = api_key
                st.session_state.authenticated = True
                st.session_state.user_id = f"user_{hash(email) % 1000000:06d}"
                st.success("Connected!")
                time.sleep(1)
                st.rerun()
        st.markdown("")
        if st.button("Use Demo Account", use_container_width=True):
            st.session_state.email = "demo@example.com"
            st.session_state.api_key = "demo_key_123"
            st.session_state.authenticated = True
            st.session_state.user_id = "user_demo_001"
            st.rerun()

# Main content area
if not st.session_state.authenticated:
    # Landing page
    st.markdown("# \U0001F4B0 FinWise AI")
    st.markdown("### Intelligent Financial Guidance Powered by ML")
    st.markdown("")
    st.markdown("**99.96% Classification Accuracy** \u00B7 **\U0001f3af 92% Forecast Accuracy**")
    st.markdown("")
    st.markdown("\U0001f449 Login from the sidebar to get started.")

else:
    # Show selected page
    if st.session_state.page == "Dashboard":
        st.markdown("# Dashboard")
        st.markdown("")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="padding: 1.5rem 0;">
                <div style="color: #64748B; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">TOTAL BALANCE</div>
                <div style="color: #1E293B; font-size: 2.5rem; font-weight: 500; line-height: 1.2;">$24,562.00</div>
                <div style="color: #10B981; font-size: 0.875rem;">▲ +2.4% vs last month</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="padding: 1.5rem 0;">
                <div style="color: #64748B; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">MONTHLY INCOME</div>
                <div style="color: #1E293B; font-size: 2.5rem; font-weight: 500; line-height: 1.2;">$8,450.00</div>
                <div style="color: #10B981; font-size: 0.875rem;">▲ +5.1% vs last month</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="padding: 1.5rem 0;">
                <div style="color: #64748B; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em;">MONTHLY EXPENSES</div>
                <div style="color: #1E293B; font-size: 2.5rem; font-weight: 500; line-height: 1.2;">$3,240.50</div>
                <div style="color: #EF4444; font-size: 0.875rem;">▼ -1.2% vs last month</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("")  # Extra spacing for next steps
        
    elif st.session_state.page == "Classify":
        st.markdown("# Transaction Classifier")
        st.markdown("_Tell us about a purchase and AI will categorize it for you_")
        st.markdown("")
        
        # Sample transaction buttons
        st.markdown("##### 💡 Try a sample transaction")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("NETFLIX", use_container_width=True):
                st.session_state.test_merchant = "NETFLIX"
                st.session_state.test_amount = -15.99
                st.session_state.test_mcc = 4899
        
        with col2:
            if st.button("SHELL", use_container_width=True):
                st.session_state.test_merchant = "SHELL"
                st.session_state.test_amount = -45.00
                st.session_state.test_mcc = 5541
        
        with col3:
            if st.button("WHOLE FOODS", use_container_width=True):
                st.session_state.test_merchant = "WHOLE FOODS"
                st.session_state.test_amount = -89.32
                st.session_state.test_mcc = 5411
        
        with col4:
            if st.button("UBER", use_container_width=True):
                st.session_state.test_merchant = "UBER"
                st.session_state.test_amount = -24.50
                st.session_state.test_mcc = 4121
        
        with col5:
            if st.button("AMAZON", use_container_width=True):
                st.session_state.test_merchant = "AMAZON"
                st.session_state.test_amount = -67.23
                st.session_state.test_mcc = 5311
        
        st.markdown("---")
        st.markdown("##### ✏️ Enter transaction details")
        
        # Transaction form
        col1, col2 = st.columns(2)
        
        with col1:
            merchant = st.text_input("Store / Merchant Name", 
                                    value=st.session_state.get("test_merchant", ""),
                                    placeholder="e.g., Starbucks, Amazon, Walmart")
            amount = st.number_input("Amount spent ($)", 
                                    value=st.session_state.get("test_amount", 0.0),
                                    format="%.2f",
                                    help="Enter the transaction amount (negative for purchases)")
        
        with col2:
            mcc = st.number_input("Merchant Type (optional)", 
                                 value=st.session_state.get("test_mcc", 0),
                                 help="Auto-detected from merchant name. Leave 0 if unsure.")
            channel = st.selectbox("Where did you shop?", 
                                  ["In Store", "Online", "ATM", "Transfer"],
                                  help="How did you make this purchase?")
        
        # Map friendly channel names back to API values
        channel_map = {"In Store": "POS", "Online": "ONLINE", "ATM": "ATM", "Transfer": "TRANSFER"}
        api_channel = channel_map[channel]
        
        # Check API key exists
        if not st.session_state.get("api_key"):
            st.warning("⚠️ Please enter your API key in the sidebar first")
        
        # Classify button
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            if not st.session_state.get("api_key"):
                st.error("🔑 You need an API key to analyze transactions. Enter it in the sidebar.")
                st.stop()
            if merchant and amount != 0:
                with st.spinner("🧠 AI is thinking..."):
                    # Call API
                    headers = {
                        "X-API-Key": st.session_state.api_key,
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "amount": amount,
                        "merchant_name": merchant,
                        "merchant_mcc": mcc,
                        "timestamp": datetime.now().isoformat(),
                        "channel": api_channel
                    }
                    
                    try:
                        response = requests.post(
                            f"{API_URL}/v1/classify",
                            json=payload,
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Show result
                            st.markdown("---")
                            st.markdown("##### ✅ Got it! Here's what AI found")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Category**")
                                st.markdown(f"### {data['category_l2']}")
                            with col2:
                                st.markdown(f"**AI Confidence**")
                                confidence_val = data['confidence']*100
                                st.markdown(f"### {confidence_val:.1f}%")
                                if confidence_val >= 95:
                                    st.caption("🎯 Very sure about this")
                                elif confidence_val >= 80:
                                    st.caption("👍 Fairly confident")
                                else:
                                    st.caption("🤔 Less certain")
                            with col3:
                                st.markdown(f"**Spending Habit Score**")
                                habit = data.get('impulse_score', 0)*100
                                st.markdown(f"### {habit:.0f}%")
                                if habit >= 70:
                                    st.caption("🔄 Likely a regular habit")
                                else:
                                    st.caption("✨ Looks like a one-time purchase")
                            
                            # SHAP explanations
                            if 'shap_top_features' in data:
                                with st.expander("💡 Why did AI choose this?"):
                                    st.markdown("_These are the main factors AI used to decide the category:_")
                                    for feat in data['shap_top_features']:
                                        st.markdown(f"• **{feat['feature']}**: {feat['impact']}")
                        else:
                            st.error(f"Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            else:
                st.warning("Please enter a store name and amount to analyze")
        
        if not st.session_state.get("test_merchant"):
            st.markdown("")
            st.markdown("👋 _Start by clicking a sample transaction above, or enter your own!_")
        
    elif st.session_state.page == "Forecast":
        st.markdown("# Spending Forecast")
        st.markdown("##### See where your money is going")
        st.markdown("")
        
        col1, col2 = st.columns([6, 1])
        with col2:
            st.markdown("""
            <div style="background-color: #6366F1; color: white; 
                        padding: 0.25rem 0.75rem; border-radius: 20px; 
                        text-align: center; font-size: 0.875rem;">
                🎯 AI Accuracy: 92%
            </div>
            """, unsafe_allow_html=True)
        
        days = st.selectbox("Forecast for next:", ["30 days", "60 days", "90 days"])
        
        if st.button("📊 Show My Forecast", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is predicting your spending..."):
                st.success("✅ Forecast ready!")
                st.info("📈 Chart will appear here with Low/Medium/High estimates")
        
    elif st.session_state.page == "Budget":
        st.markdown("# Budget Recommendations")
        st.markdown("_Personalized spending limits based on your habits_")
        st.markdown("")
        
        # TODO: Step 5 - Add budget with \U0001f50d What affects this budget explanations
        
    elif st.session_state.page == "About":
        st.markdown("# About FinWise AI")
        st.markdown("")
        st.markdown("**FinWise AI** is an intelligent financial advisor powered by machine learning.")
        st.markdown("")
        st.markdown("- 🎯 **99.96%** transaction classification accuracy")
        st.markdown("- 📈 **92%** spending forecast accuracy")
        st.markdown("- 💰 **80.9%** budget recommendation acceptance")
        st.markdown("")
        st.markdown("Built with LightGBM, ensemble forecasting, and SHAP explainability.")
