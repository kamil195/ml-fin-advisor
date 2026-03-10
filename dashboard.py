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
    .stCard, .stContainer, div[data-testid="stVerticalBlock"] {
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
    
    /* Remove default Streamlit borders */
    div[data-testid="stThumbnail"] {
        border: none !important;
    }
    
    /* Clean sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: none !important;
    }
    
    /* Spacing only, no lines */
    hr {
        display: none !important;
    }
    
    /* Metric cards styling */
    .metric-container {
        padding: 1rem 0;
    }
    .metric-label {
        color: #64748B;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #1E293B;
        font-size: 2rem;
        font-weight: 500;
    }
    .metric-change {
        color: #10B981;
        font-size: 0.875rem;
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
    # Logo and brand
    st.markdown("## \U0001F4B0 FinWise AI")
    st.markdown("---")
    
    # Navigation
    if st.button("\U0001F4CA Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
    if st.button("\U0001F4DD Classify", use_container_width=True):
        st.session_state.page = "Classify"
    if st.button("\U0001F4C8 Forecast", use_container_width=True):
        st.session_state.page = "Forecast"
    if st.button("\U0001F4B0 Budget", use_container_width=True):
        st.session_state.page = "Budget"
    if st.button("\u2699\uFE0F Settings", use_container_width=True):
        st.session_state.page = "Settings"
    
    st.markdown("---")
    
    # User profile
    if st.session_state.authenticated:
        st.markdown(f"**\U0001F464 {st.session_state.email}**")
        if st.button("\U0001F6AA Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.api_key = None
            st.rerun()
    else:
        st.markdown("### \U0001F510 Login")
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

# Main content area
if not st.session_state.authenticated:
    # Landing page
    st.markdown("# \U0001F4B0 FinWise AI")
    st.markdown("### Intelligent Financial Guidance Powered by ML")
    st.markdown("")
    st.markdown("**99.96% Classification Accuracy** \u00B7 **8.19% Forecast MAPE**")
    st.markdown("")
    st.markdown("Login from the sidebar to get started.")

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
        st.markdown("")
        
        # Sample transaction buttons
        st.markdown("##### Test with sample transactions")
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
        st.markdown("##### Enter transaction details")
        
        # Transaction form
        col1, col2 = st.columns(2)
        
        with col1:
            merchant = st.text_input("Merchant Name", 
                                    value=st.session_state.get("test_merchant", ""),
                                    placeholder="e.g., STARBUCKS")
            amount = st.number_input("Amount ($)", 
                                    value=st.session_state.get("test_amount", 0.0),
                                    format="%.2f")
        
        with col2:
            mcc = st.number_input("MCC Code", 
                                 value=st.session_state.get("test_mcc", 0),
                                 help="Merchant Category Code")
            channel = st.selectbox("Channel", ["POS", "ONLINE", "ATM", "TRANSFER"])
        
        # Classify button
        if st.button("🔍 Classify Transaction", type="primary", use_container_width=True):
            if merchant and amount != 0:
                with st.spinner("AI is classifying..."):
                    # Call API
                    headers = {"X-API-Key": st.session_state.api_key}
                    payload = {
                        "amount": amount,
                        "merchant_name": merchant,
                        "merchant_mcc": mcc,
                        "timestamp": datetime.now().isoformat(),
                        "channel": channel
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
                            st.markdown("##### Classification Result")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Category**")
                                st.markdown(f"### {data['category_l2']}")
                            with col2:
                                st.markdown(f"**Confidence**")
                                st.markdown(f"### {data['confidence']*100:.2f}%")
                            with col3:
                                st.markdown(f"**Impulse Score**")
                                st.markdown(f"### {data.get('impulse_score', 0)*100:.1f}%")
                            
                            # SHAP explanations
                            if 'shap_top_features' in data:
                                with st.expander("🧠 Why this category?"):
                                    for feat in data['shap_top_features']:
                                        st.markdown(f"• **{feat['feature']}**: {feat['impact']}")
                        else:
                            st.error(f"Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
            else:
                st.warning("Please enter merchant name and amount")
        
    elif st.session_state.page == "Forecast":
        st.markdown("# Spending Forecast")
        st.markdown("")
        
        # TODO: Step 4 - Add forecast chart here
        
    elif st.session_state.page == "Budget":
        st.markdown("# Budget Recommendations")
        st.markdown("")
        
        # TODO: Step 5 - Add SHAP explanations here
        
    elif st.session_state.page == "Settings":
        st.markdown("# Settings")
        st.markdown("")
        st.info("Settings page coming soon...")
