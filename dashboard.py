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
        
        # TODO: Step 2 - Add metrics here
        
    elif st.session_state.page == "Classify":
        st.markdown("# Transaction Classifier")
        st.markdown("")
        
        # TODO: Step 3 - Add classifier here
        
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
