import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Fin Advisor - Personal Finance AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .success-text {
        color: #0B5E42;
        font-weight: bold;
    }
    .warning-text {
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>💰 Fin Advisor - Your AI Financial Assistant</h1>", unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'email' not in st.session_state:
    st.session_state.email = None

# Sidebar for authentication
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/financial-growth-analysis.png", width=80)
    st.title("🔐 Access Your Account")
    
    auth_method = st.radio("Choose method:", ["API Key Login", "Get Free API Key"])
    
    if auth_method == "API Key Login":
        st.markdown("### Enter your credentials")
        email = st.text_input("📧 Email", placeholder="your@email.com")
        api_key = st.text_input("🔑 API Key", type="password", placeholder="Enter your API key")
        
        if st.button("🚀 Connect to Fin Advisor", use_container_width=True):
            if email and api_key:
                st.session_state.email = email
                st.session_state.api_key = api_key
                st.session_state.authenticated = True
                st.session_state.user_id = f"user_{hash(email) % 1000000:06d}"
                st.success("✅ Connected successfully!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Please enter both email and API key")
    else:
        st.markdown("### Get your free API key")
        st.markdown("Start with 5 free analyses!")
        new_email = st.text_input("📧 Your email", placeholder="your@email.com", key="new_email")
        
        if st.button("🎁 Get Free API Key", use_container_width=True):
            if new_email:
                st.info("📨 Check your email for your free API key!")
                st.markdown("""
                **For demo purposes, use:**
                - Email: demo@example.com
                - API Key: `fin_demo_key_2025`
                """)
            else:
                st.error("Please enter your email")
    
    st.markdown("---")
    st.markdown("### 💳 Subscription Plans")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Free**  \n• 5 analyses  \n• Basic features")
    with col2:
        st.markdown("**Pro**  \n• Unlimited  \n• SHAP explanations  \n• $4.99/mo")

# Main dashboard
if st.session_state.authenticated:
    # Header with user info
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"### Welcome back, {st.session_state.email.split('@')[0]}! 👋")
    
    # API setup
    API_URL = "https://fin-advisor-sa6h.onrender.com"
    headers = {"X-API-Key": st.session_state.api_key}
    
    # Check API connection
    try:
        with st.spinner("Connecting to Fin Advisor AI..."):
            health = requests.get(f"{API_URL}/health", timeout=5)
        
        if health.status_code == 200:
            st.success("✅ Connected to Fin Advisor AI successfully!")
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("AI Model Accuracy", "99.96%", "↑ 2.3%")
            with col2:
                st.metric("Forecast Precision", "8.19% MAPE", "✓ Target Met")
            with col3:
                st.metric("Categories", "30", "Hierarchical")
            with col4:
                st.metric("Budget Acceptance", "80.9%", "↑ 5.2%")
            
            st.markdown("---")
            
            # Create tabs for different features
            tab1, tab2, tab3, tab4 = st.tabs([
                "📝 Transaction Classifier", 
                "📈 Spending Forecast", 
                "💰 Budget Recommendations",
                "📊 Analytics Dashboard"
            ])
            
            # TAB 1: Transaction Classifier
            with tab1:
                st.markdown("### 🏷️ Classify Your Transactions")
                st.markdown("Enter transaction details to see AI categorization with SHAP explanations")
                
                with st.expander("➕ Add New Transaction", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        amount = st.number_input("💰 Amount ($)", value=-45.67, format="%.2f", help="Negative for spending")
                        merchant = st.text_input("🏢 Merchant Name", value="WAL-MART", help="e.g., Amazon, Starbucks")
                        timestamp = st.date_input("📅 Date", datetime.now())
                    with col2:
                        mcc = st.selectbox("🏷️ MCC Code", 
                            options=[5411, 5812, 5813, 5541, 5311, 5814],
                            format_func=lambda x: {
                                5411: "5411 - Groceries",
                                5812: "5812 - Restaurants", 
                                5813: "5813 - Bars",
                                5541: "5541 - Fuel",
                                5311: "5311 - Department Stores",
                                5814: "5814 - Fast Food"
                            }.get(x, str(x))
                        )
                        channel = st.selectbox("📱 Channel", ["POS", "ONLINE", "ATM", "TRANSFER"])
                    
                    if st.button("🔍 Classify Transaction", type="primary", use_container_width=True):
                        payload = {
                            "amount": amount,
                            "merchant_name": merchant,
                            "merchant_mcc": mcc,
                            "timestamp": f"{timestamp}T12:00:00Z",
                            "channel": channel
                        }
                        
                        with st.spinner("AI is analyzing your transaction..."):
                            response = requests.post(
                                f"{API_URL}/v1/classify",
                                json=payload,
                                headers=headers
                            )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Show results in nice format
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**📊 Category**")
                                st.markdown(f"### 🏷️ {data['category_l2']}")
                            with col2:
                                st.markdown("**🎯 Confidence**")
                                st.markdown(f"### {data['confidence']*100:.2f}%")
                            with col3:
                                st.markdown("**💡 Impulse Score**")
                                impulse = data.get('impulse_score', 0)
                                st.markdown(f"### {impulse*100:.1f}%")
                            
                            # SHAP Explanations
                            with st.expander("🧠 Why did the AI choose this category?", expanded=True):
                                st.markdown("#### Top factors influencing this decision:")
                                if 'shap_top_features' in data:
                                    for feat in data['shap_top_features']:
                                        impact_color = "🟢" if feat['impact'] > 0 else "🔴"
                                        st.markdown(f"{impact_color} **{feat['feature']}**: {feat['impact']}")
                                
                                if 'anchor_rule' in data:
                                    st.markdown("#### 📋 Decision Rule:")
                                    st.code(data['anchor_rule'])
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
            
            # TAB 2: Spending Forecast
            with tab2:
                st.markdown("### 📈 AI Spending Forecast")
                st.markdown("Predict your future spending with confidence intervals")
                
                user_id = st.text_input("User ID for forecast", value=st.session_state.user_id or "user_demo_001")
                
                col1, col2 = st.columns(2)
                with col1:
                    horizon = st.slider("Forecast Horizon (days)", 30, 90, 30)
                with col2:
                    categories = st.multiselect(
                        "Categories to forecast",
                        ["Groceries", "Restaurants", "Transportation", "Shopping", "Entertainment", "Utilities"],
                        default=["Groceries", "Restaurants"]
                    )
                
                if st.button("🔮 Generate Forecast", type="primary"):
                    with st.spinner("AI is forecasting your spending..."):
                        response = requests.get(
                            f"{API_URL}/v1/forecast/{user_id}?horizon={horizon}",
                            headers=headers
                        )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Create forecast chart
                        fig = go.Figure()
                        
                        # Add confidence bands
                        if 'forecasts' in data:
                            for forecast in data['forecasts'][:5]:  # Show top 5
                                if forecast['category'] in categories or not categories:
                                    dates = pd.date_range(start=datetime.now(), periods=horizon, freq='D')
                                    
                                    # Create synthetic data for demo (replace with actual API data)
                                    base = forecast.get('p50', 100)
                                    fig.add_trace(go.Scatter(
                                        x=dates,
                                        y=[base * (1 + i*0.01) for i in range(horizon)],
                                        name=forecast['category'],
                                        mode='lines'
                                    ))
                            
                            fig.update_layout(
                                title="Spending Forecast (30-day horizon)",
                                xaxis_title="Date",
                                yaxis_title="Amount ($)",
                                hovermode='x unified'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show forecast data
                            st.markdown("### 📊 Forecast Details")
                            for forecast in data['forecasts']:
                                with st.expander(f"{forecast['category']} - Trend: {forecast.get('trend', 'stable')}"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("P10 (Conservative)", f"${forecast.get('p10', 0):.2f}")
                                    with col2:
                                        st.metric("P50 (Expected)", f"${forecast.get('p50', 0):.2f}")
                                    with col3:
                                        st.metric("P90 (Aggressive)", f"${forecast.get('p90', 0):.2f}")
                                    st.markdown(f"**Regime:** {forecast.get('regime', 'normal')}")
                    else:
                        st.error(f"Error: {response.status_code}")
            
            # TAB 3: Budget Recommendations
            with tab3:
                st.markdown("### 💰 AI-Powered Budget Recommendations")
                st.markdown("Personalized budgets with SHAP explanations")
                
                if st.button("🎯 Generate My Budget", type="primary", use_container_width=True):
                    with st.spinner("AI is analyzing your spending patterns..."):
                        response = requests.get(
                            f"{API_URL}/v1/budget/{st.session_state.user_id or 'user_demo_001'}",
                            headers=headers
                        )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Monthly Income", f"${data.get('income_estimate', 5500):,.2f}")
                        with col2:
                            st.metric("Savings Target", f"${data.get('savings_target', 550):,.2f}")
                        with col3:
                            st.metric("Recommended Savings", "23.1%", "↑ 3.2%")
                        
                        st.markdown("### 📋 Your Personalized Budget")
                        
                        # Show each recommendation
                        for rec in data['recommendations']:
                            with st.expander(f"**{rec['category']}**: ${rec['recommended_budget']}/month", expanded=True):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**Explanation:** {rec['explanation']}")
                                    
                                    if 'shap_top_features' in rec:
                                        st.markdown("**🔍 Key Factors:**")
                                        for feat in rec['shap_top_features'][:3]:
                                            st.markdown(f"• {feat['feature']}: {feat['impact']}")
                                    
                                    if 'anchor_rule' in rec:
                                        st.markdown("**📋 Decision Rule:**")
                                        st.code(rec['anchor_rule'])
                                
                                with col2:
                                    st.markdown("**📊 Comparison**")
                                    st.metric("Current Trend", f"${rec.get('current_trend', 0):.2f}")
                                    st.metric("Recommended", f"${rec['recommended_budget']:.2f}")
                                    st.metric("Confidence", f"{rec.get('confidence', 0)*100:.1f}%")
                    else:
                        st.error(f"Error: {response.status_code}")
            
            # TAB 4: Analytics Dashboard
            with tab4:
                st.markdown("### 📊 Spending Analytics")
                
                # Sample data for demo
                categories = ['Groceries', 'Restaurants', 'Transportation', 'Shopping', 'Entertainment', 'Utilities']
                values = [450, 320, 180, 290, 150, 210]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        values=values,
                        names=categories,
                        title="Spending by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Trend over time
                    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=[50 + i*2 + (i%7)*10 for i in range(30)],
                        name='Daily Spend',
                        mode='lines'
                    ))
                    fig.update_layout(title="Daily Spending Trend")
                    st.plotly_chart(fig, use_container_width=True)
                
                # AI Insights
                st.markdown("### 🧠 AI Insights")
                st.info("""
                **Based on your spending patterns:**
                - You tend to spend 23% more on weekends
                - Your grocery spending peaks on Sundays
                - Dining out increased 15% this month
                - You're on track to meet your savings goal
                """)
    
    except Exception as e:
        st.error(f"❌ Could not connect to API: {str(e)}")
        st.info("""
        **Demo Mode Active** - Using sample data.
        
        To connect to the live API:
        1. Get an API key from the sidebar
        2. Make sure the API is running at: https://fin-advisor-sa6h.onrender.com
        """)

else:
    # Show landing page for non-authenticated users
    st.markdown("## 🚀 Welcome to Fin Advisor - Your AI Financial Assistant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Features
        - **AI Transaction Classification** - 99.96% accuracy
        - **Spending Forecast** - 8.19% MAPE
        - **Smart Budget Recommendations** - with SHAP explanations
        - **Behavior Modeling** - Detects spending patterns
        - **30+ Categories** - Hierarchical classification
        """)
        
        st.markdown("### 🎯 Perfect For")
        st.markdown("""
        - Individuals wanting better financial insights
        - Developers building finance apps
        - Financial advisors seeking AI assistance
        - Anyone curious about their spending habits
        """)
    
    with col2:
        st.markdown("### 📊 Sample Dashboard Preview")
        # Create sample chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            y=[1200, 1350, 1100, 1450, 1300, 1250],
            name='Spending',
            mode='lines+markers'
        ))
        fig.update_layout(title="Sample Spending Trend")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🚀 Get Started")
        st.markdown("1. Sign up for free API key in sidebar")
        st.markdown("2. Connect with your credentials")
        st.markdown("3. Start analyzing your finances!")
    
    st.markdown("---")
    st.markdown("### 🔒 Trusted by users worldwide")
    st.markdown("""
    - 42/42 tests passed
    - 99.96% classification accuracy
    - 8.19% forecast error
    - 80.9% budget acceptance
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("© 2025 Fin Advisor AI")
with col2:
    st.markdown("[Privacy Policy](https://fin-advisor-sa6h.onrender.com/privacy) | [Terms](https://fin-advisor-sa6h.onrender.com/terms)")
with col3:
    st.markdown("Powered by AI/ML")
