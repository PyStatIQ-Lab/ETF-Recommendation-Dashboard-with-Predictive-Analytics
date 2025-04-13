import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ETF Recommendation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# List of ETFs
etf_list = [
    "MAFANG.NS", "FMCGIETF.NS", "MOGSEC.NS", "TATAGOLD.NS", "GOLDIETF.NS",
    "GOLDCASE.NS", "HDFCGOLD.NS", "GOLD1.NS", "AXISGOLD.NS", "GOLD360.NS",
    "ABGSEC.NS", "SETFGOLD.NS", "GOLDBEES.NS", "LICMFGOLD.NS", "QGOLDHALF.NS",
    "GSEC5IETF.NS", "IVZINGOLD.NS", "GOLDSHARE.NS", "BSLGOLDETF.NS", "LICNFNHGP.NS",
    "GOLDETFADD.NS", "UNIONGOLD.NS", "CONSUMBEES.NS", "SDL26BEES.NS", "AXISCETF.NS",
    "GROWWGOLD.NS", "GOLDETF.NS", "MASPTOP50.NS", "SETF10GILT.NS", "EBBETF0433.NS",
    "NV20BEES.NS", "BBNPPGOLD.NS", "CONSUMIETF.NS", "AUTOBEES.NS", "BSLSENETFG.NS",
    "LTGILTBEES.NS", "AUTOIETF.NS", "AXISBPSETF.NS", "GILT5YBEES.NS", "LIQUIDCASE.NS",
    "GROWWLIQID.NS", "GSEC10YEAR.NS", "LIQUIDBETF.NS", "LIQUIDADD.NS", "LIQUID1.NS",
    "HDFCLIQUID.NS", "MOLOWVOL.NS", "AONELIQUID.NS", "CASHIETF.NS", "LIQUIDPLUS.NS",
    "LIQUIDSHRI.NS", "ABSLLIQUID.NS", "LIQUIDETF.NS", "CONS.NS", "LIQUIDSBI.NS",
    "LIQUID.NS", "EGOLD.NS", "BBNPNBETF.NS", "LIQUIDIETF.NS", "IVZINNIFTY.NS",
    "GSEC10ABSL.NS", "LIQUIDBEES.NS", "EBBETF0430.NS", "SBIETFCON.NS", "MON100.NS",
    "LICNETFGSC.NS", "GSEC10IETF.NS", "QUAL30IETF.NS", "SILVRETF.NS", "LICNETFSEN.NS",
    "HDFCLOWVOL.NS", "EBANKNIFTY.NS", "LOWVOLIETF.NS", "EBBETF0431.NS", "TOP100CASE.NS",
    "NIFTYQLITY.NS", "HDFCGROWTH.NS", "SHARIABEES.NS", "BBETF0432.NS"
]

# Function to fetch ETF data
@st.cache_data(ttl=3600)
def get_etf_data(etf_list, period='1y'):
    data = {}
    for etf in etf_list:
        try:
            ticker = yf.Ticker(etf)
            df = ticker.history(period=period)
            if not df.empty:
                data[etf] = df
        except:
            continue
    return data

# Function to calculate metrics
def calculate_metrics(df):
    if df.empty:
        return None
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate metrics
    metrics = {
        'Last_Price': df['Close'].iloc[-1],
        '1D_Return': df['Daily_Return'].iloc[-1],
        '1W_Return': (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) if len(df) >= 5 else np.nan,
        '1M_Return': (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else np.nan,
        '3M_Return': (df['Close'].iloc[-1] / df['Close'].iloc[-60] - 1) if len(df) >= 60 else np.nan,
        'YTD_Return': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1),
        'Volatility': df['Daily_Return'].std() * np.sqrt(252),
        'Sharpe_Ratio': (df['Daily_Return'].mean() / df['Daily_Return'].std()) * np.sqrt(252),
        'Max_Drawdown': (df['Close'] / df['Close'].cummax() - 1).min()
    }
    return metrics

# Function to train predictive model
def train_predictive_model(df, days_to_predict=5):
    if df.empty or len(df) < 30:
        return None, None, None
    
    # Prepare data
    df = df[['Close']].copy()
    df['Date'] = df.index
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag5'] = df['Close'].shift(5)
    df['Close_Lag10'] = df['Close'].shift(10)
    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df = df.dropna()
    
    if len(df) < 10:
        return None, None, None
    
    # Features and target
    X = df[['Days', 'Close_Lag1', 'Close_Lag5', 'Close_Lag10', 'MA_5', 'MA_10']]
    y = df['Close']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Predict future
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
    future_days = [(date - df['Date'].min()).days for date in future_dates]
    
    future_X = pd.DataFrame({
        'Days': future_days,
        'Close_Lag1': [df['Close'].iloc[-1]] * days_to_predict,
        'Close_Lag5': [df['Close'].iloc[-5]] * days_to_predict if len(df) >= 5 else [df['Close'].iloc[-1]] * days_to_predict,
        'Close_Lag10': [df['Close'].iloc[-10]] * days_to_predict if len(df) >= 10 else [df['Close'].iloc[-1]] * days_to_predict,
        'MA_5': [df['Close'].rolling(5).mean().iloc[-1]] * days_to_predict,
        'MA_10': [df['Close'].rolling(10).mean().iloc[-1]] * days_to_predict
    })
    
    future_predictions = model.predict(future_X)
    
    return future_dates, future_predictions, mse

# Function to categorize ETFs
def categorize_etf(etf_name):
    etf_name = etf_name.lower()
    if 'gold' in etf_name:
        return 'Gold'
    elif 'liquid' in etf_name or 'cash' in etf_name:
        return 'Liquid'
    elif 'cons' in etf_name or 'consum' in etf_name:
        return 'Consumer'
    elif 'auto' in etf_name:
        return 'Automobile'
    elif 'gilt' in etf_name or 'gsec' in etf_name:
        return 'Government Securities'
    elif 'nifty' in etf_name or 'top' in etf_name or 'mon' in etf_name:
        return 'Equity Index'
    elif 'sharia' in etf_name:
        return 'Islamic'
    elif 'silver' in etf_name:
        return 'Silver'
    else:
        return 'Other'

# Main app
def main():
    st.title("📈 ETF Recommendation Dashboard")
    st.write("Analyzing Indian ETFs for optimal investment recommendations")
    
    # Sidebar controls
    st.sidebar.header("Controls")
    analysis_period = st.sidebar.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    recommendation_horizon = st.sidebar.selectbox(
        "Recommendation Horizon",
        ["1 Week", "1 Month"],
        index=0
    )
    
    min_volatility = st.sidebar.slider(
        "Minimum Volatility Threshold",
        0.0, 1.0, 0.1, 0.01
    )
    
    max_volatility = st.sidebar.slider(
        "Maximum Volatility Threshold",
        0.0, 1.0, 0.5, 0.01
    )
    
    # Fetch data
    with st.spinner("Fetching ETF data..."):
        etf_data = get_etf_data(etf_list, analysis_period)
    
    if not etf_data:
        st.error("Failed to fetch data for any ETFs. Please try again later.")
        return
    
    # Calculate metrics for all ETFs
    metrics_data = []
    for etf, df in etf_data.items():
        metrics = calculate_metrics(df)
        if metrics:
            metrics['ETF'] = etf
            metrics['Category'] = categorize_etf(etf)
            metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Filter by volatility
    metrics_df = metrics_df[
        (metrics_df['Volatility'] >= min_volatility) & 
        (metrics_df['Volatility'] <= max_volatility)
    ]
    
    if metrics_df.empty:
        st.warning("No ETFs match your volatility criteria. Please adjust the filters.")
        return
    
    # Sort by performance based on horizon
    if recommendation_horizon == "1 Week":
        metrics_df = metrics_df.sort_values('1W_Return', ascending=False)
    else:
        metrics_df = metrics_df.sort_values('1M_Return', ascending=False)
    
    # Display top recommendations
    st.header("🏆 Top ETF Recommendations")
    top_n = min(5, len(metrics_df))
    cols = st.columns(top_n)
    
    for i in range(top_n):
        etf = metrics_df.iloc[i]['ETF']
        with cols[i]:
            st.metric(
                label=f"{etf}",
                value=f"₹{metrics_df.iloc[i]['Last_Price']:.2f}",
                delta=f"{metrics_df.iloc[i]['1W_Return' if recommendation_horizon == '1 Week' else '1M_Return']*100:.2f}%"
            )
            st.caption(f"Category: {metrics_df.iloc[i]['Category']}")
            st.caption(f"Volatility: {metrics_df.iloc[i]['Volatility']:.2f}")
            st.caption(f"Sharpe Ratio: {metrics_df.iloc[i]['Sharpe_Ratio']:.2f}")
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Metrics", "Category Analysis", "Predictive Analytics", "ETF Explorer"])
    
    with tab1:
        st.subheader("Performance Metrics Overview")
        
        # Select metrics to display
        selected_metrics = st.multiselect(
            "Select metrics to display",
            ['1D_Return', '1W_Return', '1M_Return', '3M_Return', 'YTD_Return', 
             'Volatility', 'Sharpe_Ratio', 'Max_Drawdown'],
            default=['1W_Return', '1M_Return', 'Volatility', 'Sharpe_Ratio']
        )
        
        if selected_metrics:
            display_df = metrics_df[['ETF', 'Category'] + selected_metrics]
            st.dataframe(
                display_df.style.format({
                    '1D_Return': '{:.2%}',
                    '1W_Return': '{:.2%}',
                    '1M_Return': '{:.2%}',
                    '3M_Return': '{:.2%}',
                    'YTD_Return': '{:.2%}',
                    'Volatility': '{:.2f}',
                    'Sharpe_Ratio': '{:.2f}',
                    'Max_Drawdown': '{:.2%}'
                }).background_gradient(cmap='RdYlGn', subset=selected_metrics),
                height=600
            )
        
        # Correlation heatmap
        st.subheader("Metric Correlations")
        corr_matrix = metrics_df[['1D_Return', '1W_Return', '1M_Return', '3M_Return', 
                                'YTD_Return', 'Volatility', 'Sharpe_Ratio']].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("ETF Category Analysis")
        
        # Category distribution
        category_counts = metrics_df['Category'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        category_counts.plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("ETF Category Distribution")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)
        
        # Category performance
        st.subheader("Average Performance by Category")
        category_perf = metrics_df.groupby('Category').agg({
            '1W_Return': 'mean',
            '1M_Return': 'mean',
            'Volatility': 'mean',
            'Sharpe_Ratio': 'mean'
        }).reset_index()
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        category_perf.plot(x='Category', y=['1W_Return', '1M_Return'], kind='bar', ax=ax2)
        ax2.set_title("Average Returns by Category")
        ax2.set_ylabel("Return")
        ax2.legend(["1 Week Return", "1 Month Return"])
        st.pyplot(fig2)
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        category_perf.plot(x='Category', y=['Volatility', 'Sharpe_Ratio'], kind='bar', ax=ax3)
        ax3.set_title("Risk Metrics by Category")
        ax3.set_ylabel("Value")
        ax3.legend(["Volatility", "Sharpe Ratio"])
        st.pyplot(fig3)
    
    with tab3:
        st.subheader("Predictive Analytics")
        
        selected_etf = st.selectbox(
            "Select ETF for prediction",
            metrics_df['ETF'].sort_values()
        )
        
        if selected_etf:
            df = etf_data[selected_etf]
            future_dates, future_predictions, mse = train_predictive_model(df)
            
            if future_dates:
                st.write(f"**Predicted Prices for {selected_etf}**")
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price': future_predictions
                })
                
                # Display predictions
                st.dataframe(pred_df.style.format({
                    'Predicted_Price': '{:.2f}'
                }))
                
                # Plot historical and predicted prices
                fig, ax = plt.subplots(figsize=(12, 6))
                df['Close'].plot(ax=ax, label='Historical Prices')
                pred_df.set_index('Date')['Predicted_Price'].plot(
                    ax=ax, 
                    style='r--', 
                    marker='o', 
                    label='Predicted Prices'
                )
                ax.set_title(f"Price Prediction for {selected_etf}")
                ax.set_ylabel("Price (₹)")
                ax.legend()
                st.pyplot(fig)
                
                st.write(f"Model Mean Squared Error: {mse:.4f}")
                
                # Calculate predicted returns
                current_price = df['Close'].iloc[-1]
                if recommendation_horizon == "1 Week":
                    predicted_return = (future_predictions[4] / current_price - 1) * 100
                    st.write(f"**Predicted 1 Week Return:** {predicted_return:.2f}%")
                else:
                    # For 1 month, we'd need more predictions (20 trading days)
                    st.warning("For 1 month predictions, we need more historical data.")
            else:
                st.warning("Insufficient data for reliable predictions for this ETF.")
    
    with tab4:
        st.subheader("ETF Explorer")
        
        selected_etf = st.selectbox(
            "Select ETF to explore",
            metrics_df['ETF'].sort_values()
        )
        
        if selected_etf:
            df = etf_data[selected_etf]
            metrics = metrics_df[metrics_df['ETF'] == selected_etf].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Price", f"₹{metrics['Last_Price']:.2f}")
                st.metric("1 Week Return", f"{metrics['1W_Return']*100:.2f}%")
                st.metric("1 Month Return", f"{metrics['1M_Return']*100:.2f}%")
                st.metric("YTD Return", f"{metrics['YTD_Return']*100:.2f}%")
            
            with col2:
                st.metric("Volatility", f"{metrics['Volatility']:.4f}")
                st.metric("Sharpe Ratio", f"{metrics['Sharpe_Ratio']:.2f}")
                st.metric("Max Drawdown", f"{metrics['Max_Drawdown']*100:.2f}%")
                st.metric("Category", metrics['Category'])
            
            # Price chart
            st.subheader("Price Chart")
            fig, ax = plt.subplots(figsize=(12, 6))
            df['Close'].plot(ax=ax)
            ax.set_title(f"{selected_etf} Price History")
            ax.set_ylabel("Price (₹)")
            st.pyplot(fig)
            
            # Volume chart
            st.subheader("Trading Volume")
            fig, ax = plt.subplots(figsize=(12, 4))
            df['Volume'].plot(ax=ax, color='green')
            ax.set_title(f"{selected_etf} Trading Volume")
            ax.set_ylabel("Volume")
            st.pyplot(fig)
            
            # Daily returns distribution
            st.subheader("Daily Returns Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            df['Daily_Return'].hist(bins=50, ax=ax)
            ax.set_title("Distribution of Daily Returns")
            ax.set_xlabel("Daily Return")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
