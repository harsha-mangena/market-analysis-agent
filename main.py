import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import re
from agent import init_agent
from vector_store import init_vector_store
from data_ingestion import get_stock_data, get_stock_historical_data, get_company_overview
from technical_analysis import (calculate_moving_averages, calculate_rsi, calculate_macd, 
                               calculate_bollinger_bands, calculate_trend_indicators,
                               generate_trading_signals, create_technical_analysis_charts,
                               predict_price_trend, create_forecast_chart,
                               generate_strategy_recommendations)  # Tab 3: Technical Analysis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Initialize vector store (for future use)
vector_store = init_vector_store()

# Initialize agent
agent = init_agent()

# Set page configuration
st.set_page_config(
    page_title="Market Analysis Agent",
    page_icon="📊",
    layout="wide"
)

# Use Streamlit's native components for the header
st.title("Market Analysis Agent")
st.caption("Advanced Technical Analysis & Trading Strategies")

# Add minimal CSS for necessary styling that Streamlit doesn't provide natively
st.markdown("""
    <style>
    /* Only keep essential styling for tabs */
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for ticker selection
with st.sidebar:
    st.header("Stock Selection")

    # Popular tickers for quick selection
    popular_tickers = [
        "AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "TSLA", 
        "NVDA", "AMD", "INTC", "JPM", "BAC", "GS", "V", "MA",
        "DIS", "NFLX", "CMCSA", "T", "VZ"
    ]

    # Group tickers by sector
    tech_tickers = ["AAPL", "GOOGL", "GOOG", "MSFT", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC"]
    finance_tickers = ["JPM", "BAC", "GS", "V", "MA", "AXP", "C", "WFC", "COF"]
    consumer_tickers = ["DIS", "NFLX", "CMCSA", "SBUX", "MCD", "KO", "PEP", "WMT", "TGT", "HD"]

    # Tabs for different sectors
    sector = st.radio("Select Sector", ["Technology", "Finance", "Consumer", "Custom"])

    if sector == "Technology":
        selected_ticker = st.selectbox("Select a technology stock:", tech_tickers)
    elif sector == "Finance":
        selected_ticker = st.selectbox("Select a financial stock:", finance_tickers)
    elif sector == "Consumer":
        selected_ticker = st.selectbox("Select a consumer stock:", consumer_tickers)
    else:
        selected_ticker = st.text_input("Enter stock ticker:", "AAPL")

    # Analysis type
    st.subheader("Analysis Options")
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Basic Analysis", "Technical Analysis", "Fundamental Analysis", "Custom Query"]
    )

    # Date range selection for historical data
    st.subheader("Time Range")
    time_range = st.select_slider(
        "Select time range:",
        options=["1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],
        value="1 Month"
    )

    # Query construction
    if analysis_type != "Custom Query":
        if analysis_type == "Basic Analysis":
            query = f"Analyze {selected_ticker} stock with basic metrics and short-term outlook"
        elif analysis_type == "Technical Analysis":
            query = f"Perform technical analysis on {selected_ticker} stock with {time_range.lower()} data, including moving averages, MACD, RSI"
        else:  # Fundamental Analysis
            query = f"Provide fundamental analysis for {selected_ticker} including P/E ratio, growth metrics, and competitive position"
        
        st.subheader("Query Preview")
        st.info(query)
    else:
        query = st.text_input("Enter your custom query:", 
                          f"Analyze {selected_ticker} stock")

    # Run analysis button
    run_analysis = st.button("Run Analysis", type="primary")

# Function to parse and enhance the agent's response
def enhance_response(ticker, response, time_range):
    # Extract key information using regex or simple string operations
    try:
        price_match = re.search(r"\$(\d+\.\d+)", response)
        current_price = float(price_match.group(1)) if price_match else None
    except:
        current_price = None
    
    # Get historical data for charts
    try:
        historical_data = get_stock_historical_data(ticker, time_range)
        df = pd.DataFrame(historical_data)
        
        # Apply technical analysis indicators
        df = calculate_moving_averages(df, columns=['4. close'], periods=[20, 50, 200])
        df = calculate_rsi(df, column='4. close')
        df = calculate_macd(df, column='4. close')
        df = calculate_bollinger_bands(df, column='4. close')
        df = calculate_trend_indicators(df)
        df = generate_trading_signals(df)
        
        # Get company overview data
        company_data = get_company_overview(ticker)
        
        # Create main dashboard with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Company Profile", "Technical Analysis", "Trading Strategy"])
        
        # Tab 1: Overview
        with tab1:
            # Stock Overview Section - AI Generated insights at the top
            st.header(f"{ticker} Stock Analysis")
            
            # AI-generated overview card with clean styling
            with st.container():
                st.subheader("Overview")
                # Parse recent price and create a clean summary
                try:
                    # Extract price from response using regex
                    price_match = re.search(r"\$(\d+\.\d+)", response)
                    current_price = float(price_match.group(1)) if price_match else df.iloc[-1]['4. close']
                    
                    # Extract high, low, and date info if available
                    high_match = re.search(r"52-week high\D*(\$?\d+\.\d+)", response)
                    low_match = re.search(r"52-week low\D*(\$?\d+\.\d+)", response)
                    
                    high_value = high_match.group(1) if high_match else f"${df['4. close'].max():.2f}"
                    low_value = low_match.group(1) if low_match else f"${df['4. close'].min():.2f}"
                    
                    # Clean up the response to create a concise overview
                    overview_text = re.sub(r'\n+', '\n', response)  # Remove multiple newlines
                    overview_text = re.sub(r'\$\d+\.\d+', f"${current_price:.2f}", overview_text, count=1)  # Replace first price
                    
                    # Display the overview in a clean card
                    st.markdown(f"""
                    #### {ticker} Recent Performance
                    {overview_text.split('.')[0]}.  
                    {'.'.join(overview_text.split('.')[1:3])}.
                    """)
                except Exception as e:
                    st.markdown(f"#### {ticker} Recent Performance\n{response[:200]}...")
            
            # Horizontal line to separate sections
            st.markdown("---")
            
            # Continue with original layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price chart with key moving averages
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df['date'],
                    open=df['1. open'],
                    high=df['2. high'],
                    low=df['3. low'],
                    close=df['4. close'],
                    name='Price'
                ))
                
                # Add moving averages
                fig.add_trace(go.Scatter(x=df['date'], y=df['MA20_4. close'], name='20-Day MA', line=dict(color='orange')))
                fig.add_trace(go.Scatter(x=df['date'], y=df['MA50_4. close'], name='50-Day MA', line=dict(color='blue')))
                
                # Add Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=df['date'], 
                    y=df['BB_upper'], 
                    name='Upper Band', 
                    line=dict(color='lightgray'),
                    fill=None
                ))
                
                fig.add_trace(go.Scatter(
                    x=df['date'], 
                    y=df['BB_lower'], 
                    name='Lower Band', 
                    line=dict(color='lightgray'),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.2)'
                ))
                
                fig.update_layout(
                    title=f'{ticker} Stock Price',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    template='plotly_white',
                    height=500,
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volume chart
                volume_fig = px.bar(
                    df, 
                    x='date', 
                    y='5. volume',
                    title=f'{ticker} Trading Volume',
                    color_discrete_sequence=['#1E3A8A']
                )
                volume_fig.update_layout(height=250)
                st.plotly_chart(volume_fig, use_container_width=True)
            
            with col2:
                with st.container():
                    # Latest price metrics
                    latest = df.iloc[-1]
                    prev_day = df.iloc[-2] if len(df) > 1 else df.iloc[0]
                    price_change = latest['4. close'] - prev_day['4. close']
                    pct_change = (price_change / prev_day['4. close']) * 100
                    
                    st.metric(
                        label="Current Price", 
                        value=f"${latest['4. close']:.2f}",
                        delta=f"{price_change:.2f} ({pct_change:.2f}%)"
                    )
                    
                    # Key metrics
                    st.subheader("Key Metrics")
                    metrics_data = {
                        "Open": f"${latest['1. open']:.2f}",
                        "High": f"${latest['2. high']:.2f}",
                        "Low": f"${latest['3. low']:.2f}",
                        "Volume": f"{int(latest['5. volume']):,}"
                    }
                    
                    # Display as a small table
                    metrics_df = pd.DataFrame([metrics_data])
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                    
                    # Disclaimer for potentially simulated data
                    if "generate_sample_data" in str(historical_data):
                        st.warning("⚠️ Using simulated data due to API limitations. The charts and metrics shown may not reflect actual market conditions.")
                    
                    # Technical indicators
                    st.subheader("Technical Indicators")
                    
                    latest_rsi = latest['RSI_14'] if 'RSI_14' in latest else np.nan
                    latest_macd = latest['MACD_line'] if 'MACD_line' in latest else np.nan
                    latest_signal = latest['MACD_signal'] if 'MACD_signal' in latest else np.nan
                    
                    # Generate status based on indicators
                    rsi_status = "Oversold" if latest_rsi < 30 else "Overbought" if latest_rsi > 70 else "Neutral"
                    macd_status = "Bullish" if latest_macd > latest_signal else "Bearish"
                    
                    col_rsi, col_macd = st.columns(2)
                    
                    with col_rsi:
                        st.metric("RSI (14)", f"{latest_rsi:.2f}", rsi_status)
                    
                    with col_macd:
                        st.metric("MACD", f"{latest_macd:.2f}", macd_status)
            
                # Agent insights section
                with st.container():
                    st.subheader("Agent Insights")
                    # Fix overflow by using an expander with st.markdown for better formatting
                    with st.expander("View AI Analysis", expanded=True):
                        st.markdown(response)
                
                # Simple forecast section
                with st.container():
                    st.subheader("Quick Forecast")
                    
                    # Simple prediction of price trend
                    predictions, confidence = predict_price_trend(df, column='4. close', days=7)
                    pred_change = ((predictions[-1] / latest['4. close']) - 1) * 100
                    
                    st.metric(
                        label="7-Day Price Forecast", 
                        value=f"${predictions[-1]:.2f}",
                        delta=f"{pred_change:.2f}%"
                    )
                    
                    # Add confidence indicator
                    st.progress(min(confidence, 1.0), text=f"Model Confidence: {confidence*100:.1f}%")
        
        # Tab 2: Company Profile
        with tab2:
            st.header(f"Company Profile: {company_data.get('Name', ticker)}")
            
            # Company description
            with st.expander("About the Company", expanded=True):
                st.write(company_data.get('Description', f"No description available for {ticker}"))
                
                # Company basics in a clean card format
                st.subheader("Company Basics")
                
                # Display company information in 2 columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Company basic info
                    basic_info = {
                        "Sector": company_data.get('Sector', 'N/A'),
                        "Industry": company_data.get('Industry', 'N/A'),
                        "Exchange": company_data.get('Exchange', 'N/A'),
                        "Country": company_data.get('Country', 'N/A'),
                        "Currency": company_data.get('Currency', 'USD')
                    }
                    
                    for key, value in basic_info.items():
                        st.metric(key, value)
                
                with col2:
                    # Market data
                    market_cap = float(company_data.get('MarketCapitalization', 0))
                    if market_cap > 0:
                        if market_cap >= 1_000_000_000:
                            formatted_cap = f"${market_cap/1_000_000_000:.2f}B"
                        else:
                            formatted_cap = f"${market_cap/1_000_000:.2f}M"
                    else:
                        formatted_cap = 'N/A'
                    
                    market_data = {
                        "Market Cap": formatted_cap,
                        "52-Week High": f"${company_data.get('52WeekHigh', 'N/A')}",
                        "52-Week Low": f"${company_data.get('52WeekLow', 'N/A')}",
                        "Beta": company_data.get('Beta', 'N/A')
                    }
                    
                    for key, value in market_data.items():
                        st.metric(key, value)
            
            # Financial metrics
            st.subheader("Financial Metrics")
            
            # Create financial metrics cards in 3 columns
            fin_col1, fin_col2, fin_col3 = st.columns(3)
            
            with fin_col1:
                with st.container():
                    st.markdown("**Valuation Metrics**")
                    valuation_metrics = {
                        "P/E Ratio": company_data.get('PERatio', 'N/A'),
                        "Forward P/E": company_data.get('ForwardPE', 'N/A'),
                        "PEG Ratio": company_data.get('PEGRatio', 'N/A'),
                        "Price/Book": company_data.get('PriceToBookRatio', 'N/A'),
                        "Price/Sales": company_data.get('PriceToSalesRatioTTM', 'N/A')
                    }
                    
                    val_df = pd.DataFrame({"Value": valuation_metrics})
                    st.dataframe(val_df, use_container_width=True)
            
            with fin_col2:
                with st.container():
                    st.markdown("**Profitability Metrics**")
                    profitability_metrics = {
                        "Profit Margin": f"{float(company_data.get('ProfitMargin', 0)) * 100:.2f}%" if company_data.get('ProfitMargin') else 'N/A',
                        "Operating Margin": f"{float(company_data.get('OperatingMarginTTM', 0)) * 100:.2f}%" if company_data.get('OperatingMarginTTM') else 'N/A',
                        "ROA": f"{float(company_data.get('ReturnOnAssetsTTM', 0)) * 100:.2f}%" if company_data.get('ReturnOnAssetsTTM') else 'N/A',
                        "ROE": f"{float(company_data.get('ReturnOnEquityTTM', 0)) * 100:.2f}%" if company_data.get('ReturnOnEquityTTM') else 'N/A'
                    }
                    
                    prof_df = pd.DataFrame({"Value": profitability_metrics})
                    st.dataframe(prof_df, use_container_width=True)
            
            with fin_col3:
                with st.container():
                    st.markdown("**Growth Metrics**")
                    
                    # Format the growth metrics as percentages if available
                    eps = company_data.get('EPS', 'N/A')
                    qearly_growth = f"{float(company_data.get('QuarterlyEarningsGrowthYOY', 0)) * 100:.2f}%" if company_data.get('QuarterlyEarningsGrowthYOY') else 'N/A'
                    qrev_growth = f"{float(company_data.get('QuarterlyRevenueGrowthYOY', 0)) * 100:.2f}%" if company_data.get('QuarterlyRevenueGrowthYOY') else 'N/A'
                    
                    growth_metrics = {
                        "EPS": eps,
                        "Dividend Yield": f"{float(company_data.get('DividendYield', 0)) * 100:.2f}%" if company_data.get('DividendYield') else 'N/A',
                        "Quarterly Earnings Growth (YoY)": qearly_growth,
                        "Quarterly Revenue Growth (YoY)": qrev_growth
                    }
                    
                    growth_df = pd.DataFrame({"Value": growth_metrics})
                    st.dataframe(growth_df, use_container_width=True)
            
            # Analyst recommendations
            st.subheader("Analyst Recommendations")
            
            # Display analyst target price
            target_price = company_data.get('AnalystTargetPrice', 'N/A')
            if target_price != 'N/A':
                current_price = df.iloc[-1]['4. close']
                target_price = float(target_price)
                price_difference = ((target_price - current_price) / current_price) * 100
                
                st.metric(
                    label="Analyst Target Price", 
                    value=f"${target_price:.2f}",
                    delta=f"{price_difference:.2f}%"
                )
                
                # Visual representation of current price vs target price
                if target_price > 0:
                    fig = go.Figure()
                    
                    # Add current price and target price as bar chart
                    fig.add_trace(go.Bar(
                        x=['Current Price', 'Target Price'],
                        y=[current_price, target_price],
                        text=[f'${current_price:.2f}', f'${target_price:.2f}'],
                        textposition='auto',
                        marker_color=['#1E3A8A', '#4F46E5'],
                        width=[0.4, 0.4]
                    ))
                    
                    fig.update_layout(
                        title='Current Price vs Analyst Target',
                        template='plotly_white',
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No analyst target price available for this stock.")
            
            # Disclaimer for potentially simulated data
            if "generate_sample_company_overview" in str(company_data):
                st.warning("⚠️ Using simulated company data due to API limitations. The information shown may not reflect actual company details.")
                
        # Tab 3: Technical Analysis
        with tab3:
            st.header("Technical Analysis")
            price_chart, indicator_chart, macd_chart = create_technical_analysis_charts(df, ticker)
            
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Technical indicator metrics in a row
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                trend = "Bullish" if df['MA20_4. close'].iloc[-1] > df['MA50_4. close'].iloc[-1] else "Bearish"
                st.metric("Trend", trend)
                
            with metrics_cols[1]:
                st.metric("RSI (14)", f"{latest_rsi:.2f}", rsi_status)
                
            with metrics_cols[2]:
                bb_position = ((latest['4. close'] - latest['BB_lower']) / 
                              (latest['BB_upper'] - latest['BB_lower'])) * 100
                st.metric("BB Position", f"{bb_position:.1f}%")
                
            with metrics_cols[3]:
                st.metric("MACD", f"{latest_macd:.2f}", macd_status)
                
            # RSI Chart
            st.plotly_chart(indicator_chart, use_container_width=True)
            
            # MACD Chart
            if macd_chart:
                st.plotly_chart(macd_chart, use_container_width=True)
                
            # Forecast chart
            forecast = create_forecast_chart(df, predictions, ticker)
            st.plotly_chart(forecast, use_container_width=True)
        
        # Tab 4: Trading Strategy
        with tab4:
            st.header("Trading Strategy Recommendations")
            
            # Get strategy recommendations
            recommendations = generate_strategy_recommendations(df)
            
            # Performance metrics
            st.subheader("Performance Metrics")
            performance_cols = st.columns(4)
            
            # Calculate performance metrics
            start_price = df.iloc[0]['4. close']
            end_price = df.iloc[-1]['4. close']
            total_return = ((end_price - start_price) / start_price) * 100
            
            volatility = df['4. close'].pct_change().std() * (252 ** 0.5) * 100  # Annualized
            
            # High/low based on available data
            high_value = df['4. close'].max()
            low_value = df['4. close'].min()
            
            with performance_cols[0]:
                st.metric("Total Return", f"{total_return:.2f}%")
            
            with performance_cols[1]:
                st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
                
            with performance_cols[2]:
                st.metric("Period High", f"${high_value:.2f}")
                
            with performance_cols[3]:
                st.metric("Period Low", f"${low_value:.2f}")
            
            # Strategy recommendations using Streamlit expanders
            st.subheader("Strategy Recommendations")
            strategy_cols = st.columns(3)
            
            with strategy_cols[0]:
                with st.expander("Short-term Strategy (1-5 days)", expanded=True):
                    action = recommendations['short_term'].get('action', 'hold')
                    reason = recommendations['short_term'].get('reason', 'No clear signals')
                    
                    if action == 'buy':
                        st.success(f"**Action:** {action.upper()}")
                    elif action == 'sell':
                        st.error(f"**Action:** {action.upper()}")
                    else:
                        st.info(f"**Action:** {action.upper()}")
                        
                    st.write(f"**Reason:** {reason}")
            
            with strategy_cols[1]:
                with st.expander("Medium-term Strategy (1-3 weeks)", expanded=True):
                    action = recommendations['medium_term'].get('action', 'hold')
                    reason = recommendations['medium_term'].get('reason', 'No clear signals')
                    
                    if action in ['buy', 'accumulate']:
                        st.success(f"**Action:** {action.upper()}")
                    elif action in ['sell', 'reduce']:
                        st.error(f"**Action:** {action.upper()}")
                    else:
                        st.info(f"**Action:** {action.upper()}")
                        
                    st.write(f"**Reason:** {reason}")
            
            with strategy_cols[2]:
                with st.expander("Long-term Strategy (months)", expanded=True):
                    action = recommendations['long_term'].get('action', 'hold')
                    reason = recommendations['long_term'].get('reason', 'No clear signals')
                    
                    if action in ['buy', 'accumulate']:
                        st.success(f"**Action:** {action.upper()}")
                    elif action in ['sell', 'reduce']:
                        st.error(f"**Action:** {action.upper()}")
                    else:
                        st.info(f"**Action:** {action.upper()}")
                        
                    st.write(f"**Reason:** {reason}")
            
            # Risk management section using Streamlit native components
            st.subheader("Risk Management")
            
            risk_cols = st.columns(2)
            
            with risk_cols[0]:
                with st.container():
                    st.markdown("**Position Sizing**")
                    
                    # Calculate suggested position size based on volatility
                    max_risk_pct = 2.0  # Max 2% risk per trade
                    stop_loss_pct = 5.0  # 5% stop loss
                    
                    # Create a table for risk management data
                    risk_data = {
                        "Metric": ["Maximum Risk", "Suggested Stop Loss", "Position Size Formula"],
                        "Value": [f"{max_risk_pct}% of portfolio per trade", 
                                 f"{stop_loss_pct}% below entry price",
                                 "(Portfolio * Max Risk) / Stop Loss Distance"]
                    }
                    
                    risk_df = pd.DataFrame(risk_data)
                    st.table(risk_df)
            
            with risk_cols[1]:
                with st.container():
                    st.markdown("**Key Levels**")
                    
                    # Calculate potential support/resistance levels
                    support1 = latest['4. close'] * 0.95  # 5% below
                    support2 = latest['4. close'] * 0.90  # 10% below
                    resistance1 = latest['4. close'] * 1.05  # 5% above
                    resistance2 = latest['4. close'] * 1.10  # 10% above
                    
                    # Create a table for key levels
                    levels_data = {
                        "Level": ["Support Level 1", "Support Level 2", "Resistance Level 1", "Resistance Level 2"],
                        "Price": [f"${support1:.2f}", f"${support2:.2f}", f"${resistance1:.2f}", f"${resistance2:.2f}"]
                    }
                    
                    levels_df = pd.DataFrame(levels_data)
                    st.table(levels_df)
        
        return True
        
    except Exception as e:
        st.error(f"Error generating enhanced visualization: {str(e)}")
        st.write(response)  # Fallback to just showing the response
        return False

# Main content area
if run_analysis or query:
    try:
        with st.spinner(f"Analyzing {selected_ticker}..."):
            # Get response from agent
            response = agent.run(query)
            
            # Enhance response with visualizations
            success = enhance_response(selected_ticker, response, time_range)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    # Default welcome screen with native Streamlit components
    col1, col2 = st.columns([3, 2])
    
    with col1:
        with st.container():
            st.header("Welcome to the Market Analysis Agent")
            st.write("Select a stock ticker and analysis type from the sidebar to get started.")
            
            st.subheader("This agent can provide:")
            
            features = [
                "📈 Basic stock analysis and price predictions",
                "📊 Technical analysis with key indicators",
                "📋 Trading strategies and risk management",
                "🔍 Custom analysis based on your queries"
            ]
            
            for feature in features:
                st.markdown(feature)
                
            st.caption("Select a ticker from the sidebar and click \"Run Analysis\" to begin.")
    
    with col2:
        with st.container():
            st.subheader("Featured Stocks")
            
            # Featured stocks with native Streamlit components
            featured_stocks = {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corp.",
                "NVDA": "NVIDIA Corp.",
                "GOOGL": "Alphabet Inc.",
                "TSLA": "Tesla Inc."
            }
            
            # Create a DataFrame for better display
            featured_df = pd.DataFrame(
                {"Company": featured_stocks.values()},
                index=featured_stocks.keys()
            )
            
            # Display with Streamlit dataframe styling
            st.dataframe(
                featured_df,
                use_container_width=True,
                hide_index=False
            )
            
            # Sample analysis preview
            st.image("https://static.streamlit.io/examples/stock.jpg", 
                     use_column_width=True, 
                     caption="Market data visualization")