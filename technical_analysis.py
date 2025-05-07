import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px

def calculate_moving_averages(df, columns=['5. adjusted close', '4. close'], periods=[20, 50, 200]):
    """
    Calculate moving averages for the given dataframe.
    
    Args:
        df: DataFrame with price data
        columns: List of columns to calculate MAs for
        periods: List of periods for the moving averages
    
    Returns:
        DataFrame with added moving average columns
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns:
            for period in periods:
                result_df[f'MA{period}_{col}'] = result_df[col].rolling(window=period).mean()
    
    return result_df

def calculate_rsi(df, column='4. close', periods=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        df: DataFrame with price data
        column: Column name for close price
        periods: RSI period (default 14)
    
    Returns:
        DataFrame with added RSI column
    """
    result_df = df.copy()
    
    # Calculate price changes
    delta = result_df[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    result_df[f'RSI_{periods}'] = 100 - (100 / (1 + rs))
    
    return result_df

def calculate_macd(df, column='4. close', fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        df: DataFrame with price data
        column: Column name for close price
        fast: Fast period
        slow: Slow period
        signal: Signal period
    
    Returns:
        DataFrame with added MACD columns
    """
    result_df = df.copy()
    
    # Calculate EMA values
    result_df[f'EMA_{fast}'] = result_df[column].ewm(span=fast, adjust=False).mean()
    result_df[f'EMA_{slow}'] = result_df[column].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line and signal line
    result_df['MACD_line'] = result_df[f'EMA_{fast}'] - result_df[f'EMA_{slow}']
    result_df['MACD_signal'] = result_df['MACD_line'].ewm(span=signal, adjust=False).mean()
    result_df['MACD_histogram'] = result_df['MACD_line'] - result_df['MACD_signal']
    
    return result_df

def calculate_bollinger_bands(df, column='4. close', window=20, num_std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with price data
        column: Column name for close price
        window: Moving average window
        num_std_dev: Number of standard deviations for the bands
    
    Returns:
        DataFrame with added Bollinger Bands columns
    """
    result_df = df.copy()
    
    # Calculate middle band (SMA)
    result_df['BB_middle'] = result_df[column].rolling(window=window).mean()
    
    # Calculate standard deviation
    result_df['BB_std_dev'] = result_df[column].rolling(window=window).std()
    
    # Calculate upper and lower bands
    result_df['BB_upper'] = result_df['BB_middle'] + (result_df['BB_std_dev'] * num_std_dev)
    result_df['BB_lower'] = result_df['BB_middle'] - (result_df['BB_std_dev'] * num_std_dev)
    
    return result_df

def calculate_trend_indicators(df, column='4. close'):
    """
    Calculate various trend indicators and signals.
    
    Args:
        df: DataFrame with price data and moving averages
        column: Column name for close price
        
    Returns:
        DataFrame with added trend indicators
    """
    result_df = df.copy()
    
    # Check if we have moving averages to determine trend
    if 'MA20_4. close' in result_df.columns and 'MA50_4. close' in result_df.columns:
        # Golden/Death Cross
        result_df['golden_cross'] = (
            (result_df['MA20_4. close'] > result_df['MA50_4. close']) & 
            (result_df['MA20_4. close'].shift(1) <= result_df['MA50_4. close'].shift(1))
        )
        
        result_df['death_cross'] = (
            (result_df['MA20_4. close'] < result_df['MA50_4. close']) & 
            (result_df['MA20_4. close'].shift(1) >= result_df['MA50_4. close'].shift(1))
        )
        
        # Trend direction
        result_df['trend'] = 'neutral'
        result_df.loc[result_df['MA20_4. close'] > result_df['MA50_4. close'], 'trend'] = 'bullish'
        result_df.loc[result_df['MA20_4. close'] < result_df['MA50_4. close'], 'trend'] = 'bearish'
    
    # Simple price direction (comparing to n periods ago)
    result_df['price_change_5d'] = result_df[column].diff(5)
    result_df['price_change_pct_5d'] = result_df[column].pct_change(5) * 100
    
    return result_df

def generate_trading_signals(df):
    """
    Generate trading signals based on technical indicators.
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with added signal columns
    """
    result_df = df.copy()
    
    # RSI signals
    if 'RSI_14' in result_df.columns:
        result_df['rsi_signal'] = 'neutral'
        result_df.loc[result_df['RSI_14'] < 30, 'rsi_signal'] = 'oversold'
        result_df.loc[result_df['RSI_14'] > 70, 'rsi_signal'] = 'overbought'
    
    # MACD signals
    if 'MACD_line' in result_df.columns and 'MACD_signal' in result_df.columns:
        # Bullish crossover (MACD line crosses above signal line)
        result_df['macd_crossover_bullish'] = (
            (result_df['MACD_line'] > result_df['MACD_signal']) & 
            (result_df['MACD_line'].shift(1) <= result_df['MACD_signal'].shift(1))
        )
        
        # Bearish crossover (MACD line crosses below signal line)
        result_df['macd_crossover_bearish'] = (
            (result_df['MACD_line'] < result_df['MACD_signal']) & 
            (result_df['MACD_line'].shift(1) >= result_df['MACD_signal'].shift(1))
        )
    
    # Combine signals for buy/sell recommendations
    result_df['signal'] = 'hold'
    
    # Buy signals
    if 'RSI_14' in result_df.columns and 'MACD_line' in result_df.columns:
        buy_condition = (
            (result_df['RSI_14'] < 40) & 
            (result_df['MACD_line'] > result_df['MACD_signal'])
        )
        result_df.loc[buy_condition, 'signal'] = 'buy'
    
    # Sell signals
    if 'RSI_14' in result_df.columns and 'MACD_line' in result_df.columns:
        sell_condition = (
            (result_df['RSI_14'] > 60) & 
            (result_df['MACD_line'] < result_df['MACD_signal'])
        )
        result_df.loc[sell_condition, 'signal'] = 'sell'
    
    return result_df

def create_technical_analysis_charts(df, ticker):
    """
    Create technical analysis charts.
    
    Args:
        df: DataFrame with price data and indicators
        ticker: Stock ticker symbol
        
    Returns:
        tuple: (price_chart, indicator_charts) as Plotly figures
    """
    # Price chart with moving averages and Bollinger Bands
    price_chart = go.Figure()
    
    # Add candlestick chart
    price_chart.add_trace(go.Candlestick(
        x=df['date'],
        open=df['1. open'],
        high=df['2. high'],
        low=df['3. low'],
        close=df['4. close'],
        name='Price'
    ))
    
    # Add moving averages if available
    if 'MA20_4. close' in df.columns:
        price_chart.add_trace(go.Scatter(
            x=df['date'],
            y=df['MA20_4. close'],
            name='20-Day MA',
            line=dict(color='orange')
        ))
    
    if 'MA50_4. close' in df.columns:
        price_chart.add_trace(go.Scatter(
            x=df['date'],
            y=df['MA50_4. close'],
            name='50-Day MA',
            line=dict(color='blue')
        ))
    
    # Add Bollinger Bands if available
    if 'BB_upper' in df.columns:
        price_chart.add_trace(go.Scatter(
            x=df['date'],
            y=df['BB_upper'],
            name='Upper Band',
            line=dict(color='green', dash='dash')
        ))
        
        price_chart.add_trace(go.Scatter(
            x=df['date'],
            y=df['BB_lower'],
            name='Lower Band',
            line=dict(color='red', dash='dash')
        ))
    
    price_chart.update_layout(
        title=f'{ticker} Price Chart with Indicators',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    # Create indicator charts (RSI, MACD)
    indicator_charts = go.Figure()
    
    # Add RSI if available
    if 'RSI_14' in df.columns:
        indicator_charts.add_trace(go.Scatter(
            x=df['date'],
            y=df['RSI_14'],
            name='RSI (14)'
        ))
        
        # Add RSI reference lines
        indicator_charts.add_shape(
            type='line',
            x0=df['date'].iloc[0],
            y0=70,
            x1=df['date'].iloc[-1],
            y1=70,
            line=dict(color='red', dash='dash')
        )
        
        indicator_charts.add_shape(
            type='line',
            x0=df['date'].iloc[0],
            y0=30,
            x1=df['date'].iloc[-1],
            y1=30,
            line=dict(color='green', dash='dash')
        )
    
    # Add MACD if available
    if 'MACD_line' in df.columns and 'MACD_signal' in df.columns:
        macd_chart = go.Figure()
        
        # MACD line and signal line
        macd_chart.add_trace(go.Scatter(
            x=df['date'],
            y=df['MACD_line'],
            name='MACD'
        ))
        
        macd_chart.add_trace(go.Scatter(
            x=df['date'],
            y=df['MACD_signal'],
            name='Signal'
        ))
        
        # MACD histogram
        macd_chart.add_trace(go.Bar(
            x=df['date'],
            y=df['MACD_histogram'],
            name='Histogram'
        ))
        
        macd_chart.update_layout(
            title='MACD',
            xaxis_title='Date',
            template='plotly_white',
            height=300
        )
    
    indicator_charts.update_layout(
        title='Technical Indicators',
        xaxis_title='Date',
        template='plotly_white',
        height=300
    )
    
    return price_chart, indicator_charts, macd_chart if 'MACD_line' in df.columns else None

def predict_price_trend(df, column='4. close', days=30):
    """
    Predict price trend using linear regression.
    
    Args:
        df: DataFrame with price data
        column: Column name for close price
        days: Number of days to predict
        
    Returns:
        tuple: (forecasted_values, confidence)
    """
    # Prepare data
    data = df[-60:].copy()  # Use last 60 days for prediction
    data = data.reset_index(drop=True)
    
    # Create features (just using index as feature for simple linear regression)
    X = np.array(range(len(data))).reshape(-1, 1)
    y = data[column].values
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate future dates for prediction
    future_X = np.array(range(len(data), len(data) + days)).reshape(-1, 1)
    
    # Predict
    predictions = model.predict(future_X)
    
    # Calculate confidence (using R² of model)
    confidence = model.score(X, y)
    
    return predictions, confidence

def create_forecast_chart(df, predictions, ticker, column='4. close', days=30):
    """
    Create forecast chart with predictions.
    
    Args:
        df: DataFrame with price data
        predictions: Array of predicted values
        ticker: Stock ticker symbol
        column: Column name for close price
        days: Number of days that were predicted
        
    Returns:
        Plotly figure
    """
    # Create a date range for predictions
    last_date = pd.to_datetime(df['date'].iloc[-1])
    date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
    
    # Create chart
    forecast_chart = go.Figure()
    
    # Add historical data
    forecast_chart.add_trace(go.Scatter(
        x=df['date'],
        y=df[column],
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add predictions
    forecast_chart.add_trace(go.Scatter(
        x=date_range,
        y=predictions,
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    forecast_chart.update_layout(
        title=f'{ticker} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        height=400
    )
    
    return forecast_chart

def generate_strategy_recommendations(df):
    """
    Generate trading strategy recommendations based on technical analysis.
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        dict: Strategy recommendations
    """
    recommendations = {
        'short_term': {},
        'medium_term': {},
        'long_term': {}
    }
    
    # Get the latest data
    latest = df.iloc[-1]
    
    # Short-term strategy (1-5 days)
    if 'RSI_14' in df.columns and 'MACD_line' in df.columns:
        if latest['RSI_14'] < 30 and latest['MACD_line'] > latest['MACD_signal']:
            recommendations['short_term']['action'] = 'buy'
            recommendations['short_term']['reason'] = 'RSI indicates oversold conditions with positive MACD momentum'
        elif latest['RSI_14'] > 70 and latest['MACD_line'] < latest['MACD_signal']:
            recommendations['short_term']['action'] = 'sell'
            recommendations['short_term']['reason'] = 'RSI indicates overbought conditions with weakening momentum'
        else:
            recommendations['short_term']['action'] = 'hold'
            recommendations['short_term']['reason'] = 'Mixed signals, no clear direction'
    
    # Medium-term strategy (1-3 weeks)
    if 'trend' in df.columns:
        if latest['trend'] == 'bullish':
            if 'golden_cross' in df.columns and latest['golden_cross']:
                recommendations['medium_term']['action'] = 'buy'
                recommendations['medium_term']['reason'] = 'Golden cross suggests starting uptrend'
            else:
                recommendations['medium_term']['action'] = 'hold'
                recommendations['medium_term']['reason'] = 'Established uptrend, maintain positions'
        elif latest['trend'] == 'bearish':
            if 'death_cross' in df.columns and latest['death_cross']:
                recommendations['medium_term']['action'] = 'sell'
                recommendations['medium_term']['reason'] = 'Death cross suggests starting downtrend'
            else:
                recommendations['medium_term']['action'] = 'reduce'
                recommendations['medium_term']['reason'] = 'Established downtrend, consider reducing positions'
    
    # Long-term strategy (months)
    if 'MA50_4. close' in df.columns and 'MA200_4. close' in df.columns:
        if latest['MA50_4. close'] > latest['MA200_4. close']:
            recommendations['long_term']['action'] = 'accumulate'
            recommendations['long_term']['reason'] = 'Long-term uptrend based on 50-200 day moving averages'
        else:
            recommendations['long_term']['action'] = 'reduce'
            recommendations['long_term']['reason'] = 'Long-term downtrend based on 50-200 day moving averages'
    else:
        # Fallback to overall trend if we don't have 200-day MA
        price_5d_change = df['4. close'].pct_change(5).iloc[-1] * 100
        price_20d_change = df['4. close'].pct_change(20).iloc[-1] * 100
        
        if price_5d_change > 0 and price_20d_change > 0:
            recommendations['long_term']['action'] = 'hold'
            recommendations['long_term']['reason'] = f'Positive momentum: {price_20d_change:.1f}% over 20 days'
        elif price_20d_change < -5:
            recommendations['long_term']['action'] = 'reduce'
            recommendations['long_term']['reason'] = f'Negative momentum: {price_20d_change:.1f}% over 20 days'
        else:
            recommendations['long_term']['action'] = 'hold'
            recommendations['long_term']['reason'] = 'No clear long-term direction'
    
    return recommendations
