import pandas as pd
import numpy as np


def pro(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Projection Oscillator (PO), which measures the angle or slope of
    price movement to identify trend strength and direction. It combines linear
    regression slope with volatility normalization.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 10.
            - smooth_period (int): Smoothing period for the oscillator. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Projection Oscillator series and a list of column names.

    Calculation Steps:
    1. Calculate Linear Regression Slope:
       Calculate the slope of the closing prices over the period.
    2. Normalize Slope:
       Normalized Slope = Slope / Standard Deviation (over period)
    3. Apply Smoothing:
       PO = SMA(Normalized Slope, smooth_period) * 100

    Interpretation:
    - Positive PO: Upward trend, bullish.
    - Negative PO: Downward trend, bearish.
    - PO near 0: No clear trend, sideways.
    - High absolute PO: Strong trend.
    - Low absolute PO: Weak trend or consolidation.

    Use Cases:
    - Trend strength measurement.
    - Direction identification.
    - Divergence detection.
    - Entry/exit signals (crossovers of zero line).
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 10))
    smooth_period = int(parameters.get('smooth_period', 3))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling linear regression slope
    def calculate_slope(window):
        if len(window) < 2:
            return np.nan
        x = np.arange(len(window))
        y = window.values
        # Simple linear regression: slope = covariance(x,y) / variance(x)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    slopes = close.rolling(window=period).apply(calculate_slope, raw=False)
    
    # Calculate rolling standard deviation for normalization
    std_dev = close.rolling(window=period).std()
    
    # Normalize slope by standard deviation
    normalized_slope = slopes / std_dev
    
    # Apply smoothing
    po_values = normalized_slope.rolling(window=smooth_period).mean()
    
    # Convert to percentage-like scale
    po_values = po_values * 100
    
    po_values.name = f'PO_{period}_{smooth_period}'
    columns_list = [po_values.name]
    return po_values, columns_list


def strategy_pro(
    data: pd.DataFrame,
    parameters: dict = None,
    config = None,
    trading_type: str = 'long',
    day1_position: str = 'none',
    risk_free_rate: float = 0.0,
    long_entry_pct_cash: float = 1.0,
    short_entry_pct_cash: float = 1.0
) -> tuple:
    """
    PRO (Projection Oscillator) - Zero Line Crossover Strategy
    
    LOGIC: Buy when PO crosses above zero (bullish trend),
           sell when PO crosses below zero (bearish trend).
    WHY: PO measures slope of price movement normalized by volatility.
         Positive = uptrend, negative = downtrend.
    BEST MARKETS: Trending markets. Stocks, forex, futures.
    TIMEFRAME: Daily charts. 10-period with 3-period smoothing is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 10), 'smooth_period' (default 3),
                    'upper' (default 50), 'lower' (default -50)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_band_trade_strategies import run_band_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    period = int(parameters.get('period', 10))
    smooth_period = int(parameters.get('smooth_period', 3))
    upper = float(parameters.get('upper', 50))
    lower = float(parameters.get('lower', -50))
    price_col = 'Close'
    indicator_col = f'PO_{period}_{smooth_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='pro',
        parameters={"period": period, "smooth_period": smooth_period},
        figure=False
    )
    
    data['upper'] = upper
    data['lower'] = lower
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
