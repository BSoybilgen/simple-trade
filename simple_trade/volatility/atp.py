import pandas as pd


def atp(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Average True Range Percent (atp), which expresses the Average True Range
    as a percentage of the closing price. This normalization allows for comparison of volatility
    across different assets and price levels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for ATR calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ATP series and a list of column names.

    Calculation Steps:
    1. Calculate the True Range (TR) for each period:
       TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    2. Calculate the Average True Range (ATR):
       Smoothed average of TR over the specified window (typically using Wilder's smoothing).
    3. Convert ATR to percentage of closing price:
       ATP = (ATR / Close) * 100

    Interpretation:
    - Low ATP (<2%): Low volatility, stable price movements.
    - Medium ATP (2-5%): Normal volatility, typical market conditions.
    - High ATP (5-10%): Elevated volatility, increased price swings.
    - Very High ATP (>10%): Extreme volatility, highly unstable market.

    Use Cases:
    - Cross-asset comparison: Compare volatility across assets with different prices.
    - Position sizing: Normalize position sizes across different assets based on relative volatility.
    - Stop-loss placement: Set percentage-based stops using ATP multiples.
    - Volatility screening: Screen for low or high volatility assets.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 14))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low  # Current high - current low
    tr2 = (high - prev_close).abs()  # Current high - previous close
    tr3 = (low - prev_close).abs()  # Current low - previous close
    
    # True Range is the maximum of the three calculations
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using Wilder's smoothing method
    atr_values = pd.Series(index=close.index, dtype=float)
    
    # First ATR value is simple average of first 'window' TRs
    first_atr = tr.iloc[:window].mean()
    
    # Use the first value to start the smoothing process
    atr_values.iloc[window-1] = first_atr
    
    # Apply Wilder's smoothing method for the rest of the values
    for i in range(window, len(close)):
        atr_values.iloc[i] = ((atr_values.iloc[i-1] * (window-1)) + tr.iloc[i]) / window
    
    # Convert ATR to percentage of closing price
    atp_values = (atr_values / close) * 100
    
    atp_values.name = f'ATP_{window}'
    columns_list = [atp_values.name]
    return atp_values, columns_list


def strategy_atp(
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
    atp (Average True Range Percent) - Volatility Threshold Strategy
    
    LOGIC: Buy when atp drops below lower threshold (low volatility squeeze),
           sell when rises above upper threshold (high volatility).
    WHY: atp normalizes ATR as percentage of price. Low atp indicates
         consolidation (potential breakout setup), high atp indicates overextension.
    BEST MARKETS: All markets. Good for identifying volatility regimes.
                  Use low ATP for breakout setups, high ATP for mean reversion.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper' (default 5.0), 'lower' (default 2.0)
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
    
    window = int(parameters.get('window', 14))
    upper = float(parameters.get('upper', 5.0))
    lower = float(parameters.get('lower', 2.0))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='atp',
        parameters={"window": window},
        figure=False
    )
    
    data['upper'] = upper
    data['lower'] = lower
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=f'ATP_{window}',
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
    
    indicator_cols_to_plot = [f'ATP_{window}', 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
