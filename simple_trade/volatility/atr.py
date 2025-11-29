import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Average True Range (ATR), a volatility indicator that measures market volatility
    by decomposing the entire range of an asset price for a given period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ATR series and a list of column names.

    Calculation Steps:
    1. Calculate the True Range (TR) for each period:
       TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    2. Calculate Average True Range (ATR):
       - First value: Simple average of TR over the window.
       - Subsequent values: ((Prior ATR * (window-1)) + Current TR) / window (Wilder's Smoothing).

    Interpretation:
    - Higher ATR values indicate higher volatility.
    - Lower ATR values indicate lower volatility.
    - Does not indicate trend direction, only magnitude of price movement.

    Use Cases:
    - Volatility measurement: Gauging market activity.
    - Position sizing: Adjusting trade size inversely to volatility.
    - Stop-loss placement: Setting stops based on a multiple of ATR (e.g., 2 * ATR).
    - Breakout identification: Rising ATR often accompanies breakouts.
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
    # First ATR value is the simple average of TR over the window
    atr_values = pd.Series(index=close.index)
    
    # First ATR value is simple average of first 'window' TRs
    first_atr = tr.iloc[:window].mean()
    
    # Use the first value to start the smoothing process
    atr_values.iloc[window-1] = first_atr
    
    # Apply Wilder's smoothing method for the rest of the values
    for i in range(window, len(close)):
        atr_values.iloc[i] = ((atr_values.iloc[i-1] * (window-1)) + tr.iloc[i]) / window

    atr_values.name = f'ATR_{window}'
    columns_list = [atr_values.name]
    return atr_values, columns_list


def strategy_atr(
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
    ATR (Average True Range) - Volatility Threshold Strategy
    
    LOGIC: Buy when ATR drops below lower percentile (low volatility squeeze),
           sell when rises above upper percentile (high volatility).
    WHY: ATR measures market volatility. Low ATR indicates consolidation and
         potential breakout setup. High ATR indicates strong moves or overextension.
    BEST MARKETS: All markets. Good for volatility-based position sizing.
                  Combine with trend indicators for directional trades.
    TIMEFRAME: Daily charts. 14-period is standard.
    NOTE: Uses rolling percentile bands since ATR is in price units.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper_pct' (default 80),
                    'lower_pct' (default 20), 'lookback' (default 100)
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
    upper_pct = float(parameters.get('upper_pct', 80))
    lower_pct = float(parameters.get('lower_pct', 20))
    lookback = int(parameters.get('lookback', 100))
    price_col = 'Close'
    indicator_col = f'ATR_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='atr',
        parameters={"window": window},
        figure=False
    )
    
    # Calculate rolling percentile bands for ATR
    data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(upper_pct / 100)
    data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(lower_pct / 100)
    
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
