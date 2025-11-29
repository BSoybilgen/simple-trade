import pandas as pd


def nat(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Normalized Average True Range (NATR), which expresses ATR as a
    percentage of the closing price, similar to ATRP but with additional normalization
    commonly used in technical analysis platforms.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for ATR calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the NATR series and a list of column names.

    Calculation Steps:
    1. Calculate ATR:
       Using Wilder's smoothing method over the specified window.
    2. Normalize:
       NATR = (ATR / Close) * 100

    Interpretation:
    - Low NATR (<2%): Low relative volatility.
    - Medium NATR (2-5%): Normal volatility.
    - High NATR (>5%): High relative volatility.

    Use Cases:
    - Cross-asset volatility comparison.
    - Normalized position sizing.
    - Historical volatility analysis.
    - Percentage-based stop-loss placement.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    window = int(parameters.get('window', 14))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using Wilder's smoothing
    atr_values = pd.Series(index=close.index, dtype=float)
    first_atr = tr.iloc[:window].mean()
    atr_values.iloc[window-1] = first_atr
    
    for i in range(window, len(close)):
        atr_values.iloc[i] = ((atr_values.iloc[i-1] * (window-1)) + tr.iloc[i]) / window
    
    # Normalize to percentage
    natr_values = (atr_values / close) * 100
    
    natr_values.name = f'NATR_{window}'
    columns_list = [natr_values.name]
    return natr_values, columns_list


def strategy_nat(
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
    NAT (Normalized ATR) - Volatility Threshold Strategy
    
    LOGIC: Buy when NATR drops below lower threshold (low volatility squeeze),
           sell when rises above upper threshold (high volatility).
    WHY: NATR normalizes ATR as percentage of close price. Allows cross-asset
         volatility comparison and percentage-based stop-loss placement.
    BEST MARKETS: All markets. Good for normalized volatility analysis.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper' (default 5.0),
                    'lower' (default 2.0)
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
    indicator_col = f'NATR_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='nat',
        parameters={"window": window},
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
