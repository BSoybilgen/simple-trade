import pandas as pd


def zsc(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Z-Score (zsc), a statistical measure that indicates how many standard
    deviations a data point is from the mean.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Z-Score series and a list of column names.

    Calculation Steps:
    1. Calculate rolling mean over the window.
    2. Calculate rolling standard deviation over the window.
    3. Z-Score = (Price - Mean) / Standard Deviation

    Interpretation:
    - Z-Score > 2: Price is significantly above average (overbought).
    - Z-Score > 0: Price is above average.
    - Z-Score = 0: Price is at the average.
    - Z-Score < 0: Price is below average.
    - Z-Score < -2: Price is significantly below average (oversold).

    Use Cases:
    - Mean reversion strategies.
    - Overbought/oversold detection.
    - Statistical arbitrage.
    - Pairs trading.
    - Outlier detection.

    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    window = int(window_param if window_param is not None else 20)
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling mean and standard deviation
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()
    
    # Calculate Z-Score
    zsc_values = (close - rolling_mean) / rolling_std
    
    zsc_values.name = f'ZSC_{window}'
    columns_list = [zsc_values.name]
    return zsc_values, columns_list


def strategy_zsc(
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
    zsc (Z-Score) - Mean Reversion Strategy
    
    LOGIC: Buy when zsc drops below lower threshold (oversold),
           sell when zsc rises above upper threshold (overbought).
    WHY: zsc measures how far price has deviated from its mean.
         Extreme zsc values indicate potential mean reversion opportunities.
    BEST MARKETS: Range-bound markets. Classic mean reversion strategy.
    TIMEFRAME: Daily charts. 20-period is standard.
    NOTE: Uses fixed thresholds (typically +/- 2 standard deviations).
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'upper_threshold' (default 2.0),
                    'lower_threshold' (default -2.0)
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    window = int(window_param if window_param is not None else 20)
    upper_threshold = float(parameters.get('upper_threshold', 2.0))
    lower_threshold = float(parameters.get('lower_threshold', -2.0))
    price_col = 'Close'
    indicator_col = f'ZSC_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='zsc',
        parameters={"window": window},
        figure=False
    )
    
    # Use fixed thresholds for Z-Score
    data['upper'] = upper_threshold
    data['lower'] = lower_threshold
    
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
