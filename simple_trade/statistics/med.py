import pandas as pd


def med(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Rolling Median (med), a statistical measure of central tendency that
    represents the middle value in a sorted dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Median series and a list of column names.

    Calculation Steps:
    1. Sort values in the rolling window.
    2. Select the middle value (or average of two middle values for even counts).

    Interpretation:
    - Median is more robust to outliers than mean.
    - Price above median: Bullish bias.
    - Price below median: Bearish bias.
    - Median crossovers can signal trend changes.

    Use Cases:
    - Trend identification (robust alternative to SMA).
    - Support/resistance levels.
    - Outlier-resistant price smoothing.
    - Mean reversion strategies.
    - Noise filtering in volatile markets.

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
    
    # Calculate rolling median
    med_values = close.rolling(window=window).median()
    
    med_values.name = f'MED_{window}'
    columns_list = [med_values.name]
    return med_values, columns_list


def strategy_med(
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
    med (Rolling Median) - Price Crossover Strategy
    
    LOGIC: Buy when price crosses above med (bullish),
           sell when price crosses below med (bearish).
    WHY: Median is robust to outliers and provides reliable trend signals.
         Price above median indicates bullish momentum.
    BEST MARKETS: All markets. Good for trend-following strategies.
    TIMEFRAME: Daily charts. 20-period is standard.
    NOTE: Uses price vs median comparison for signals.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20)
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
    price_col = 'Close'
    indicator_col = f'MED_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='med',
        parameters={"window": window},
        figure=False
    )
    
    # Use median as both upper and lower band (crossover strategy)
    data['upper'] = data[indicator_col]
    data['lower'] = data[indicator_col]
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=price_col,
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
    
    indicator_cols_to_plot = [indicator_col, price_col]
    
    return results, portfolio, indicator_cols_to_plot, data
