import pandas as pd


def vra(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volatility Ratio (vra), which compares short-term volatility to
    long-term volatility to identify changes in volatility regimes. A ratio above 1
    indicates increasing volatility, while below 1 indicates decreasing volatility.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_period (int): Short-term volatility period. Default is 5.
            - long_period (int): Long-term volatility period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Volatility Ratio series and a list of column names.

    Calculation Steps:
    1. Calculate Short-Term Volatility:
       Short Vol = Standard Deviation(Close, short_period)
    2. Calculate Long-Term Volatility:
       Long Vol = Standard Deviation(Close, long_period)
    3. Calculate Ratio:
       VRA = Short Vol / Long Vol

    Interpretation:
    - VRA > 1: Short-term volatility is expanding relative to long-term.
    - VRA < 1: Short-term volatility is contracting (consolidation).
    - Breakout Signal: A crossing above 1 often signals a price breakout from range.

    Use Cases:
    - Volatility regime detection.
    - Strategy switching (Breakout vs. Mean Reversion).
    - Breakout confirmation.

    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    short_window_param = parameters.get('short_window')
    short_period_param = parameters.get('short_period')
    if short_window_param is None and short_period_param is not None:
        short_window_param = short_period_param
    elif short_window_param is not None and short_period_param is not None:
        if int(short_window_param) != int(short_period_param):
            raise ValueError("Provide either 'short_window' or 'short_period' (aliases) with the same value if both are set.")

    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    short_period = int(short_window_param if short_window_param is not None else 5)
    long_period = int(long_window_param if long_window_param is not None else 20)
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate short-term and long-term standard deviation
    short_std = close.rolling(window=short_period).std()
    long_std = close.rolling(window=long_period).std()
    
    # Calculate volatility ratio, keeping NaNs until both windows are populated
    vra_values = short_std / long_std
    vra_values = vra_values.where(long_std != 0)
    
    vra_values.name = f'VRA_{short_period}_{long_period}'
    columns_list = [vra_values.name]
    return vra_values, columns_list


def strategy_vra(
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
    vra (Volatility Ratio) - Volatility Regime Strategy
    
    LOGIC: Buy when vra rises above upper threshold (volatility expansion),
           sell when drops below lower threshold (volatility contraction).
    WHY: vra compares short-term to long-term volatility. vra > 1 indicates
         expanding volatility, vra < 1 indicates contracting volatility.
    BEST MARKETS: All markets. Good for volatility regime detection.
    TIMEFRAME: Daily charts. 5/20 periods is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_period' (default 5), 'long_period' (default 20),
                    'upper' (default 1.5), 'lower' (default 0.7)
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
    
    short_window_param = parameters.get('short_window')
    short_period_param = parameters.get('short_period')
    if short_window_param is None and short_period_param is not None:
        short_window_param = short_period_param
    elif short_window_param is not None and short_period_param is not None:
        if int(short_window_param) != int(short_period_param):
            raise ValueError("Provide either 'short_window' or 'short_period' (aliases) with the same value if both are set.")

    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    short_period = int(short_window_param if short_window_param is not None else 5)
    long_period = int(long_window_param if long_window_param is not None else 20)
    upper = float(parameters.get('upper', 1.5))
    lower = float(parameters.get('lower', 0.7))
    price_col = 'Close'
    indicator_col = f'VRA_{short_period}_{long_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='vra',
        parameters={"short_period": short_period, "long_period": long_period},
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
