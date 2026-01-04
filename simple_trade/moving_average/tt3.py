import pandas as pd


def tt3(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the T3 Moving Average (tt3).
    tt3 is a smoother, less laggy moving average developed by Tim Tillson.
    It uses exponential moving averages and a volume factor to control
    responsiveness.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 5.
            - v_factor (float): The volume factor (0 to 1). Default is 0.7.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TT3 series and a list of column names.

    The tt3 is calculated as follows:

    1. Calculate GD (Generalized DEMA):
       GD = EMA(Price) * (1 + v_factor) - EMA(EMA(Price)) * v_factor

    2. Apply GD three times:
       tt3 = GD(GD(GD(Price)))

    Interpretation:
    - tt3 is extremely smooth while maintaining low lag.
    - Higher v_factor (closer to 1) gives faster response, lower values give smoother output.

    Use Cases:
    - Trend Following: Excellent for identifying smooth trends with minimal noise.
    - Crossovers: Using tt3 in place of EMA for smoother signals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    window = int(window_param if window_param is not None else 5)
    v_factor = float(parameters.get('v_factor', 0.7))

    series = df[close_col].copy()

    c1 = -v_factor ** 3
    c2 = 3 * v_factor ** 2 + 3 * v_factor ** 3
    c3 = -6 * v_factor ** 2 - 3 * v_factor - 3 * v_factor ** 3
    c4 = 1 + 3 * v_factor + v_factor ** 3 + 3 * v_factor ** 2

    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    ema4 = ema3.ewm(span=window, adjust=False).mean()
    ema5 = ema4.ewm(span=window, adjust=False).mean()
    ema6 = ema5.ewm(span=window, adjust=False).mean()

    tt3_series = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
    tt3_series.name = f'TT3_{window}'

    columns_list = [tt3_series.name]
    return tt3_series, columns_list


def strategy_tt3(
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
    tt3 (T3 Moving Average) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast tt3 crosses above slow tt3, sell when crosses below.
    WHY: tt3 uses multiple EMAs with volume factor for extremely smooth output.
         Minimal lag while maintaining smoothness. Developed by Tim Tillson.
    BEST MARKETS: All markets. Particularly effective for trend following
                  where smooth signals are needed. Stocks, forex, crypto.
    TIMEFRAME: All timeframes. 5-period is standard, adjust v_factor for responsiveness.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 5), 'long_window' (default 10)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_cross_trade_strategies import run_cross_trade
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

    short_window = int(short_window_param if short_window_param is not None else 5)
    long_window = int(long_window_param if long_window_param is not None else 10)
    v_factor = float(parameters.get('v_factor', 0.7))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'TT3_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='tt3',
            parameters={"window": short_window, "v_factor": v_factor},
            figure=False
        )
    
    long_window_indicator = f'TT3_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='tt3',
        parameters={"window": long_window, "v_factor": v_factor},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data
