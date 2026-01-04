import pandas as pd


def ult(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ultimate Oscillator (ult), developed by Larry Williams.
    It combines short, medium, and long-term buying pressure to reduce volatility 
    and false signals associated with single-timeframe oscillators.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - short_window (int): Short-term period length. Default is 7.
            - medium_window (int): Medium-term period length. Default is 14.
            - long_window (int): Long-term period length. Default is 28.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.

    Returns:
        tuple: A tuple containing the Ultimate Oscillator series and a list of column names.

    The Ultimate Oscillator is calculated as follows:

    1. Calculate Buying Pressure (BP):
       BP = Close - Min(Low, Prev Close)

    2. Calculate True Range (TR):
       TR = Max(High, Prev Close) - Min(Low, Prev Close)

    3. Calculate Averages for each window:
       Avg7 = Sum(BP, 7) / Sum(TR, 7)
       Avg14 = Sum(BP, 14) / Sum(TR, 14)
       Avg28 = Sum(BP, 28) / Sum(TR, 28)

    4. Calculate Ultimate Oscillator:
       ult = 100 * ((4 * Avg7) + (2 * Avg14) + Avg28) / 7

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 70 indicate potential overbought conditions.
    - Oversold: Values below 30 indicate potential oversold conditions.

    Use Cases:
    - Divergence: Bullish divergence (Price lower low, Oscillator higher low) in oversold territory is a strong buy signal.
    - Breakouts: Can be used to confirm trend reversals.
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

    medium_window_param = parameters.get('medium_window')
    medium_period_param = parameters.get('medium_period')
    if medium_window_param is None and medium_period_param is not None:
        medium_window_param = medium_period_param
    elif medium_window_param is not None and medium_period_param is not None:
        if int(medium_window_param) != int(medium_period_param):
            raise ValueError("Provide either 'medium_window' or 'medium_period' (aliases) with the same value if both are set.")

    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    short_window = int(short_window_param if short_window_param is not None else 7)
    medium_window = int(medium_window_param if medium_window_param is not None else 14)
    long_window = int(long_window_param if long_window_param is not None else 28)

    close_col = columns.get('close_col', 'Close')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')

    close = df[close_col]
    high = df[high_col]
    low = df[low_col]
    prev_close = close.shift(1)

    min_low_close = pd.concat([low, prev_close], axis=1).min(axis=1)
    max_high_close = pd.concat([high, prev_close], axis=1).max(axis=1)

    buying_pressure = close - min_low_close
    true_range = max_high_close - min_low_close

    def _avg_bp_tr(window: int):
        bp_sum = buying_pressure.rolling(window=window, min_periods=window).sum()
        tr_sum = true_range.rolling(window=window, min_periods=window).sum()
        return bp_sum / tr_sum.where(tr_sum != 0)

    avg_short = _avg_bp_tr(short_window)
    avg_medium = _avg_bp_tr(medium_window)
    avg_long = _avg_bp_tr(long_window)

    ultimate = 100 * ((4 * avg_short) + (2 * avg_medium) + avg_long) / 7
    ultimate.name = f'ULT_{short_window}_{medium_window}_{long_window}'

    columns_list = [ultimate.name]
    return ultimate, columns_list


def strategy_ult(
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
    ult (Ultimate Oscillator) - Mean Reversion Strategy
    
    LOGIC: Buy when ult drops below 30 (oversold), sell when above 70 (overbought).
    WHY: ult combines 3 timeframes to reduce volatility and false signals.
         Multi-timeframe approach provides more reliable overbought/oversold readings.
    BEST MARKETS: Range-bound markets. Stocks, forex, commodities. Particularly good
                  for divergence trading in oversold/overbought zones.
    TIMEFRAME: Daily charts. 7/14/28 periods are standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 7), 'medium_window' (default 14),
                   'long_window' (default 28), 'upper' (default 70), 'lower' (default 30)
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

    medium_window_param = parameters.get('medium_window')
    medium_period_param = parameters.get('medium_period')
    if medium_window_param is None and medium_period_param is not None:
        medium_window_param = medium_period_param
    elif medium_window_param is not None and medium_period_param is not None:
        if int(medium_window_param) != int(medium_period_param):
            raise ValueError("Provide either 'medium_window' or 'medium_period' (aliases) with the same value if both are set.")

    long_window_param = parameters.get('long_window')
    long_period_param = parameters.get('long_period')
    if long_window_param is None and long_period_param is not None:
        long_window_param = long_period_param
    elif long_window_param is not None and long_period_param is not None:
        if int(long_window_param) != int(long_period_param):
            raise ValueError("Provide either 'long_window' or 'long_period' (aliases) with the same value if both are set.")

    short_window = int(short_window_param if short_window_param is not None else 7)
    medium_window = int(medium_window_param if medium_window_param is not None else 14)
    long_window = int(long_window_param if long_window_param is not None else 28)
    upper = int(parameters.get('upper', 70))
    lower = int(parameters.get('lower', 30))
    
    indicator_params = {
        "short_window": short_window,
        "medium_window": medium_window,
        "long_window": long_window
    }
    indicator_col = f'ULT_{short_window}_{medium_window}_{long_window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='ult',
        parameters=indicator_params,
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
