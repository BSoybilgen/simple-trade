import pandas as pd


def stc(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Schaff Trend Cycle (STC), an indicator developed by Doug Schaff.
    It is a product of combining the MACD with the Stochastic Oscillator to identify faster, 
    more accurate trends with less lag.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window_fast (int): Fast EMA length used in MACD. Default is 23.
            - window_slow (int): Slow EMA length used in MACD. Default is 50.
            - cycle (int): Look-back window for stochastic calculations. Default is 10.
            - smooth (int): EMA smoothing factor for the cycle. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the STC series and a list of column names.

    The Schaff Trend Cycle is calculated in multiple steps:

    1. Calculate MACD Line:
       MACD = EMA(Close, window_fast) - EMA(Close, window_slow)

    2. Calculate Stochastic of MACD (%K):
       %K = (MACD - Min(MACD)) / (Max(MACD) - Min(MACD)) * 100
       (Min/Max over 'cycle' period)

    3. Smooth %K (First Smoothing):
       Smoothed %K = EMA(%K, smooth)

    4. Calculate Stochastic of Smoothed %K (%D-ish):
       %D = (Smoothed %K - Min) / (Max - Min) * 100

    5. Smooth %D (Second Smoothing - Final STC):
       STC = EMA(%D, smooth)

    Interpretation:
    - Range: 0 to 100.
    - Overbought: Values above 75 indicate overbought conditions.
    - Oversold: Values below 25 indicate oversold conditions.
    - Trend: Rising STC suggests an uptrend; falling STC suggests a downtrend.

    Use Cases:
    - Early Trend Detection: STC is designed to identify trends earlier than MACD.
    - Cycle Tops and Bottoms: Identifying cyclical turning points.
    - Filters: Using STC direction to filter trades from other strategies.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window_fast = int(parameters.get('window_fast', 23))
    window_slow = int(parameters.get('window_slow', 50))
    cycle = int(parameters.get('cycle', 10))
    smooth = int(parameters.get('smooth', 3))
    close_col = columns.get('close_col', 'Close')

    series = pd.to_numeric(df[close_col], errors='coerce')
    ema_fast = series.ewm(span=window_fast, adjust=False).mean()
    ema_slow = series.ewm(span=window_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow

    macd_min = macd_line.rolling(window=cycle, min_periods=1).min()
    macd_max = macd_line.rolling(window=cycle, min_periods=1).max()
    macd_range = macd_max - macd_min
    macd_range = macd_range.replace(0, float('nan'))

    percent_k = ((macd_line - macd_min) / macd_range) * 100
    percent_k = percent_k.fillna(0).clip(lower=0, upper=100).astype(float)

    smoothed_k = percent_k.ewm(span=smooth, adjust=False).mean()

    smoothed_min = smoothed_k.rolling(window=cycle, min_periods=1).min()
    smoothed_max = smoothed_k.rolling(window=cycle, min_periods=1).max()
    smoothed_range = smoothed_max - smoothed_min
    smoothed_range = smoothed_range.replace(0, float('nan'))

    stc_values = ((smoothed_k - smoothed_min) / smoothed_range) * 100
    stc_values = stc_values.fillna(0).clip(lower=0, upper=100).astype(float)
    stc_values.name = f'STC_{window_fast}_{window_slow}_{cycle}'

    columns_list = [stc_values.name]
    return stc_values, columns_list


def strategy_stc(
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
    STC (Schaff Trend Cycle) - Mean Reversion Strategy
    
    LOGIC: Buy when STC drops below 25 (oversold), sell when above 75 (overbought).
    WHY: STC combines MACD with stochastic, identifying trends faster with less lag.
         Designed for early trend detection and cycle identification.
    BEST MARKETS: Trending and cyclical markets. Forex, stocks, futures.
                  Good for identifying cycle tops and bottoms.
    TIMEFRAME: All timeframes. 23/50/10 settings are common for daily charts.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window_fast' (default 23), 'window_slow' (default 50),
                   'cycle' (default 10), 'smooth' (default 3),
                   'upper' (default 75), 'lower' (default 25)
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
    
    window_fast = int(parameters.get('window_fast', 23))
    window_slow = int(parameters.get('window_slow', 50))
    cycle = int(parameters.get('cycle', 10))
    smooth = int(parameters.get('smooth', 3))
    upper = int(parameters.get('upper', 75))
    lower = int(parameters.get('lower', 25))
    
    indicator_params = {
        "window_fast": window_fast,
        "window_slow": window_slow,
        "cycle": cycle,
        "smooth": smooth
    }
    indicator_col = f'STC_{window_fast}_{window_slow}_{cycle}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='stc',
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
