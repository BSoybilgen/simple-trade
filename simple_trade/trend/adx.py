import pandas as pd


def adx(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Average Directional Index (adx) along with the Positive
    Directional Indicator (+DI) and Negative Directional Indicator (-DI).
    The adx is a technical indicator used to measure the strength of a trend.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the adx calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the adx DataFrame (adx, +DI, -DI) and a list of column names.

    Calculation Steps:
    1. Calculate True Range (TR):
       TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))

    2. Calculate Directional Movements (+DM, -DM):
       +DM = Current High - Previous High (if > 0 and > -DM, else 0)
       -DM = Previous Low - Current Low (if > 0 and > +DM, else 0)

    3. Smooth TR, +DM, and -DM (using rolling mean/RMA):
       ATR = Smooth(TR)
       +Smoothed = Smooth(+DM)
       -Smoothed = Smooth(-DM)

    4. Calculate Directional Indicators (+DI, -DI):
       +DI = 100 * (+Smoothed / ATR)
       -DI = 100 * (-Smoothed / ATR)

    5. Calculate Directional Index (DX):
       DX = 100 * Abs(+DI - -DI) / (+DI + -DI)

    6. Calculate adx:
       adx = Smooth(DX)

    Interpretation:
    - adx > 25: Strong trend.
    - adx < 20: Weak trend or non-trending market.
    - +DI > -DI: Bullish trend.
    - -DI > +DI: Bearish trend.

    Use Cases:
    - Identifying trend strength: The adx can be used to determine whether a
      trend is strong or weak.
    - Identifying trend direction: The +DI and -DI can be used to determine
      the direction of the trend.
    - Generating buy and sell signals: Crossovers of +DI and -DI.
    - Filtering: Using adx to filter out trades in sideways markets.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 14))

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_ = dx.rolling(window=window).mean()
    df_adx = pd.DataFrame({
        f'ADX_{window}': adx_,
        f'+DI_{window}': plus_di,
        f'-DI_{window}': minus_di
    })
    df_adx.index = close.index

    columns = list(df_adx.columns)

    return df_adx, columns


def strategy_adx(
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
    adx (Average Directional Index) - DI Crossover Strategy
    
    LOGIC: Trade the crossovers between +DI and -DI to capture directional shifts.
    WHY: adx package produces +DI/-DI which directly encode bullish vs bearish pressure.
         Using their crossover keeps the trading logic consistent with the plotted values.
    BEST MARKETS: Any market. adx provides additional context on trend strength if you
                  choose to inspect it, but signals rely purely on DI crossovers now.
    TIMEFRAME: Daily or weekly. Window 14 remains the default.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14)
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
    
    window = int(parameters.get('window', 14))
    adx_col = f'ADX_{window}'
    price_col = 'Close'
    
    # Compute ADX indicator
    data, _, _ = compute_indicator(
        data=data,
        indicator='adx',
        parameters={"window": window},
        figure=False
    )
    
    plus_di_col = f'+DI_{window}'
    minus_di_col = f'-DI_{window}'
    
    # ADX Strategy: trade the +DI / -DI crossover directly
    data_filtered = data.copy()
    short_window_indicator = plus_di_col
    long_window_indicator = minus_di_col
    
    results, portfolio = run_cross_trade(
        data=data_filtered,
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
    
    indicator_cols_to_plot = [adx_col, plus_di_col, minus_di_col]
    
    return results, portfolio, indicator_cols_to_plot, data