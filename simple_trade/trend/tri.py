import pandas as pd


def tri(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the TRIX (tri) indicator.
    tri is a momentum oscillator that displays the percent rate of change of a triple
    exponentially smoothed moving average. It oscillates around a zero line and can be
    used to identify overbought/oversold conditions, divergences, and trend direction.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the EMA. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TRI DataFrame and a list of column names.

    The TRI is calculated as follows:

    1. Calculate Single-Smoothed EMA:
       EMA1 = EMA(Close, window)

    2. Calculate Double-Smoothed EMA:
       EMA2 = EMA(EMA1, window)

    3. Calculate Triple-Smoothed EMA:
       EMA3 = EMA(EMA2, window)

    4. Calculate TRI:
       TRI = 100 * (EMA3 - Previous EMA3) / Previous EMA3

    5. Calculate Signal Line:
       Signal = EMA(TRI, 9)

    Interpretation:
    - Zero Line Crossover: Crossing above zero is bullish; below zero is bearish.
    - Signal Line Crossover: TRI crossing above Signal is bullish; below is bearish.
    - Divergences: Price making new highs while TRI fails to do so indicates a potential reversal.

    Use Cases:
    - Momentum Measurement: Gauging the rate of change of the triple smoothed average.
    - Trend Reversals: Identifying early signs of trend shifts via divergences.
    - Filtering: TRI filters out insignificant price movements better than standard rate-of-change.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    window = int(window_param if window_param is not None else 14)

    series = df[close_col]
    # Step 1: Calculate the single-smoothed EMA
    ema1 = series.ewm(span=window, adjust=False).mean()
    
    # Step 2: Calculate the double-smoothed EMA
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    
    # Step 3: Calculate the triple-smoothed EMA
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    # Step 4: Calculate the 1-period percent rate of change of the triple-smoothed EMA
    tri_line = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    
    # Calculate signal line (9-period EMA of TRI)
    signal_line = tri_line.ewm(span=9, adjust=False).mean()
    
    # Create result DataFrame
    df_tri = pd.DataFrame({
        f'TRI_{window}': tri_line,
        f'TRI_SIGNAL_{window}': signal_line
    })
    df_tri.index = series.index

    columns_list = list(df_tri.columns)
    return df_tri, columns_list


def strategy_tri(
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
    tri (TRIX) - Signal Line Crossover Strategy
    
    LOGIC: Buy when tri crosses above signal line, sell when crosses below.
    WHY: tri is a momentum oscillator showing rate of change of triple-smoothed EMA.
         Signal crossovers indicate momentum shifts. Filters out noise well.
    BEST MARKETS: Trending markets. Stocks, forex, commodities. Good for
         identifying momentum shifts and divergences.
    TIMEFRAME: Daily charts. 14-period is standard.
    
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    window = int(window_param if window_param is not None else 14)
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='tri',
        parameters={"window": window},
        figure=False
    )
    
    short_window_indicator = f'TRI_{window}'
    long_window_indicator = f'TRI_SIGNAL_{window}'
    
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
