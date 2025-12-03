import pandas as pd


def dem(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Double Exponential Moving Average (dem).
    dem reduces the lag of traditional EMAs by subtracting the EMA of the EMA from the doubled EMA.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the DEM series and a list of column names.

    The dem is calculated as follows:

    1. Calculate EMA1:
       EMA1 = EMA(Close, window)

    2. Calculate EMA2:
       EMA2 = EMA(EMA1, window)

    3. Calculate dem:
       dem = 2 * EMA1 - EMA2

    Interpretation:
    - dem is faster and more responsive than a standard EMA.
    - It aims to reduce the inherent lag of moving averages.

    Use Cases:
    - Trend Following: Identifying trends earlier than SMA/EMA.
    - Crossovers: DEMA crossings can signal entries/exits faster.
    - Support/Resistance: Can act as dynamic support/resistance levels.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))

    series = df[close_col]

    ema1 = series.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()

    dema_series = 2 * ema1 - ema2
    dema_series.name = f'DEM_{window}'

    columns_list = [dema_series.name]
    return dema_series, columns_list


def strategy_dem(
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
    dem (Double EMA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast dem crosses above slow dem, sell when crosses below.
    WHY: dem reduces lag by subtracting EMA of EMA from doubled EMA.
         Faster and more responsive than standard EMA.
    BEST MARKETS: Trending markets where quick response is needed. Stocks, forex.
                  Good for swing trading with reduced lag.
    TIMEFRAME: All timeframes. Particularly effective on daily charts.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 12), 'long_window' (default 26)
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
    
    short_window = int(parameters.get('short_window', 12))
    long_window = int(parameters.get('long_window', 26))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'DEM_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='dem',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'DEM_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='dem',
        parameters={"window": long_window},
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
