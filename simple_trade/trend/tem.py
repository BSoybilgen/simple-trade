import pandas as pd


def tem(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Triple Exponential Moving Average (TEMA).
    TEMA uses triple smoothing to further reduce lag compared to DEMA and EMA.
    It was developed by Patrick Mulloy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TEMA series and a list of column names.

    The TEMA is calculated as follows:

    1. Calculate EMA1:
       EMA1 = EMA(Close, window)

    2. Calculate EMA2:
       EMA2 = EMA(EMA1, window)

    3. Calculate EMA3:
       EMA3 = EMA(EMA2, window)

    4. Calculate TEMA:
       TEMA = 3 * EMA1 - 3 * EMA2 + EMA3

    Interpretation:
    - TEMA reacts extremely quickly to price changes.
    - It eliminates the lag associated with single and double EMAs.

    Use Cases:
    - Scalping/Day Trading: Highly responsive indicator for short-term trades.
    - Trend Confirmation: Fast confirmation of new trends.
    - Crossovers: Using TEMA in place of EMA in MACD or other crossover strategies.
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
    ema3 = ema2.ewm(span=window, adjust=False).mean()

    tema_series = 3 * ema1 - 3 * ema2 + ema3
    tema_series.name = f'TEMA_{window}'

    columns_list = [tema_series.name]
    return tema_series, columns_list


def strategy_tem(
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
    TEM (Triple EMA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast TEMA crosses above slow TEMA, sell when crosses below.
    WHY: TEMA uses triple smoothing to minimize lag while maintaining smoothness.
         Extremely responsive to price changes.
    BEST MARKETS: Fast-moving markets. Scalping, day trading. Stocks, forex, crypto.
                  Best for short-term trend confirmation.
    TIMEFRAME: All timeframes. Particularly effective on lower timeframes.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 9), 'long_window' (default 21)
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
    
    short_window = int(parameters.get('short_window', 9))
    long_window = int(parameters.get('long_window', 21))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'TEMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='tem',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'TEMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='tem',
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
