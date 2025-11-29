import pandas as pd


def zma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Zero-Lag Moving Average (ZLEMA).
    ZLEMA reduces lag by pre-adjusting the input series with its own momentum
    before applying an EMA. This keeps responsiveness similar to shorter EMAs
    while preserving smoothness.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ZLEMA series and a list of column names.

    The Zero-Lag Moving Average is calculated as follows:

    1. Calculate Lag:
       Lag = Window / 2

    2. Adjust Data for Lag (De-lagged Data):
       Adjusted Price = Price + (Price - Price(Lag periods ago))

    3. Calculate EMA of Adjusted Data:
       ZLEMA = EMA(Adjusted Price, window)

    Interpretation:
    - ZLEMA removes much of the lag associated with trend-following indicators.
    - It tracks prices very closely, making it sensitive to reversals.

    Use Cases:
    - Trend Following: Catching trends earlier than SMA/EMA.
    - Reversal Trading: Sharp turns in ZLEMA can signal reversals.
    - Crossovers: ZLEMA crossing price or another MA.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = max(1, int(parameters.get('window', 20)))
    lag = max(1, window // 2)

    price = df[close_col]
    adjusted_price = price + (price - price.shift(lag))
    zlema_series = adjusted_price.ewm(span=window, adjust=False).mean()
    zlema_series.name = f'ZMA_{window}'

    return zlema_series, [zlema_series.name]


def strategy_zma(
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
    ZMA (Zero-Lag MA) - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast ZLEMA crosses above slow ZLEMA, sell when crosses below.
    WHY: ZLEMA pre-adjusts input with momentum before applying EMA, removing lag.
         Tracks prices very closely while maintaining smoothness.
    BEST MARKETS: Trending markets where quick response is needed. Stocks, forex.
                  Good for catching trends earlier than standard MAs.
    TIMEFRAME: All timeframes. Effective for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 10), 'long_window' (default 20)
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
    
    short_window = int(parameters.get('short_window', 10))
    long_window = int(parameters.get('long_window', 20))
    price_col = 'Close'
    
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'ZMA_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='zma',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'ZMA_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='zma',
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
