import pandas as pd


def qst(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Qstick Indicator (qst), a technical indicator developed by Tushar Chande.
    It quantifies the buying and selling pressure by averaging the difference between closing and opening prices.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The rolling window length for averaging. Default is 10.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - open_col (str): The column name for opening prices. Default is 'Open'.

    Returns:
        tuple: A tuple containing the QST series and a list of column names.

    The QST Indicator is calculated as follows:

    1. Calculate Candle Body:
       Body = Close - Open

    2. Calculate Moving Average of Body:
       QST = SMA(Body, window)

    Interpretation:
    - Positive QST: Buying pressure is dominant (Closes > Opens on average).
    - Negative QST: Selling pressure is dominant (Opens > Closes on average).
    - Zero Crossing: Crossing the zero line acts as a signal for trend change.

    Use Cases:
    - Trend Confirmation: Confirming the validity of a trend (e.g., price rising but QST falling is a divergence).
    - Signal Generation: Crossovers of the zero line.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 10))
    close_col = columns.get('close_col', 'Close')
    open_col = columns.get('open_col', 'Open')

    close = df[close_col]
    open_price = df[open_col]
    body = close - open_price

    qst_values = body.rolling(window=window, min_periods=window).mean()
    qst_values.name = f'QST_{window}'

    columns_list = [qst_values.name]
    return qst_values, columns_list


def strategy_qst(
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
    qst - Zero Line Crossover Strategy
    
    LOGIC: Buy when qst crosses above zero (buying pressure), sell when crosses below.
    WHY: qst averages candle bodies (Close - Open). Positive = buyers dominating,
         negative = sellers dominating. Zero crossings signal pressure shifts.
    BEST MARKETS: Trending markets with clear candlestick patterns. Stocks, forex,
                  and futures. Good for confirming trend direction.
    TIMEFRAME: Daily charts. 10-period is common. Adjust window for sensitivity.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 10)
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
    
    window = int(parameters.get('window', 10))
    
    indicator_params = {"window": window}
    short_window_indicator = f'QST_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='qst',
        parameters=indicator_params,
        figure=False
    )
    
    # Create zero line for crossover strategy
    data['zero_line'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator='zero_line',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, 'zero_line']
    
    return results, portfolio, indicator_cols_to_plot, data
