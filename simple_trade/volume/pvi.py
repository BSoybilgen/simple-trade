import pandas as pd


def pvi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Positive Volume Index (pvi), an indicator that tracks price changes
    on days when volume increases from the previous day. It is based on the premise
    that the "crowd" or uninformed investors trade on high volume days.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - initial_value (float): The starting value for PVI. Default is 1000.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the PVI series and a list of column names.

    The Positive Volume Index is calculated as follows:

    1. Initialize PVI:
       Start with an arbitrary value (e.g., 1000).

    2. Iterate through each period:
       - If Current Volume > Previous Volume:
         PVI = Previous PVI + (Previous PVI * Price ROC)
         where Price ROC = (Close - Previous Close) / Previous Close
       - If Current Volume <= Previous Volume:
         PVI = Previous PVI (No Change)

    Interpretation:
    - Rising PVI: Crowd is buying (Bullish sentiment among general public).
    - Falling PVI: Crowd is selling (Bearish sentiment among general public).
    - Above EMA: Bullish trend.
    - Below EMA: Bearish trend.

    Use Cases:
    - Sentiment Analysis: Gauge the activity of uninformed investors.
    - Trend Confirmation: Confirm trends driven by high volume.
    - Market Phase Identification: Used with NVI to identify Bull/Bear markets.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    initial_value = parameters.get('initial_value', 1000)
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    close = df[close_col]
    volume = df[volume_col]
    
    # Initialize PVI series
    pvi_values = pd.Series(index=df.index, dtype=float)
    pvi_values.iloc[0] = initial_value
    
    # Calculate PVI
    for i in range(1, len(df)):
        if volume.iloc[i] > volume.iloc[i-1]:
            # Volume increased - update PVI based on price change
            price_change_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            pvi_values.iloc[i] = pvi_values.iloc[i-1] + (pvi_values.iloc[i-1] * price_change_pct)
        else:
            # Volume decreased or stayed same - PVI unchanged
            pvi_values.iloc[i] = pvi_values.iloc[i-1]
    
    pvi_values.name = 'PVI'
    columns_list = [pvi_values.name]
    return pvi_values, columns_list


def strategy_pvi(
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
    pvi (Positive Volume Index) - EMA Crossover Strategy
    
    LOGIC: Buy when pvi crosses above its EMA (crowd buying),
           sell when pvi crosses below its EMA (crowd selling).
    WHY: pvi tracks price changes on high volume days. Based on premise that
         the "crowd" or uninformed investors trade on high volume days.
    BEST MARKETS: Stocks, ETFs. Good for sentiment analysis.
    TIMEFRAME: Daily charts. Often used with 255-day EMA.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'ema_period' (default 255)
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
    
    ema_period = int(parameters.get('ema_period', 255))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='pvi',
        parameters={},
        figure=False
    )
    
    # Calculate EMA of PVI for crossover signals
    data[f'PVI_EMA_{ema_period}'] = data['PVI'].ewm(span=ema_period, adjust=False).mean()
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='PVI',
        long_window_indicator=f'PVI_EMA_{ema_period}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['PVI', f'PVI_EMA_{ema_period}']
    
    return results, portfolio, indicator_cols_to_plot, data
