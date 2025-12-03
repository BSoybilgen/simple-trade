import pandas as pd


def nvi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Negative Volume Index (nvi), an indicator that tracks price changes
    on days when volume decreases from the previous day. It is based on the premise
    that "smart money" trades on low volume days.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - initial_value (float): The starting value for NVI. Default is 1000.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the NVI series and a list of column names.

    The Negative Volume Index is calculated as follows:

    1. Initialize NVI:
       Start with an arbitrary value (e.g., 1000).

    2. Iterate through each period:
       - If Current Volume < Previous Volume:
         NVI = Previous NVI + (Previous NVI * Price ROC)
         where Price ROC = (Close - Previous Close) / Previous Close
       - If Current Volume >= Previous Volume:
         NVI = Previous NVI (No Change)

    Interpretation:
    - Rising NVI: Smart money is accumulating.
    - Falling NVI: Smart money is distributing.
    - Above EMA: Bullish trend.
    - Below EMA: Bearish trend.

    Use Cases:
    - Smart Money Tracking: identifying institutional accumulation/distribution.
    - Trend Identification: Long-term trend direction.
    - Confirmation: Often used with Positive Volume Index (PVI).
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
    
    # Initialize NVI series
    nvi_values = pd.Series(index=df.index, dtype=float)
    nvi_values.iloc[0] = initial_value
    
    # Calculate NVI
    for i in range(1, len(df)):
        if volume.iloc[i] < volume.iloc[i-1]:
            # Volume decreased - update NVI based on price change
            price_change_pct = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            nvi_values.iloc[i] = nvi_values.iloc[i-1] + (nvi_values.iloc[i-1] * price_change_pct)
        else:
            # Volume increased or stayed same - NVI unchanged
            nvi_values.iloc[i] = nvi_values.iloc[i-1]
    
    nvi_values.name = 'NVI'
    columns_list = [nvi_values.name]
    return nvi_values, columns_list


def strategy_nvi(
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
    nvi (Negative Volume Index) - EMA Crossover Strategy
    
    LOGIC: Buy when nvi crosses above its EMA (smart money accumulating),
           sell when nvi crosses below its EMA (smart money distributing).
    WHY: nvi tracks price changes on low volume days. Based on premise that
         "smart money" trades on low volume days.
    BEST MARKETS: Stocks, ETFs. Good for long-term trend identification.
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
        indicator='nvi',
        parameters={},
        figure=False
    )
    
    # Calculate EMA of NVI for crossover signals
    data[f'NVI_EMA_{ema_period}'] = data['NVI'].ewm(span=ema_period, adjust=False).mean()
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='NVI',
        long_window_indicator=f'NVI_EMA_{ema_period}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['NVI', f'NVI_EMA_{ema_period}']
    
    return results, portfolio, indicator_cols_to_plot, data
