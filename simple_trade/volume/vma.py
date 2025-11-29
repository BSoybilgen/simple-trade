import pandas as pd


def vma(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volume Moving Average (VMA), which is a weighted moving average
    that uses volume as the weighting factor. It gives more weight to prices
    accompanied by higher volume.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VMA series and a list of column names.

    The Volume Moving Average is calculated as follows:

    1. Calculate Weighted Price:
       Weighted Price = Price * Volume

    2. Calculate VMA:
       VMA = Sum(Weighted Price, window) / Sum(Volume, window)

    Interpretation:
    - Rising VMA: Bullish trend supported by volume.
    - Falling VMA: Bearish trend supported by volume.
    - Support/Resistance: VMA often acts as dynamic support or resistance.

    Use Cases:
    - Trend Identification: Identify volume-supported trends.
    - Dynamic Support/Resistance: Use VMA lines for entry/exit points.
    - Filtering: Validate price moves (price above VMA in uptrend).
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')

    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate the volume-weighted price
    weighted_price = close * volume
    
    # Calculate the VMA using rolling windows
    # For each window, sum(price * volume) / sum(volume)
    vma_values = weighted_price.rolling(window=window).sum() / volume.rolling(window=window).sum()
    vma_values.name = f'VMA_{window}'
    columns_list = [vma_values.name]
    return vma_values, columns_list


def strategy_vma(
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
    VMA (Volume Moving Average) - Price vs VMA Crossover Strategy
    
    LOGIC: Buy when price crosses above VMA (bullish volume-weighted trend),
           sell when price crosses below VMA (bearish volume-weighted trend).
    WHY: VMA uses volume as weighting factor. Gives more weight to prices
         accompanied by higher volume. Acts as dynamic support/resistance.
    BEST MARKETS: Stocks, ETFs. Good for trend identification.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20)
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
    
    window = int(parameters.get('window', 20))
    price_col = 'Close'
    indicator_col = f'VMA_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='vma',
        parameters={"window": window},
        figure=False
    )
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='Close',
        long_window_indicator=indicator_col,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['Close', indicator_col]
    
    return results, portfolio, indicator_cols_to_plot, data
