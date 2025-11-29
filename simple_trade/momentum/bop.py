import pandas as pd


def bop(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Balance of Power (BOP), an indicator that measures the strength of buyers 
    versus sellers by assessing the ability of each side to drive prices to extreme levels.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The smoothing window for the BOP. Default is 14.
            - smooth (bool): Whether to smooth the result using an SMA. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Balance of Power series and a list of column names.

    The Balance of Power is calculated as follows:

    1. Calculate the Price Range:
       Range = High - Low

    2. Calculate the Raw BOP:
       BOP_Raw = (Close - Open) / Range

    3. (Optional) Smooth the BOP:
       If smooth=True:
           BOP = SMA(BOP_Raw, window)
       Else:
           BOP = BOP_Raw

    Interpretation:
    - BOP > 0: Buyers are in control (Bullish pressure).
    - BOP < 0: Sellers are in control (Bearish pressure).
    - BOP near 0: Market is in equilibrium or indecision.
    - Extremes: High positive values indicate strong buying; low negative values indicate strong selling.

    Use Cases:
    - Trend Identification: Confirming the direction and strength of a trend.
    - Divergence: Divergence between price and BOP can signal potential reversals.
    - Overbought/Oversold: Extreme values can indicate potential exhaustion of the current trend.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    smooth = parameters.get('smooth', True)
    open_col = columns.get('open_col', 'Open')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

    o = df[open_col]
    h = df[high_col]
    low_vals = df[low_col]
    c = df[close_col]

    # Formula: (Close - Open) / (High - Low)
    range_hl = h - low_vals
    bop_raw = (c - o) / range_hl.replace(0, float('nan'))

    if smooth:
        bop_val = bop_raw.rolling(window=window).mean()
        name = f'BOP_{window}'
    else:
        bop_val = bop_raw
        name = 'BOP'

    bop_val.name = name
    return bop_val, [name]


def strategy_bop(
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
    BOP (Balance of Power) - Zero Line Crossover Strategy
    
    LOGIC: Buy when BOP crosses above zero (buyers in control), sell when crosses below.
    WHY: BOP measures the strength of buyers vs sellers by comparing close-open to high-low.
         Positive BOP = buyers driving prices up, negative = sellers in control.
    BEST MARKETS: Stocks and indices with clear institutional participation.
                  Works well in trending markets with strong volume.
    TIMEFRAME: Daily charts. Smoothed version (default) reduces noise.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'smooth' (default True)
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
    smooth = parameters.get('smooth', True)
    
    indicator_params = {"window": window, "smooth": smooth}
    
    if smooth:
        short_window_indicator = f'BOP_{window}'
    else:
        short_window_indicator = 'BOP'
    
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='bop',
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
