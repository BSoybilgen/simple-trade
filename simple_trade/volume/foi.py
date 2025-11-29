import pandas as pd


def foi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Force Index (FI), an indicator that uses price and volume
    to assess the power behind a price move and identify potential turning points.
    It combines price change, extent of price change, and trading volume.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The EMA smoothing period. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the Force Index series and a list of column names.

    The Force Index is calculated as follows:

    1. Calculate Raw Force Index:
       Raw FI = (Current Close - Previous Close) * Current Volume

    2. Calculate Smoothed Force Index:
       FI = EMA(Raw FI, period)

    Interpretation:
    - Positive FI: Buying pressure dominates (Bulls are in control).
    - Negative FI: Selling pressure dominates (Bears are in control).
    - Zero Line Crossovers: Signal trend changes.

    Use Cases:
    - Trend Confirmation: Positive FI confirms uptrend; negative FI confirms downtrend.
    - Divergence: Price making new highs/lows while FI fails to do so signals weakness.
    - Entry Signals: Buying on negative spikes in uptrends (pullbacks).
    - Breakout Validation: High volume moves produce large FI spikes.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = parameters.get('period', 13)
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate price change
    price_change = close.diff()
    
    # Calculate Raw Force Index
    raw_force_index = price_change * volume
    
    # Apply EMA smoothing
    force_index = raw_force_index.ewm(span=period, adjust=False).mean()
    
    force_index.name = f'FI_{period}'
    columns_list = [force_index.name]
    return force_index, columns_list


def strategy_foi(
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
    FOI (Force Index) - Zero Line Cross Strategy
    
    LOGIC: Buy when Force Index crosses above zero (buying pressure),
           sell when crosses below zero (selling pressure).
    WHY: Force Index combines price change and volume to assess power behind moves.
         Positive FI indicates bulls in control, negative FI indicates bears.
    BEST MARKETS: Stocks, ETFs. Good for trend confirmation and divergence.
    TIMEFRAME: Daily charts. 13-period EMA is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 13)
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
    
    period = int(parameters.get('period', 13))
    price_col = 'Close'
    indicator_col = f'FI_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='foi',
        parameters={"period": period},
        figure=False
    )
    
    # Create zero line for crossover
    data['zero'] = 0
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=indicator_col,
        long_window_indicator='zero',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'zero']
    
    return results, portfolio, indicator_cols_to_plot, data
