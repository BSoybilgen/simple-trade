import pandas as pd


def roc(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Rate of Change (ROC), a momentum oscillator that measures the percentage 
    change in price between the current price and the price a specified number of periods ago.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 12.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ROC series and a list of column names.

    Calculation Steps:
    1. Calculate the difference between the current price and the price n periods ago.
    2. Divide the result by the price n periods ago.
    3. Multiply by 100 to convert to a percentage.

    Interpretation:
    - Positive values: Price is higher than it was n periods ago (Uptrend/Momentum).
    - Negative values: Price is lower than it was n periods ago (Downtrend/Momentum).
    - Zero Line: Crossing above zero indicates increasing upward momentum; below zero indicates increasing downward momentum.
    - Slope: A steep slope indicates strong momentum.

    Use Cases:
    - Identifying overbought/oversold conditions: Extreme positive values may indicate overbought 
      conditions, while extreme negative values may indicate oversold conditions.
    - Divergence analysis: When price makes a new high or low but ROC doesn't, it may signal 
      a potential reversal.
    - Zero-line crossovers: When ROC crosses above zero, it may signal a buy opportunity; 
      when it crosses below zero, it may signal a sell opportunity.
    - Trend confirmation: Strong positive ROC values confirm an uptrend, while strong negative 
      values confirm a downtrend.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 12))
    close_col = columns.get('close_col', 'Close')
    
    series = df[close_col]
    
    # Calculate the Rate of Change
    # ROC = ((Current Price - Price n periods ago) / Price n periods ago) * 100
    roc_values = ((series / series.shift(window)) - 1) * 100
    
    roc_values.name = f'ROC_{window}'
    columns_list = [roc_values.name]
    return roc_values, columns_list


def strategy_roc(
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
    ROC (Rate of Change) - Zero Line Crossover Strategy
    
    LOGIC: Buy when ROC crosses above zero (upward momentum), sell when crosses below.
    WHY: ROC measures percentage price change over N periods. Positive = price rising,
         negative = price falling. Zero crossings indicate momentum direction changes.
    BEST MARKETS: Trending markets across all asset classes. Stocks, forex, crypto.
                  Simple but effective for identifying momentum shifts.
    TIMEFRAME: All timeframes. 12-period is common for daily charts.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 12)
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
    
    window = int(parameters.get('window', 12))
    
    indicator_params = {"window": window}
    short_window_indicator = f'ROC_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='roc',
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
