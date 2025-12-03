import numpy as np


def emv(df, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ease of Movement (emv), an indicator that relates price change
    to volume and shows how easily a price can move up or down. High EMV values
    occur when price moves upward on low volume.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The smoothing period for the EMV. Default is 14.
            - divisor (int): Divisor to scale the EMV values. Default is 10000.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the EMV series and a list of column names.

    The Ease of Movement is calculated as follows:

    1. Calculate Distance Moved (Midpoint Move):
       Distance = ((High + Low) / 2) - ((PrevHigh + PrevLow) / 2)

    2. Calculate Box Ratio:
       Box Ratio = (Volume / divisor) / (High - Low)

    3. Calculate 1-Period EMV:
       EMV_1 = Distance / Box Ratio

    4. Calculate Smoothed EMV:
       EMV = SMA(EMV_1, period)

    Interpretation:
    - Positive EMV: Price rising with ease (Buying pressure).
    - Negative EMV: Price falling with ease (Selling pressure).
    - Near Zero: Heavy volume required to move price, or little price movement.

    Use Cases:
    - Trend Confirmation: Confirm strength of trend (Ease of movement in trend direction).
    - Volume Analysis: Assess the "fuel" behind price moves.
    - Entry Signals: Crossovers of the zero line.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = parameters.get('period', 14)
    divisor = parameters.get('divisor', 10000)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]
    
    # Calculate midpoint
    midpoint = (high + low) / 2
    
    # Calculate Distance Moved
    distance_moved = midpoint.diff()
    
    # Calculate Box Ratio (handle division by zero)
    high_low_diff = high - low
    box_ratio = (volume / divisor) / high_low_diff.replace(0, np.nan)
    
    # Calculate 1-Period EMV
    one_period_emv = distance_moved / box_ratio
    one_period_emv = one_period_emv.fillna(0)
    
    # Calculate EMV as simple moving average
    emv_values = one_period_emv.rolling(window=period, min_periods=1).mean()
    
    emv_values.name = f'EMV_{period}'
    columns_list = [emv_values.name]
    return emv_values, columns_list


def strategy_emv(
    data,
    parameters: dict = None,
    config = None,
    trading_type: str = 'long',
    day1_position: str = 'none',
    risk_free_rate: float = 0.0,
    long_entry_pct_cash: float = 1.0,
    short_entry_pct_cash: float = 1.0
) -> tuple:
    """
    emv (Ease of Movement) - Zero Line Cross Strategy
    
    LOGIC: Buy when emv crosses above zero (price rising easily),
           sell when emv crosses below zero (price falling easily).
    WHY: emv relates price change to volume. Positive emv means price moves up
         with ease, negative emv means price moves down with ease.
    BEST MARKETS: Stocks, ETFs. Good for volume-weighted momentum analysis.
    TIMEFRAME: Daily charts. 14-period is standard. Good for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14)
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
    
    period = int(parameters.get('period', 14))
    price_col = 'Close'
    indicator_col = f'EMV_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='emv',
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
