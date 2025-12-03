import pandas as pd


def obv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the On-Balance Volume (obv), a volume-based momentum indicator that
    relates volume flow to price changes. It measures buying and selling pressure
    as a cumulative indicator that adds volume on up days and subtracts it on down days.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the OBV series and a list of column names.

    The On-Balance Volume is calculated as follows:

    1. Determine Price Direction:
       If Close > Previous Close: Direction = +1
       If Close < Previous Close: Direction = -1
       If Close = Previous Close: Direction = 0

    2. Calculate OBV:
       OBV = Previous OBV + (Volume * Direction)

    Interpretation:
    - Rising OBV: Buying pressure (Accumulation).
    - Falling OBV: Selling pressure (Distribution).
    - Trend Confirmation: OBV should move in the direction of the price trend.

    Use Cases:
    - Trend Confirmation: Confirm the strength of a trend.
    - Divergence Detection: Divergences between Price and OBV often precede reversals.
    - Breakout Validation: Rising OBV during consolidation can signal a breakout.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    close = df[close_col]
    volume = df[volume_col]

    # Calculate the daily price change direction
    # 1 for price up, -1 for price down, 0 for unchanged
    price_direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # First OBV value is equal to the first period's volume
    obv_values = pd.Series(index=close.index, dtype=float)
    obv_values.iloc[0] = volume.iloc[0]
    
    # Cumulative sum of volume multiplied by price direction
    for i in range(1, len(close)):
        obv_values.iloc[i] = obv_values.iloc[i-1] + (volume.iloc[i] * price_direction.iloc[i])
    
    obv_values.name = 'OBV'
    columns_list = [obv_values.name]
    return obv_values, columns_list


def strategy_obv(
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
    obv (On-Balance Volume) - SMA Crossover Strategy
    
    LOGIC: Buy when obv crosses above its SMA (accumulation),
           sell when obv crosses below its SMA (distribution).
    WHY: obv measures buying/selling pressure as cumulative volume.
         Rising obv indicates accumulation, falling indicates distribution.
    BEST MARKETS: Stocks, ETFs. Good for trend confirmation and divergence.
    TIMEFRAME: Daily charts. Good for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'sma_period' (default 20)
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
    
    sma_period = int(parameters.get('sma_period', 20))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='obv',
        parameters={},
        figure=False
    )
    
    # Calculate SMA of OBV for crossover signals
    data[f'OBV_SMA_{sma_period}'] = data['OBV'].rolling(window=sma_period).mean()
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator='OBV',
        long_window_indicator=f'OBV_SMA_{sma_period}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['OBV', f'OBV_SMA_{sma_period}']
    
    return results, portfolio, indicator_cols_to_plot, data
