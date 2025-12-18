import pandas as pd
import numpy as np


def cmf(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Chaikin Money Flow (cmf), a volume-based indicator that measures
    the amount of Money Flow Volume over a specific period. It combines price and
    volume to identify buying and selling pressure.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the CMF series and a list of column names.

    The Chaikin Money Flow is calculated as follows:

    1. Calculate Money Flow Multiplier (MFM):
       MFM = ((Close - Low) - (High - Close)) / (High - Low)

    2. Calculate Money Flow Volume (MFV):
       MFV = MFM * Volume

    3. Calculate CMF:
       CMF = Sum(MFV, period) / Sum(Volume, period)

    Interpretation:
    - CMF > 0: Buying pressure (Accumulation).
    - CMF < 0: Selling pressure (Distribution).
    - Magnitude: The further from zero, the stronger the pressure.

    Use Cases:
    - Trend Confirmation: Positive CMF confirms uptrends; negative confirms downtrends.
    - Divergence: Price making new highs while CMF declines suggests weakening trend.
    - Support/Resistance: Increasing CMF during breakouts confirms the move.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else 20)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Handle division by zero - if high and low are the same,
    # money flow multiplier is zero (neutral)
    price_range = high - low
    price_range_nonzero = price_range.replace(0, np.nan)
    
    # Calculate Money Flow Multiplier (MFM)
    # MFM = ((Close - Low) - (High - Close)) / (High - Low)
    # This simplifies to MFM = (2*Close - High - Low) / (High - Low)
    mfm = ((2 * close - high - low) / price_range_nonzero).fillna(0)
    
    # Calculate Money Flow Volume (MFV)
    mfv = mfm * volume
    
    # Calculate CMF as sum(MFV)/sum(Volume) over the period
    cmf_values = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    cmf_values.name = f'CMF_{period}'
    columns_list = [cmf_values.name]
    return cmf_values, columns_list


def strategy_cmf(
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
    cmf (Chaikin Money Flow) - Zero Line Cross Strategy
    
    LOGIC: Buy when cmf crosses above zero (buying pressure),
           sell when cmf crosses below zero (selling pressure).
    WHY: cmf measures money flow over a period. Positive cmf indicates accumulation,
         negative cmf indicates distribution. Good for trend confirmation.
    BEST MARKETS: Stocks, ETFs. Good for confirming breakouts and trends.
    TIMEFRAME: Daily charts. 20-period is standard. Good for swing trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 20)
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else 20)
    price_col = 'Close'
    indicator_col = f'CMF_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='cmf',
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
