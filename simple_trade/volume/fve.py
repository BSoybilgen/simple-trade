import pandas as pd
import numpy as np


def fve(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Finite Volume Elements (fve), a money flow indicator that resolves
    volatility by separating volume into "bullish" and "bearish" components based on
    intra-period price action and a volatility threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 22.
            - factor (float): The cutoff factor (percent). Default is 0.3 (0.3%).
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the FVE series and a list of column names.

    The Finite Volume Elements is calculated as follows:

    1. Calculate Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate Price Change:
       Change = TP - TP(previous)

    3. Determine Cutoff Threshold:
       Cutoff = (factor / 100) * Close

    4. Assign Volume Direction:
       - If Change > Cutoff: Bullish Volume (+Volume)
       - If Change < -Cutoff: Bearish Volume (-Volume)
       - Otherwise: Neutral Volume (0)

    5. Calculate FVE:
       FVE = (Sum(Volume Direction, period) / Sum(Volume, period)) * 100

    Interpretation:
    - FVE > 0: Bullish money flow.
    - FVE < 0: Bearish money flow.
    - Rising FVE: Buying pressure increasing.

    Use Cases:
    - Trend Confirmation: Confirm price trends with money flow.
    - Divergence: Spot reversals when Price and FVE diverge.
    - Breakout Validation: FVE crossing zero can signal new trends.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = parameters.get('period', 22)
    factor = parameters.get('factor', 0.3)
    
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # 1. Typical Price
    tp = (high + low + close) / 3
    
    # 2. Price Change
    tp_change = tp.diff()
    
    # 3. Cutoff
    cutoff = (factor / 100.0) * close
    
    # 4. Volume Direction
    vol_direction = pd.Series(0.0, index=df.index)
    
    # Bullish
    mask_bull = tp_change > cutoff
    vol_direction[mask_bull] = volume[mask_bull]
    
    # Bearish
    mask_bear = tp_change < -cutoff
    vol_direction[mask_bear] = -volume[mask_bear]
    
    # 5. FVE Calculation
    vol_sum = volume.rolling(window=period).sum()
    dir_sum = vol_direction.rolling(window=period).sum()
    
    fve_values = (dir_sum / vol_sum.replace(0, np.nan)) * 100
    
    fve_values.name = f'FVE_{period}'
    columns_list = [fve_values.name]
    return fve_values, columns_list


def strategy_fve(
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
    fve (Finite Volume Elements) - Zero Line Cross Strategy
    
    LOGIC: Buy when fve crosses above zero (bullish money flow),
           sell when crosses below zero (bearish money flow).
    WHY: fve separates volume into bullish/bearish components based on
         intra-period price action and volatility threshold.
    BEST MARKETS: Stocks, ETFs. Good for money flow analysis.
    TIMEFRAME: Daily charts. 22-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 22), 'factor' (default 0.3)
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
    
    period = int(parameters.get('period', 22))
    price_col = 'Close'
    indicator_col = f'FVE_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='fve',
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
