import pandas as pd
import numpy as np

def str(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the SuperTrend indicator (str).
    str is a trend following indicator similar to moving averages.
    It plots on price charts as a line that follows price but stays a certain
    distance from it, reacting to volatility.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for ATR. Default is 14.
            - multiplier (float): The multiplier for ATR. Default is 3.0.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
    
    Returns:
        tuple: A tuple containing the SuperTrend DataFrame (Supertrend, Direction) and a list of column names.
        
    The SuperTrend indicator is calculated as follows:
    
    1. Calculate Average True Range (ATR) over the specified period.
    
    2. Calculate Basic Upper and Lower Bands:
       Basic Upper Band = (High + Low) / 2 + (Multiplier * ATR)
       Basic Lower Band = (High + Low) / 2 - (Multiplier * ATR)
       
    3. Calculate Final Bands (with Ratchet Effect):
       - If Price < Previous Final Upper Band, Final Upper Band = Min(Basic Upper Band, Previous Final Upper Band)
       - If Price > Previous Final Lower Band, Final Lower Band = Max(Basic Lower Band, Previous Final Lower Band)
       
    4. Determine SuperTrend:
       - If Previous Trend was Down and Close > Previous Final Upper Band: Trend flips to Up. SuperTrend = Final Lower Band.
       - If Previous Trend was Up and Close < Previous Final Lower Band: Trend flips to Down. SuperTrend = Final Upper Band.
    
    Interpretation:
    - Price above SuperTrend line: Uptrend (Bullish).
    - Price below SuperTrend line: Downtrend (Bearish).
    - The line changes color (Green/Red) based on direction in most charting software.
    
    Use Cases:
    - Trend detection: SuperTrend helps identify the current market trend direction.
    - Stop loss placement: The SuperTrend line can serve as a trailing stop level.
    - Trade filtering: Use SuperTrend to only take trades in the direction of the trend.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 14))
    multiplier = float(parameters.get('multiplier', 3.0))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]

    # Calculate True Range
    hl = high - low
    hc = np.abs(high - close.shift(1))
    lc = np.abs(low - close.shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1, skipna=False)
    
    # Calculate ATR
    atr = tr.rolling(window=period, min_periods=1).mean() # Use min_periods=1 for ATR
    
    # Calculate basic bands
    mid_price = (high + low) / 2 # Though not directly used in final ST, useful for initial band calc
    up_band = mid_price + (multiplier * atr)
    low_band = mid_price - (multiplier * atr)
    
    # Create result dataframe
    result = pd.DataFrame(index=df.index)
    result['ATR'] = atr # For reference if needed
    result['Basic_up_band'] = up_band # Basic upper band before adjustment
    result['Basic_low_band'] = low_band # Basic lower band before adjustment
    result['STR'] = np.nan
    result['Direction'] = 0  # 1 for uptrend, -1 for downtrend, 0 for undetermined

    # The first 'period-1' ATR values might be less reliable or NaN if min_periods=period.
    # We start calculations from the first valid ATR.
    # With min_periods=1 for ATR, we can start earlier, but Supertrend logic itself needs a previous ST value.
    # Iterative calculation for SuperTrend
    for i in range(len(df)):
        if i < period -1 : # ATR might not be stable enough or is NaN. Or first period elements for rolling
            result.loc[result.index[i], 'STR'] = np.nan # Or some initial value if preferred
            result.loc[result.index[i], 'Direction'] = 0
            continue

        # Initial SuperTrend and direction value (e.g., at index `period-1` or first calculable)
        if i == period -1: # First point where ATR is based on `period` lookback
                           # or first point to set an initial trend
            if close.iloc[i] > result.loc[result.index[i], 'Basic_low_band']:
                result.loc[result.index[i], 'Direction'] = 1
                result.loc[result.index[i], 'STR'] = result.loc[result.index[i], 'Basic_low_band']
            else: # close <= basic_low_band (could also be close < basic_up_band)
                result.loc[result.index[i], 'Direction'] = -1
                result.loc[result.index[i], 'STR'] = result.loc[result.index[i], 'Basic_up_band']
            continue

        # Previous values
        prev_direction = result.loc[result.index[i-1], 'Direction']
        prev_str = result.loc[result.index[i-1], 'STR']
        
        curr_close = close.iloc[i]
        curr_basic_up_band = result.loc[result.index[i], 'Basic_up_band']
        curr_basic_low_band = result.loc[result.index[i], 'Basic_low_band']
        
        curr_direction = prev_direction
        curr_supertrend = np.nan

        if prev_direction == 1: # Previous trend was UP
            curr_supertrend = max(prev_str, curr_basic_low_band) # Ratchet: ST cannot go down in uptrend
            if curr_close < curr_supertrend: # Price crossed below ST line
                curr_direction = -1 # Change trend to DOWN
                curr_supertrend = curr_basic_up_band # New ST is the upper band
        elif prev_direction == -1: # Previous trend was DOWN
            curr_supertrend = min(prev_str, curr_basic_up_band) # Ratchet: ST cannot go up in downtrend
            if curr_close > curr_supertrend: # Price crossed above ST line
                curr_direction = 1 # Change trend to UP
                curr_supertrend = curr_basic_low_band # New ST is the lower band
        else: # Previous direction was 0 (e.g. initial state before period-1)
            # This case should ideally be handled by the i == period-1 block
            # For robustness, re-evaluate based on current price vs bands
            if curr_close > curr_basic_low_band:
                curr_direction = 1
                curr_supertrend = curr_basic_low_band
            else:
                curr_direction = -1
                curr_supertrend = curr_basic_up_band
                
        result.loc[result.index[i], 'Direction'] = curr_direction
        result.loc[result.index[i], 'STR'] = curr_supertrend

        df = result[['STR', 'Direction']].copy()
        df.rename(columns={'STR': f'STR_{period}_{multiplier}', 
                           'Direction': f'Direction_{period}_{multiplier}'},
                           inplace=True)

        # Initialize Bullish and Bearish SuperTrend values
        df[f'STR_Bullish_{period}_{multiplier}'] = np.nan
        df[f'STR_Bearish_{period}_{multiplier}'] = np.nan
        
        # Set Bullish and Bearish values directly
        df.loc[df[f'Direction_{period}_{multiplier}'] == 1, f'STR_Bullish_{period}_{multiplier}'] = df.loc[df[f'Direction_{period}_{multiplier}'] == 1, f'STR_{period}_{multiplier}']
        df.loc[df[f'Direction_{period}_{multiplier}'] == -1, f'STR_Bearish_{period}_{multiplier}'] = df.loc[df[f'Direction_{period}_{multiplier}'] == -1, f'STR_{period}_{multiplier}']
        
        # Fill NaN values with scaled close prices
        df[f'STR_Bullish_{period}_{multiplier}'] = df[f'STR_Bullish_{period}_{multiplier}'].fillna(close * 1.5)
        df[f'STR_Bearish_{period}_{multiplier}'] = df[f'STR_Bearish_{period}_{multiplier}'].fillna(close * 0.5)
        
    columns_list = list(df.columns)
    return df, columns_list


def strategy_str(
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
    str (SuperTrend) - Price vs str Crossover Strategy
    
    LOGIC: Buy when price crosses above str line, sell when crosses below.
    WHY: str is a volatility-based trend indicator. Price above = uptrend,
         below = downtrend. The line acts as dynamic support/resistance.
    BEST MARKETS: Trending markets. Stocks, forex, crypto. Excellent for
                  trend following and trailing stop placement.
    TIMEFRAME: All timeframes. 14-period with 3x multiplier is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14), 'multiplier' (default 3.0)
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
    multiplier = float(parameters.get('multiplier', 3.0))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='str',
        parameters={"period": period, "multiplier": multiplier},
        figure=False
    )
    
    short_window_indicator = 'Close'
    long_window_indicator = f'STR_{period}_{multiplier}'
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    # Include Close price in plot so users can see the crossover signals
    indicator_cols_to_plot = [long_window_indicator, 'Close']
    
    return results, portfolio, indicator_cols_to_plot, data