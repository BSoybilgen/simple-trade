import pandas as pd
import numpy as np

def psa(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Parabolic SAR (psa).
    psa (Stop And Reverse) is a trend-following indicator developed by J. Welles Wilder
    that helps identify potential reversals in price direction. It appears as a series of dots
    placed either above or below the price, depending on the trend direction.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - af_initial (float): Initial acceleration factor. Default is 0.02.
            - af_step (float): Acceleration factor step. Default is 0.02.
            - af_max (float): Maximum acceleration factor. Default is 0.2.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame with PSA values and a list of column names.

    Calculation Steps:
    1. Initial SAR value:
       - In an uptrend, SAR starts at the lowest low of the previous data range.
       - In a downtrend, SAR starts at the highest high of the previous data range.

    2. Extreme Points (EP):
       - Uptrend: EP is the highest high reached during the current trend.
       - Downtrend: EP is the lowest low reached during the current trend.

    3. Acceleration Factor (AF):
       - Starts at af_initial. Increases by af_step when a new EP is reached.
       - Capped at af_max.

    4. SAR Calculation:
       SAR = Previous SAR + AF * (EP - Previous SAR)
       (Constraints: SAR cannot be above/below previous period's High/Low depending on trend).

    5. Trend Reversal:
       - When price crosses SAR, trend reverses. SAR resets to EP.

    Interpretation:
    - Dots Below Price: Uptrend (Bullish).
    - Dots Above Price: Downtrend (Bearish).
    - The gap between price and dots tightens as the trend matures (acceleration).

    Use Cases:
    - Stop loss placement: The SAR value can be used as a trailing stop loss.
    - Exit signal generation: A cross of price through the SAR dots indicates a potential reversal.
    - Trend identification: Determining the current market bias.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    af_initial = float(parameters.get('af_initial', 0.02))
    af_step = float(parameters.get('af_step', 0.02))
    af_max = float(parameters.get('af_max', 0.2))
    
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    length = len(high)
    if length < 2: # Need at least 2 points
        # Return empty DataFrame with all NaN values
        result = pd.DataFrame(
            {
                'PSA': np.nan,
                'PSA_Bullish': np.nan,
                'PSA_Bearish': np.nan
            }, 
            index=high.index
        )
        return result, list(result.columns)

    psa_values = np.zeros(length)
    psa_bullish = np.full(length, np.nan)  # Initialize with NaN
    psa_bearish = np.full(length, np.nan)  # Initialize with NaN
    trend_is_bull = np.zeros(length, dtype=bool)  # Track the trend direction
    
    bull = True # Initial trend assumption
    af = af_initial
    ep = high.iloc[0] # Initial Extreme Point (assuming initial uptrend)

    # Initialize first SAR value
    psa_values[0] = low.iloc[0]
    trend_is_bull[0] = bull
    
    # Set initial values based on initial trend
    if bull:
        psa_bullish[0] = psa_values[0]
    else:
        psa_bearish[0] = psa_values[0]

    # A slightly more robust initial trend check (optional, needs 'Close' if used)
    # if length > 1 and close.iloc[1] < close.iloc[0]:
    #     bull = False
    #     ep = low.iloc[0]
    #     psa_values[0] = high.iloc[0]

    for i in range(1, length):
        prev_psa = psa_values[i-1]
        prev_ep = ep
        prev_af = af

        if bull:
            current_psa = prev_psa + prev_af * (prev_ep - prev_psa)
            # SAR cannot be higher than the low of the previous two periods
            current_psa = min(current_psa, low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

            if low.iloc[i] < current_psa: # Trend reversal to Bear
                bull = False
                current_psa = prev_ep # SAR starts at the last extreme high
                ep = low.iloc[i]     # New extreme point is the current low
                af = af_initial # Reset AF
            else: # Continue Bull trend
                # If new high is made, update EP and increment AF
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(prev_af + af_step, af_max)
                else:
                    af = prev_af # AF doesn't change if EP not exceeded
        else: # Bear trend
            current_psa = prev_psa + prev_af * (prev_ep - prev_psa)
            # SAR cannot be lower than the high of the previous two periods
            current_psa = max(current_psa, high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

            if high.iloc[i] > current_psa: # Trend reversal to Bull
                bull = True
                current_psa = prev_ep # SAR starts at the last extreme low
                ep = high.iloc[i]     # New extreme point is the current high
                af = af_initial # Reset AF
            else: # Continue Bear trend
                # If new low is made, update EP and increment AF
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(prev_af + af_step, af_max)
                else:
                    af = prev_af # AF doesn't change if EP not exceeded

        psa_values[i] = current_psa
        trend_is_bull[i] = bull
        
        # Populate the appropriate trend-specific array
        if bull:
            psa_bullish[i] = current_psa
        else:
            psa_bearish[i] = current_psa

    # Create a DataFrame with the base PSA values
    result = pd.DataFrame(
        {
            f'PSA_{af_initial}_{af_step}_{af_max}': psa_values,
            f'PSA_Bullish_{af_initial}_{af_step}_{af_max}': psa_bullish,
            f'PSA_Bearish_{af_initial}_{af_step}_{af_max}': psa_bearish
        }, 
        index=high.index
    )
    
    # Replace NaN values in PSA_Bullish with half the price and PSA_Bearish with 1.5x price
    result[f'PSA_Bullish_{af_initial}_{af_step}_{af_max}'] = result[f'PSA_Bullish_{af_initial}_{af_step}_{af_max}'].fillna(close * 1.5)
    result[f'PSA_Bearish_{af_initial}_{af_step}_{af_max}'] = result[f'PSA_Bearish_{af_initial}_{af_step}_{af_max}'].fillna(close * 0.5)
    
    columns_list = list(result.columns)
    return result, columns_list


def strategy_psa(
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
    psa (Parabolic SAR) - Price vs SAR Crossover Strategy
    
    LOGIC: Buy when price crosses above psa, sell when crosses below.
    WHY: psa is a trend-following indicator that provides stop-and-reverse
         signals. Dots below price = uptrend, above = downtrend.
    BEST MARKETS: Trending markets. Stocks, forex, commodities. Excellent for
                  trailing stop placement and trend identification.
    TIMEFRAME: Daily charts. Standard: af_initial=0.02, af_step=0.02, af_max=0.2.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'af_initial', 'af_step', 'af_max'
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
    
    af_initial = float(parameters.get('af_initial', 0.02))
    af_step = float(parameters.get('af_step', 0.02))
    af_max = float(parameters.get('af_max', 0.2))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='psa',
        parameters={"af_initial": af_initial, "af_step": af_step, "af_max": af_max},
        figure=False
    )
    
    short_window_indicator = 'Close'
    long_window_indicator = f'PSA_{af_initial}_{af_step}_{af_max}'
    
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
