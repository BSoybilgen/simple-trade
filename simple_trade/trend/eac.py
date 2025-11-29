import pandas as pd


def eac(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Ehlers Adaptive CyberCycle (EAC) indicator.
    The Adaptive CyberCycle is based on John Ehlers' work on cycle analysis.
    It adapts to the dominant market cycle and provides a smooth, low-lag
    trend indicator that oscillates around the price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - alpha (float): The smoothing factor. Default is 0.07.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the EAC series and a list of column names.

    The Ehlers Adaptive CyberCycle is calculated as follows:

    1. Smooth the Price Data:
       Smooth = (Price + 2*Price[1] + 2*Price[2] + Price[3]) / 6

    2. Calculate Cycle:
       Cycle[i] = (1 - 0.5*alpha)^2 * (Smooth[i] - 2*Smooth[i-1] + Smooth[i-2]) 
                  + 2*(1 - alpha)*Cycle[i-1] 
                  - (1 - alpha)^2 * Cycle[i-2]

    3. Calculate Trend Line:
       Trend = Smooth - Cycle

    Interpretation:
    - The indicator separates the cycle component from the trend component.
    - It provides a very smooth trendline with minimal lag.

    Use Cases:
    - Trend Following: The resulting trendline tracks the underlying trend.
    - Cycle Analysis: The removed cycle component can be analyzed separately.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    alpha = float(parameters.get('alpha', 0.07))

    series = df[close_col].copy()
    
    # Smooth the price data
    smooth = pd.Series(index=series.index, dtype=float)
    smooth.iloc[0] = series.iloc[0]
    
    for i in range(1, len(series)):
        if i < 3:
            smooth.iloc[i] = series.iloc[i]
        else:
            # Simple 4-bar average for smoothing
            smooth.iloc[i] = (series.iloc[i] + 2*series.iloc[i-1] + 
                             2*series.iloc[i-2] + series.iloc[i-3]) / 6.0
    
    # Initialize cycle
    cycle = pd.Series(index=series.index, dtype=float)
    cycle.iloc[:2] = 0.0
    
    # Calculate CyberCycle using recursive filter
    for i in range(2, len(series)):
        if pd.notna(smooth.iloc[i]) and pd.notna(smooth.iloc[i-1]) and pd.notna(smooth.iloc[i-2]):
            # Ehlers CyberCycle formula (simplified)
            cycle.iloc[i] = ((1 - 0.5*alpha)**2 * (smooth.iloc[i] - 2*smooth.iloc[i-1] + smooth.iloc[i-2]) + 
                            2*(1 - alpha)*cycle.iloc[i-1] - 
                            (1 - alpha)**2 * cycle.iloc[i-2])
        else:
            cycle.iloc[i] = 0.0
    
    # Create the trend line by subtracting cycle from smoothed price
    trend = smooth - cycle
    trend.name = f'EAC_{int(alpha*100)}'
    
    columns_list = [trend.name]
    return trend, columns_list


def strategy_eac(
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
    EAC (Ehlers Adaptive CyberCycle) - Price vs Trend Crossover Strategy
    
    LOGIC: Buy when price crosses above EAC trendline, sell when crosses below.
    WHY: EAC separates cycle from trend using Ehlers' cycle analysis.
         Provides smooth, low-lag trendline that tracks underlying trend.
    BEST MARKETS: All markets. Particularly good for cycle analysis.
                  Stocks, forex, futures.
    TIMEFRAME: Daily charts. Alpha of 0.07 is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'alpha' (default 0.07)
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
    
    alpha = float(parameters.get('alpha', 0.07))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='eac',
        parameters={"alpha": alpha},
        figure=False
    )
    
    short_window_indicator = 'Close'
    long_window_indicator = f'EAC_{int(alpha*100)}'
    
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
