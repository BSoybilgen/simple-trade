import pandas as pd


def ads(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Adaptive Deviation-Scaled Moving Average (ads).
    ads adjusts its smoothing factor based on price changes relative to the moving average.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): Base period for EMA calculation. Default is 20.
            - sensitivity (float): Multiplier for price change impact. Default is 0.5.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the ADS series and a list of column names.

    The ads is calculated as follows:

    1. Calculate Price Change Ratio:
       Ratio = Abs(Price - Previous ads) / Previous ads

    2. Scale the Ratio:
       Scaled Change = Ratio * sensitivity

    3. Calculate Adaptive Alpha:
       Base Alpha = 2 / (window + 1)
       Adaptive Alpha = Base Alpha * (1 + Scaled Change)
       (Capped at 1.0)

    4. Calculate ads:
       ads = (Adaptive Alpha * Price) + ((1 - Adaptive Alpha) * Previous ads)

    Interpretation:
    - When price changes are large, ads becomes more responsive (higher alpha) to capture the move.
    - When price changes are small (consolidation), ads becomes smoother (lower alpha) to avoid noise.

    Use Cases:
    - Trend Following: Filters out noise during consolidation while reacting quickly to breakouts.
    - Dynamic Support/Resistance: Acts as a support line in uptrends and resistance in downtrends.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    window = int(parameters.get('window', 20))
    sensitivity = float(parameters.get('sensitivity', 0.5))

    series = df[close_col].copy()
    
    # Initialize ADSMA with first value
    adsma = pd.Series(index=series.index, dtype=float)
    adsma.iloc[0] = series.iloc[0]
    
    # Calculate adaptive EMA with dynamic alpha based on price changes
    for i in range(1, len(series)):
        if pd.notna(series.iloc[i]) and pd.notna(adsma.iloc[i-1]):
            # Calculate price change ratio
            price_change = abs(series.iloc[i] - adsma.iloc[i-1]) / (adsma.iloc[i-1] + 1e-10)
            
            # Scale by sensitivity
            scaled_change = price_change * sensitivity
            
            # Convert to alpha: larger changes = higher alpha (more responsive)
            # Smaller changes = lower alpha (more smoothing)
            base_alpha = 2.0 / (window + 1)  # Base EMA alpha
            
            # Adaptive multiplier: ranges from 1.0 (no change) to higher values (large changes)
            # Use a bounded multiplier to prevent convergence of different windows
            adaptive_multiplier = 1.0 + scaled_change
            
            # Calculate adaptive alpha with bounds that respect the window parameter
            # Min bound: half of base alpha (more smoothing during consolidation)
            # Max bound: 3x base alpha or 0.9, whichever is smaller (prevents hitting 1.0 too easily)
            adaptive_alpha = base_alpha * adaptive_multiplier
            adaptive_alpha = max(adaptive_alpha, base_alpha * 0.5)
            adaptive_alpha = min(adaptive_alpha, min(base_alpha * 3.0, 0.9))
            
            # Apply adaptive EMA formula
            adsma.iloc[i] = adaptive_alpha * series.iloc[i] + (1 - adaptive_alpha) * adsma.iloc[i-1]
        else:
            adsma.iloc[i] = series.iloc[i]
    
    adsma.name = f'ADS_{window}'
    
    columns_list = [adsma.name]
    return adsma, columns_list


def strategy_ads(
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
    ads - Dual MA Crossover Strategy
    
    LOGIC: Buy when fast ads crosses above slow ads, sell when crosses below.
    WHY: Adaptive smoothing reduces lag while filtering noise. Crossovers signal
         trend changes with better timing than traditional MAs.
    BEST MARKETS: Trending stocks, forex, and indices. Reduces whipsaws in
                  moderately volatile markets compared to SMA/EMA.
    TIMEFRAME: Daily or weekly charts for position trading.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'short_window' (default 10), 'long_window' (default 30)
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
    
    short_window = int(parameters.get('short_window', 10))
    long_window = int(parameters.get('long_window', 30))
    price_col = 'Close'
    
    # If short_window is 0, use actual price instead of indicator
    if short_window == 0:
        short_window_indicator = 'Close'
    else:
        short_window_indicator = f'ADS_{short_window}'
        data, _, _ = compute_indicator(
            data=data,
            indicator='ads',
            parameters={"window": short_window},
            figure=False
        )
    
    long_window_indicator = f'ADS_{long_window}'
    data, _, _ = compute_indicator(
        data=data,
        indicator='ads',
        parameters={"window": long_window},
        figure=False
    )
    
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
    
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data
