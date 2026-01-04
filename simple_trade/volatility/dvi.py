import pandas as pd


def dvi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Dynamic Volatility Indicator (dvi), a composite indicator that combines
    multiple volatility measures and price momentum to create a normalized oscillator that
    identifies overbought/oversold conditions based on volatility-adjusted price movements.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - magnitude_period (int): Period for magnitude calculation (price position). Default is 5.
            - stretch_period (int): Period for stretch calculation (consecutive moves). Default is 100.
            - smooth_period (int): Period for final smoothing. Default is 3.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the DVI series and a list of column names.

    Calculation Steps:
    1. Calculate Magnitude Component:
       - Ratio = Close / SMA(Close, magnitude_period)
       - Normalize to 0-1 rank over stretch_period.

    2. Calculate Stretch Component:
       - Net consecutive up/down days count.
       - Normalize to 0-1 rank over stretch_period.

    3. Combine Components:
       - DVI = 0.5 * Magnitude + 0.5 * Stretch

    4. Apply Smoothing:
       - Final DVI = SMA(DVI, smooth_period) * 100

    Interpretation:
    - DVI < 30: Oversold (potential buy).
    - DVI > 70: Overbought (potential sell).
    - DVI near 50: Neutral.

    Use Cases:
    - Mean reversion trading: Counter-trend entries at extremes.
    - Divergence detection: Reversals.
    - Trend filtering: Confirming entries in direction of larger trend.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    magnitude_window_param = parameters.get('magnitude_window')
    magnitude_period_param = parameters.get('magnitude_period')
    if magnitude_window_param is None and magnitude_period_param is not None:
        magnitude_window_param = magnitude_period_param
    elif magnitude_window_param is not None and magnitude_period_param is not None:
        if int(magnitude_window_param) != int(magnitude_period_param):
            raise ValueError("Provide either 'magnitude_window' or 'magnitude_period' (aliases) with the same value if both are set.")

    stretch_window_param = parameters.get('stretch_window')
    stretch_period_param = parameters.get('stretch_period')
    if stretch_window_param is None and stretch_period_param is not None:
        stretch_window_param = stretch_period_param
    elif stretch_window_param is not None and stretch_period_param is not None:
        if int(stretch_window_param) != int(stretch_period_param):
            raise ValueError("Provide either 'stretch_window' or 'stretch_period' (aliases) with the same value if both are set.")

    smooth_window_param = parameters.get('smooth_window')
    smooth_period_param = parameters.get('smooth_period')
    if smooth_window_param is None and smooth_period_param is not None:
        smooth_window_param = smooth_period_param
    elif smooth_window_param is not None and smooth_period_param is not None:
        if int(smooth_window_param) != int(smooth_period_param):
            raise ValueError("Provide either 'smooth_window' or 'smooth_period' (aliases) with the same value if both are set.")

    magnitude_period = int(magnitude_window_param if magnitude_window_param is not None else 5)
    stretch_period = int(stretch_window_param if stretch_window_param is not None else 100)
    smooth_period = int(smooth_window_param if smooth_window_param is not None else 3)
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate Magnitude component
    # Ratio of current price to its moving average
    sma = close.rolling(window=magnitude_period).mean()
    magnitude_ratio = close / sma
    
    # Normalize magnitude using percentile rank over stretch_period
    def percentile_rank(series, period):
        """Calculate percentile rank of current value within rolling window"""
        result = pd.Series(index=series.index, dtype=float)
        for i in range(period - 1, len(series)):
            window = series.iloc[max(0, i - period + 1):i + 1]
            current_value = series.iloc[i]
            if len(window) > 0:
                rank = (window < current_value).sum() / len(window)
                result.iloc[i] = rank
        return result
    
    magnitude_normalized = percentile_rank(magnitude_ratio, stretch_period)
    
    # Calculate Stretch component
    # Count consecutive up/down moves
    price_change = close.diff()
    
    # Calculate consecutive up days
    consecutive_up = pd.Series(0, index=close.index, dtype=int)
    consecutive_down = pd.Series(0, index=close.index, dtype=int)
    
    up_count = 0
    down_count = 0
    
    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            up_count += 1
            down_count = 0
        elif price_change.iloc[i] < 0:
            down_count += 1
            up_count = 0
        else:
            # No change, maintain previous counts
            pass
        
        consecutive_up.iloc[i] = up_count
        consecutive_down.iloc[i] = down_count
    
    # Calculate stretch score (negative for down moves)
    stretch_score = consecutive_up - consecutive_down
    
    # Normalize stretch using percentile rank over stretch_period
    stretch_normalized = percentile_rank(stretch_score, stretch_period)
    
    # Combine Magnitude and Stretch (equal weighting)
    dvi_raw = 0.5 * magnitude_normalized + 0.5 * stretch_normalized
    
    # Apply smoothing
    dvi_smoothed = dvi_raw.rolling(window=smooth_period).mean()
    
    # Scale to 0-100 range
    dvi_values = dvi_smoothed * 100
    
    dvi_values.name = f'DVI_{magnitude_period}_{stretch_period}_{smooth_period}'
    columns_list = [dvi_values.name]
    return dvi_values, columns_list


def strategy_dvi(
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
    dvi (Dynamic Volatility Indicator) - Mean Reversion Strategy
    
    LOGIC: Buy when dvi drops below lower threshold (oversold),
           sell when rises above upper threshold (overbought).
    WHY: dvi combines magnitude and stretch components to identify
         overbought/oversold conditions based on volatility-adjusted moves.
    BEST MARKETS: Range-bound markets. Stocks, ETFs. Good for mean reversion.
    TIMEFRAME: Daily charts. Standard periods: 5/100/3.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'magnitude_period', 'stretch_period', 'smooth_period',
                    'upper' (default 70), 'lower' (default 30)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_band_trade_strategies import run_band_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    magnitude_window_param = parameters.get('magnitude_window')
    magnitude_period_param = parameters.get('magnitude_period')
    if magnitude_window_param is None and magnitude_period_param is not None:
        magnitude_window_param = magnitude_period_param
    elif magnitude_window_param is not None and magnitude_period_param is not None:
        if int(magnitude_window_param) != int(magnitude_period_param):
            raise ValueError("Provide either 'magnitude_window' or 'magnitude_period' (aliases) with the same value if both are set.")

    stretch_window_param = parameters.get('stretch_window')
    stretch_period_param = parameters.get('stretch_period')
    if stretch_window_param is None and stretch_period_param is not None:
        stretch_window_param = stretch_period_param
    elif stretch_window_param is not None and stretch_period_param is not None:
        if int(stretch_window_param) != int(stretch_period_param):
            raise ValueError("Provide either 'stretch_window' or 'stretch_period' (aliases) with the same value if both are set.")

    smooth_window_param = parameters.get('smooth_window')
    smooth_period_param = parameters.get('smooth_period')
    if smooth_window_param is None and smooth_period_param is not None:
        smooth_window_param = smooth_period_param
    elif smooth_window_param is not None and smooth_period_param is not None:
        if int(smooth_window_param) != int(smooth_period_param):
            raise ValueError("Provide either 'smooth_window' or 'smooth_period' (aliases) with the same value if both are set.")

    magnitude_period = int(magnitude_window_param if magnitude_window_param is not None else 5)
    stretch_period = int(stretch_window_param if stretch_window_param is not None else 100)
    smooth_period = int(smooth_window_param if smooth_window_param is not None else 3)
    upper = float(parameters.get('upper', 70))
    lower = float(parameters.get('lower', 30))
    price_col = 'Close'
    indicator_col = f'DVI_{magnitude_period}_{stretch_period}_{smooth_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='dvi',
        parameters={"magnitude_period": magnitude_period, "stretch_period": stretch_period, "smooth_period": smooth_period},
        figure=False
    )
    
    data['upper'] = upper
    data['lower'] = lower
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [indicator_col, 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
