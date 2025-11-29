import pandas as pd


def vqi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Volatility Quality Index (VQI), an indicator that measures the quality
    of price movements by analyzing the relationship between price changes, volume, and
    volatility to identify genuine trends versus noise.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculations. Default is 9.
            - smooth_period (int): The period for smoothing the VQI. Default is 9.
            - volatility_cutoff (float): Multiplier for ATR to filter noise. Default is 0.1.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the VQI series and a list of column names.

    Calculation Steps:
    1. Calculate True Range (TR) and ATR.
    2. Calculate Price Change:
       Change = Close - Previous Close
    3. Filter Noise:
       If abs(Change) < (volatility_cutoff * ATR), then Change is treated as 0.
    4. Calculate Directional Volume:
       If Change > 0: Move = Change * Volume
       If Change < 0: Move = Change * Volume
       Else: Move = 0
    5. Sum Movements:
       VQI_Raw = Sum(Move, period)
    6. Smooth VQI:
       VQI = EMA(VQI_Raw, smooth_period)

    Interpretation:
    - Positive VQI: Quality uptrend supported by volume.
    - Negative VQI: Quality downtrend supported by volume.
    - Near Zero: Choppy/Noisy market or low volume movements.

    Use Cases:
    - Trend Quality Assessment: Distinguishing between real trends and false moves.
    - Trade Filtering: Avoiding trades in low VQI conditions.
    - Divergence: Spotting disagreements between price and VQI.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 9))
    smooth_period = int(parameters.get('smooth_period', 9))
    volatility_cutoff = float(parameters.get('volatility_cutoff', 0.1))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR using EMA
    atr = tr.ewm(span=period, adjust=False).mean()
    
    # Calculate price change
    price_change = close.diff()
    
    # Calculate cutoff threshold
    cutoff = volatility_cutoff * atr
    
    # Calculate volume-weighted directional movement
    vqi_raw = pd.Series(0.0, index=close.index)
    
    # Positive movements (above cutoff)
    positive_mask = price_change > cutoff
    vqi_raw[positive_mask] = price_change[positive_mask] * volume[positive_mask]
    
    # Negative movements (below -cutoff)
    negative_mask = price_change < -cutoff
    vqi_raw[negative_mask] = price_change[negative_mask] * volume[negative_mask]
    
    # Sum over period
    vqi_sum = vqi_raw.rolling(window=period).sum()
    
    # Apply smoothing
    vqi_smoothed = vqi_sum.ewm(span=smooth_period, adjust=False).mean()
    
    vqi_smoothed.name = f'VQI_{period}_{smooth_period}'
    columns_list = [vqi_smoothed.name]
    return vqi_smoothed, columns_list


def strategy_vqi(
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
    VQI (Volatility Quality Index) - Trend Quality Strategy
    
    LOGIC: Buy when VQI rises above upper threshold (quality uptrend),
           sell when drops below lower threshold (quality downtrend).
    WHY: VQI measures trend quality by analyzing price, volume, and volatility.
         Positive = quality uptrend, negative = quality downtrend.
    BEST MARKETS: All markets with volume data. Good for trend quality assessment.
    TIMEFRAME: Daily charts. 9-period with 9-period smoothing is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period', 'smooth_period', 'upper', 'lower'
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
    
    period = int(parameters.get('period', 9))
    smooth_period = int(parameters.get('smooth_period', 9))
    upper = float(parameters.get('upper', 0))
    lower = float(parameters.get('lower', 0))
    price_col = 'Close'
    indicator_col = f'VQI_{period}_{smooth_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='vqi',
        parameters={"period": period, "smooth_period": smooth_period},
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
