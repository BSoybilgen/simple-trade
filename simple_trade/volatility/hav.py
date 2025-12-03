import pandas as pd


def hav(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Heikin-Ashi Volatility (hav), a volatility indicator that applies
    the Heikin-Ashi smoothing technique to price data and then measures the volatility
    of the smoothed candles to filter out market noise.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for volatility calculation. Default is 14.
            - method (str): Volatility calculation method - 'atr' or 'std'. Default is 'atr'.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the HAV series and a list of column names.

    Calculation Steps:
    1. Calculate Heikin-Ashi candles:
       HA_Close = (Open + High + Low + Close) / 4
       HA_Open = (Previous HA_Open + Previous HA_Close) / 2
       HA_High = max(High, HA_Open, HA_Close)
       HA_Low = min(Low, HA_Open, HA_Close)

    2. Calculate Volatility on Heikin-Ashi candles:
       
       If method = 'atr':
       - Calculate True Range on HA candles:
         TR = max(HA_High - HA_Low, abs(HA_High - prev_HA_Close), abs(HA_Low - prev_HA_Close))
       - Apply Wilder's smoothing:
         HAV = smoothed average of TR over period
       
       If method = 'std':
       - Calculate standard deviation of HA_Close over period:
         HAV = std(HA_Close, period)

    Interpretation:
    - Lower HAV: Low volatility, potential consolidation.
    - Higher HAV: High volatility, potential trending conditions.
    - Rising HAV: Increasing volatility, potential breakout.
    - Falling HAV: Decreasing volatility, potential consolidation.

    Use Cases:
    - Trend identification: High HAV indicates active trends.
    - Breakout detection: Sharp increases in HAV.
    - Noise reduction: Smoother than standard ATR.
    - Volatility compression: Very low HAV precedes explosive moves.
    - Position sizing: Use HAV to adjust position sizes based on current market volatility levels.
    - Market regime filtering: Filter trades based on HAV levels - avoid trading during extremely high or low volatility periods.
    - Smoother signals: HAV provides cleaner volatility signals compared to regular ATR.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    period = int(parameters.get('period', 14))
    method = parameters.get('method', 'atr').lower()
    open_col = columns.get('open_col', 'Open')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    open_price = df[open_col]
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate Heikin-Ashi candles
    ha_close = (open_price + high + low + close) / 4
    
    # Initialize HA_Open
    ha_open = pd.Series(index=close.index, dtype=float)
    ha_open.iloc[0] = (open_price.iloc[0] + close.iloc[0]) / 2
    
    # Calculate HA_Open iteratively
    for i in range(1, len(close)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    
    # Calculate HA_High and HA_Low
    ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)
    
    # Calculate volatility based on method
    if method == 'atr':
        # Calculate True Range on Heikin-Ashi candles
        prev_ha_close = ha_close.shift(1)
        tr1 = ha_high - ha_low
        tr2 = (ha_high - prev_ha_close).abs()
        tr3 = (ha_low - prev_ha_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Apply Wilder's smoothing method
        hav_values = pd.Series(index=close.index, dtype=float)
        first_hav = tr.iloc[:period].mean()
        hav_values.iloc[period-1] = first_hav
        
        for i in range(period, len(close)):
            hav_values.iloc[i] = ((hav_values.iloc[i-1] * (period-1)) + tr.iloc[i]) / period
    
    elif method == 'std':
        # Calculate standard deviation of HA_Close
        hav_values = ha_close.rolling(window=period).std()
    
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'atr' or 'std'.")
    
    hav_values.name = f'HAV_{period}_{method.upper()}'
    columns_list = [hav_values.name]
    return hav_values, columns_list


def strategy_hav(
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
    hav (Heikin-Ashi Volatility) - Volatility Threshold Strategy
    
    LOGIC: Buy when hav drops below lower percentile (low volatility squeeze),
           sell when rises above upper percentile (high volatility).
    WHY: hav applies Heikin-Ashi smoothing to filter noise before measuring
         volatility. Provides cleaner signals than standard ATR.
    BEST MARKETS: All markets. Good for volatility-based strategies.
    TIMEFRAME: Daily charts. 14-period is standard.
    NOTE: Uses rolling percentile bands since HAV is in price units.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14), 'upper_pct' (default 80),
                    'lower_pct' (default 20), 'lookback' (default 100)
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
    
    period = int(parameters.get('period', 14))
    method = parameters.get('method', 'atr').upper()
    upper_pct = float(parameters.get('upper_pct', 80))
    lower_pct = float(parameters.get('lower_pct', 20))
    lookback = int(parameters.get('lookback', 100))
    price_col = 'Close'
    indicator_col = f'HAV_{period}_{method}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='hav',
        parameters={"period": period, "method": method.lower()},
        figure=False
    )
    
    # Calculate rolling percentile bands for HAV
    data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=period).quantile(upper_pct / 100)
    data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=period).quantile(lower_pct / 100)
    
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
