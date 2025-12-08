import pandas as pd


def tsv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the True Strength Index (tsv) Volatility, which applies the TSI momentum
    indicator formula to volatility measures (ATR or standard deviation) to create a
    double-smoothed volatility momentum indicator.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - atr_period (int): Period for ATR calculation. Default is 14.
            - long_period (int): Long smoothing period. Default is 25.
            - short_period (int): Short smoothing period. Default is 13.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the TSV series and a list of column names.

    Calculation Steps:
    1. Calculate ATR (Average True Range).
    2. Calculate ATR Momentum:
       Momentum = ATR - ATR(previous)
    3. Double Smooth Momentum:
       Smooth1 = EMA(Momentum, long_period)
       Smooth2 = EMA(Smooth1, short_period)
    4. Double Smooth Absolute Momentum:
       AbsSmooth1 = EMA(|Momentum|, long_period)
       AbsSmooth2 = EMA(AbsSmooth1, short_period)
    5. Calculate TSV:
       TSV = 100 * (Smooth2 / AbsSmooth2)

    Interpretation:
    - Positive TSV: Rising volatility (Momentum Up).
    - Negative TSV: Falling volatility (Momentum Down).
    - Zero Line Crossovers: Trend changes in volatility.

    Use Cases:
    - Volatility trend identification.
    - Divergence detection (price vs volatility momentum).
    - Filtering noise with double smoothing.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    atr_period = int(parameters.get('atr_period', 14))
    long_period = int(parameters.get('long_period', 25))
    short_period = int(parameters.get('short_period', 13))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.ewm(span=atr_period, adjust=False).mean()
    
    # Calculate ATR momentum (change)
    atr_momentum = atr.diff()
    
    # Double smoothing of momentum
    smooth1 = atr_momentum.ewm(span=long_period, adjust=False).mean()
    smooth2 = smooth1.ewm(span=short_period, adjust=False).mean()
    
    # Double smoothing of absolute momentum
    abs_momentum = atr_momentum.abs()
    abs_smooth1 = abs_momentum.ewm(span=long_period, adjust=False).mean()
    abs_smooth2 = abs_smooth1.ewm(span=short_period, adjust=False).mean()
    
    # Calculate TSV
    tsv_values = 100 * (smooth2 / abs_smooth2)
    
    # Set warmup period to NaN to avoid extreme values during initialization
    # Warmup needs: 1 for prev_close, atr_period for ATR EMA, 1 for diff, 
    # long_period for first EMA, short_period for second EMA
    warmup_period = 1 + atr_period + 1 + long_period + short_period
    tsv_values.iloc[:warmup_period] = float('nan')
    
    tsv_values.name = f'TSV_{atr_period}_{long_period}_{short_period}'
    columns_list = [tsv_values.name]
    return tsv_values, columns_list


def strategy_tsv(
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
    tsv (TSV) - Volatility Momentum Strategy
    
    LOGIC: Buy when tsv crosses above zero (rising volatility momentum),
           sell when crosses below zero (falling volatility momentum).
    WHY: tsv applies TSI formula to ATR for double-smoothed volatility momentum.
         Positive = rising volatility, negative = falling volatility.
    BEST MARKETS: All markets. Good for volatility trend identification.
    TIMEFRAME: Daily charts. 14 ATR with 25/13 smoothing is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'atr_period', 'long_period', 'short_period',
                    'upper' (default 25), 'lower' (default -25)
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
    
    atr_period = int(parameters.get('atr_period', 14))
    long_period = int(parameters.get('long_period', 25))
    short_period = int(parameters.get('short_period', 13))
    upper = float(parameters.get('upper', 25))
    lower = float(parameters.get('lower', -25))
    price_col = 'Close'
    indicator_col = f'TSV_{atr_period}_{long_period}_{short_period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='tsv',
        parameters={"atr_period": atr_period, "long_period": long_period, "short_period": short_period},
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
