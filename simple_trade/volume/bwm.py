import numpy as np


def bwm(df, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Bill Williams Market Facilitation Index (BW MFI), which measures
    the efficiency of price movement by analyzing the change in price per unit of volume.
    It helps determine the willingness of the market to move the price.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters.
            No parameters are used.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the BW MFI series and a list of column names.

    The Market Facilitation Index is calculated as follows:

    1. Calculate Range:
       Range = High - Low

    2. Calculate BW MFI:
       BW MFI = Range / Volume

    Interpretation (when combined with Volume):
    - Green (MFI Up, Vol Up): Strong trend, increasing participation.
    - Fade (MFI Down, Vol Down): Market losing interest, potential reversal.
    - Fake (MFI Up, Vol Down): Price moving without volume support (speculative).
    - Squat (MFI Down, Vol Up): High volume but little movement, battle between bulls/bears.

    Use Cases:
    - Trend Strength: Identify if price moves are supported by volume.
    - Reversal Warning: "Squat" bars often precede reversals.
    - Filtering: Avoid trading during "Fade" conditions.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    volume = df[volume_col]
    
    # Calculate MFI
    # Handle division by zero by replacing 0 volume with NaN (or a very small number if preferred)
    # Using NaN results in NaN MFI, which is safer than Infinity
    mfi_values = (high - low) / volume.replace(0, np.nan)
    mfi_values = mfi_values.fillna(0)
    
    mfi_values.name = 'BWMFI'
    columns_list = [mfi_values.name]
    return mfi_values, columns_list


def strategy_bwm(
    data,
    parameters: dict = None,
    config = None,
    trading_type: str = 'long',
    day1_position: str = 'none',
    risk_free_rate: float = 0.0,
    long_entry_pct_cash: float = 1.0,
    short_entry_pct_cash: float = 1.0
) -> tuple:
    """
    BWM (Bill Williams Market Facilitation Index) - Percentile Strategy
    
    LOGIC: Buy when BWMFI drops below lower percentile (low facilitation),
           sell when rises above upper percentile (high facilitation).
    WHY: BWMFI measures price movement efficiency per unit of volume.
         Low values indicate consolidation, high values indicate strong moves.
    BEST MARKETS: All markets. Good for identifying breakout potential.
    TIMEFRAME: Daily charts. Good for swing trading.
    NOTE: Uses rolling percentile bands since BWMFI values vary by asset.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'upper_pct' (default 80), 'lower_pct' (default 20),
                    'lookback' (default 100)
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
    
    upper_pct = float(parameters.get('upper_pct', 80))
    lower_pct = float(parameters.get('lower_pct', 20))
    lookback = int(parameters.get('lookback', 100))
    price_col = 'Close'
    indicator_col = 'BWMFI'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='bwm',
        parameters={},
        figure=False
    )
    
    # Calculate rolling percentile bands
    data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=20).quantile(upper_pct / 100)
    data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=20).quantile(lower_pct / 100)
    
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
    
    indicator_cols_to_plot = ['BWMFI', 'lower', 'upper']
    
    return results, portfolio, indicator_cols_to_plot, data
