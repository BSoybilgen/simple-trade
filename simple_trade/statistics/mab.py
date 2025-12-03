import pandas as pd
import numpy as np


def mab(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Mean Absolute Deviation (mab), a statistical measure of dispersion that
    quantifies the average absolute distance of data points from the mean.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The lookback period for the calculation. Default is 20.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Mean Absolute Deviation series and a list of column names.

    Calculation Steps:
    1. Calculate rolling mean over the window.
    2. Calculate absolute deviation from mean for each point.
    3. Calculate average of absolute deviations.

    Interpretation:
    - High mab: High dispersion, prices are spread out from the mean.
    - Low mab: Low dispersion, prices are close to the mean.
    - mab is more robust to outliers than standard deviation.

    Use Cases:
    - Volatility measurement (robust alternative to STD).
    - Risk assessment.
    - Trend strength confirmation.
    - Position sizing adjustment.
    - Mean reversion strategies.

    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window = int(parameters.get('window', 20))
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate rolling mean absolute deviation
    def calc_mad(x):
        return np.mean(np.abs(x - np.mean(x)))
    
    mab_values = close.rolling(window=window).apply(calc_mad, raw=True)
    
    mab_values.name = f'MAB_{window}'
    columns_list = [mab_values.name]
    return mab_values, columns_list


def strategy_mab(
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
    mab (Mean Absolute Deviation) - Volatility Threshold Strategy
    
    LOGIC: Buy when mab drops below lower percentile (low volatility squeeze),
           sell when rises above upper percentile (high volatility).
    WHY: mab measures price dispersion robustly. Low mab indicates consolidation,
         high mab indicates volatility expansion.
    BEST MARKETS: All markets. Good for volatility-based strategies.
    TIMEFRAME: Daily charts. 20-period is standard.
    NOTE: Uses rolling percentile bands since MAB is in price units.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 20), 'upper_pct' (default 80),
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
    
    window = int(parameters.get('window', 20))
    upper_pct = float(parameters.get('upper_pct', 80))
    lower_pct = float(parameters.get('lower_pct', 20))
    lookback = int(parameters.get('lookback', 100))
    price_col = 'Close'
    indicator_col = f'MAB_{window}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='mab',
        parameters={"window": window},
        figure=False
    )
    
    # Calculate rolling percentile bands for MAB
    data['upper'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(upper_pct / 100)
    data['lower'] = data[indicator_col].rolling(window=lookback, min_periods=window).quantile(lower_pct / 100)
    
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
