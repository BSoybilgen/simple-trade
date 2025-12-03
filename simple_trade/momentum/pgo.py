import numpy as np
import pandas as pd


def pgo(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Pretty Good Oscillator (pgo), a momentum indicator developed by Mark Johnson.
    It measures the distance of the current close from its simple moving average, normalized by the average true range.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): The period for SMA and ATR smoothing. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the PGO series and a list of column names.

    The Pretty Good Oscillator is calculated as follows:

    1. Calculate True Range (TR):
       TR = Max(High-Low, Abs(High-PrevClose), Abs(Low-PrevClose))

    2. Calculate Average True Range (ATR):
       ATR = EMA(TR, window) (Note: Implementation uses EMA for smoothing TR)

    3. Calculate Simple Moving Average (SMA):
       SMA = SMA(Close, window)

    4. Calculate PGO:
       PGO = (Close - SMA) / ATR

    Interpretation:
    - Values > 3.0: Overbought condition.
    - Values < -3.0: Oversold condition.
    - Breakouts: Crossing above 3.0 or below -3.0 can indicate a strong trend initiation.

    Use Cases:
    - Overbought/Oversold: Identifying potential reversal points.
    - Trend Strength: High absolute values can indicate strong momentum.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window = int(parameters.get('window', 14))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')

    high = df[high_col]
    low_vals = df[low_col]
    close = df[close_col]

    # Calculate True Range
    prev_close = close.shift(1)
    tr1 = high - low_vals
    tr2 = (high - prev_close).abs()
    tr3 = (low_vals - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # EMA of True Range (often used for PGO)
    atr = tr.ewm(span=window, adjust=False).mean()

    # SMA of Close
    sma = close.rolling(window=window).mean()

    pgo_val = (close - sma) / atr.replace(0, np.nan)
    pgo_val.name = f'PGO_{window}'

    return pgo_val, [pgo_val.name]


def strategy_pgo(
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
    pgo (Pretty Good Oscillator) - Mean Reversion Strategy
    
    LOGIC: Buy when pgo drops below -3 (oversold), sell when rises above +3 (overbought).
    WHY: pgo measures distance from SMA normalized by ATR. Values beyond ±3 indicate
         price has moved significantly from its average relative to volatility.
    BEST MARKETS: Range-bound markets and mean-reverting assets. Stocks, forex, commodities.
                  Good for identifying overextended moves that may revert.
    TIMEFRAME: Daily charts. 14-period is standard. Adjust thresholds based on asset.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'upper' (default 3), 'lower' (default -3)
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
    
    window = int(parameters.get('window', 14))
    upper = float(parameters.get('upper', 3))
    lower = float(parameters.get('lower', -3))
    
    indicator_params = {"window": window}
    indicator_col = f'PGO_{window}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='pgo',
        parameters=indicator_params,
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
