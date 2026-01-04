import pandas as pd
from ..moving_average.ema import ema
from .atr import atr


def kel(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Keltner Channels (kel), a volatility-based envelope set above and below an exponential moving average.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - ema_window (int): The window for the EMA calculation. Default is 20.
            - atr_window (int): The window for the ATR calculation. Default is 10.
            - atr_multiplier (float): Multiplier for the ATR to set channel width. Default is 2.0.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
    
    Returns:
        tuple: A tuple containing the Keltner Channels DataFrame and a list of column names.
    
    Calculation Steps:
    1. Middle Line:
       Exponential Moving Average (EMA) of the closing price over ema_window.
    2. Average True Range (ATR):
       ATR calculated over atr_window.
    3. Upper Band:
       Middle Line + (ATR * atr_multiplier)
    4. Lower Band:
       Middle Line - (ATR * atr_multiplier)
    
    Interpretation:
    - Price above Upper Band: Strong uptrend, potential overbought.
    - Price below Lower Band: Strong downtrend, potential oversold.
    - Middle Line slope indicates trend direction.
    
    Use Cases:
    - Identifying trend direction.
    - Spotting breakouts (price closing outside channels).
    - Overbought/oversold conditions (mean reversion to middle line).
    - Support and resistance (bands act as dynamic levels).
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    ema_window_param = parameters.get('ema_window')
    ema_period_param = parameters.get('ema_period')
    if ema_window_param is None and ema_period_param is not None:
        ema_window_param = ema_period_param
    elif ema_window_param is not None and ema_period_param is not None:
        if int(ema_window_param) != int(ema_period_param):
            raise ValueError("Provide either 'ema_window' or 'ema_period' (aliases) with the same value if both are set.")

    atr_window_param = parameters.get('atr_window')
    atr_period_param = parameters.get('atr_period')
    if atr_window_param is None and atr_period_param is not None:
        atr_window_param = atr_period_param
    elif atr_window_param is not None and atr_period_param is not None:
        if int(atr_window_param) != int(atr_period_param):
            raise ValueError("Provide either 'atr_window' or 'atr_period' (aliases) with the same value if both are set.")

    ema_window = int(ema_window_param if ema_window_param is not None else 20)
    atr_window = int(atr_window_param if atr_window_param is not None else 10)
    atr_multiplier = float(parameters.get('atr_multiplier', 2.0))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    close = df[close_col]
    
    # Calculate the middle line (EMA of close)
    ema_parameters = {'window': ema_window}
    ema_columns = {'close_col': close_col}
    middle_line_series, _ = ema(df, parameters=ema_parameters, columns=ema_columns)
    
    # Calculate ATR for the upper and lower bands
    atr_parameters = {'window': atr_window}
    atr_columns = {'high_col': high_col, 'low_col': low_col, 'close_col': close_col}
    atr_values_series, _ = atr(df, parameters=atr_parameters, columns=atr_columns)
    
    # Calculate the upper and lower bands
    upper_band = middle_line_series + (atr_values_series * atr_multiplier)
    lower_band = middle_line_series - (atr_values_series * atr_multiplier)
    
    # Prepare the result DataFrame
    result = pd.DataFrame({
        f'KEL_Middle_{ema_window}_{atr_window}_{atr_multiplier}': middle_line_series,
        f'KEL_Upper_{ema_window}_{atr_window}_{atr_multiplier}': upper_band,
        f'KEL_Lower_{ema_window}_{atr_window}_{atr_multiplier}': lower_band
    }, index=close.index)
    
    columns_list = list(result.columns)
    return result, columns_list


def strategy_kel(
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
    kel (Keltner Channels) - Mean Reversion Strategy
    
    LOGIC: Buy when price touches lower band (oversold),
           sell when price touches upper band (overbought).
    WHY: kel uses EMA with ATR-based bands. Price at lower band
         suggests oversold, at upper band suggests overbought.
    BEST MARKETS: Range-bound markets. Stocks, forex. Good for mean reversion.
                  Can also be used for breakout trading.
    TIMEFRAME: Daily charts. 20 EMA with 10 ATR and 2x multiplier is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'ema_window' (default 20), 'atr_window' (default 10),
                    'atr_multiplier' (default 2.0)
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
    
    ema_window_param = parameters.get('ema_window')
    ema_period_param = parameters.get('ema_period')
    if ema_window_param is None and ema_period_param is not None:
        ema_window_param = ema_period_param
    elif ema_window_param is not None and ema_period_param is not None:
        if int(ema_window_param) != int(ema_period_param):
            raise ValueError("Provide either 'ema_window' or 'ema_period' (aliases) with the same value if both are set.")

    atr_window_param = parameters.get('atr_window')
    atr_period_param = parameters.get('atr_period')
    if atr_window_param is None and atr_period_param is not None:
        atr_window_param = atr_period_param
    elif atr_window_param is not None and atr_period_param is not None:
        if int(atr_window_param) != int(atr_period_param):
            raise ValueError("Provide either 'atr_window' or 'atr_period' (aliases) with the same value if both are set.")

    ema_window = int(ema_window_param if ema_window_param is not None else 20)
    atr_window = int(atr_window_param if atr_window_param is not None else 10)
    atr_multiplier = float(parameters.get('atr_multiplier', 2.0))
    price_col = 'Close'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='kel',
        parameters={"ema_window": ema_window, "atr_window": atr_window, "atr_multiplier": atr_multiplier},
        figure=False
    )
    
    results, portfolio = run_band_trade(
        data=data,
        indicator_col='Close',
        upper_band_col=f'KEL_Upper_{ema_window}_{atr_window}_{atr_multiplier}',
        lower_band_col=f'KEL_Lower_{ema_window}_{atr_window}_{atr_multiplier}',
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = ['Close', f'KEL_Upper_{ema_window}_{atr_window}_{atr_multiplier}',
                              f'KEL_Middle_{ema_window}_{atr_window}_{atr_multiplier}',
                              f'KEL_Lower_{ema_window}_{atr_window}_{atr_multiplier}']
    
    return results, portfolio, indicator_cols_to_plot, data
