import pandas as pd
import numpy as np


def mfi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Money Flow Index (mfi), a momentum indicator that uses price
    and volume to identify overbought or oversold conditions. It is often referred
    to as volume-weighted RSI.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for MFI calculation. Default is 14.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.
            - volume_col (str): The column name for volume. Default is 'Volume'.

    Returns:
        tuple: A tuple containing the MFI series and a list of column names.

    The Money Flow Index is calculated as follows:

    1. Calculate Typical Price (TP):
       TP = (High + Low + Close) / 3

    2. Calculate Raw Money Flow:
       RMF = TP * Volume

    3. Separate Money Flow:
       - Positive Flow: RMF if TP > Previous TP
       - Negative Flow: RMF if TP < Previous TP

    4. Calculate Money Flow Ratio (MFR):
       MFR = Sum(Positive Flow, period) / Sum(Negative Flow, period)

    5. Calculate MFI:
       MFI = 100 - (100 / (1 + MFR))

    Interpretation:
    - MFI > 80: Overbought (potential sell).
    - MFI < 20: Oversold (potential buy).
    - Divergence: Price vs MFI disagreement signals potential reversal.

    Use Cases:
    - Overbought/Oversold: Identifying market extremes.
    - Trend Confirmation: MFI trending with price confirms strength.
    - Divergence Trading: Spotting reversals earlier than price alone.
    - Failure Swings: MFI failing to reach extremes in a trend.
    """
    # Set default values
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    # Extract parameters with defaults
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else 14)
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    volume_col = columns.get('volume_col', 'Volume')
    
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col]
    
    # Calculate Typical Price
    typical_price = (high + low + close) / 3
    
    # Calculate Raw Money Flow
    raw_money_flow = typical_price * volume
    
    # Identify positive and negative money flow
    typical_price_change = typical_price.diff()
    
    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)
    
    positive_flow[typical_price_change > 0] = raw_money_flow[typical_price_change > 0]
    negative_flow[typical_price_change < 0] = raw_money_flow[typical_price_change < 0]
    
    # Calculate rolling sums of positive and negative money flow
    positive_mf = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=period, min_periods=1).sum()
    
    # Calculate Money Flow Ratio (handle division by zero)
    money_flow_ratio = positive_mf / negative_mf.replace(0, np.nan)
    
    # Calculate Money Flow Index
    mfi_values = 100 - (100 / (1 + money_flow_ratio))
    mfi_values = mfi_values.fillna(50)  # Neutral value when undefined
    
    mfi_values.name = f'MFI_{period}'
    columns_list = [mfi_values.name]
    return mfi_values, columns_list


def strategy_mfi(
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
    mfi (Money Flow Index) - Overbought/Oversold Strategy
    
    LOGIC: Buy when mfi drops below lower threshold (oversold),
           sell when rises above upper threshold (overbought).
    WHY: mfi is volume-weighted RSI. Combines price and volume to identify
         overbought/oversold conditions. Good for divergence detection.
    BEST MARKETS: Stocks, ETFs. Good for mean reversion and divergence.
    TIMEFRAME: Daily charts. 14-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 14), 'upper' (default 80),
                    'lower' (default 20)
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else 14)
    upper = float(parameters.get('upper', 80))
    lower = float(parameters.get('lower', 20))
    price_col = 'Close'
    indicator_col = f'MFI_{period}'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='mfi',
        parameters={"period": period},
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
