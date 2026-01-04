import numpy as np
import pandas as pd


def msi(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the Momentum Strength Index (msi), a custom momentum indicator.
    MSI aims to quantify the strength of price momentum by comparing recent gains 
    and losses, with an optional power parameter to emphasize stronger moves.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - window (int): Rolling window for averaging gains/losses. Default is 14.
            - power (float): Exponent applied to gains/losses to accentuate strong moves. Default is 1.0.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the MSI series and a list of column names.

    The Momentum Strength Index is calculated as follows:

    1. Calculate Price Differences:
       Diff = Close - Close(prev)

    2. Separate and Power-Scale Gains and Losses:
       Gain = (Diff if Diff > 0 else 0) ^ power
       Loss = (Abs(Diff) if Diff < 0 else 0) ^ power

    3. Calculate Average Gain and Loss:
       Avg Gain = SMA(Gain, window)
       Avg Loss = SMA(Loss, window)

    4. Calculate MSI:
       Strength Ratio = Avg Gain / Avg Loss
       MSI = 100 * Strength Ratio / (1 + Strength Ratio)

    Interpretation:
    - Range: 0 to 100.
    - Power Parameter: If power > 1, large moves have a disproportionately larger effect on the index, making it more sensitive to volatility spikes.
    - High Values: Strong upside momentum.
    - Low Values: Strong downside momentum.

    Use Cases:
    - Volatility-Adjusted Momentum: Using power > 1 allows traders to filter out low-volatility noise and focus on high-momentum moves.
    - Overbought/Oversold: Similar to RSI, can identify extremes.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 14)
    power = float(parameters.get('power', 1.0))
    close_col = columns.get('close_col', 'Close')

    series = df[close_col]
    delta = series.diff()

    gains = delta.clip(lower=0).pow(power)
    losses = (-delta.clip(upper=0)).pow(power)

    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()

    strength_ratio = avg_gain / avg_loss.replace({0: np.nan})
    msi_values = 100 * strength_ratio / (1 + strength_ratio)
    msi_values = msi_values.astype(float)
    msi_values.name = f'MSI_{window}_{power}'

    columns_list = [msi_values.name]
    return msi_values, columns_list


def strategy_msi(
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
    msi (Momentum Strength Index) - Mean Reversion Strategy
    
    LOGIC: Buy when msi drops below lower threshold (oversold), sell when above upper.
    WHY: msi quantifies momentum strength with optional power scaling. Similar to RSI
         but with adjustable sensitivity to large moves via power parameter.
    BEST MARKETS: Range-bound markets. Stocks and ETFs in consolidation.
                  Power > 1 emphasizes volatile moves, useful for breakout detection.
    TIMEFRAME: Daily charts. 14-period is standard. Adjust power for volatility sensitivity.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'window' (default 14), 'power' (default 1.0),
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
    
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is None and period_param is not None:
        window_param = period_param
    elif window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")
    window = int(window_param if window_param is not None else 14)
    power = float(parameters.get('power', 1.0))
    upper = int(parameters.get('upper', 70))
    lower = int(parameters.get('lower', 30))
    
    indicator_params = {"window": window, "power": power}
    indicator_col = f'MSI_{window}_{power}'
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='msi',
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
