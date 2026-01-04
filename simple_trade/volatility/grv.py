import pandas as pd
import numpy as np


def grv(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Garman-Klass Volatility (grv), a more efficient volatility estimator that uses
    OHLC data instead of just closing prices. It provides better volatility estimates
    with the same amount of data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period for calculation. Default is 20.
            - trading_periods (int): Trading periods per year for annualization. Default is 252.
            - annualized (bool): Whether to annualize the volatility. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - open_col (str): The column name for open prices. Default is 'Open'.
            - high_col (str): The column name for high prices. Default is 'High'.
            - low_col (str): The column name for low prices. Default is 'Low'.
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing the Garman-Klass Volatility series and a list of column names.

    Calculation Steps:
    1. Calculate Log Ratios:
       log_hl = ln(High / Low)
       log_co = ln(Close / Open)

    2. Calculate GRV Component for each period:
       GRV = 0.5 * (log_hl)^2 - (2 * ln(2) - 1) * (log_co)^2

    3. Average and Square Root:
       GRV_Vol = sqrt(Average(GRV, period))

    4. Annualize (optional):
       Annualized GRV = GRV_Vol * sqrt(trading_periods)

    Interpretation:
    - Lower values: Low volatility.
    - Higher values: High volatility.
    - More efficient than close-to-close volatility as it uses intrabar information.

    Use Cases:
    - Superior volatility estimation compared to standard deviation.
    - Options pricing and risk management.
    - More accurate with same data as Historical Volatility.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    window_param = parameters.get('window')
    period_param = parameters.get('period')
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 20))
    trading_periods = int(parameters.get('trading_periods', 252))
    annualized = bool(parameters.get('annualized', True))
    open_col = columns.get('open_col', 'Open')
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    close_col = columns.get('close_col', 'Close')
    
    open_price = df[open_col]
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    
    # Calculate log ratios
    log_hl = np.log(high / low)
    log_co = np.log(close / open_price)
    
    # Garman-Klass formula
    # GRV = sqrt(0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2)
    grv_component = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    # Take rolling mean and square root
    grv_variance = grv_component.rolling(window=period).mean()
    grv_volatility = np.sqrt(grv_variance)
    
    # Annualize if requested
    if annualized:
        grv_volatility = grv_volatility * np.sqrt(trading_periods)
    
    # Convert to percentage
    grv_values = grv_volatility * 100
    
    if annualized:
        grv_values.name = f'GRV_VOL_{period}_Ann'
    else:
        grv_values.name = f'GRV_VOL_{period}'
    
    columns_list = [grv_values.name]
    return grv_values, columns_list


def strategy_grv(
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
    grv (Garman-Klass Volatility) - Volatility Threshold Strategy
    
    LOGIC: Buy when grv drops below lower threshold (low vol squeeze),
           sell when rises above upper threshold (high volatility).
    WHY: grv is more efficient than close-to-close volatility,
         using OHLC data. Low volatility often precedes breakouts.
    BEST MARKETS: All markets. Good for volatility-based strategies.
    TIMEFRAME: Daily charts. 20-period is standard.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'period' (default 20), 'upper' (default 30),
                    'lower' (default 15)
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
    if window_param is not None and period_param is not None:
        if int(window_param) != int(period_param):
            raise ValueError("Provide either 'window' or 'period' (aliases) with the same value if both are set.")

    period = int(window_param if window_param is not None else (period_param if period_param is not None else 20))
    upper = float(parameters.get('upper', 30))
    lower = float(parameters.get('lower', 15))
    price_col = 'Close'
    indicator_col = f'GRV_VOL_{period}_Ann'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='grv',
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
