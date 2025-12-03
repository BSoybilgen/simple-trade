import pandas as pd
import numpy as np


def pav(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates Parkinson Volatility (pav), an efficient volatility estimator that uses only
    the high-low range. It's more efficient than close-to-close volatility when there
    are no overnight gaps.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - period (int): The lookback period. Default is 20.
            - trading_periods (int): Trading periods per year. Default is 252.
            - annualized (bool): Whether to annualize. Default is True.
        columns (dict, optional): Dictionary containing column name mappings:
            - high_col (str): High prices column. Default is 'High'.
            - low_col (str): Low prices column. Default is 'Low'.

    Returns:
        tuple: Parkinson Volatility series and column names list.

    Calculation Steps:
    1. Calculate Log Ratios:
       log_hl = ln(High / Low)
    2. Calculate Parkinson Component:
       Comp = (1 / (4 * ln(2))) * (log_hl)^2
    3. Average and Square Root:
       Vol = sqrt(Average(Comp, period))
    4. Annualize (optional):
       Parkinson = Vol * sqrt(trading_periods) * 100

    Interpretation:
    - Lower values: Low volatility.
    - Higher values: High volatility.
    - Uses intraday range, so it captures volatility missed by close-to-close measures.

    Use Cases:
    - Efficient volatility estimation using high-low range.
    - Better than standard deviation when no overnight gaps.
    - Risk management and options pricing.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}
        
    period = int(parameters.get('period', 20))
    trading_periods = int(parameters.get('trading_periods', 252))
    annualized = bool(parameters.get('annualized', True))
    high_col = columns.get('high_col', 'High')
    low_col = columns.get('low_col', 'Low')
    
    high = df[high_col]
    low = df[low_col]
    
    # Parkinson formula
    log_hl = np.log(high / low)
    parkinson_component = (1 / (4 * np.log(2))) * (log_hl ** 2)
    
    # Rolling mean and square root
    pav_variance = parkinson_component.rolling(window=period).mean()
    pav_volatility = np.sqrt(pav_variance)
    
    if annualized:
        pav_volatility = pav_volatility * np.sqrt(trading_periods)
    
    pav_values = pav_volatility * 100
    
    pav_values.name = f'PAV_VOL_{period}_Ann' if annualized else f'PAV_VOL_{period}'
    return pav_values, [pav_values.name]


def strategy_pav(
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
    pav (Parkinson Volatility) - Volatility Threshold Strategy
    
    LOGIC: Buy when pav drops below lower threshold (low volatility),
           sell when rises above upper threshold (high volatility).
    WHY: pav uses high-low range for efficient volatility estimation.
         Better than close-to-close when no overnight gaps.
    BEST MARKETS: All markets. Good for intraday volatility analysis.
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
    
    period = int(parameters.get('period', 20))
    upper = float(parameters.get('upper', 30))
    lower = float(parameters.get('lower', 15))
    price_col = 'Close'
    indicator_col = f'PAV_VOL_{period}_Ann'
    
    data, _, _ = compute_indicator(
        data=data,
        indicator='pav',
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
