import numpy as np
import pandas as pd


def mam(df: pd.DataFrame, parameters: dict = None, columns: dict = None) -> tuple:
    """
    Calculates the MESA Adaptive Moving Average (mam).
    mam is an adaptive moving average developed by John Ehlers that adjusts to
    the market's phase rate of change. It uses a Hilbert Transform to measure
    the dominant cycle period.

    Args:
        df (pd.DataFrame): The input DataFrame.
        parameters (dict, optional): Dictionary containing calculation parameters:
            - fast_limit (float): Fast limit for alpha. Default is 0.5.
            - slow_limit (float): Slow limit for alpha. Default is 0.05.
        columns (dict, optional): Dictionary containing column name mappings:
            - close_col (str): The column name for closing prices. Default is 'Close'.

    Returns:
        tuple: A tuple containing a DataFrame with MAM and FAMA columns, and a list of column names.

    The mam is calculated as follows:

    1. Calculate Smooth Price using weighted average.
    2. Compute Hilbert Transform components (InPhase, Quadrature).
    3. Calculate Phase and DeltaPhase.
    4. Determine Alpha based on phase rate of change.
    5. Apply adaptive smoothing:
       MAM = alpha * Price + (1 - alpha) * Prev MAM
       FAMA = 0.5 * alpha * MAM + (1 - 0.5 * alpha) * Prev FAMA

    Interpretation:
    - MAM (MAMA) is the fast adaptive line.
    - FAMA (Following Adaptive MA) is the slow line.
    - When MAM crosses above FAMA, it's a bullish signal.

    Use Cases:
    - Trend Following: Excellent for distinguishing trending vs cycling markets.
    - Crossovers: MAM/FAMA crossover for entry/exit signals.
    """
    if parameters is None:
        parameters = {}
    if columns is None:
        columns = {}

    close_col = columns.get('close_col', 'Close')
    fast_limit = float(parameters.get('fast_limit', 0.5))
    slow_limit = float(parameters.get('slow_limit', 0.05))

    price = df[close_col].values
    n = len(price)

    # Initialize arrays
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    i1 = np.zeros(n)
    q1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.zeros(n)
    smooth_period = np.zeros(n)
    phase = np.zeros(n)
    mama = np.zeros(n)
    fama = np.zeros(n)

    # Handle initial values
    # Initialize with price to avoid startup artifacts
    mama[:6] = price[:6]
    fama[:6] = price[:6]
    smooth[:6] = price[:6]
    period[:6] = 10.0
    smooth_period[:6] = 10.0

    # Hilbert Transform constants
    for i in range(6, n):
        # Smooth price
        smooth[i] = (4 * price[i] + 3 * price[i-1] + 2 * price[i-2] + price[i-3]) / 10.0

        # Compute Detrender
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i-2] -
                       0.5769 * smooth[i-4] - 0.0962 * smooth[i-6]) * (0.075 * period[i-1] + 0.54)

        # Compute InPhase and Quadrature components
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i-2] -
                0.5769 * detrender[i-4] - 0.0962 * detrender[i-6]) * (0.075 * period[i-1] + 0.54)
        i1[i] = detrender[i-3]

        # Advance the phase of I1 and Q1 by 90 degrees
        ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i-2] -
                0.5769 * i1[i-4] - 0.0962 * i1[i-6]) * (0.075 * period[i-1] + 0.54)
        jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i-2] -
                0.5769 * q1[i-4] - 0.0962 * q1[i-6]) * (0.075 * period[i-1] + 0.54)

        # Phasor addition for 3-bar averaging
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]

        # Smooth the I and Q components before applying the discriminator
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i-1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i-1]

        # Homodyne Discriminator
        re[i] = i2[i] * i2[i-1] + q2[i] * q2[i-1]
        im[i] = i2[i] * q2[i-1] - q2[i] * i2[i-1]
        re[i] = 0.2 * re[i] + 0.8 * re[i-1]
        im[i] = 0.2 * im[i] + 0.8 * im[i-1]

        if im[i] != 0 and re[i] != 0:
            period[i] = 2 * np.pi / np.arctan(im[i] / re[i])
        if period[i] > 1.5 * period[i-1]:
            period[i] = 1.5 * period[i-1]
        if period[i] < 0.67 * period[i-1]:
            period[i] = 0.67 * period[i-1]
        if period[i] < 6:
            period[i] = 6
        if period[i] > 50:
            period[i] = 50
        period[i] = 0.2 * period[i] + 0.8 * period[i-1]
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i-1]

        # Compute Phase
        if i1[i] != 0:
            phase[i] = np.degrees(np.arctan(q1[i] / i1[i]))

        # Compute DeltaPhase
        delta_phase = phase[i-1] - phase[i]
        if delta_phase < 1:
            delta_phase = 1

        # Compute Alpha
        alpha = fast_limit / delta_phase
        if alpha < slow_limit:
            alpha = slow_limit
        if alpha > fast_limit:
            alpha = fast_limit

        # Compute MAMA and FAMA
        mama[i] = alpha * price[i] + (1 - alpha) * mama[i-1]
        fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i-1]

    # Handle initial values
    # (Moved to before loop)

    result_df = pd.DataFrame({
        'MAM': mama,
        'MAM_FAMA': fama
    }, index=df.index)

    columns_list = ['MAM', 'MAM_FAMA']
    return result_df, columns_list


def strategy_mam(
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
    mam (MESA Adaptive Moving Average) - MAMA/FAMA Crossover Strategy
    
    LOGIC: Buy when MAM crosses above FAMA, sell when crosses below.
    WHY: mam adapts to market cycles using Hilbert Transform. Distinguishes
         between trending and cycling markets. Developed by John Ehlers.
    BEST MARKETS: All markets. Particularly effective for cyclical instruments.
                  Stocks, forex, commodities with clear cycles.
    TIMEFRAME: Daily or higher. Works best on instruments with clear cyclical behavior.
    
    Args:
        data: DataFrame with OHLCV data
        parameters: Dict with 'fast_limit' (default 0.5), 'slow_limit' (default 0.05)
        config: BacktestConfig object for backtest settings
        trading_type: 'long', 'short', or 'both'
        day1_position: Initial position ('none', 'long', 'short')
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        long_entry_pct_cash: Percentage of cash to use for long entries
        short_entry_pct_cash: Percentage of cash to use for short entries
        
    Returns:
        tuple: (results_dict, portfolio_df, indicator_cols_to_plot, data_with_indicators)
    """
    from ..run_cross_trade_strategies import run_cross_trade
    from ..compute_indicators import compute_indicator
    
    if parameters is None:
        parameters = {}
    
    price_col = 'Close'
    
    data, columns, _ = compute_indicator(
        data=data,
        indicator='mam',
        parameters=parameters,
        figure=False
    )
    
    short_window_indicator = 'MAM'
    long_window_indicator = 'MAM_FAMA'
    
    results, portfolio = run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        config=config,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate
    )
    
    indicator_cols_to_plot = [short_window_indicator, long_window_indicator]
    
    return results, portfolio, indicator_cols_to_plot, data
