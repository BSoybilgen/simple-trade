import pandas as pd

from .indicator_handler import compute_indicator
from .cross_trade import CrossTradeBacktester
from .band_trade import BandTradeBacktester
from .plot_test import BacktestPlotter


def premade_backtest(data:pd.DataFrame, strategy_name:str, parameters:dict=None):
    
    if parameters is None:
        parameters = {}

    initial_cash = float(parameters.get('initial_cash', 10000.0))
    commission_long = float(parameters.get('commission_long', 0.001))
    commission_short = float(parameters.get('commission_short', 0.001))
    short_borrow_fee_inc_rate = float(parameters.get('short_borrow_fee_inc_rate', 0.0))
    long_borrow_fee_inc_rate = float(parameters.get('long_borrow_fee_inc_rate', 0.0))
    long_entry_pct_cash = float(parameters.get('long_entry_pct_cash', 1))
    short_entry_pct_cash = float(parameters.get('short_entry_pct_cash', 1))
    trading_type = str(parameters.get('trading_type', 'long'))
    day1_position = str(parameters.get('day1_position', 'none'))
    risk_free_rate = float(parameters.get('risk_free_rate', 0.0))

    plotter = BacktestPlotter()
    cross_backtester = CrossTradeBacktester(initial_cash=initial_cash, commission_long=commission_long, 
                                            commission_short=commission_short, short_borrow_fee_inc_rate=short_borrow_fee_inc_rate, 
                                            long_borrow_fee_inc_rate=long_borrow_fee_inc_rate)
    band_backtester = BandTradeBacktester(initial_cash=initial_cash, commission_long=commission_long, 
                                          commission_short=commission_short, short_borrow_fee_inc_rate=short_borrow_fee_inc_rate, 
                                          long_borrow_fee_inc_rate=long_borrow_fee_inc_rate)
    parameters_indicators = dict()

    fig_control = int(parameters.get('fig_control', 0))
    
    # Initialize default values for results and portfolio
    results = None
    portfolio = pd.DataFrame()
    fig = None

    if strategy_name == 'adx':
        window = int(parameters.get('window', 14))
        parameters_indicators["window"] = window
        short_window_indicator=f'+DI_{window}'
        long_window_indicator=f'-DI_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='adx',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (+DI_{window} vs -DI_{window})")

    elif strategy_name == 'aroon':
        period = int(parameters.get('period', 14))
        parameters_indicators["period"] = period
        short_window_indicator=f'AROON_UP_{period}'
        long_window_indicator=f'AROON_DOWN_{period}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='aroon',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (AROON_DOWN_{period} vs AROON_UP_{period})")

    elif strategy_name == 'ema':
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        short_window_indicator=f'EMA_{short_window}'
        long_window_indicator=f'EMA_{long_window}'
        price_col='Close'

        parameters["window"] = short_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='ema',
        parameters=parameters,
        figure=False)

        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='ema',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (EMA-{short_window} vs EMA-{long_window})")

    elif strategy_name=='ichimoku':
        tenkan_period = int(parameters.get('tenkan_period', 9))
        kijun_period = int(parameters.get('kijun_period', 26))
        senkou_b_period = int(parameters.get('senkou_b_period', 52))
        displacement = int(parameters.get('displacement', 26))
        parameters_indicators["tenkan_period"] = tenkan_period
        parameters_indicators["kijun_period"] = kijun_period
        parameters_indicators["senkou_b_period"] = senkou_b_period
        parameters_indicators["displacement"] = displacement
        short_window_indicator=f'tenkan_sen_{tenkan_period}'
        long_window_indicator=f'kijun_sen_{kijun_period}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='ichimoku',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (tenkan_sen_{tenkan_period}, kijun_sen_{kijun_period})")

    elif strategy_name=='psar':
        af_initial = float(parameters.get('af_initial', 0.03))
        af_step = float(parameters.get('af_step', 0.03))
        af_max = float(parameters.get('af_max', 0.3))
        parameters_indicators["af_initial"] = af_initial
        parameters_indicators["af_step"] = af_step
        parameters_indicators["af_max"] = af_max
        short_window_indicator="Close"
        long_window_indicator=f"PSAR_Bullish_{af_initial}_{af_step}_{af_max}"
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='psar',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (PSAR_Bullish_{af_initial}_{af_step}_{af_max} vs Close)")

    elif strategy_name == 'sma':
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        short_window_indicator=f'SMA_{short_window}'
        long_window_indicator=f'SMA_{long_window}'
        price_col='Close'

        parameters["window"] = short_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='sma',
        parameters=parameters,
        figure=False)

        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='sma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (SMA-{short_window} vs SMA-{long_window})")

    elif strategy_name == 'hma':
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        short_window_indicator=f'HMA_{short_window}'
        long_window_indicator=f'HMA_{long_window}'
        price_col='Close'

        parameters["window"] = short_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='hma',
        parameters=parameters,
        figure=False)

        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='hma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (HMA-{short_window} vs HMA-{long_window})")

    elif strategy_name=='strend':
        period = int(parameters.get('period', 7))
        multiplier = float(parameters.get('multiplier', 3.0))
        parameters_indicators["period"] = period
        parameters_indicators["multiplier"] = multiplier
        short_window_indicator="Close"
        long_window_indicator=f'Supertrend_Bullish_{period}_{multiplier}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='strend',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (f'Supertrend_Bullish_{period}_{multiplier}' vs Close)")

    elif strategy_name=='trix':
        window = int(parameters.get('window', 7))
        parameters_indicators["window"] = window
        short_window_indicator=f'TRIX_{window}'
        long_window_indicator=f'TRIX_SIGNAL_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='trix',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (TRIX-{window} vs TRIX_SIGNAL_{window})")

    elif strategy_name == 'wma':
        short_window = int(parameters.get('short_window', 25))
        long_window = int(parameters.get('long_window', 75))
        short_window_indicator=f'WMA_{short_window}'
        long_window_indicator=f'WMA_{long_window}'
        price_col='Close'

        parameters["window"] = short_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='wma',
        parameters=parameters,
        figure=False)

        parameters["window"] = long_window
        data, columns, fig = compute_indicator(
        data=data,
        indicator='wma',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (WMA-{short_window} vs WMA-{long_window})")

    elif strategy_name=='cci':
        window = int(parameters.get('window', 20))
        constant = float(parameters.get('constant', 0.015))
        upper = int(parameters.get('upper', 150))
        lower = int(parameters.get('lower', -150))
        parameters_indicators["window"] = window
        parameters_indicators["constant"] = constant
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'CCI_{window}_{constant}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cci',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'CCI_{window}_{constant}', 'lower', 'upper']

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"CCI Threshold (CCI_{window}_{constant} {lower}/{upper})")

    elif strategy_name=='macd':
        window_fast = int(parameters.get('window_fast', 12))
        window_slow = int(parameters.get('window_slow', 26))
        window_signal = int(parameters.get('window_signal', 26))
        parameters_indicators["window_fast"] = window_fast
        parameters_indicators["window_slow"] = window_slow
        parameters_indicators["window_signal"] = window_signal
        short_window_indicator=f'MACD_{window_fast}_{window_slow}'
        long_window_indicator=f'Signal_{window_signal}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='macd',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Cross Trade (MACD_{window_fast}_{window_slow} vs MACD_Signal_{window_signal})")

    elif strategy_name=='rsi':
        window = int(parameters.get('window', 14))
        upper = int(parameters.get('upper', 80))
        lower = int(parameters.get('lower', 20))
        parameters_indicators["window"] = window
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'RSI_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='rsi',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'RSI_{window}', 'lower', 'upper']

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"RSI Threshold (RSI_{window} {lower}/{upper})")

    elif strategy_name=='stoch':
        k_period = int(parameters.get('k_period', 14))
        d_period = int(parameters.get('d_period', 14))
        smooth_k = int(parameters.get('smooth_k', 14))
        upper = int(parameters.get('upper', 80))
        lower = int(parameters.get('lower', 20))
        parameters_indicators["k_period"] = k_period
        parameters_indicators["d_period"] = d_period
        parameters_indicators["smooth_k"] = smooth_k
        parameters_indicators["upper"] = upper
        parameters_indicators["lower"] = lower
        indicator_col=f'STOCH_D_{k_period}_{d_period}_{smooth_k}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='stoch',
        parameters=parameters_indicators,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = [f'STOCH_D_{k_period}_{d_period}_{smooth_k}', 'lower', 'upper']

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"STOCH Threshold (STOCH_D_{k_period}_{d_period}_{smooth_k} {lower}/{upper})")

    elif strategy_name=='bollin':
        window = int(parameters.get('window', 20))
        num_std = float(parameters.get('num_std', 2))
        parameters_indicators["window"] = window
        parameters_indicators["num_std"] = num_std
        indicator_col='Close'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='bollin',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'BB_Upper_{window}_{num_std}',
        lower_band_col=f'BB_Middle_{window}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['Close', f'BB_Upper_{window}_{num_std}', f'BB_Middle_{window}']

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Bollinger Threshold (Close BB_Upper_{window}_{num_std}/BB_Middle_{window})")

    elif strategy_name=='kelt':
        ema_window = int(parameters.get('ema_window', 20))
        atr_window = int(parameters.get('atr_window', 10))
        atr_multiplier = float(parameters.get('atr_multiplier', 2.0))
        parameters_indicators["ema_window"] = ema_window
        parameters_indicators["atr_window"] = atr_window
        parameters_indicators["atr_multiplier"] = atr_multiplier
        indicator_col='Close'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='kelt',
        parameters=parameters_indicators,
        figure=False)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}',
        lower_band_col=f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}',
        price_col=price_col,
        long_entry_pct_cash=long_entry_pct_cash,
        short_entry_pct_cash=short_entry_pct_cash,
        trading_type=trading_type,
        day1_position=day1_position,
        risk_free_rate=risk_free_rate)

        indicator_cols_to_plot = ['Close', f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}']

        if fig_control==1:
            fig = plotter.plot_results(
            data_df=data,
            history_df=portfolio,
            price_col=price_col,
            indicator_cols=indicator_cols_to_plot, 
            title=f"Keltner Threshold (Close KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}/KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier})")

    if fig_control==1:
        return results, portfolio, fig
    else:
        return results, portfolio, None