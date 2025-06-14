import pandas as pd

from simple_trade.indicator_handler import download_data, compute_indicator
from simple_trade.cross_trade import CrossTradeBacktester
from simple_trade.band_trade import BandTradeBacktester
from simple_trade.plot_test import BacktestPlotter


def premade_trade(data:pd.DataFrame, strategy_name:str, parameters:dict=None):

    initial_cash = parameters.get('initial_cash', 10000.0)
    commission_long = parameters.get('commission_long', 0.001)
    commission_short = parameters.get('commission_short', 0.001)
    short_borrow_fee_inc_rate = parameters.get('short_borrow_fee_inc_rate', 0.0)
    long_borrow_fee_inc_rate = parameters.get('long_borrow_fee_inc_rate', 0.0)

    plotter = BacktestPlotter()
    cross_backtester = CrossTradeBacktester(initial_cash=initial_cash, commission_long=commission_long, 
                                            commission_short=commission_short, short_borrow_fee_inc_rate=short_borrow_fee_inc_rate, 
                                            long_borrow_fee_inc_rate=long_borrow_fee_inc_rate)
    band_backtester = BandTradeBacktester(initial_cash=initial_cash, commission_long=commission_long, 
                                          commission_short=commission_short, short_borrow_fee_inc_rate=short_borrow_fee_inc_rate, 
                                          long_borrow_fee_inc_rate=long_borrow_fee_inc_rate)

    if strategy_name == 'adx':
        window = parameters.get('window', 14)
        short_window_indicator=f'+DI_{window}'
        long_window_indicator=f'-DI_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='adx',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (+DI_{window} vs -DI_{window})")

    elif strategy_name == 'aroon':
        period = parameters.get('period', 14)
        short_window_indicator=f'AROON_UP_{period}'
        long_window_indicator=f'AROON_DOWN_{period}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='aroon',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (AROON_DOWN_{period} vs AROON_UP_{period})")

    elif strategy_name == 'ema':
        short_window = parameters.get('short_window', 25)
        long_window = parameters.get('long_window', 75)
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
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (EMA-{short_window} vs EMA-{long_window})")

    elif strategy_name=='ichimoku':
        tenkan_period = parameters.get('tenkan_period', 9)
        kijun_period = parameters.get('kijun_period', 26)
        senkou_b_period = parameters.get('senkou_b_period', 52)
        displacement = parameters.get('displacement', 26)
        short_window_indicator=f'tenkan_sen_{tenkan_period}'
        long_window_indicator=f'kijun_sen_{kijun_period}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='ichimoku',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (tenkan_sen_{tenkan_period}, kijun_sen_{kijun_period})")

    elif strategy_name=='psar':
        af_initial = parameters.get('af_initial', 0.03)
        af_step = parameters.get('af_step', 0.03)
        af_max = parameters.get('af_max', 0.3)
        short_window_indicator="Close"
        long_window_indicator=f"PSAR_Bullish_{af_initial}_{af_step}_{af_max}"
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='psar',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (PSAR_Bullish_{af_initial}_{af_step}_{af_max} vs Close)")

    elif strategy_name == 'sma':
        short_window = parameters.get('short_window', 25)
        long_window = parameters.get('long_window', 75)
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
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (SMA-{short_window} vs SMA-{long_window})")

    elif strategy_name=='strend':
        period = parameters.get('period', 7)
        multiplier = parameters.get('multiplier', 3.0)
        short_window_indicator="Close"
        long_window_indicator=f'Supertrend_Bullish_{period}_{multiplier}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='strend',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (f'Supertrend_Bullish_{period}_{multiplier}' vs Close)")

    elif strategy_name=='trix':
        window = parameters.get('window', 7)
        short_window_indicator=f'TRIX_{window}'
        long_window_indicator=f'TRIX_SIGNAL_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='trix',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (TRIX-{window} vs TRIX_SIGNAL_{window})")

    elif strategy_name == 'wma':
        short_window = parameters.get('short_window', 25)
        long_window = parameters.get('long_window', 75)
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
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (WMA-{short_window} vs WMA-{long_window})")

    elif strategy_name=='cci':
        window = parameters.get('window', 20)
        constant = parameters.get('constant', 0.015)
        upper = parameters.get('upper', 150)
        lower = parameters.get('lower', -150)
        indicator_col=f'CCI_{window}_{constant}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='cci',
        parameters=parameters,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col)

        indicator_cols_to_plot = [f'CCI_{window}_{constant}', 'lower', 'upper']

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"CCI Threshold (CCI_{window}_{constant} {lower}/{upper})")

    elif strategy_name=='macd':
        window_fast = parameters.get('window_fast', 12)
        window_slow = parameters.get('window_slow', 26)
        window_signal = parameters.get('window_signal', 26)
        short_window_indicator=f'MACD_{window_fast}_{window_slow}'
        long_window_indicator=f'Signal_{window_signal}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='macd',
        parameters=parameters,
        figure=False)

        results, portfolio = cross_backtester.run_cross_trade(
        data=data,
        short_window_indicator=short_window_indicator,
        long_window_indicator=long_window_indicator,
        price_col=price_col)

        indicator_cols_to_plot = [short_window_indicator, long_window_indicator]

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Cross Trade (MACD_{window_fast}_{window_slow} vs MACD_Signal_{window_signal})")

    elif strategy_name=='rsi':
        window = parameters.get('window', 14)
        upper = parameters.get('upper', 80)
        lower = parameters.get('lower', 20)
        indicator_col=f'RSI_{window}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='rsi',
        parameters=parameters,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col)

        indicator_cols_to_plot = [f'RSI_{window}', 'lower', 'upper']

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"RSI Threshold (RSI_{window} {lower}/{upper})")

    elif strategy_name=='stoch':
        k_period = parameters.get('k_period', 14)
        d_period = parameters.get('d_period', 14)
        smooth_k = parameters.get('smooth_k', 14)
        upper = parameters.get('upper', 80)
        lower = parameters.get('lower', 20)
        indicator_col=f'STOCH_D_{k_period}_{d_period}_{smooth_k}'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='stoch',
        parameters=parameters,
        figure=False)

        data['upper'] = upper
        data['lower'] = lower

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col="upper",
        lower_band_col="lower",
        price_col=price_col)

        indicator_cols_to_plot = [f'STOCH_D_{k_period}_{d_period}_{smooth_k}', 'lower', 'upper']

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"STOCH Threshold (STOCH_D_{k_period}_{d_period}_{smooth_k} {lower}/{upper})")

    elif strategy_name=='bollin':
        window = parameters.get('window', 20)
        num_std = parameters.get('num_std', 2)
        indicator_col='Close'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='bollin',
        parameters=parameters,
        figure=False)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'BB_Upper_{window}_{num_std}',
        lower_band_col=f'BB_Middle_{window}',
        price_col=price_col)

        indicator_cols_to_plot = ['Close', f'BB_Upper_{window}_{num_std}', f'BB_Middle_{window}']

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Bollinger Threshold (Close BB_Upper_{window}_{num_std}/BB_Middle_{window})")

    elif strategy_name=='kelt':
        ema_window = parameters.get('ema_window', 20)
        atr_window = parameters.get('atr_window', 10)
        atr_multiplier = parameters.get('atr_multiplier', 2.0)
        indicator_col='Close'
        price_col='Close'

        data, columns, fig = compute_indicator(
        data=data,
        indicator='kelt',
        parameters=parameters,
        figure=False)

        results, portfolio = band_backtester.run_band_trade(
        data=data,
        indicator_col=indicator_col,
        upper_band_col=f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}',
        lower_band_col=f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}',
        price_col=price_col)

        indicator_cols_to_plot = ['Close', f'KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}', f'KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier}']

        fig = plotter.plot_results(
        data_df=data,
        history_df=portfolio,
        price_col=price_col,
        indicator_cols=indicator_cols_to_plot, 
        title=f"Keltner Threshold (Close KELT_Upper_{ema_window}_{atr_window}_{atr_multiplier}/KELT_Middle_{ema_window}_{atr_window}_{atr_multiplier})")


    return results, portfolio, fig