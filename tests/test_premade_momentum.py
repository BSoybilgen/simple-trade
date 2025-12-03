"""
Tests for momentum premade strategies.
"""
import pandas as pd
from simple_trade.run_premade_strategies import run_premade_trade


class TestMomentumStrategies:
    """Test all momentum trading strategies"""

    def test_awo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test AWO (Awesome Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_window': 5, 'slow_window': 34})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'awo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
        assert 'total_return_pct' in results

    def test_bop_strategy(self, sample_ohlcv_data, default_parameters):
        """Test BOP (Balance of Power) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'smooth': True})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bop', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_bop_strategy_unsmoothed(self, sample_ohlcv_data, default_parameters):
        """Test BOP strategy without smoothing"""
        params = default_parameters.copy()
        params.update({'window': 14, 'smooth': False})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'bop', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cmo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CMO (Chande Momentum Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 50, 'lower': -50})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cmo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_cog_strategy(self, sample_ohlcv_data, default_parameters):
        """Test COG (Center of Gravity) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10, 'signal_window': 3})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'cog', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_crs_strategy(self, sample_ohlcv_data, default_parameters):
        """Test CRS (Connors RSI) strategy"""
        params = default_parameters.copy()
        params.update({
            'rsi_window': 3, 'streak_window': 2, 'rank_window': 100,
            'upper': 90, 'lower': 10
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'crs', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_dpo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test DPO (Detrended Price Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'dpo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_eri_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ERI (Elder-Ray Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'eri', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_fis_strategy(self, sample_ohlcv_data, default_parameters):
        """Test FIS (Fisher Transform) strategy"""
        params = default_parameters.copy()
        params.update({'window': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'fis', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_imi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test IMI (Intraday Momentum Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 70, 'lower': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'imi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_kst_strategy(self, sample_ohlcv_data, default_parameters):
        """Test KST (Know Sure Thing) strategy"""
        params = default_parameters.copy()
        params.update({'signal': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'kst', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_lsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test LSI (Laguerre RSI) strategy"""
        params = default_parameters.copy()
        params.update({'gamma': 0.5, 'upper': 80, 'lower': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'lsi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_msi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test MSI (Momentum Strength Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'power': 1.0, 'upper': 70, 'lower': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'msi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_pgo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PGO (Pretty Good Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': 3.0, 'lower': -3.0})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'pgo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ppo_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PPO (Percentage Price Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({'fast_window': 12, 'slow_window': 26, 'signal_window': 9})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ppo', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_psy_strategy(self, sample_ohlcv_data, default_parameters):
        """Test PSY (Psychological Line) strategy"""
        params = default_parameters.copy()
        params.update({'window': 12, 'upper': 75, 'lower': 25})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'psy', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_qst_strategy(self, sample_ohlcv_data, default_parameters):
        """Test QST (Qstick) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'qst', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rmi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RMI (Relative Momentum Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 20, 'momentum_period': 5, 'upper': 70, 'lower': 30})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rmi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_roc_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ROC (Rate of Change) strategy"""
        params = default_parameters.copy()
        params.update({'window': 12})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_rvg_strategy(self, sample_ohlcv_data, default_parameters):
        """Test RVG (Relative Vigor Index) strategy"""
        params = default_parameters.copy()
        params.update({'window': 10})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'rvg', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_sri_strategy(self, sample_ohlcv_data, default_parameters):
        """Test SRI (Stochastic RSI) strategy"""
        params = default_parameters.copy()
        params.update({
            'rsi_window': 14, 'stoch_window': 14, 'k_window': 3, 'd_window': 3,
            'upper': 80, 'lower': 20
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'sri', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_stc_strategy(self, sample_ohlcv_data, default_parameters):
        """Test STC (Schaff Trend Cycle) strategy"""
        params = default_parameters.copy()
        params.update({
            'window_fast': 23, 'window_slow': 50, 'cycle': 10, 'smooth': 3,
            'upper': 75, 'lower': 25
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'stc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_tsi_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TSI (True Strength Index) strategy"""
        params = default_parameters.copy()
        params.update({'slow': 25, 'fast': 13})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'tsi', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ttm_strategy(self, sample_ohlcv_data, default_parameters):
        """Test TTM (TTM Squeeze) strategy"""
        params = default_parameters.copy()
        params.update({
            'length': 20, 'std_dev': 2.0, 'atr_length': 20,
            'atr_multiplier': 1.5, 'smooth': 3
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ttm', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_ult_strategy(self, sample_ohlcv_data, default_parameters):
        """Test ULT (Ultimate Oscillator) strategy"""
        params = default_parameters.copy()
        params.update({
            'short_window': 7, 'medium_window': 14, 'long_window': 28,
            'upper': 70, 'lower': 30
        })
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'ult', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_vor_strategy(self, sample_ohlcv_data, default_parameters):
        """Test VOR (Vortex Indicator) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'vor', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_wil_strategy(self, sample_ohlcv_data, default_parameters):
        """Test WIL (Williams %R) strategy"""
        params = default_parameters.copy()
        params.update({'window': 14, 'upper': -20, 'lower': -80})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'wil', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)

    def test_wad_strategy(self, sample_ohlcv_data, default_parameters):
        """Test WAD (Williams Accumulation/Distribution) strategy"""
        params = default_parameters.copy()
        params.update({'sma_period': 20})
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'wad', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)


class TestMomentumStrategiesWithTradingTypes:
    """Test momentum strategies with different trading types"""

    def test_momentum_strategy_long_only(self, sample_ohlcv_data, default_parameters):
        """Test momentum strategy with long-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'long'
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'short').any()

    def test_momentum_strategy_short_only(self, sample_ohlcv_data, default_parameters):
        """Test momentum strategy with short-only trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'short'
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert not (portfolio['PositionType'] == 'long').any()

    def test_momentum_strategy_mixed(self, sample_ohlcv_data, default_parameters):
        """Test momentum strategy with mixed trading"""
        params = default_parameters.copy()
        params['trading_type'] = 'mixed'
        
        results, portfolio, fig = run_premade_trade(sample_ohlcv_data, 'roc', params)
        
        assert isinstance(results, dict)
        assert isinstance(portfolio, pd.DataFrame)
